from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import os, tempfile, traceback, hashlib, asyncio, time
from dotenv import load_dotenv
import httpx
import fitz  # PyMuPDF for fast PDF parsing

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Load env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEAM_TOKEN = "4989feddb4a3e9804496d494285681b5632a15eada95a81c0afe811ca23b1e4b"

app = FastAPI(title="High Accuracy Policy RAG (Fast Version)")
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

# Embedding and LLM setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="Gemma2-9b-It",
    temperature=0.1,
    model_kwargs={"top_p": 0.9}
)

class QueryPayload(BaseModel):
    documents: str
    questions: list[str]

# Vectorstore cache (in-memory)
vectorstore_cache = {}

def get_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

async def get_api_key(api_key: str = Security(api_key_header), request: Request = None):
    token = api_key.strip().lower().removeprefix("bearer ").strip('"\' ')
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token

def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 10) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [doc[i].get_text() for i in range(min(max_pages, len(doc)))]

@app.post("/api/v1/hackrx/run")
async def ask_questions(payload: QueryPayload, api_key: str = Security(get_api_key)):
    try:
        start_time = time.time()
        cache_key = get_cache_key(payload.documents)

        # Use cache if available
        if cache_key in vectorstore_cache:
            vectorstore = vectorstore_cache[cache_key]
        else:
            # Download PDF
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(payload.documents)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download PDF.")
            pdf_bytes = response.content

            # Fast PDF parse (first 10 pages)
            raw_text_pages = extract_text_from_pdf(pdf_bytes, max_pages=10)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text("\n".join(raw_text_pages))

            if not chunks:
                raise HTTPException(status_code=400, detail="Empty or unreadable document.")

            vectorstore = FAISS.from_texts(chunks, embeddings)
            vectorstore_cache[cache_key] = vectorstore

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        SYSTEM_PROMPT = (
    "You are an intelligent and thoughtful document question-answering assistant.\n\n"
    "Instructions:\n"
    "- Use the provided document context to answer questions as accurately and completely as possible.\n"
    "- Apply reasoning and synthesis if the answer is not directly stated but can be logically inferred.\n"
    "- Prefer using the document's language and terminology when responding.\n"
    "- If the answer truly cannot be found or inferred from the document, respond with: 'Not mentioned in the document.'\n"
    "- Be concise, precise, and avoid hallucinating or assuming facts not grounded in the content.\n\n"
    "Context:\n{context}"
)


        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )

        # Answer all questions in parallel
        async def answer(q):
            try:
                result = await asyncio.to_thread(qa_chain.run, q)
                cleaned = result.strip()
                print(f"Q: {q}\nA: {cleaned}\n")
                if not cleaned or cleaned.lower() in {"n/a", "not found"}:
                    return "Not mentioned in the policy."
                return cleaned.split(":", 1)[-1].strip() if ":" in cleaned else cleaned
            except Exception:
                return "Error processing question."

        answers = await asyncio.gather(*[answer(q) for q in payload.questions])

        print(f"Total time: {time.time() - start_time:.2f}s")
        return JSONResponse({"answers": answers})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/api/v1")
def root():
    return {"message": "High Accuracy Policy RAG API running fast."}
