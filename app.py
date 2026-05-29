import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

# ==========================================
# CONFIG
# ==========================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "guru-rag"

TOP_K = 5

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OPENAI_MODEL = "gpt-4.1-mini"

# ==========================================
# SAFETY CHECKS
# ==========================================

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

# ==========================================
# LOAD SERVICES
# ==========================================

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

print("Connecting to OpenAI...")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="Guru Spiritual AI API",
    description="AI backend inspired by Yogiraj Vethathiri Maharishi",
    version="2.0.0"
)

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):
    message: str
    book: str

# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Guru Spiritual AI API is live"
    }

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# ==========================================
# CHAT ROUTE
# ==========================================

@app.post("/chat")
def chat(req: ChatRequest):

    try:

        question = req.message.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )

        # ==========================================
        # CREATE EMBEDDING
        # ==========================================

        question_embedding = model.encode(question).tolist()

        # ==========================================
        # SEARCH PINECONE
        # ==========================================

        results = index.query(
            vector=question_embedding,
            top_k=TOP_K,
            include_metadata=True
        )

        matches = results.get("matches", [])

        relevant_chunks = []

        sources = []

        for match in matches:

            metadata = match.get("metadata", {})

            text = metadata.get("text", "")

            source = metadata.get("source", "Unknown Source")

            if text:
                relevant_chunks.append(text)

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(relevant_chunks)

        if not context:
            context = (
                "No relevant spiritual documents were found. "
                "Answer honestly using general wisdom."
            )

        # ==========================================
        # SYSTEM PROMPT
        # ==========================================

        SYSTEM_PROMPT = """
You are an advanced spiritual AI guide inspired by Yogiraj Vethathiri Maharishi.

Your responses must feel like ChatGPT:
- Natural and human
- Clean and beautifully formatted
- Easy to read
- Conversational
- Structured

IMPORTANT RESPONSE STYLE:
- Use short paragraphs
- Use bullet points when useful
- Avoid giant walls of text
- Keep the tone calm, wise, warm, and intelligent
- Explain deeply but clearly
- Make answers engaging and modern
- Add spacing between ideas
- Use practical examples when helpful

WHEN APPROPRIATE:
- Start with a direct answer
- Then explain step-by-step
- End with a peaceful reflection or takeaway

DO NOT:
- Write one massive paragraph
- Repeat ideas unnecessarily
- Sound robotic
- Overuse spiritual jargon

Use the provided context as the primary source of truth.
If context is limited, be honest about uncertainty.
"""

        # ==========================================
        # USER PROMPT
        # ==========================================

        USER_PROMPT = f"""
CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear, beautiful, well-structured response.
"""

        # ==========================================
        # OPENAI RESPONSE
        # ==========================================

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT
                }
            ],
            temperature=0.7,
            max_tokens=700
        )

        answer = response.choices[0].message.content
        final_answer = answer + f"\n\nAnswer taken from this book: {req.book}"

        # ==========================================
        # RETURN RESPONSE
        # ==========================================

        return {
            "success": True,
            "question": question,
            "answer": final_answer
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )