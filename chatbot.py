import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================
# CONFIG
# =========================

API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_FILE = "knowledge_base.index"
CHUNKS_FILE = "knowledge_chunks.json"

TOP_K = 3

# =========================
# CHECK API KEY
# =========================

if not API_KEY:
    print("ERROR: OPENAI_API_KEY not found")
    print("\nSet it in PowerShell using:")
    print('$env:OPENAI_API_KEY="your-api-key"')
    exit()

# =========================
# CHECK FILES
# =========================

if not os.path.exists(INDEX_FILE):
    print(f"ERROR: {INDEX_FILE} not found")
    exit()

if not os.path.exists(CHUNKS_FILE):
    print(f"ERROR: {CHUNKS_FILE} not found")
    exit()

# =========================
# LOAD OPENAI
# =========================

try:
    client = OpenAI(api_key=API_KEY)

except Exception as e:
    print("OPENAI ERROR:", e)
    exit()

# =========================
# LOAD EMBEDDING MODEL
# =========================

try:
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

except Exception as e:
    print("MODEL LOAD ERROR:", e)
    exit()

# =========================
# LOAD FAISS INDEX
# =========================

try:
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)

except Exception as e:
    print("FAISS LOAD ERROR:", e)
    exit()

# =========================
# LOAD KNOWLEDGE CHUNKS
# =========================

try:
    print("Loading knowledge chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

except Exception as e:
    print("CHUNKS LOAD ERROR:", e)
    exit()

# =========================
# READY MESSAGE
# =========================

print("\n====================================")
print(" Spiritual AI Assistant is Ready ")
print(" Type 'exit' to quit")
print("====================================\n")

# =========================
# CHAT LOOP
# =========================

while True:

    try:
        # User input
        question = input("You: ").strip()

        # Exit condition
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Skip empty input
        if len(question) == 0:
            continue

        # =========================
        # CREATE QUESTION EMBEDDING
        # =========================

        question_embedding = model.encode(
            [question],
            convert_to_numpy=True
        )

        question_embedding = question_embedding.astype("float32")

        # =========================
        # SEARCH RELEVANT CHUNKS
        # =========================

        distances, indices = index.search(
            question_embedding,
            TOP_K
        )

        relevant_chunks = []

        for idx in indices[0]:

            if 0 <= idx < len(chunks):
                relevant_chunks.append(chunks[idx])

        # Combine context
        context = "\n\n".join(relevant_chunks)

        # =========================
        # ASK OPENAI
        # =========================

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a calm spiritual AI assistant. "
                        "Answer clearly, peacefully, and only using the provided context. "
                        "If the answer is not in the context, say you do not know."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{question}
"""
                }
            ],
            temperature=0.7
        )

        # Extract answer
        answer = response.choices[0].message.content

        # =========================
        # PRINT ANSWER
        # =========================

        print("\nAI:")
        print(answer)

        print("\n" + "=" * 50 + "\n")

    except KeyboardInterrupt:
        print("\nProgram stopped safely.")
        break

    except Exception as e:
        print("\nERROR:", e)
        print()