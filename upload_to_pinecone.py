import os
import json
import time
from typing import List

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone.exceptions import PineconeException

# ==========================================
# CONFIG
# ==========================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "guru-rag"

CHUNKS_FILE = "knowledge_chunks.json"

BATCH_SIZE = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==========================================
# SAFETY CHECKS
# ==========================================

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not found")
    print("\nSet it using:")
    print('$env:PINECONE_API_KEY="your-key"')
    exit()

if not os.path.exists(CHUNKS_FILE):
    print(f"ERROR: {CHUNKS_FILE} not found")
    exit()

# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

try:
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model loaded successfully.\n")

except Exception as e:
    print("MODEL LOAD ERROR:")
    print(e)
    exit()

# ==========================================
# LOAD CHUNKS
# ==========================================

try:
    print("Loading knowledge chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("knowledge_chunks.json must contain a list")

    if len(chunks) == 0:
        raise ValueError("knowledge_chunks.json is empty")

    print(f"Loaded {len(chunks)} chunks.\n")

except Exception as e:
    print("CHUNKS LOAD ERROR:")
    print(e)
    exit()

# ==========================================
# CONNECT TO PINECONE
# ==========================================

try:
    print("Connecting to Pinecone...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index = pc.Index(INDEX_NAME)

    print("Connected to Pinecone successfully.\n")

except PineconeException as e:
    print("PINECONE CONNECTION ERROR:")
    print(e)
    exit()

except Exception as e:
    print("UNKNOWN CONNECTION ERROR:")
    print(e)
    exit()

# ==========================================
# CREATE EMBEDDINGS
# ==========================================

def create_vectors(batch_chunks: List[str], start_index: int):

    vectors = []

    try:
        embeddings = model.encode(batch_chunks)

        for i, embedding in enumerate(embeddings):

            chunk_text = batch_chunks[i]

            vectors.append({
                "id": str(start_index + i),
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk_text
            })

        return vectors

    except Exception as e:
        print(f"EMBEDDING ERROR at batch starting {start_index}")
        print(e)
        return []

# ==========================================
# UPLOAD TO PINECONE
# ==========================================

total_chunks = len(chunks)

print("Starting upload process...\n")

uploaded_count = 0

for start in range(0, total_chunks, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_chunks)

    batch = chunks[start:end]

    print(f"Processing batch {start} → {end - 1}")

    vectors = create_vectors(batch, start)

    if len(vectors) == 0:
        print("Skipping failed batch.\n")
        continue

    try:
        index.upsert(vectors=vectors)

        uploaded_count += len(vectors)

        print(f"Uploaded {len(vectors)} vectors successfully.")
        print(f"Total uploaded: {uploaded_count}/{total_chunks}\n")

        # Small delay for stability
        time.sleep(0.5)

    except PineconeException as e:
        print("PINECONE UPSERT ERROR:")
        print(e)
        print()

    except Exception as e:
        print("UNKNOWN UPLOAD ERROR:")
        print(e)
        print()

# ==========================================
# FINISHED
# ==========================================

print("====================================")
print("UPLOAD PROCESS FINISHED")
print(f"Total uploaded: {uploaded_count}/{total_chunks}")
print("====================================")