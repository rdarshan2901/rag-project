import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "knowledge_base.index"
SAVE_CHUNKS_FILE = "knowledge_chunks.json"

# Check chunks file exists
if not os.path.exists(CHUNKS_FILE):
    print("ERROR: chunks.json not found")
    exit()

# Load chunks
try:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"TOTAL CHUNKS LOADED: {len(chunks)}")

except Exception as e:
    print("ERROR LOADING CHUNKS:", e)
    exit()

# Remove empty chunks
clean_chunks = []

for chunk in chunks:
    if isinstance(chunk, str):
        chunk = chunk.strip()

        if len(chunk) > 20:
            clean_chunks.append(chunk)

print(f"CLEAN CHUNKS: {len(clean_chunks)}")

if len(clean_chunks) == 0:
    print("ERROR: No valid chunks found")
    exit()

# Load embedding model
try:
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

except Exception as e:
    print("MODEL LOAD ERROR:", e)
    exit()

# Create embeddings
try:
    print("Creating embeddings...")
    
    embeddings = model.encode(
        clean_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

except Exception as e:
    print("EMBEDDING ERROR:", e)
    exit()

# Convert datatype
embeddings = embeddings.astype("float32")

# Create FAISS index
try:
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

except Exception as e:
    print("FAISS ERROR:", e)
    exit()

# Save FAISS index
try:
    faiss.write_index(index, INDEX_FILE)

except Exception as e:
    print("INDEX SAVE ERROR:", e)
    exit()

# Save cleaned chunks
try:
    with open(SAVE_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_chunks, f, ensure_ascii=False, indent=2)

except Exception as e:
    print("CHUNK SAVE ERROR:", e)
    exit()

print("\nSUCCESS!")
print("Embeddings created successfully")
print("Total vectors:", index.ntotal)
print("Index saved as:", INDEX_FILE)
print("Chunks saved as:", SAVE_CHUNKS_FILE)