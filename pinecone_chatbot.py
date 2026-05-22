import os
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

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
    print("ERROR: OPENAI_API_KEY not found")
    exit()

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not found")
    exit()

# ==========================================
# LOAD OPENAI
# ==========================================

try:
    print("Connecting to OpenAI...")

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("OpenAI connected successfully.\n")

except Exception as e:
    print("OPENAI ERROR:")
    print(e)
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
# CONNECT TO PINECONE
# ==========================================

try:
    print("Connecting to Pinecone...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index = pc.Index(INDEX_NAME)

    print("Pinecone connected successfully.\n")

except Exception as e:
    print("PINECONE CONNECTION ERROR:")
    print(e)
    exit()

# ==========================================
# READY MESSAGE
# ==========================================

print("==========================================")
print("      Guru Spiritual AI Assistant")
print("==========================================")
print("Type 'exit' to quit\n")

# ==========================================
# CHAT LOOP
# ==========================================

while True:

    try:

        question = input("You: ").strip()

        # ==========================================
        # EXIT
        # ==========================================

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        # ==========================================
        # EMPTY CHECK
        # ==========================================

        if len(question) == 0:
            continue

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

        # ==========================================
        # NO RESULTS
        # ==========================================

        if len(matches) == 0:

            print("\nAI:\n")
            print("I could not find relevant spiritual knowledge.")
            print("\n" + "=" * 60 + "\n")

            continue

        # ==========================================
        # BUILD CONTEXT
        # ==========================================

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

        # ==========================================
        # SYSTEM PROMPT
        # ==========================================

        SYSTEM_PROMPT = """
You are an advanced spiritual AI guide inspired by Yogiraj Vethathiri Maharishi.

Your responses should feel similar to ChatGPT:
- Natural
- Human
- Calm
- Intelligent
- Deep but easy to understand

RESPONSE STYLE RULES:
- Use short paragraphs
- Avoid giant walls of text
- Use bullet points when useful
- Make answers conversational
- Keep responses beautifully formatted
- Explain clearly and deeply
- Use practical examples and analogies
- Sound emotionally intelligent and peaceful

WHEN APPROPRIATE:
- Start with a direct answer
- Then explain gradually
- End with a reflective takeaway

DO NOT:
- Repeat the same ideas
- Sound robotic
- Use overly complex spiritual language
- Write one massive paragraph

Use the provided context as the main source of truth.
If context is limited, answer honestly.
"""

        # ==========================================
        # USER PROMPT
        # ==========================================

        USER_PROMPT = f"""
SPIRITUAL KNOWLEDGE CONTEXT:
{context}

USER QUESTION:
{question}

Give a thoughtful, structured, natural, and emotionally engaging answer.
"""

        # ==========================================
        # ASK OPENAI
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

        # ==========================================
        # PRINT RESPONSE
        # ==========================================

        print("\nAI:\n")

        print(answer)

        # ==========================================
        # SHOW SOURCES
        # ==========================================

        if len(sources) > 0:

            print("\nSources Used:")

            for i, source in enumerate(sources, start=1):
                print(f"{i}. {source}")

        print("\n" + "=" * 60 + "\n")

    except KeyboardInterrupt:

        print("\nProgram stopped safely.")
        break

    except Exception as e:

        print("\nERROR:")
        print(e)
        print()