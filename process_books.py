import os
import json
import fitz
import easyocr

# =========================
# FOLDERS
# =========================

PDF_FOLDER = "books"
TEMP_FOLDER = "temp"

# Create temp folder safely
if os.path.exists(TEMP_FOLDER):

    if not os.path.isdir(TEMP_FOLDER):
        os.remove(TEMP_FOLDER)

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# =========================
# OCR READER
# =========================

reader = easyocr.Reader(['en'])

# =========================
# STORAGE
# =========================

all_chunks = []

# =========================
# PDF FILES
# =========================

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

# =========================
# PROCESS PDFS
# =========================

for file in pdf_files:

    print(f"\nProcessing: {file}")

    pdf_path = os.path.join(PDF_FOLDER, file)

    try:

        doc = fitz.open(pdf_path)

        text = ""

        # =========================
        # PROCESS PAGES
        # =========================

        for page_num in range(len(doc)):

            page = doc.load_page(page_num)

            pix = page.get_pixmap()

            image_path = os.path.join(
                TEMP_FOLDER,
                f"{file}_{page_num}.png"
            )

            pix.save(image_path)

            result = reader.readtext(image_path, detail=0)

            page_text = " ".join(result)

            text += page_text + "\n"

            # Safe delete
            try:
                os.remove(image_path)
            except:
                pass

        # =========================
        # TEXT INFO
        # =========================

        print("TEXT LENGTH:", len(text))

        if len(text.strip()) == 0:
            print("NO TEXT EXTRACTED")
            continue

        # =========================
        # CHUNKING
        # =========================

        chunk_size = 500

        chunks = []

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            chunks.append(chunk)

        print("Chunks created:", len(chunks))

        # =========================
        # SAVE CHUNKS
        # =========================

        all_chunks.extend(chunks)

        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print("Progress saved!")

    except Exception as e:

        print("ERROR:", e)

# =========================
# FINAL OUTPUT
# =========================

print("\nTOTAL CHUNKS:", len(all_chunks))

if len(all_chunks) > 0:

    print("\nFIRST CHUNK:\n")

    print(all_chunks[0])

print("\nAll chunks saved to chunks.json")