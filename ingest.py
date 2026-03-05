import os
import sys
import hashlib
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import *

# -------------------------
# Args
# -------------------------
if len(sys.argv) < 3:
    print("Usage: python ingest.py <subject> <file1.pdf> [file2.pdf ...]")
    sys.exit(1)

SUBJECT = sys.argv[1]
FILES = sys.argv[2:]

BASE_DIR = f"subjects/{SUBJECT}"
DATA_DIR = f"{BASE_DIR}/data"
DB_DIR = f"{BASE_DIR}/vectordb"

# -------------------------
# Helpers
# -------------------------
def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------------------
# Init DB
# -------------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

# Get already-ingested file hashes
existing = db.get(include=["metadatas"])
existing_hashes = {
    m["file_hash"]
    for m in existing["metadatas"]
    if "file_hash" in m
}

print(f"[INFO] Found {len(existing_hashes)} already-ingested files")

# -------------------------
# Process Selected Files
# -------------------------
for filename in FILES:
    pdf_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(pdf_path):
        print(f"[WARN] File not found: {filename}")
        continue

    fh = file_hash(pdf_path)

    if fh in existing_hashes:
        print(f"[SKIP] Already embedded: {filename}")
        continue

    print(f"\n[INGEST] {filename}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    if "slide" in filename.lower():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=SLIDES_CHUNK_SIZE,
            chunk_overlap=SLIDES_OVERLAP
        )
        doc_type = "slides"
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXTBOOK_CHUNK_SIZE,
            chunk_overlap=TEXTBOOK_OVERLAP
        )
        doc_type = "textbook"

    chunks = splitter.split_documents(pages)

    clean_chunks = []
    for c in chunks:
        if c.page_content and len(c.page_content.strip()) > 20:
            c.metadata.update({
                "subject": SUBJECT,
                "type": doc_type,
                "source": filename,
                "file_hash": fh,
                "page": c.metadata.get("page"),
            })
            clean_chunks.append(c)

    print(f"[INFO] Embedding {len(clean_chunks)} chunks")

    for c in tqdm(clean_chunks, desc="Embedding"):
        try:
            db.add_documents([c])
        except Exception as e:
            print(f"[WARN] Skipped chunk (page {c.metadata.get('page')}): {e}")

    print(f"[✓] Completed: {filename}")

print("\n[✓] Selective ingestion finished.")
