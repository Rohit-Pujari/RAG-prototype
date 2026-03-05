import sys
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import *

# -------------------------
# Args
# -------------------------
if len(sys.argv) < 3:
    print("Usage: python query.py <subject> <your question>")
    sys.exit(1)

SUBJECT = sys.argv[1]
QUERY = " ".join(sys.argv[2:])

DB_DIR = f"subjects/{SUBJECT}/vectordb"

# -------------------------
# Startup Info
# -------------------------
print("\n[INFO] Querying subject:", SUBJECT)
print("[INFO] Loading vector database...")
print("[INFO] Question:", QUERY, "\n")

# -------------------------
# Init Models
# -------------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

llm = Ollama(
    model=LLM_MODEL,
    temperature=0.2
)

# -------------------------
# Retrieval
# -------------------------
print("[STEP] Retrieving relevant context...")

retriever = db.as_retriever(
    search_kwargs={"k": 6}
)

docs: list[Document] = retriever.invoke(QUERY)

if not docs:
    print("[WARN] No relevant context found.")
    sys.exit(0)

print(f"[INFO] Retrieved {len(docs)} chunks\n")

# -------------------------
# Show Retrieved Context
# -------------------------
print("========= Retrieved Context =========\n")

for i, d in enumerate(docs, 1):
    meta = d.metadata
    source = meta.get("source", "unknown")
    page = meta.get("page", "N/A")
    doc_type = meta.get("type", "unknown")

    print(f"[{i}] Source: {source}")
    print(f"    Type  : {doc_type}")
    print(f"    Page  : {page}")
    print("-" * 50)

# -------------------------
# Build Prompt
# -------------------------
context_text = "\n\n".join(
    f"Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}\n{d.page_content}"
    for d in docs
)

prompt = f"""
You are an academic study assistant.

Answer the question strictly using the provided context.
If the answer is not present, say so clearly.
Cite the source and page numbers in your answer.

Context:
{context_text}

Question:
{QUERY}

Answer:
"""

print("\n[STEP] Generating answer...\n")

# -------------------------
# LLM Call
# -------------------------
response = llm.invoke(prompt)

# -------------------------
# Output
# -------------------------
print("========= Answer =========\n")
print(response)

print("\n========= Sources Used =========\n")

seen = set()
for d in docs:
    src = d.metadata.get("source")
    page = d.metadata.get("page")
    key = (src, page)

    if key not in seen:
        print(f"- {src} (page {page})")
        seen.add(key)

print("\n[✓] Query complete.\n")
