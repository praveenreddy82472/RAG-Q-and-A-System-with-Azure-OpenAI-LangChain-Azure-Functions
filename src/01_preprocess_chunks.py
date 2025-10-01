# preprocess_chunks.py
"""
This script ingests documents (PDF, DOCX, TXT), splits them into
overlapping text chunks for RAG, and saves them into a JSON file.
Output file will always be stored in the project root under /output.
"""

import os
import uuid
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables if needed (for future use in embeddings, etc.)
load_dotenv()

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
# Project root directory (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Input folder where documents are stored
INPUT_DIR = os.path.join(BASE_DIR, "docs_to_ingest")

# Output file where processed chunks will be saved
OUT_FILE = os.path.join(BASE_DIR, "output", "chunks.json")

# --------------------------------------------------------------------
# Chunking setup
# --------------------------------------------------------------------
# Recursive splitter ensures text is cut at logical boundaries if possible
chunker = RecursiveCharacterTextSplitter(
    chunk_size=800,   # max characters per chunk
    chunk_overlap=100 # overlap between chunks to preserve context
)

# --------------------------------------------------------------------
# File readers
# --------------------------------------------------------------------
def read_pdf(path):
    """Extract text from a PDF file."""
    text = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text.append(page.extract_text() or "")
    return "\n".join(text)

def read_docx(path):
    """Extract text from a DOCX file."""
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def read_txt(path):
    """Read plain text from a TXT file."""
    with open(path, "r", encoding="utf8") as f:
        return f.read()

# --------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------
all_chunks = []

for fname in sorted(os.listdir(INPUT_DIR)):
    path = os.path.join(INPUT_DIR, fname)

    # Choose parser depending on file extension
    if fname.lower().endswith(".pdf"):
        text = read_pdf(path)
    elif fname.lower().endswith(".docx"):
        text = read_docx(path)
    elif fname.lower().endswith(".txt"):
        text = read_txt(path)
    else:
        print("Skipping unsupported file:", fname)
        continue

    # Split into chunks
    chunks = chunker.split_text(text)

    # Save chunks with metadata
    for i, c in enumerate(chunks):
        all_chunks.append({
            "id": str(uuid.uuid4()),  # unique chunk ID
            "source": fname,          # original file name
            "chunk_index": i,         # chunk order in file
            "content": c              # actual text content
        })

print(f"Prepared {len(all_chunks)} chunks from {len(os.listdir(INPUT_DIR))} files")

# --------------------------------------------------------------------
# Save results
# --------------------------------------------------------------------
# Ensure output folder exists
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# Write all chunks into JSON file
with open(OUT_FILE, "w", encoding="utf8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Chunks written to {OUT_FILE}")
