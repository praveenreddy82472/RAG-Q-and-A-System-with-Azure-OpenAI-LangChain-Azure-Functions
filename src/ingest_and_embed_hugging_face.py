import os, uuid, json
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")  # Reads token from .env
embedder = SentenceTransformer(MODEL_NAME,use_auth_token=HF_TOKEN)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Function to read PDF
def read_pdf(path):
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)

# Function to chunk and embed
def chunk_and_embed(doc_text, metadata):
    chunks = splitter.split_text(doc_text)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "metadata": metadata,
            "vector": embeddings[i].tolist()
        })
    return docs

if __name__ == "__main__":
    folder = "docs_to_ingest"  # put your documents here
    all_docs = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.lower().endswith(".pdf"):
            txt = read_pdf(path)
        elif fname.lower().endswith(".txt"):
            with open(path, "r", encoding="utf8") as f:
                txt = f.read()
        else:
            continue
        docs = chunk_and_embed(txt, {"source": fname})
        all_docs.extend(docs)

    # Save embeddings locally
    with open("prepared_docs.json", "w", encoding="utf8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print("Prepared", len(all_docs), "chunks with embeddings.")
