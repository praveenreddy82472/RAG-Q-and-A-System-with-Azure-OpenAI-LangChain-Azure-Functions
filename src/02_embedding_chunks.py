# 02_embedding_chunks.py
import os
import json
import requests
from tqdm import tqdm

# -------------------------------
# Azure OpenAI Embedding Config
# -------------------------------
from dotenv import load_dotenv
import os

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_emb_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_emb_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_emb_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_emb_API_VERSION")

# Embedding API URL
EMBED_URL = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/embeddings?api-version={API_VERSION}"

HEADERS = {
    "api-key": AZURE_OPENAI_KEY,
    "Content-Type": "application/json"
}

# -------------------------------
# Load preprocessed chunks
# -------------------------------
CHUNKS_FILE = "../output/chunks.json"  # adjust path if needed
with open(CHUNKS_FILE, "r", encoding="utf8") as f:
    all_chunks = json.load(f)

print(f"Loaded {len(all_chunks)} chunks")

# -------------------------------
# Embed all chunks
# -------------------------------
all_embeddings = []

for chunk in tqdm(all_chunks, desc="Embedding chunks"):
    try:
        data = {"input": chunk["content"]}
        response = requests.post(EMBED_URL, headers=HEADERS, json=data)
        response.raise_for_status()  # raises error if request failed
        embedding = response.json()["data"][0]["embedding"]

        all_embeddings.append({
            "id": chunk["id"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "content":chunk["content"],
            "embedding": embedding,
            "content_vector":embedding
        })
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for chunk {chunk['id']}: {e}")
        print(response.text)
    except KeyError:
        print(f"Unexpected response for chunk {chunk['id']}: {response.text}")

# -------------------------------
# Save embeddings
# -------------------------------
OUTPUT_FILE = "../output/chunks_embeddings.json"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf8") as f:
    json.dump(all_embeddings, f, ensure_ascii=False, indent=2)

print(f"Saved embeddings for {len(all_embeddings)} chunks to {OUTPUT_FILE}")
