import os
import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # this will read .env from current directory

# Read variables
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX")
key = os.getenv("AZURE_SEARCH_KEY")

if not all([endpoint, index_name, key]):
    raise ValueError("Please set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, and AZURE_SEARCH_KEY in the .env file!")

# Connect to Azure Search
search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(key)
)

# Load embeddings
with open("../output/chunks_embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Allowed fields matching your index schema
allowed_fields = ["id", "content", "embedding", "content_vector"]

# Upload documents
for chunk in chunks:
    # Keep only allowed fields
    doc = {k: chunk[k] for k in allowed_fields if k in chunk}

    # Convert embeddings and content_vector to float lists
    if "embedding" in doc:
        doc["embedding"] = [float(x) for x in doc["embedding"]]
    if "content_vector" in doc:
        doc["content_vector"] = [float(x) for x in doc["content_vector"]]

    # Upload to Azure Cognitive Search
    try:
        search_client.upload_documents(documents=[doc])
        print(f"Uploaded chunk {doc.get('id')}")
    except Exception as e:
        print(f"Error uploading chunk {doc.get('id')}: {e}")

