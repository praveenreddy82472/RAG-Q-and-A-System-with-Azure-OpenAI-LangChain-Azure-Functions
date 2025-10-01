import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.chains import RetrievalQA
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Initialize embeddings
# ---------------------------
embeddings = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_emb_DEPLOYMENT"),
    model=os.getenv("AZURE_OPENAI_emb_EMBEDDING_MODEL"),
    api_key=os.getenv("AZURE_OPENAI_emb_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_emb_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_emb_API_VERSION"),
)

# ---------------------------
# Connect to Azure Cognitive Search
# ---------------------------
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    embedding_function=embeddings.embed_query,
    content_field="content",        # check your index schema!
    vector_field="content_vector",  # check your index schema!
    top_k=3
)

retriever = vector_store.as_retriever(search_type="similarity")

# ---------------------------
# Initialize LLM
# ---------------------------
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    model=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    temperature=0,
)

# ---------------------------
# Build RetrievalQA chain
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="RAG Q&A with Azure OpenAI + Cognitive Search")
# Serve frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static\index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    result = qa_chain.invoke(request.question)
    sources = [doc.metadata.get("id") for doc in result["source_documents"]]
    return {
        "question": request.question,
        "answer": result["result"],
        "sources": sources
    }

@app.get("/")
async def root():
    return {"message": "RAG Q&A API is running!"}
