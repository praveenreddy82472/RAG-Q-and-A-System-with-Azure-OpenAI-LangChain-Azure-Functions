import os
import json
import logging
import azure.functions as func
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.chains import RetrievalQA

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
    embedding_function=embeddings,   # ✅ safer
    content_field="content",
    vector_field="content_vector",
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
# Azure Function entrypoint
# ---------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing RAG Q&A request...")

    try:
        req_body = req.get_json()
        question = req_body.get("question")

        if not question:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'question' in request body"}),
                mimetype="application/json",
                status_code=400
            )

        result = qa_chain.invoke(question)
        sources = [doc.metadata.get("id") for doc in result["source_documents"]]

        response = {
            "question": question,
            "answer": result["result"],
            "sources": sources
        }

        return func.HttpResponse(
            body=json.dumps(response),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
