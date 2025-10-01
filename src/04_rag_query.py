import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.chains import RetrievalQA

# Load environment variables
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
    embedding_function=embeddings.embed_query,  # Use embedding function explicitly
    content_field="content",                     # your field containing the text
    vector_field="content_vector",                 # your vector field
    top_k=3,  # retrieve top 3 relevant chunks
)

# ---------------------------
# Retriever
# ---------------------------
retriever = vector_store.as_retriever(
    search_type="similarity"  # or "hybrid" depending on your use case
)

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
# RetrievalQA chain
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# ---------------------------
# Run a query
# ---------------------------
query = "What services does Praveen Industries provide?"
result = qa_chain.invoke(query)

print("\nAnswer:\n", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("id"))
