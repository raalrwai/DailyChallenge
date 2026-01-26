import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents
documents = [
    "Large Language Models (LLMs) are transforming AI.",
    "FAISS is a powerful vector database for search.",
    "Retrieval-Augmented Generation (RAG) enhances LLM responses.",
    "Vector embeddings represent text numerically.",
    "LangChain makes working with LLMs easier.",
]

# Convert to LangChain Documents
docs = [Document(page_content=d) for d in documents]

# Create FAISS vector store
faiss_vector_store = FAISS.from_documents(docs, embedding_model)

# LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3
)

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_vector_store.as_retriever()
)

# Query
query_text = "How does RAG work?"

# Run retrieval
retrieved_docs = faiss_vector_store.similarity_search(query_text)

# Run QA
result = qa_chain.run(query_text)

# Output
print("Retrieved Documents:")
for doc in retrieved_docs:
    print("-", doc.page_content)

print("\nGenerated Answer:")
print(result)

