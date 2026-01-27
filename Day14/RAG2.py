import re
from typing import List

import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def standardize_query(query: str) -> str:
    """
    Standardize the query string:
    - Strip extra whitespace
    - Convert to lowercase
    - Remove unnecessary special characters
    """
    query = query.strip().lower()
    query = re.sub(r"[^\w\s]", "", query)  # remove punctuation
    return query


def create_documents(texts: List[str]) -> List[Document]:
    """Convert a list of strings into LangChain Document objects."""
    return [Document(page_content=text) for text in texts]


def build_vector_store(docs: List[Document], embedding_model) -> FAISS:
    """Create a FAISS vector store from documents and embeddings."""
    return FAISS.from_documents(docs, embedding_model)


def run_qa(query: str, qa_chain: RetrievalQA, vector_store: FAISS, top_k: int = 3):
    """
    Perform retrieval and QA:
    - Returns top_k retrieved documents and the generated answer
    """
    standardized_query = standardize_query(query)
    retrieved_docs = vector_store.similarity_search(standardized_query, k=top_k)
    answer = qa_chain.run(standardized_query)
    return retrieved_docs, answer


# -----------------------------
# Main pipeline
# -----------------------------

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

docs = create_documents(documents)

# Vector store
faiss_vector_store = build_vector_store(docs, embedding_model)

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_vector_store.as_retriever()
)

# Example query
query_text = "How does RAG work?"

retrieved_docs, answer = run_qa(query_text, qa_chain, faiss_vector_store, top_k=3)

# Output
print("Retrieved Documents:")
for doc in retrieved_docs:
    print("-", doc.page_content)

print("\nGenerated Answer:")
print(answer)
