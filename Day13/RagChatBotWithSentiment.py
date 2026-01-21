"""
Customer-Facing RAG Chatbot with Sentiment Classification
- RAG pipeline using Pinecone + OpenAI embeddings
- Two sentiment models:
    1. TF-IDF + Logistic Regression (CPU-friendly)
    2. DistilBERT (GPU, optional)
- Dynamic prompt restructuring based on sentiment
"""

import os
from dotenv import load_dotenv
import re
from typing import List
import numpy as np

# ML packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optional transformer model
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    USE_TRANSFORMER = True
except ImportError:
    USE_TRANSFORMER = False

# OpenAI & Pinecone
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# -----------------------------
# 1a. Preprocessing
# -----------------------------
def preprocess_text(text: str) -> str:
    """Basic text cleaning for sentiment classification."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# 1b. TF-IDF + Logistic Regression
# -----------------------------
def train_simple_sentiment(texts: List[str], labels: List[int]):
    """
    Trains a simple TF-IDF + Logistic Regression pipeline.
    labels: 0 = negative, 1 = neutral, 2 = positive
    """
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=500))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("[TFIDF MODEL] Classification report:")
    print(classification_report(y_test, preds))
    return pipeline

# -----------------------------
# 1c. Transformer-based classifier (optional GPU)
# -----------------------------
if USE_TRANSFORMER:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    transformer_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def predict_sentiment_transformer(text: str):
    """
    Predict sentiment with DistilBERT.
    Returns 0 = negative, 1 = positive
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        return pred, probs[0][pred].item()

# -----------------------------
# 2. RAG pipeline helpers (from previous example)
# -----------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    """Chunk text with overlap"""
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({
            "chunk_id": chunk_id,
            "text": text[start:end]
        })
        start = end - overlap
        chunk_id += 1
    return chunks

def embed_texts(texts):
    """Convert list of texts to embedding vectors using OpenAI"""
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]

def embed_query(query):
    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
    return resp.data[0].embedding

def retrieve_similar_chunks(query_embedding, top_k=4):
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results.matches

# -----------------------------
# 3. Dynamic prompt based on sentiment
# -----------------------------
def build_prompt(user_query: str, retrieved_chunks, sentiment_label: int):
    """
    Build LLM prompt. Adjust tone based on sentiment:
    - 0 = negative → empathetic
    - 1 = neutral → standard
    - 2 = positive → upbeat
    """
    context = "\n\n".join([chunk.metadata["text"] for chunk in retrieved_chunks])

    if sentiment_label == 0:
        tone_instruction = "The user seems frustrated or upset. Respond empathetically and clearly."
    elif sentiment_label == 1:
        tone_instruction = "Respond politely and factually."
    else:
        tone_instruction = "The user seems happy. Respond positively and cheerfully."

    prompt = f"""
{tone_instruction}
Use the context below to answer the question. Do not make up facts.

Context:
{context}

Question:
{user_query}

Answer:
"""
    return prompt

def generate_answer(prompt):
    """Call OpenAI GPT to generate response"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return resp.choices[0].message.content

# -----------------------------
# 4. Full query pipeline
# -----------------------------
def answer_user_query(user_query: str, sentiment_model_pipeline=None):
    """End-to-end RAG + sentiment response"""
    preprocessed = preprocess_text(user_query)
    
    # 1. Predict sentiment (TF-IDF if available, else DistilBERT if GPU)
    if sentiment_model_pipeline:
        sentiment_label = sentiment_model_pipeline.predict([preprocessed])[0]
    elif USE_TRANSFORMER:
        sentiment_label, _ = predict_sentiment_transformer(preprocessed)
    else:
        sentiment_label = 1  # default neutral

    # 2. Embed query and retrieve chunks
    q_emb = embed_query(user_query)
    chunks = retrieve_similar_chunks(q_emb, top_k=4)

    # 3. Build prompt dynamically
    prompt = build_prompt(user_query, chunks, sentiment_label)

    # 4. Generate answer
    answer = generate_answer(prompt)

    return answer, sentiment_label

# -----------------------------
# 5. Example interactive session
# -----------------------------
if __name__ == "__main__":
    # For demonstration, train a simple sentiment classifier
    sample_texts = ["I love this product!", "This is terrible", "Meh, it's okay"]
    sample_labels = [2, 0, 1]
    tfidf_pipeline = train_simple_sentiment(sample_texts, sample_labels)

    print("Ready for interactive chat. Type 'exit' to quit.")
    while True:
        user_q = input("\nYou: ")
        if user_q.lower() == "exit":
            break
        ans, sent = answer_user_query(user_q, sentiment_model_pipeline=tfidf_pipeline)
        print(f"Bot (sentiment={sent}): {ans}")
