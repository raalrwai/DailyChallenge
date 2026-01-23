"""
Customer-Facing RAG Chatbot with Sentiment Classification

Capabilities:
- Retrieval-Augmented Generation using Pinecone + OpenAI embeddings
- Dual sentiment classification strategies:
    1. TF-IDF + Logistic Regression (CPU-friendly, deterministic)
    2. DistilBERT sentiment classifier (GPU-accelerated, optional)
- Dynamic prompt restructuring based on detected sentiment

Intended usage:
- Lightweight customer support assistant
- Tone-aware response generation
- Safe default behavior when ML components are unavailable
"""

import os
import re
import logging
from typing import List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Environment / Config
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

DEFAULT_TOP_K = 4
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Sentiment label conventions
NEGATIVE = 0
NEUTRAL = 1
POSITIVE = 2

# -----------------------------
# ML packages (CPU)
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# Optional transformer model
# -----------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    USE_TRANSFORMER = torch.cuda.is_available()
except ImportError:
    USE_TRANSFORMER = False

# -----------------------------
# External services
# -----------------------------
from openai import OpenAI
from pinecone import Pinecone

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# -----------------------------
# 1. Text preprocessing
# -----------------------------
def preprocess_text(text: str) -> str:
    """
    Normalize user input for sentiment classification.
    This should be lightweight and deterministic.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# 2. TF-IDF sentiment model
# -----------------------------
def train_simple_sentiment(
    texts: List[str],
    labels: List[int]
) -> Pipeline:
    """
    Train a TF-IDF + Logistic Regression classifier.

    Labels:
        0 = negative
        1 = neutral
        2 = positive
    """
    logger.info("Training TF-IDF sentiment model")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=500, n_jobs=1))
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    logger.info("TF-IDF sentiment classification report:\n%s",
                classification_report(y_test, preds))

    return pipeline

# -----------------------------
# 3. Transformer-based sentiment (optional)
# -----------------------------
if USE_TRANSFORMER:
    logger.info("Loading DistilBERT sentiment model (GPU enabled)")
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ).cuda()


def predict_sentiment_transformer(text: str) -> Tuple[int, float]:
    """
    Predict sentiment using DistilBERT.
    Returns:
        label: 0 (negative) or 1 (positive)
        confidence: softmax probability
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True
    ).to("cuda")

    with torch.no_grad():
        outputs = transformer_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        label = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, label].item())

    return label, confidence

# -----------------------------
# 4. RAG helpers
# -----------------------------
def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
):
    """Split long documents into overlapping chunks."""
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(
            {"chunk_id": chunk_id, "text": text[start:end]}
        )
        start = max(end - overlap, 0)
        chunk_id += 1

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for documents."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def embed_query(query: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    return response.data[0].embedding


def retrieve_similar_chunks(
    query_embedding: List[float],
    top_k: int = DEFAULT_TOP_K
):
    """Retrieve semantically similar chunks from Pinecone."""
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = []
    for match in results.matches:
        if match.metadata and "text" in match.metadata:
            matches.append(match)

    return matches

# -----------------------------
# 5. Prompt construction
# -----------------------------
def build_prompt(
    user_query: str,
    retrieved_chunks,
    sentiment_label: int
) -> str:
    """
    Build an LLM prompt with sentiment-aware tone control.
    """
    context = "\n\n".join(
        chunk.metadata["text"] for chunk in retrieved_chunks
    )

    if sentiment_label == NEGATIVE:
        tone = "The user seems frustrated. Respond empathetically and calmly."
    elif sentiment_label == POSITIVE:
        tone = "The user seems positive. Respond helpfully and enthusiastically."
    else:
        tone = "Respond professionally and clearly."

    return f"""
{tone}

Use the context below to answer the question.
If the answer is not contained in the context, say you do not know.

Context:
{context}

Question:
{user_query}

Answer:
""".strip()

# -----------------------------
# 6. LLM invocation
# -----------------------------
def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# -----------------------------
# 7. End-to-end pipeline
# -----------------------------
def answer_user_query(
    user_query: str,
    sentiment_model_pipeline: Optional[Pipeline] = None
) -> Tuple[str, int]:
    """
    Full RAG + sentiment-aware response generation.
    """
    cleaned = preprocess_text(user_query)

    # Sentiment selection logic
    if sentiment_model_pipeline:
        sentiment = int(
            sentiment_model_pipeline.predict([cleaned])[0]
        )
    elif USE_TRANSFORMER:
        raw_label, _ = predict_sentiment_transformer(cleaned)
        sentiment = POSITIVE if raw_label == 1 else NEGATIVE
    else:
        sentiment = NEUTRAL

    query_embedding = embed_query(user_query)
    chunks = retrieve_similar_chunks(query_embedding)

    prompt = build_prompt(user_query, chunks, sentiment)
    answer = generate_answer(prompt)

    return answer, sentiment

# -----------------------------
# 8. Interactive demo
# -----------------------------
if __name__ == "__main__":
    logger.info("Bootstrapping demo sentiment model")

    sample_texts = [
        "I love this product",
        "This is terrible",
        "It's okay, nothing special"
    ]
    sample_labels = [POSITIVE, NEGATIVE, NEUTRAL]

    tfidf_pipeline = train_simple_sentiment(
        sample_texts, sample_labels
    )

    logger.info("Interactive RAG chatbot ready (type 'exit')")

    while True:
        user_q = input("\nYou: ")
        if user_q.lower() == "exit":
            break

        answer, sentiment = answer_user_query(
            user_q,
            sentiment_model_pipeline=tfidf_pipeline
        )
        print(f"\nBot [sentiment={sentiment}]: {answer}")
