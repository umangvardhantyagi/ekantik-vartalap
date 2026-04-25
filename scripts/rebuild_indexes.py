import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL, EMBEDDING_MODEL, FAISS_PATH, BM25_PATH

# Ensure the indexes directory exists
os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

sync_url = DATABASE_URL.replace("+asyncpg", "").replace("+psycopg2", "")
engine = create_engine(sync_url)
Session = sessionmaker(bind=engine)
session = Session()

print("Fetching search_text from database...")
result = session.execute(text("SELECT search_text FROM questions"))
texts = [row[0] for row in result.fetchall()]
session.close()

if not texts:
    print("No data in database.")
    sys.exit(1)

print(f"Loaded {len(texts)} records.")

# BM25
print("Building BM25 index...")
tokenized = [doc.lower().split() for doc in texts]
bm25 = BM25Okapi(tokenized)
with open(BM25_PATH, 'wb') as f:
    pickle.dump(bm25, f)

# FAISS
print("Building FAISS index...")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype('float32'))
faiss.write_index(index, FAISS_PATH)

print("Indexes rebuilt successfully.")