import asyncio
import os
import re
import pickle
import time
from datetime import datetime
import pandas as pd
import numpy as np
import faiss
import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import (
    FAISS_PATH, BM25_PATH, FALLBACK_VIDEOS, OFF_TOPIC_KEYWORDS,
    SPIRITUAL_KEYWORDS, SYNONYMS, DATABASE_URL, QUERY_EXPANSIONS,
    SPIRITUAL_QUESTION_PATTERNS
)

# ---------------------------
# Database setup
# ---------------------------
sync_url = DATABASE_URL.replace("+asyncpg", "").replace("+psycopg2", "")
engine = create_engine(sync_url)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class SearchLog(Base):
    __tablename__ = "search_logs"
    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    was_off_topic = Column(Integer, default=0)
    recommended_count = Column(Integer, default=0)
    user_ip = Column(String, nullable=True)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    question_id = Column(Integer, nullable=False)
    rating = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_ip = Column(String, nullable=True)

Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

# ---------------------------
# Load models and indexes (with fallback rebuild)
# ---------------------------
print("Loading embedding model (intfloat/multilingual-e5-small)...")
model = SentenceTransformer("intfloat/multilingual-e5-small")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

# Check if pre-computed index files exist
if os.path.exists(FAISS_PATH) and os.path.exists(BM25_PATH):
    print("🚀 Pre-computed index found! Loading instantly...")
    index = faiss.read_index(FAISS_PATH)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print("✅ Indexes loaded successfully!")
else:
    print("⚠️ Index files not found. Starting manual rebuild from database...")
    sync_engine = create_engine(sync_url)
    with sync_engine.connect() as conn:
        result = conn.execute(text("SELECT search_text FROM questions ORDER BY id"))
        texts = [row[0] for row in result.fetchall()]
        if not texts:
            raise RuntimeError("No questions found in database. Cannot rebuild indexes.")
    tokenized = [doc.lower().split() for doc in texts]
    bm25_obj = BM25Okapi(tokenized)
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25_obj, f)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings.astype('float32'))
    faiss.write_index(faiss_index, FAISS_PATH)
    index = faiss_index
    bm25 = bm25_obj
    print("✅ Indexes rebuilt successfully!")

print("Ready.")

# ---------------------------
# Helper functions
# ---------------------------
def contains_hindi(text: str) -> bool:
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def sanitize_query(query: str) -> str:
    query = re.sub(r'[\\\x00-\x1f\x7f]', '', query)
    return query.strip()

def is_gibberish(query: str) -> bool:
    if contains_hindi(query):
        return False
    q = query.strip().lower()
    if len(q) < 2:
        return True
    vowels = set('aeiou')
    vowel_count = sum(1 for ch in q if ch in vowels)
    if vowel_count >= 1 and len(q) >= 4:
        return False
    if re.search(r'(.)\1{4,}', q):
        return True
    kb_rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm', 'lkjhgfdsa', 'poiuytrewq']
    for row in kb_rows:
        if row in q or any(seq in q for seq in [row[i:i+4] for i in range(len(row)-3)]):
            return True
    if vowel_count == 0 and len(q) >= 4:
        return True
    return False

def is_spiritual_english_query(query: str) -> bool:
    q_lower = query.lower().strip()
    for pattern in SPIRITUAL_QUESTION_PATTERNS:
        if pattern.endswith(' '):
            if q_lower.startswith(pattern):
                return True
        else:
            if pattern in q_lower:
                return True
    return False

def is_off_topic(query: str) -> bool:
    if contains_hindi(query):
        return False
    
    q_lower = query.lower().strip()
    
    if is_spiritual_english_query(q_lower):
        return False
    
    has_spiritual = any(kw in q_lower for kw in SPIRITUAL_KEYWORDS)
    has_off = any(kw in q_lower for kw in OFF_TOPIC_KEYWORDS)
    
    if len(q_lower) < 4 and not has_spiritual:
        return True
    
    if has_off and not has_spiritual:
        return True
    return False

def clean_noisy_query(query: str) -> tuple:
    q = query.strip()
    if len(q) < 3:
        return (False, q)
    tokens = re.findall(r'[\w\u0900-\u097F]+', q)
    valid_tokens = []
    for token in tokens:
        if contains_hindi(token):
            valid_tokens.append(token)
            continue
        vowel_count = sum(1 for ch in token if ch.lower() in 'aeiou')
        if vowel_count > 0 and len(token) <= 15:
            valid_tokens.append(token)
    cleaned = ' '.join(valid_tokens)
    if len(cleaned) < 3:
        return (True, cleaned)
    return (False, cleaned)

def expand_synonyms(query: str) -> str:
    q_lower = query.lower()
    expanded = query
    for word, syns in SYNONYMS.items():
        if word in q_lower:
            for s in syns:
                if s not in expanded.lower():
                    expanded += f" {s}"
    return expanded

def expand_query(query: str) -> str:
    expanded = query
    q_lower = query.lower()
    for term, expansions in QUERY_EXPANSIONS.items():
        if term in q_lower:
            for exp in expansions:
                if exp not in expanded:
                    expanded += f" {exp}"
    return expanded

def get_yt_link(url: str, ts: str) -> str:
    try:
        parts = str(ts).strip().split(':')
        if len(parts) == 2:
            seconds = int(parts[0])*60 + int(parts[1])
        elif len(parts) == 3:
            seconds = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
        else:
            seconds = 0
        vid_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
        if not vid_match:
            return url
        vid = vid_match.group(1)
        return f"https://www.youtube.com/watch?v={vid}&t={seconds}s"
    except Exception:
        return url

def fuzzy_match_score(query: str, text: str) -> int:
    if not text:
        return 0
    return fuzz.token_set_ratio(query.lower(), text.lower())

# ---------------------------
# Cached DataFrame
# ---------------------------
_df_cache = None
_cache_time = 0

async def get_questions_df(db):
    global _df_cache, _cache_time
    if _df_cache is not None and (time.time() - _cache_time) < 3600:
        return _df_cache
    def fetch():
        result = db.execute(text("SELECT id, question, hienglish_question, english_question, search_text, video_title, video_url, timestamp FROM questions ORDER BY id"))
        rows = result.fetchall()
        data = []
        for row in rows:
            data.append({
                "id": row[0], "question": row[1], "hienglish_question": row[2],
                "english_question": row[3], "search_text": row[4],
                "video_title": row[5], "video_url": row[6], "timestamp": row[7]
            })
        return pd.DataFrame(data)
    _df_cache = await asyncio.to_thread(fetch)
    _cache_time = time.time()
    return _df_cache

async def log_search(query: str, was_off_topic: int, rec_count: int, ip: str, db):
    def insert():
        db.execute(
            text("INSERT INTO search_logs (query, timestamp, was_off_topic, recommended_count, user_ip) VALUES (:q, :ts, :off, :cnt, :ip)"),
            {"q": query[:500], "ts": datetime.utcnow(), "off": was_off_topic, "cnt": rec_count, "ip": ip}
        )
        db.commit()
    await asyncio.to_thread(insert)

# ---------------------------
# Feedback and Rating endpoints
# ---------------------------
@app.post("/feedback")
async def submit_feedback(request: Request, db=Depends(get_db)):
    try:
        data = await request.json()
        query = data.get('query')
        question_id = data.get('question_id')
        rating = data.get('rating')
        if not all([query, question_id, rating]) or rating not in (1, -1):
            return JSONResponse(status_code=400, content={"message": "Invalid data"})
        fb = Feedback(query=query[:500], question_id=question_id, rating=rating)
        db.add(fb)
        db.commit()
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/rating")
async def submit_rating(request: Request, db=Depends(get_db)):
    try:
        data = await request.json()
        query = data.get('query', '')[:500]
        rating = data.get('rating')
        if not query or not isinstance(rating, int) or rating < 1 or rating > 5:
            return JSONResponse(status_code=400, content={"message": "Invalid rating data"})
        client_ip = request.client.host if request.client else "unknown"
        new_rating = Rating(query=query, rating=rating, user_ip=client_ip)
        db.add(new_rating)
        db.commit()
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# ---------------------------
# Search endpoint (FINAL with fixes)
# ---------------------------
@app.post("/search")
async def search(request: Request, db=Depends(get_db)):
    client_ip = request.client.host if request.client else "unknown"
    try:
        payload = await request.json()
        raw_query = payload.get('query', '').strip()
        if not raw_query:
            return {"status": "success", "results": []}
        
        raw_query = sanitize_query(raw_query)
        
        if is_gibberish(raw_query):
            await log_search(raw_query, 1, len(FALLBACK_VIDEOS), client_ip, db)
            return {
                "status": "success",
                "fallback": True,
                "results": FALLBACK_VIDEOS,
                "message": "This looks like random typing. Please ask a genuine spiritual question. You can enjoy these Radha Naam tracks."
            }
        
        is_noisy, cleaned_query = clean_noisy_query(raw_query)
        if is_noisy:
            await log_search(raw_query, 1, 0, client_ip, db)
            return {
                "status": "success",
                "results": [],
                "message": "No relevant answer found. Please ask a clear spiritual question."
            }
        query = cleaned_query
        
        df_local = await get_questions_df(db)
        if df_local.empty:
            await log_search(raw_query, 1, 0, client_ip, db)
            return {"status": "success", "results": []}
        
        # -------------------------
        # NEW: Off‑topic override – if query matches any existing question (fuzzy), allow it
        # -------------------------
        def has_matching_question(q, df):
            for _, row in df.iterrows():
                if row['hienglish_question'] and fuzz.token_set_ratio(q.lower(), row['hienglish_question'].lower()) >= 70:
                    return True
                if row['question'] and fuzz.token_set_ratio(q.lower(), row['question'].lower()) >= 70:
                    return True
            return False
        
        # Check off‑topic only if no direct match
        if not has_matching_question(query, df_local):
            if is_off_topic(query):
                await log_search(raw_query, 1, len(FALLBACK_VIDEOS), client_ip, db)
                return {
                    "status": "success",
                    "fallback": True,
                    "results": FALLBACK_VIDEOS,
                    "message": "Your search does not seem related to spiritual topics. Please ask a spiritual question. You can enjoy these Radha Naam tracks."
                }
        
        # -------------------------
        # Romanized Hindi fast path (improved threshold 65 + fallback)
        # -------------------------
        is_romanized_hindi = (not contains_hindi(query) and 
                              any(ch in 'aeiou' for ch in query.lower()) and
                              any(word in query.lower() for word in ['ka', 'ki', 'ko', 'se', 'mein', 'hai', 'kya', 'kyu', 'kaise', 'lo', 'ji', 'maharaj', 'guru', 'radha', 'krishna', 'ram']))
        
        if is_romanized_hindi:
            best_matches = []
            for idx, row in df_local.iterrows():
                if row['hienglish_question']:
                    # Token set ratio
                    score = fuzz.token_set_ratio(query.lower(), row['hienglish_question'].lower())
                    if score >= 65:   # lowered threshold
                        best_matches.append((score, idx))
                    else:
                        # Fallback: partial ratio
                        score2 = fuzz.partial_ratio(query.lower(), row['hienglish_question'].lower())
                        if score2 >= 75:
                            best_matches.append((score2, idx))
            if best_matches:
                best_matches.sort(key=lambda x: x[0], reverse=True)
                results = []
                seen = set()
                for score, idx in best_matches[:12]:
                    row = df_local.iloc[idx]
                    q_text = row['question']
                    if q_text in seen:
                        continue
                    seen.add(q_text)
                    results.append({
                        'id': int(row['id']),
                        'question': q_text,
                        'video_title': row['video_title'],
                        'video_url': get_yt_link(row['video_url'], row['timestamp']),
                        'timestamp': row['timestamp']
                    })
                    if len(results) >= 12:
                        break
                if results:
                    await log_search(raw_query, 0, len(results), client_ip, db)
                    return {"status": "success", "results": results}
        
        # -------------------------
        # Normal hybrid search (unchanged from working version)
        # -------------------------
        expanded = expand_synonyms(query)
        expanded = expand_query(expanded)
        query_for_model = f"query: {expanded}"
        
        lit_scores = {}
        for idx, row in df_local.iterrows():
            hindi_q = row['question']
            hieng_q = row['hienglish_question']
            eng_q = row['english_question']
            
            score_hindi = fuzzy_match_score(expanded, hindi_q)
            score_hieng = fuzzy_match_score(expanded, hieng_q) * 1.5
            score_eng = fuzzy_match_score(expanded, eng_q)
            
            best = max(score_hindi, score_hieng, score_eng)
            if best >= 40:
                lit_scores[idx] = 1000 + best
            elif any(expanded.lower() in str(f).lower() for f in [hindi_q, hieng_q, eng_q]):
                lit_scores[idx] = 1000
            if hieng_q and expanded.lower() in hieng_q.lower():
                lit_scores[idx] = lit_scores.get(idx, 0) + 800
        
        bm25_scores = bm25.get_scores(expanded.lower().split())
        top_bm25 = np.argsort(bm25_scores)[::-1][:30]
        
        vec = model.encode([query_for_model], normalize_embeddings=True).astype('float32')
        dist, idxs = index.search(vec, k=50)
        
        final = {}
        for idx, sc in lit_scores.items():
            final[idx] = sc
        for idx in top_bm25:
            if bm25_scores[idx] > 0:
                final[idx] = final.get(idx, 0) + bm25_scores[idx] * 15
        for i, idx in enumerate(idxs[0]):
            if idx != -1:
                final[idx] = final.get(idx, 0) + float(dist[0][i]) * 100 * 2.5
        
        if not final or max(final.values()) < 3:
            await log_search(raw_query, 1, len(FALLBACK_VIDEOS), client_ip, db)
            return {
                "status": "success",
                "fallback": True,
                "results": FALLBACK_VIDEOS,
                "message": "No relevant answer found. You can enjoy these Radha Naam tracks."
            }
        
        candidates = sorted(final.items(), key=lambda x: x[1], reverse=True)[:30]
        candidate_indices = [c[0] for c in candidates]
        
        if not contains_hindi(query):
            pairs = [(expanded, df_local.iloc[idx]['search_text'][:512]) for idx in candidate_indices]
            cross_scores = cross_encoder.predict(pairs)
            final_reranked = {}
            for i, idx in enumerate(candidate_indices):
                hybrid = final[idx]
                cross = float(cross_scores[i]) * 100
                final_reranked[idx] = hybrid * 0.6 + cross * 0.4
            sorted_idx = sorted(final_reranked.keys(), key=lambda x: final_reranked[x], reverse=True)[:12]
        else:
            sorted_idx = sorted(final.items(), key=lambda x: x[1], reverse=True)[:12]
            sorted_idx = [idx for idx, score in sorted_idx]
        
        results = []
        seen = set()
        for idx in sorted_idx:
            row = df_local.iloc[idx]
            q_text = row['question']
            if q_text in seen:
                continue
            seen.add(q_text)
            results.append({
                'id': int(row['id']),
                'question': q_text,
                'video_title': row['video_title'],
                'video_url': get_yt_link(row['video_url'], row['timestamp']),
                'timestamp': row['timestamp']
            })
        
        await log_search(raw_query, 0, len(results), client_ip, db)
        return {"status": "success", "results": results}
    
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

# ---------------------------
# Improved /suggest endpoint with limited dropdown size (CSS in HTML)
# ---------------------------
@app.get("/suggest")
async def suggest(q: str = "", db=Depends(get_db)):
    if len(q) < 2:
        return []
    q = sanitize_query(q)
    if len(q) < 2:
        return []
    
    df_local = await get_questions_df(db)
    candidates = []
    for idx, row in df_local.iterrows():
        texts = [row['question'], row['hienglish_question'], row['english_question']]
        best_score = max((fuzz.partial_ratio(q.lower(), str(t).lower()) for t in texts), default=0)
        if best_score >= 55:
            candidates.append((best_score, row['question']))
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    unique = []
    for score, q_text in candidates:
        if q_text not in seen:
            seen.add(q_text)
            unique.append(q_text)
        if len(unique) >= 8:
            break
    return unique

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# ---------------------------
# Daily update scheduler
# ---------------------------
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess

def scheduled_update():
    try:
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "update_pipeline.py")
        if not os.path.exists(script_path):
            print("Update script not found. Skipping.")
            return
        result = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"[Update] Success: {result.stdout[-200:]}")
        else:
            print(f"[Update] Error: {result.stderr[-200:]}")
    except Exception as e:
        print(f"[Update] Failed to run pipeline: {e}")

if os.path.exists(os.path.join(os.path.dirname(__file__), "scripts", "update_pipeline.py")):
    scheduler = BackgroundScheduler()
    # Run every day at 20:00 (8:00 PM) server time
    scheduler.add_job(scheduled_update, CronTrigger(hour=20, minute=0))
    scheduler.start()
    print("Daily YouTube update pipeline scheduled at 8:00 PM.")

if __name__ == "__main__":
    # Force the port to 8000 regardless of environment
    uvicorn.run(app, host="0.0.0.0", port=8000)