#!/usr/bin/env python3
"""
Daily YouTube Update Pipeline for Bhajan Marg Channel
- Fetches new Ekantik Vartalaap videos
- Extracts Q&A from video description using regex
- Saves new questions to PostgreSQL database
- Rebuilds FAISS and BM25 indexes only if new questions are added
- Logs all activity to logs/update.log
- AUTO-PUSHES updated indexes to GitHub (so Railway auto-deploys)
"""

import os
import sys
import re
import pickle
import logging
import time
import subprocess
import yt_dlp
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

# Add parent directory to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL, EMBEDDING_MODEL, FAISS_PATH, BM25_PATH

# ============================================================
# Setup logging
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "update.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CHANNEL_URL = "https://www.youtube.com/@BhajanMarg"
DAYS_BACK = int(os.getenv("UPDATE_DAYS_BACK", "1"))

# ============================================================
# Database model (inline definition, matching app.py)
# ============================================================
Base = declarative_base()

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True)
    timestamp = Column(String, nullable=True)
    question = Column(String, nullable=False)
    video_url = Column(String, nullable=False)
    video_title = Column(String, nullable=True)
    hienglish_question = Column(String, nullable=True)
    english_question = Column(String, nullable=True)
    search_text = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# ============================================================
# Helper functions
# ============================================================
def get_new_videos(since_days=DAYS_BACK):
    """Fetch videos from channel uploaded after cutoff date."""
    ydl_opts = {'quiet': True, 'extract_flat': True, 'force_generic_extractor': False}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(CHANNEL_URL, download=False)
            videos = []
            cutoff = datetime.now() - timedelta(days=since_days)
            for entry in info.get('entries', []):
                upload_date = datetime.strptime(entry.get('upload_date', '19700101'), '%Y%m%d')
                if upload_date >= cutoff:
                    videos.append({
                        'id': entry['id'],
                        'title': entry['title'],
                        'url': f"https://www.youtube.com/watch?v={entry['id']}",
                        'upload_date': upload_date
                    })
            logger.info(f"Found {len(videos)} new videos since {cutoff.date()}")
            return videos
    except Exception as e:
        logger.error(f"Failed to fetch videos: {e}")
        return []

def get_video_description(video_url, retries=3):
    """Extract description using yt-dlp with retries."""
    for attempt in range(retries):
        try:
            ydl_opts = {'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return info.get('description', '')
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {video_url}: {e}")
            time.sleep(2)
    logger.error(f"All retries failed for {video_url}")
    return ""

def extract_qa_from_description(description):
    """Extract Q&A pairs from video description."""
    qa = []
    lines = description.split('\n')
    ts_pattern = re.compile(r'^(\d{1,2}:)?\d{1,2}:\d{2}')
    current_ts = None
    current_q = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ts_pattern.match(line):
            if current_ts and current_q:
                qa.append({'timestamp': current_ts, 'question': ' '.join(current_q)})
            parts = line.split(maxsplit=1)
            current_ts = parts[0]
            current_q = [parts[1]] if len(parts) > 1 else []
        else:
            if current_q:
                current_q.append(line)
    if current_ts and current_q:
        qa.append({'timestamp': current_ts, 'question': ' '.join(current_q)})
    return qa

def add_to_db(video, qa_pairs):
    """Insert new Q&A pairs into database."""
    sync_url = DATABASE_URL.replace("+asyncpg", "").replace("+psycopg2", "")
    engine = create_engine(sync_url)
    # Create tables if not exist (should already exist, but safe)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    added = 0
    for q in qa_pairs:
        # Check if question already exists (exact match)
        exists = session.execute(text("SELECT id FROM questions WHERE question = :q"), {"q": q['question']}).fetchone()
        if exists:
            continue
        # For now, hienglish and english are empty – you can add transliteration later
        hienglish = ""
        english = ""
        search_text = f"{q['question']} [hindi] {hienglish} [roman] {english} [eng]"
        new_q = Question(
            timestamp=q['timestamp'],
            question=q['question'],
            video_url=video['url'],
            video_title=video['title'],
            hienglish_question=hienglish,
            english_question=english,
            search_text=search_text
        )
        session.add(new_q)
        added += 1
    session.commit()
    session.close()
    return added

def rebuild_indexes():
    """Rebuild FAISS and BM25 indexes from current database."""
    sync_url = DATABASE_URL.replace("+asyncpg", "").replace("+psycopg2", "")
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    texts = [row[0] for row in session.execute(text("SELECT search_text FROM questions")).fetchall()]
    session.close()
    if not texts:
        logger.warning("No questions in database. Indexes not rebuilt.")
        return
    logger.info(f"Building BM25 index from {len(texts)} records...")
    tokenized = [doc.lower().split() for doc in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info("Building FAISS index...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, FAISS_PATH)
    logger.info("Indexes rebuilt successfully.")

def push_to_github():
    """Push updated index files to GitHub to trigger Railway auto-deploy."""
    try:
        # Get the git root directory
        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], 
                                           text=True).strip()
        os.chdir(git_root)
        
        # Check if index files have changed
        status = subprocess.run(['git', 'status', '--porcelain', 'indexs/'], 
                               capture_output=True, text=True)
        
        if not status.stdout.strip():
            logger.info("No changes to index files. Skipping git push.")
            return
        
        # Add index files
        subprocess.run(['git', 'add', 'indexs/'], check=True)
        
        # Commit with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        subprocess.run(['git', 'commit', '-m', f"Auto-update indexes with new data [{timestamp}]"], 
                       check=True)
        
        # Push to GitHub
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        logger.info("✅ Successfully pushed updated indexes to GitHub!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
    except Exception as e:
        logger.error(f"Failed to push to GitHub: {e}")

def main():
    logger.info("=== Starting daily update pipeline ===")
    try:
        videos = get_new_videos()
        if not videos:
            logger.info("No new videos found.")
            return
        logger.info(f"Processing {len(videos)} videos...")
        total_added = 0
        for video in videos:
            logger.info(f"Processing: {video['title']}")
            desc = get_video_description(video['url'])
            if not desc:
                logger.warning(f"No description for {video['title']}, skipping.")
                continue
            qa = extract_qa_from_description(desc)
            added = add_to_db(video, qa)
            total_added += added
            logger.info(f"Added {added} new questions from this video")
        if total_added > 0:
            logger.info(f"Total new questions: {total_added}. Rebuilding indexes...")
            rebuild_indexes()
            # NEW: Push updated indexes to GitHub
            logger.info("Pushing updated indexes to GitHub...")
            push_to_github()
        else:
            logger.info("No new questions added.")
    except Exception as e:
        logger.exception("Pipeline failed with exception")
    logger.info("=== Daily update pipeline finished ===\n")

if __name__ == "__main__":
    main()