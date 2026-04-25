from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

# Convert URL to sync format (remove asyncpg)
sync_url = DATABASE_URL.replace("+asyncpg", "").replace("+psycopg2", "")
engine = create_engine(sync_url, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=False)
    question = Column(String, nullable=False)
    video_url = Column(String, nullable=False)
    video_title = Column(String, nullable=False)
    hienglish_question = Column(String, nullable=True)
    english_question = Column(String, nullable=True)
    search_text = Column(String, nullable=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()