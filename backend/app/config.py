import os
from dotenv import load_dotenv

load_dotenv()

_DEV_ORIGINS = (
    "http://127.0.0.1:5500,"
    "http://localhost:5500,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:8080,"
    "http://127.0.0.1:8080,"
    "http://localhost,"
    "http://127.0.0.1"
)

class Config:
    SECRET_KEY        = os.environ.get('SECRET_KEY', 'dev-secret-change-in-production')

    # ── Gemini (replaces OpenAI) ──────────────────────────────────────────────
    GEMINI_API_KEY    = os.environ.get('GEMINI_API_KEY', '')

    # Free-tier models:
    #   Embeddings : models/embedding-001   (2048 dim, free)
    #   Chat       : gemini-1.5-flash       (free, 15 req/min, 1500 req/day)
    #   Local alt  : sentence-transformers/all-MiniLM-L6-v2 (no API key required)
    EMBEDDING_PROVIDER = os.environ.get('EMBEDDING_PROVIDER', 'auto')  # auto | gemini | local
    EMBEDDING_MODEL   = os.environ.get('EMBEDDING_MODEL', 'models/embedding-001')
    LOCAL_EMBEDDING_MODEL = os.environ.get('LOCAL_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    LLM_MODEL         = os.environ.get('LLM_MODEL',       'gemini-1.5-flash')
    LLM_TEMPERATURE   = float(os.environ.get('LLM_TEMPERATURE', 0.1))

    UPLOAD_FOLDER      = os.environ.get('UPLOAD_FOLDER',    'uploads')
    VECTOR_DB_FOLDER   = os.environ.get('VECTOR_DB_FOLDER', 'vector_db')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))

    CHUNK_SIZE    = int(os.environ.get('CHUNK_SIZE',    900))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 150))
    RETRIEVAL_K   = int(os.environ.get('RETRIEVAL_K',   4))
    CHAT_RETRIEVAL_MODE = os.environ.get('CHAT_RETRIEVAL_MODE', 'vector')  # vector | long_context | hybrid
    LONG_CONTEXT_MAX_CHARS = int(os.environ.get('LONG_CONTEXT_MAX_CHARS', 300000))
    MEMORY_WINDOW = int(os.environ.get('MEMORY_WINDOW', 10))

    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', _DEV_ORIGINS).split(',')


class TestConfig(Config):
    TESTING          = True
    UPLOAD_FOLDER    = 'test_uploads'
    VECTOR_DB_FOLDER = 'test_vector_db'
