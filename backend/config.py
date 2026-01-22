import os
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Auth
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
AUTH = (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Storage
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_BASE = os.getenv("FRONTEND_BASE", "http://localhost:8501")
