from backend.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


def log_chat_trace(user_id, session_id, query, chain_type, result):
    # No-op fallback for observability; implement real client if needed.
    return None
