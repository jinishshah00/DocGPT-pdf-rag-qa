import hashlib
import os
import time
from typing import List, Tuple

from sqlalchemy.orm import Session

from backend.config import STORAGE_DIR
from backend.models import Document, ChatSession, Message
from backend.rag_pipeline import RAGPipeline

_pipeline_cache: dict[Tuple[int, bool, str], RAGPipeline] = {}


def _ensure_storage() -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)


def ingest_document(db: Session, user_id: int, filename: str, content: bytes) -> Document:
    _ensure_storage()
    file_hash = hashlib.sha256(content).hexdigest()
    existing = db.query(Document).filter(Document.owner_id == user_id, Document.file_hash == file_hash).first()
    if existing:
        return existing

    storage_path = os.path.join(STORAGE_DIR, f"{user_id}_{file_hash}.pdf")
    with open(storage_path, "wb") as f:
        f.write(content)

    doc = Document(owner_id=user_id, filename=filename, file_hash=file_hash, storage_path=storage_path)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Build index into Chroma
    pipeline = RAGPipeline()
    pipeline.add_pdf(storage_path, orig_filename=filename)
    return doc


def create_session(db: Session, user_id: int, doc_id: int, title: str | None) -> ChatSession:
    # Allow creating sessions without an attached document. Default title when missing.
    if title is None:
        title = "New Chat Session"
    sess = ChatSession(user_id=user_id, doc_id=doc_id, title=title)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


def _get_pipeline(doc_id: int, use_compression: bool, chain_type: str) -> RAGPipeline:
    key = (doc_id, use_compression, chain_type)
    if key in _pipeline_cache:
        return _pipeline_cache[key]
    pipeline = RAGPipeline(use_compression=use_compression, chain_type=chain_type)
    _pipeline_cache[key] = pipeline
    return pipeline


def answer_question(db: Session, session_id: int, question: str, chain_type: str = "stuff", use_compression: bool = False):
    sess = db.query(ChatSession).get(session_id)
    if not sess:
        raise ValueError("Session not found")
    # If session has no attached document, surface a clear error before querying
    if not sess.doc_id:
        raise ValueError("Document not found")
    doc = db.query(Document).get(sess.doc_id)
    if not doc:
        raise ValueError("Document not found")

    pipeline = _get_pipeline(doc.id, use_compression, chain_type)
    # Make sure document is indexed
    if os.path.exists(doc.storage_path):
        pipeline.add_pdf(doc.storage_path, orig_filename=doc.filename)

    start = time.time()
    response = pipeline.ask(question)
    latency_ms = int((time.time() - start) * 1000)

    # Normalize sources for API response
    sources = []
    for doc_obj in response.get("source_documents", []):
        sources.append({
            "source": doc_obj.metadata.get("source", "?"),
            "snippet": doc_obj.page_content[:500]
        })

    msg = Message(session_id=session_id, role="assistant", content=response.get("answer") or response.get("result"), sources_json=sources, latency_ms=latency_ms)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    # If the session has a placeholder title, ask the LLM to suggest a concise title based on filename
    try:
        if not sess.title or sess.title.strip() == "New Chat Session":
            # Use pipeline LLM to suggest a short descriptive title from the filename
            title_prompt = (
                f"Suggest a concise (3-8 words) descriptive title for a chat about the document named '{doc.filename}'."
                " Reply with only the title text."
            )
            pipeline = _get_pipeline(doc.id, use_compression=False, chain_type=chain_type)
            try:
                title_resp = pipeline.llm.invoke(title_prompt)
                suggested = title_resp.content.strip().splitlines()[0][:255]
                if suggested:
                    sess.title = suggested
                    db.add(sess)
                    db.commit()
            except Exception:
                pass
    except Exception:
        pass
    return msg, sources, latency_ms


def clear_session_messages(db: Session, session_id: int) -> int:
    q = db.query(Message).filter(Message.session_id == session_id)
    count = q.count()
    q.delete()
    db.commit()
    return count


def delete_session(db: Session, session_id: int, user_id: int) -> bool:
    """Delete a chat session and all its messages"""
    sess = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
    if not sess:
        return False
    # Delete all messages in the session first
    db.query(Message).filter(Message.session_id == session_id).delete()
    # Delete the session
    db.delete(sess)
    db.commit()
    return True
