from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from backend.db import get_db, Base, engine
from backend.auth import router as auth_router
from backend.google_oauth import router as google_router
from backend.schemas import (
    UploadResponse,
    CreateSessionRequest, CreateSessionResponse,
    ChatMessageRequest, ChatMessageResponse,
    FeedbackRequest, EvalRunRequest, EvalRunResponse
)
from backend.models import User, Document, ChatSession, Message, Feedback
from backend.rag_service import ingest_document, create_session, answer_question, clear_session_messages, delete_session
from backend.analytics_utils import (
    save_eval_row,
    save_llm_metrics_row,
    compute_custom_self_eval,
    compute_llm_metrics,
    EVAL_RESULTS_PATH,
    LLM_METRICS_PATH,
)
from backend.reflection_service import evaluate_answer, improve_answer
from backend.agent_service import invoke_agent
from backend.config import JWT_SECRET, JWT_ALGORITHM
from backend.observability.langfuse_client import log_chat_trace

Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAG QA Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(google_router)


def get_current_user(request: Request, db: Session = Depends(get_db)):
    # Accept Authorization via header (preferred). Keep compatibility with query param.
    header_auth = request.headers.get("Authorization")
    bearer = header_auth or request.query_params.get("authorization")
    if not bearer or not bearer.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = bearer.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub"))
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.get("/auth/me")
def auth_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email}


@app.post("/docs/upload", response_model=UploadResponse)
async def upload_doc(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    content = await file.read()
    doc = ingest_document(db, current_user.id, file.filename, content)
    return UploadResponse(doc_id=doc.id, filename=doc.filename)


@app.post("/chat/session", response_model=CreateSessionResponse)
async def create_chat_session(payload: CreateSessionRequest | None = None, request: Request = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc_id = None
    title = None
    if payload is not None:
        doc_id = payload.doc_id
        title = payload.title
    else:
        data = {}
        try:
            data = await request.json()
        except Exception:
            try:
                form = await request.form()
                data = dict(form)
            except Exception:
                pass
        doc_id = int(data.get("doc_id")) if data.get("doc_id") is not None else None
        title = data.get("title")
    # Allow creating sessions without an attached document (doc_id optional)
    sess = create_session(db, current_user.id, doc_id, title)
    return CreateSessionResponse(session_id=sess.id)


@app.get("/chat/sessions")
def list_chat_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title or f"Session {s.id}",
                "doc_id": s.doc_id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in sessions
        ]
    }


@app.post("/chat/{session_id}/attach")
async def attach_doc_to_session(session_id: int, doc_id: int | None = None, request: Request = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Allow doc_id to be passed via query param, JSON body, or form
    sess = db.query(ChatSession).get(session_id)
    if not sess or sess.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    # If doc_id wasn't provided as query param, try JSON body or form
    if doc_id is None and request is not None:
        try:
            j = await request.json()
            print("[attach] json body:", j)
            if j and j.get("doc_id") is not None:
                doc_id = int(j.get("doc_id"))
        except Exception as e:
            print("[attach] json parse failed:", e)
            try:
                form = await request.form()
                print("[attach] form body:", dict(form))
                if form and form.get("doc_id") is not None:
                    doc_id = int(form.get("doc_id"))
            except Exception as e2:
                print("[attach] form parse failed:", e2)
    if doc_id is None:
        raise HTTPException(status_code=422, detail="Missing doc_id")
    doc = db.query(Document).get(doc_id)
    if not doc or doc.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    sess.doc_id = doc.id
    db.add(sess)
    db.commit()
    # Persist an assistant message noting the attachment so frontend can show it
    try:
        from backend.models import Message

        note = Message(session_id=sess.id, role="assistant", content=f"File {doc.filename} attached to session.")
        db.add(note)
        db.commit()
    except Exception:
        pass
    return {"status": "ok", "session_id": sess.id}


@app.get("/chat/{session_id}")
def get_chat_session(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.query(ChatSession).get(session_id)
    if not sess or sess.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    doc = None
    if sess.doc_id:
        d = db.query(Document).get(sess.doc_id)
        if d:
            doc = {"id": d.id, "filename": d.filename}
    messages = []
    for m in db.query(Message).filter(Message.session_id == sess.id).order_by(Message.created_at.asc()).all():
        messages.append({"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None})
    return {"id": sess.id, "title": sess.title, "doc": doc, "messages": messages}


@app.post("/chat/{session_id}/message", response_model=ChatMessageResponse)
async def send_message(session_id: int, payload: ChatMessageRequest | None = None, request: Request = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.query(ChatSession).get(session_id)
    if not sess or sess.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    content = None
    chain_type = "stuff"
    use_compression = False
    if payload is not None:
        content = payload.content
        chain_type = payload.chain_type or "stuff"
    else:
        data = {}
        try:
            data = await request.json()
        except Exception:
            try:
                form = await request.form()
                data = dict(form)
            except Exception:
                pass
        content = data.get("content")
        chain_type = data.get("chain_type") or "stuff"
        use_compression = str(data.get("use_compression", "false")).lower() in ("1", "true", "yes")
    # If payload is pydantic, also try to read use_compression from request JSON
    if not use_compression and request is not None:
        try:
            j = await request.json()
            use_compression = bool(j.get("use_compression", False))
        except Exception:
            pass
    if not content:
        raise HTTPException(status_code=422, detail="Missing content")
    # Persist the user's message so it appears when loading the session
    try:
        user_msg = Message(session_id=session_id, role="user", content=content)
        db.add(user_msg)
        db.commit()
    except Exception:
        # If persisting user message fails, continue to attempt answering
        pass
    try:
        msg, sources, latency_ms = answer_question(db, session_id, content, chain_type, use_compression)
    except ValueError as e:
        # Treat missing document or session as a client error
        raise HTTPException(status_code=422, detail=str(e))
    result = {
        "result": msg.content,
        "sources": sources,
        "num_source_docs": len(sources),
        "latency_ms": latency_ms,
    }
    log_chat_trace(current_user.id, session_id, content, chain_type, result)
    return ChatMessageResponse(
        message_id=msg.id,
        answer=msg.content,
        sources=sources,
        latency_ms=latency_ms,
    )


@app.delete("/chat/{session_id}/messages")
def clear_messages(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.query(ChatSession).get(session_id)
    if not sess or sess.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    deleted = clear_session_messages(db, session_id)
    return {"deleted": deleted}


@app.delete("/chat/{session_id}")
def delete_chat_session(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    success = delete_session(db, session_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    msg = db.query(Message).get(payload.message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    # Ensure message belongs to user's session
    sess = db.query(ChatSession).get(msg.session_id)
    if not sess or sess.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    fb = Feedback(message_id=msg.id, rating=payload.rating, note=payload.note)
    db.add(fb)
    db.commit()
    return {"status": "ok"}


@app.post("/eval/run", response_model=EvalRunResponse)
def run_eval(payload: EvalRunRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Placeholder: in a real runner, iterate over dataset and store metrics
    # For now, create a dummy eval row
    from backend.models import EvalRun
    ev = EvalRun(user_id=current_user.id, doc_id=payload.doc_id, session_id=payload.session_id, metrics_json={"status": "started"})
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return EvalRunResponse(eval_id=ev.id, status="started")


# Analytics: save eval rows and LLM metrics, and serve CSVs
@app.post("/analytics/eval")
def save_eval(question: str = Form(...), answer: str = Form(...), faithful: bool = Form(False), llm_feedback: str = Form(""), sources_json: str = Form(""), current_user: User = Depends(get_current_user)):
    import json
    try:
        sources = json.loads(sources_json) if sources_json else []
    except Exception:
        sources = []
    save_eval_row(question, answer, sources, faithful, llm_feedback)
    return {"status": "saved"}


@app.post("/analytics/llm")
def save_llm_metrics(question: str = Form(...), answer: str = Form(...), faithfulness: float = Form(...), relevance: float = Form(...), conciseness: float = Form(...), justification: str = Form(""), current_user: User = Depends(get_current_user)):
    save_llm_metrics_row(question, answer, faithfulness, relevance, conciseness, justification)
    return {"status": "saved"}


@app.get("/analytics/eval")
def get_eval_csv():
    from fastapi.responses import FileResponse
    return FileResponse(EVAL_RESULTS_PATH)


@app.get("/analytics/llm")
def get_llm_csv():
    from fastapi.responses import FileResponse
    return FileResponse(LLM_METRICS_PATH)


# Compute analytics using LLM, then save
@app.post("/analytics/compute")
async def compute_and_save_analytics(request: Request, question: str = Form(None), answer: str = Form(None), sources_json: str = Form(""), current_user: User = Depends(get_current_user)):
    import json
    if not question or not answer:
        try:
            data = await request.json()
            question = data.get("question")
            answer = data.get("answer")
            sources_json = json.dumps(data.get("sources", []))
        except Exception:
            pass
    try:
        sources_list = json.loads(sources_json) if sources_json else []
    except Exception:
        sources_list = []
    sources_text = [s.get("snippet", "") for s in sources_list]
    faithful, llm_feedback = compute_custom_self_eval(question, answer, sources_text)
    save_eval_row(question, answer, sources_list, faithful, llm_feedback)
    fth, rel, conc, just = compute_llm_metrics(question, answer, sources_text)
    save_llm_metrics_row(question, answer, fth, rel, conc, just)
    return {
        "eval": {"faithful": faithful, "llm_feedback": llm_feedback},
        "metrics": {"faithfulness": fth, "relevance": rel, "conciseness": conc, "justification": just}
    }


# Reflection endpoint
@app.post("/chat/{session_id}/reflect")
def reflect_answer(session_id: int, question: str = Form(...), answer: str = Form(...), current_user: User = Depends(get_current_user)):
    score, justification, retry_needed = evaluate_answer(question, answer)
    improved = answer
    if retry_needed:
        improved = improve_answer(question, answer, justification)
    return {"answer": improved, "justification": justification, "score": score, "retry": retry_needed}


# Agent invoke endpoint
@app.post("/agent/invoke")
async def agent_invoke(query: str = Form(None), request: Request = None, current_user: User = Depends(get_current_user)):
    if not query and request is not None:
        try:
            data = await request.json()
            query = data.get("query")
        except Exception:
            pass
    if not query:
        raise HTTPException(status_code=422, detail="Missing query")
    output = invoke_agent(query)
    return {"output": output}


@app.post("/agent/{session_id}/invoke", response_model=ChatMessageResponse)
async def agent_invoke_session(session_id: int, query: str = Form(None), request: Request = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not query and request is not None:
        try:
            data = await request.json()
            query = data.get("query")
        except Exception:
            pass
    if not query:
        raise HTTPException(status_code=422, detail="Missing query")
    sess = db.query(ChatSession).get(session_id)
    if not sess or sess.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    # Persist user query
    from backend.models import Message
    user_msg = Message(session_id=session_id, role="user", content=query)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    # Invoke agent
    import time
    start = time.time()
    output = invoke_agent(query)
    latency_ms = int((time.time() - start) * 1000)
    sources: list[dict] = []
    assistant_msg = Message(session_id=session_id, role="assistant", content=output, sources_json=sources, latency_ms=latency_ms)
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
    return ChatMessageResponse(message_id=assistant_msg.id, answer=assistant_msg.content, sources=sources, latency_ms=latency_ms)
