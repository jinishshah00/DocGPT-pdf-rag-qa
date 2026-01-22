from pydantic import BaseModel, EmailStr
from typing import Optional, List, Any

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UploadResponse(BaseModel):
    doc_id: int
    filename: str

class CreateSessionRequest(BaseModel):
    doc_id: int
    title: Optional[str] = None

class CreateSessionResponse(BaseModel):
    session_id: int

class ChatMessageRequest(BaseModel):
    content: str
    chain_type: str = "stuff"

class Source(BaseModel):
    source: str
    snippet: str

class ChatMessageResponse(BaseModel):
    message_id: int
    answer: str
    sources: List[Source]
    latency_ms: Optional[int] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_estimate: Optional[int] = None

class FeedbackRequest(BaseModel):
    message_id: int
    rating: int  # +1 or -1
    note: Optional[str]

class EvalRunRequest(BaseModel):
    doc_id: Optional[int]
    session_id: Optional[int]
    dataset_path: Optional[str]

class EvalRunResponse(BaseModel):
    eval_id: int
    status: str
