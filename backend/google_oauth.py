import base64, json
from typing import Optional
import urllib.parse
import requests
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import User
from backend.auth import create_access_token, get_password_hash
from backend.config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    FRONTEND_BASE,
)

router = APIRouter(prefix="/auth/google", tags=["auth"])


def _require_google_config():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")


@router.get("/start")
def google_start(redirect_to: Optional[str] = None):
    _require_google_config()
    # Default redirect target back to frontend
    redirect_target = redirect_to or FRONTEND_BASE
    state_payload = {"redirect_to": redirect_target}
    state = base64.urlsafe_b64encode(json.dumps(state_payload).encode()).decode()
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "consent",
        "state": state,
    }
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(auth_url)


@router.get("/callback")
def google_callback(code: str, state: str, db: Session = Depends(get_db)):
    _require_google_config()
    # Decode state
    try:
        payload = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
    except Exception:
        payload = {"redirect_to": FRONTEND_BASE}
    redirect_to = payload.get("redirect_to", FRONTEND_BASE)

    # Exchange code for tokens
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    token_resp = requests.post("https://oauth2.googleapis.com/token", data=data)
    if not token_resp.ok:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {token_resp.text}")
    tokens = token_resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Missing access_token")

    # Fetch userinfo
    uinfo_resp = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if not uinfo_resp.ok:
        raise HTTPException(status_code=400, detail=f"Userinfo failed: {uinfo_resp.text}")
    uinfo = uinfo_resp.json()
    email = uinfo.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email")

    # Upsert user
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Create without a password since Google OAuth is used for authentication.
        # Avoid calling the password hasher here (some runtimes may not have
        # a compatible bcrypt backend). An empty string indicates no local
        # password is set.
        user = User(email=email, password_hash="")
        db.add(user)
        db.commit()
        db.refresh(user)

    # Issue JWT and redirect back to frontend with token
    token = create_access_token({"sub": str(user.id)})
    sep = "&" if ("?" in redirect_to) else "?"
    target = f"{redirect_to}{sep}token={token}"
    return RedirectResponse(target)
