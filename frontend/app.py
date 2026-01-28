import os
import sys
import tempfile, uuid
import html
import warnings
import logging
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv
try:
    from streamlit_cookies_manager import EncryptedCookieManager
except Exception:
    EncryptedCookieManager = None

# Env
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("CHROMA_ENABLE_TELEMETRY", "FALSE")

# Silence telemetry loggers
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

# Silence legacy LangChain deprecation warnings in logs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

load_dotenv()
API_BASE_INTERNAL = os.getenv("API_BASE_INTERNAL", os.getenv("API_BASE", "http://localhost:8000"))
API_BASE_PUBLIC = os.getenv("API_BASE_PUBLIC", os.getenv("API_BASE", "http://localhost:8000"))
FRONTEND_BASE = os.getenv("FRONTEND_BASE", "http://localhost:8501")
COOKIE_PASSWORD = os.getenv("COOKIE_PASSWORD", "change-me")

# Allow frontend to import backend modules (split from app/)
BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

from rag_pipeline import RAGPipeline
from evaluation import custom_self_eval, save_eval_row, save_llm_metrics
from langfuse_utils import langfuse_trace_span
import analytics
from langchain.agents import initialize_agent, AgentType
from rag_tools import rag_query_tool
from web_tools import tavily_search_tool
from openai import OpenAI
import streamlit.components.v1 as components
from self_reflection import self_reflect_and_retry

# Ensure user/session IDs are persistent
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

user_id = st.session_state["user_id"]
session_id = st.session_state["session_id"]

st.set_page_config(page_title="PDF QA")
st.markdown(
    """
    <style>
    /* Sidebar styling to match form */
    [data-testid="stSidebar"] {
        background-color: var(--form-dropzone-bg) !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebarContent"] {
        background-color: var(--form-dropzone-bg) !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebarHeader"] {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: var(--form-dropzone-bg) !important;
        padding-top: 20px !important;
        color: var(--form-text-color) !important;
    }
    [data-testid="stSidebarHeader"]::before {
        content: "DocGPT";
        font-weight: 700;
        font-size: 24px;
        color: var(--form-text-color) !important;
    }
    /* Always show the sidebar collapse/expand control */
    [data-testid="stSidebarCollapseButton"] {
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stSidebarContent"] > .block-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        padding-bottom: 12px;
        background-color: var(--form-dropzone-bg) !important;
    }
    /* Hide stale duplicates only when a fresh copy exists in the same container */
    [data-testid="stElementContainer"]:has([data-stale="true"]):has([data-stale="false"]) [data-stale="true"] {
        display: none !important;
    }
    .sidebar-footer {margin-top: auto; width: 100%;}
    .composer-row {margin-top: 6px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;}
    .composer-label {font-size:12px; color:var(--form-text-color); opacity:0.7; margin-bottom:2px;}
    [data-testid="stMainBlockContainer"] > div {position: relative; height: 100vh; max-height: 100vh; padding: 0 !important;}
    .st-emotion-cache-1w723zb {padding: 0 !important;}
    /* Ensure no padding for the specific runtime-generated container seen in inspector */
    .st-emotion-cache-fis6aj {padding: 0 !important;}
    [data-testid="stMainBlockContainer"] {
        max-width: 70% !important;
        width: 70% !important;
        margin: 0 auto !important;
    }
    [data-testid="stLayoutWrapper"]:has([data-testid="stForm"]) {
        position: absolute;
        bottom: 2%;
        left: 0;
        right: 0;
        z-index: 10;
                background: transparent;
        padding-bottom: 8px;
    }
    [data-testid="stLayoutWrapper"]:has(.chat-scroll) {
        height: 60%;
        max-height: 60%;
        overflow: auto;
    }
        [data-testid="stForm"] {
            border:1px solid var(--form-border-color); 
            border-radius:12px; 
            padding:12px; 
            background: var(--form-background); 
            margin-top:0;
            color: var(--form-text-color);
        }
    .chat-scroll {height: 100%; max-height: 100%; overflow-y: auto; padding-right: 6px;}
        @media (prefers-color-scheme: light) {
            :root {
                --form-border-color: #d0d0d0;
                --form-background: #ffffff;
                --form-text-color: #1f1f1f;
                --form-dropzone-bg: #f8f9fa;
                --form-file-item-bg: #f1f3f5;
            }
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --form-border-color: #3a3a3a;
                --form-background: #1e1e1e;
                --form-text-color: #e8eaed;
                --form-dropzone-bg: #2a2a2a;
                --form-file-item-bg: #2a2a2a;
            }
        }
        [data-testid="stForm"] [data-testid="stSelectbox"] div[role="combobox"] {
            min-height: 28px;
            height: 28px;
            font-size: 12px;
            padding-top: 2px;
            padding-bottom: 2px;
        }
        [data-testid="stForm"] [data-testid="stSelectbox"] span {
            font-size: 11px;
            line-height: 28px;
        }
        [data-testid="stForm"] .stSelectbox div[data-baseweb="select"] > div {
            min-height: 28px;
            height: 28px;
            padding-top: 0;
            padding-bottom: 0;
            align-items: center;
            background: var(--form-dropzone-bg) !important;
            border-color: var(--form-border-color) !important;
        }
        [data-testid="stForm"] .stSelectbox div[data-baseweb="select"] input {
            font-size: 11px;
            line-height: 28px;
            padding-top: 0;
            padding-bottom: 0;
        }
        [data-testid="stForm"] .stSelectbox svg {
            width: 14px;
            height: 14px;
        }
        /* Text area styling for dark theme */
        [data-testid="stForm"] textarea {
            background: var(--form-dropzone-bg) !important;
            color: var(--form-text-color) !important;
            border-color: var(--form-border-color) !important;
        }
        [data-testid="stForm"] textarea::placeholder {
            color: var(--form-text-color) !important;
            opacity: 0.5 !important;
        }
        /* Checkbox styling */
        [data-testid="stForm"] [data-testid="stCheckbox"] {
            color: var(--form-text-color) !important;
        }
        /* Submit button styling */
        [data-testid="stForm"] button[kind="formSubmit"] {
            background: var(--form-dropzone-bg) !important;
            color: var(--form-text-color) !important;
            border-color: var(--form-border-color) !important;
        }
        /* Slim file uploader inside the form */
        [data-testid="stForm"] [data-testid="stFileUploader"] section {
            padding: 8px 10px !important;
            min-height: 44px;
            background: var(--form-dropzone-bg) !important;
            border-color: var(--form-border-color) !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploaderDropzone"] {
            padding: 8px 10px !important;
            min-height: 44px !important;
            height: 3rem !important;
            background: var(--form-dropzone-bg) !important;
            border-color: var(--form-border-color) !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploaderDropzone"] > div {
            padding: 0 !important;
            min-height: 44px !important;
            height: 3rem !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] section div {
            font-size: 12px !important;
            line-height: 16px !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] section div > span:first-child {
            font-weight: 600 !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] section div > span:last-child {
            font-size: 10px !important;
            line-height: 14px !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] ul {
            margin-top: 4px !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] ul li {
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] ul li svg {
            margin-right: 0 !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] > div {
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] section {
            flex: 1 1 100%;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] ul {
            display: flex !important;
            align-items: center !important;
            margin: 0 !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) {
            display: grid !important;
            grid-template-columns: 1fr 1fr !important;
            column-gap: 12px !important;
            align-items: stretch !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) label {
            grid-column: 1 / -1 !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) section {
            grid-column: 1 / 2 !important;
            width: 100% !important;
            height: 3rem !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) ul {
            grid-column: 2 / 3 !important;
            width: 100% !important;
            justify-content: flex-start !important;
            height: 3rem !important;
            display: flex !important;
            align-items: stretch !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) ul li {
            background: var(--form-file-item-bg) !important;
            border-radius: 8px !important;
            padding: 0 12px !important;
            height: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }
        /* Make the left content (filename area) stretch and truncate if needed */
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) ul li > *:first-child {
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            flex: 1 1 auto !important;
            min-width: 0 !important;
            overflow: hidden !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) ul li > *:first-child span {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            font-size: 12px !important;
        }
        /* Ensure the remove (X) button sits at the far right */
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) ul li button {
            margin-left: 12px !important;
            flex: 0 0 auto !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) > div > div {
            padding: 0 !important;
            margin: 0 !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"]:has(ul li) ul {
            padding: 0 !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] button {
            padding: 2px 8px !important;
            font-size: 12px !important;
            height: 28px !important;
            min-height: 28px !important;
        }
        [data-testid="stForm"] [data-testid="stFileUploader"] svg {
            width: 22px !important;
            height: 22px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Auth
if "token" not in st.session_state:
    st.session_state["token"] = None
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "active_panel" not in st.session_state:
    st.session_state["active_panel"] = None
if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pipeline_cache" not in st.session_state:
    st.session_state["pipeline_cache"] = {}

# Cookie manager for persistent JWT
cookies = None
if EncryptedCookieManager is not None:
    cookies = EncryptedCookieManager(prefix="docgpt", password=COOKIE_PASSWORD)
    if not cookies.ready():
        st.stop()


def clear_auth_state() -> None:
    st.session_state["token"] = None
    st.session_state["user_email"] = None
    if cookies is not None:
        cookies["jwt"] = ""
        cookies.save()


def fetch_user_email(token: str | None) -> str | None:
    if not token:
        return None
    try:
        r = requests.get(
            f"{API_BASE_INTERNAL}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code == 401:
            clear_auth_state()
            return None
        if r.ok:
            return r.json().get("email")
    except Exception:
        return None
    return None


def fetch_chat_sessions(token: str | None) -> list[dict]:
    if not token:
        return []
    try:
        r = requests.get(
            f"{API_BASE_INTERNAL}/chat/sessions",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code == 401:
            clear_auth_state()
            return []
        if r.ok:
            return r.json().get("sessions", []) or []
    except Exception:
        return []
    return []


def ensure_analytics_csvs(token: str | None) -> None:
    """Ensure `eval_results.csv` and `llm_eval_metrics.csv` exist.
    If missing, try to build them from chat sessions/messages via the backend.
    If no sessions/messages are available, write a small sample CSV so the
    dashboards render without error.
    """
    try:
        import pandas as pd
    except Exception:
        return

    base_eval = os.path.join(os.getcwd(), "eval_results.csv")
    base_llm = os.path.join(os.getcwd(), "llm_eval_metrics.csv")

    # If both files already exist, nothing to do
    if os.path.exists(base_eval) and os.path.exists(base_llm):
        return

    rows_eval = []
    rows_llm = []

    sessions = fetch_chat_sessions(token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for s in sessions or []:
        sid = s.get("id")
        try:
            r = requests.get(f"{API_BASE_INTERNAL}/chat/{sid}", headers=headers, timeout=5)
            if not r.ok:
                continue
            data = r.json()
            msgs = data.get("messages", []) or []
            # Pair user->assistant messages into Q/A rows
            for i, m in enumerate(msgs):
                if m.get("role") == "user":
                    q = m.get("content", "")
                    a = ""
                    for j in range(i + 1, len(msgs)):
                        if msgs[j].get("role") == "assistant":
                            a = msgs[j].get("content", "")
                            break
                    rows_eval.append({
                        "question": q,
                        "answer": a,
                        "faithful": True,
                        "llm_feedback": "",
                        "num_sources": 0,
                    })
                    # llm metrics placeholder row (neutral values)
                    rows_llm.append({"faithfulness": 0.0, "relevance": 0.0, "conciseness": 0.0})
        except Exception:
            continue

    # If no rows found, write a small sample so dashboard renders
    if not rows_eval:
        rows_eval = [
            {
                "question": "Sample question",
                "answer": "Sample answer",
                "faithful": True,
                "llm_feedback": "",
                "num_sources": 1,
            }
        ]
    if not rows_llm:
        rows_llm = [{"faithfulness": 0.5, "relevance": 0.5, "conciseness": 0.5}]

    try:
        pd.DataFrame(rows_eval).to_csv(base_eval, index=False)
    except Exception:
        pass
    try:
        pd.DataFrame(rows_llm).to_csv(base_llm, index=False)
    except Exception:
        pass


def get_pipeline(use_compression: bool, chain_type: str) -> RAGPipeline:
    key = (use_compression, chain_type)
    if key in st.session_state["pipeline_cache"]:
        return st.session_state["pipeline_cache"][key]
    pipeline = RAGPipeline(use_compression=use_compression, chain_type=chain_type)
    st.session_state["pipeline_cache"][key] = pipeline
    return pipeline


# Load token from cookie if present
if cookies is not None and not st.session_state["token"]:
    cookie_token = cookies.get("jwt")
    if cookie_token:
        st.session_state["token"] = cookie_token

# Handle token returned via callback (Streamlit >= 1.30)
token_param = st.query_params.get("token")
if token_param and not st.session_state["token"]:
    st.session_state["token"] = token_param

# (removed: query-param driven close control — panels are closed via sidebar actions)

# If a session is requested via query param (e.g. ?session=19), load it and clear the param
session_param = st.query_params.get("session")
if session_param:
    try:
        sid = int(session_param[0]) if isinstance(session_param, list) else int(session_param)
        if st.session_state.get("token"):
            headers = {"Authorization": f"Bearer {st.session_state['token']}"}
            r = requests.get(f"{API_BASE_INTERNAL}/chat/{sid}", headers=headers)
            if r.ok:
                data = r.json()
                st.session_state["active_session_id"] = data.get("id")
                st.session_state["active_session_is_new"] = False
                msgs = data.get("messages", []) or []
                st.session_state["messages"] = []
                for m in msgs:
                    role = m.get("role", "assistant")
                    st.session_state["messages"].append({"role": role, "content": m.get("content", "")})
                st.session_state["active_panel"] = None
    except Exception:
        pass
    st.rerun()

# If requested, create a new session via ?new=1 and open it
new_param = st.query_params.get("new")
if new_param:
    if st.session_state.get("token"):
        try:
            headers = {"Authorization": f"Bearer {st.session_state['token']}"}
            r = requests.post(f"{API_BASE_INTERNAL}/chat/session", headers=headers, timeout=10)
            if r.ok:
                st.session_state["active_session_id"] = r.json().get("session_id")
                st.session_state["active_session_is_new"] = True
                st.session_state["messages"] = []
        except Exception:
            pass
    st.rerun()

# Handle logout via query param
logout_param = st.query_params.get("logout")
if logout_param:
    clear_auth_state()
    try:
        st.query_params.pop("logout")
    except Exception:
        pass
    st.rerun()

# Ensure an active chat session exists for logged in users (created once per visit)
if st.session_state.get("token") and "active_session_id" not in st.session_state:
    try:
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        # Check if there's already an unused session
        existing_sessions = fetch_chat_sessions(st.session_state["token"])
        unused_session = None
        
        if existing_sessions:
            # Check the most recent session
            most_recent = existing_sessions[0]
            session_id = most_recent.get("id")
            # Fetch session details to check if it has messages
            r = requests.get(f"{API_BASE_INTERNAL}/chat/{session_id}", headers=headers)
            if r.ok:
                session_data = r.json()
                messages = session_data.get("messages", []) or []
                title = session_data.get("title") or ""
                # If the session has no messages or is still titled "New Chat Session", reuse it
                if len(messages) == 0 or title.startswith("New Chat Session") or title.startswith("Session"):
                    unused_session = session_data
        
        if unused_session:
            # Reuse the existing unused session
            st.session_state["active_session_id"] = unused_session.get("id")
            st.session_state["active_session_is_new"] = True
            st.session_state["messages"] = []
        else:
            # Create a new session only if there's no unused one
            r = requests.post(f"{API_BASE_INTERNAL}/chat/session", headers=headers)
            if r.ok:
                st.session_state["active_session_id"] = r.json().get("session_id")
                st.session_state["active_session_is_new"] = True
            else:
                st.session_state["active_session_id"] = None
                st.session_state["active_session_is_new"] = False
    except Exception:
        st.session_state["active_session_id"] = None
        st.session_state["active_session_is_new"] = False

with st.sidebar:
    st.markdown(
        """
        <style>
        /* Sidebar buttons to match form theme */
        .sidebar-auth {display:flex; flex-direction:column; align-items:flex-start; gap:10px;}
        .sidebar-auth {width:100%;}
        .google-btn {background:var(--form-dropzone-bg); color:var(--form-text-color); border:1px solid var(--form-border-color); padding:8px 12px; border-radius:8px; text-decoration:none; font-weight:600; display:flex; align-items:center; justify-content:center; gap:10px; width:100%; box-sizing:border-box;}
        .google-btn:hover {background:var(--form-border-color);}
        .google-icon {width:18px; height:18px;}
        .logout-btn {background:var(--form-dropzone-bg); color:var(--form-text-color); border:1px solid var(--form-border-color); padding:8px 16px; border-radius:8px; text-decoration:none; font-weight:600; display:inline-block;}
        .logout-btn:hover {background:var(--form-border-color);}
        /* All sidebar buttons default - rounded and centered */
        [data-testid="stSidebar"] .stButton > button {
            width:100%;
            background: var(--form-dropzone-bg) !important;
            color: var(--form-text-color) !important;
            border: 1px solid var(--form-border-color) !important;
            text-align: center !important;
            display: flex !important;
            justify-content: center !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--form-border-color) !important;
            border-color: var(--form-text-color) !important;
        }
        [data-testid="stSidebar"] .stButton > button > div {
            text-align: center !important;
            justify-content: center !important;
            display: flex !important;
        }
        .chat-section {color:var(--form-text-color); opacity:0.7; font-size:12px; margin:10px 0 6px; display:flex; align-items:center; gap:6px; font-weight:600; letter-spacing:.3px;}
        /* New chat button icon styling */
        [data-testid="stSidebar"] button[data-testid*="new_chat"] .plus-icon {
            color: #ffffff;
        }
        @media (prefers-color-scheme: light) {
            [data-testid="stSidebar"] button[data-testid*="new_chat"] .plus-icon {
                color: #000000;
            }
        }
        /* Chat session buttons ONLY - override to remove border radius, left align, no gaps */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stVerticalBlock"] > div {
            margin: 0 !important;
            padding: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stVerticalBlock"] .stButton {
            margin: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stVerticalBlock"] .stButton > button {
            border-radius: 0 !important;
            margin: 0 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            background: transparent !important;
            border: none !important;
            padding: 8px 12px !important;
            position: relative !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stVerticalBlock"] .stButton > button:hover {
            background: var(--form-border-color) !important;
        }
        /* Session item container with delete button */
        .session-item {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            cursor: pointer;
            background: transparent;
            transition: background 0.2s;
        }
        .session-item:hover {
            background: var(--form-border-color) !important;
        }
        .session-title {
            flex: 1;
            text-align: left;
            color: var(--form-text-color);
        }
        .delete-btn {
            opacity: 0;
            transition: opacity 0.2s;
            background: #ff4444;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            cursor: pointer;
            color: white;
            font-size: 14px;
            margin-left: 8px;
        }
        .session-item:hover .delete-btn {
            opacity: 1;
        }
        .delete-btn:hover {
            background: #cc0000;
        }
        /* Style delete button in sessions - hide by default, show on hover */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] {
            gap: 0 !important;
            margin: 0 !important;
            position: relative !important;
            align-items: stretch !important;
            display: flex !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] > div {
            padding: 0 !important;
            margin: 0 !important;
            display: flex !important;
            align-items: stretch !important;
        }
        /* Session title button - left column */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] > div:first-child {
            flex: 1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] > div:first-child button {
            padding: 8px 12px !important;
            height: 100% !important;
            width: 100% !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] > div:first-child button p {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        /* Delete button column - target last column */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] > div:last-child {
            width: 44px !important;
            min-width: 44px !important;
            max-width: 44px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            opacity: 0 !important;
            visibility: hidden !important;
            pointer-events: none !important;
            transition: opacity 0.2s ease, visibility 0.2s ease !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"]:hover > div:last-child {
            opacity: 1 !important;
            visibility: visible !important;
            pointer-events: auto !important;
            background: rgba(255, 68, 68, 0.1) !important;
        }
        /* Delete button - force all children to flex center */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] > div:last-child button {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
            height: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] > div:last-child button > div {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            text-align: center !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stHorizontalBlock"] > div:last-child button p {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            line-height: 1 !important;
            margin: 0 auto !important;
            padding: 0 !important;
            font-size: 18px !important;
            text-align: center !important;
            filter: brightness(0) saturate(100%) invert(27%) sepia(98%) saturate(7426%) hue-rotate(358deg) brightness(102%) contrast(118%) !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.chat-section) [data-testid="stVerticalBlock"] .stButton > button > div {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        /* Chat sessions list (only the list under the "Your chats" header) */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] .chat-section + .chat-list {
            display: block !important;
            max-height: 40vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            -webkit-overflow-scrolling: touch !important;
            padding-right: 4px !important;
        }
        /* Hover state for chat session buttons and sidebar action buttons */
        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] .stButton > button:focus {
            background: var(--form-border-color) !important;
            color: var(--form-text-color) !important;
        }

        /* Ensure parent sidebar blocks do not clip or become scroll containers */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            overflow: visible !important;
        }

        /* Also target the specific sidebar wrapper observed in DevTools and limit its height */
        [data-testid="stSidebar"] div.st-emotion-cache-18kf3ut {
            max-height: 40vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            -webkit-overflow-scrolling: touch !important;
        }
        /* Remove global application of max-height */
        div.st-emotion-cache-18kf3ut {
            max-height: unset !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar sessions list (only when signed in)
    if st.session_state["token"]:
        # Check if top session is "New Chat Session" (empty/unused)
        st.session_state["chat_sessions"] = fetch_chat_sessions(st.session_state["token"])
        has_unused_new_chat = False
        if st.session_state.get("chat_sessions") and len(st.session_state["chat_sessions"]) > 0:
            first_session = st.session_state["chat_sessions"][0]
            first_session_title = first_session.get("title") or ""
            if first_session_title == "New Chat Session":
                has_unused_new_chat = True
        
        # New chat button: disabled if there's already an unused "New Chat Session" at top
        if st.button("✚     New chat", use_container_width=True, key="new_chat_btn", disabled=has_unused_new_chat):
            try:
                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                r = requests.post(f"{API_BASE_INTERNAL}/chat/session", headers=headers)
                if r.ok:
                    st.session_state["active_session_id"] = r.json().get("session_id")
                    st.session_state["active_session_is_new"] = True
                    st.session_state["messages"] = []
                    # Refresh chat sessions list
                    st.session_state["chat_sessions"] = fetch_chat_sessions(st.session_state["token"])
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to create new chat: {str(e)}")
                pass
        if st.session_state.get("chat_sessions"):
            st.markdown("<div class='chat-section'>Your chats ▾</div>", unsafe_allow_html=True)
            # Wrap only the sessions list in a scrollable container so header and other buttons are excluded
            st.markdown('<div class="chat-list">', unsafe_allow_html=True)
            sessions_container = st.container()
            with sessions_container:
                for s in st.session_state["chat_sessions"]:
                    sid = s.get("id")
                    title = s.get("title") or f"Session {sid}"
                    
                    # Create columns for session title and delete button
                    col1, col2 = st.columns([0.85, 0.15])
                    
                    with col1:
                        # Render each session as a button to load it
                        if st.button(title, key=f"session_{sid}", use_container_width=True):
                            try:
                                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                                r = requests.get(f"{API_BASE_INTERNAL}/chat/{sid}", headers=headers)
                                if r.ok:
                                    data = r.json()
                                    st.session_state["active_session_id"] = data.get("id")
                                    st.session_state["active_session_is_new"] = False
                                    msgs = data.get("messages", []) or []
                                    st.session_state["messages"] = []
                                    for m in msgs:
                                        role = m.get("role", "assistant")
                                        st.session_state["messages"].append({"role": role, "content": m.get("content", "")})
                                    st.session_state["active_panel"] = None
                                    st.rerun()
                            except Exception:
                                pass
                    
                    with col2:
                        # Simple HTML delete button with form to handle click
                        if st.button("🗑️", key=f"delete_{sid}", use_container_width=True):
                            try:
                                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                                r = requests.delete(f"{API_BASE_INTERNAL}/chat/{sid}", headers=headers)
                                if r.ok:
                                    st.session_state["chat_sessions"] = fetch_chat_sessions(st.session_state["token"])
                                    # If deleted session was active, clear it
                                    if st.session_state.get("active_session_id") == sid:
                                        st.session_state["active_session_id"] = None
                                        st.session_state["messages"] = []
                                        st.rerun()
                            except Exception:
                                pass
            # close the chat-list wrapper so only the sessions themselves are scrollable
            st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar middle content actions
    if st.button("Show Analytics Dashboard", use_container_width=True, key="analytics_btn"):
        try:
            ensure_analytics_csvs(st.session_state.get("token"))
        except Exception:
            pass
        st.session_state["active_panel"] = "analytics"
        st.rerun()
    if st.button("Show LLM eval metrics", use_container_width=True, key="llm_metrics_btn"):
        try:
            ensure_analytics_csvs(st.session_state.get("token"))
        except Exception:
            pass
        st.session_state["active_panel"] = "llm"
        st.rerun()
    st.markdown("<div class=\"sidebar-footer\">", unsafe_allow_html=True)
    if st.session_state["token"]:
        if not st.session_state["user_email"]:
            st.session_state["user_email"] = fetch_user_email(st.session_state["token"])
        email = st.session_state["user_email"] or ""
        st.markdown(
            f"""
            <div class="sidebar-auth sidebar-footer">
                <div style='color:#1a7f37; font-weight:700;'>Logged in as</div>
                <div>{email}</div>
                <a class="logout-btn" href="?logout=1" target="_self" rel="noreferrer">Logout</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        google_link = f"{API_BASE_PUBLIC}/auth/google/start?redirect_to={quote(FRONTEND_BASE)}"
        st.markdown(
            f"""
            <div class="sidebar-auth sidebar-footer">
                <a class="google-btn" href="{google_link}" target="_self" rel="noreferrer">
                    Sign in with Google
                    <img class="google-icon" alt="Google" src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" />
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # Place the anonymous warning below the sign-in area (only shown when not logged in)
    if not st.session_state.get("token"):
        st.markdown(
            """
            <br>
            <div style="margin:8px 0; font-size:13px; color:var(--form-text-color);">
                ⚠️  You are using DocGPT anonymously <br> ⚠️  Your session won’t be saved.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state["token"]:
    # Intentionally do not block anonymous users; show a short note in the sidebar instead.
    pass

# Main body panel switcher — panels are shown without close buttons; use sidebar tabs to switch
if st.session_state["active_panel"]:
    if st.session_state["active_panel"] == "analytics":
        analytics.analytics_dashboard("eval_results.csv")
    else:
        analytics.llm_eval_dashboard("llm_eval_metrics.csv")

    st.stop()

client = OpenAI()

# UI layer: composer (controls) separated from chat rendering
with st.form("composer_form", clear_on_submit=True):
    # Document upload controls (stick to options section)
    uploaded_file = st.file_uploader("Upload a PDF for analysis", type=["pdf"])
    if uploaded_file is not None:
        tmp_path = None
        num_chunks = None
        # Try to build a local pipeline (may fail if Chroma/DB isn't available).
        try:
            pipeline = get_pipeline(use_compression=False, chain_type="stuff")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            # If pipeline initialized, add PDF to local index
            try:
                num_chunks = pipeline.add_pdf(tmp_path, orig_filename=uploaded_file.name)
            except Exception:
                num_chunks = None
        except Exception:
            # If local pipeline fails (e.g., chroma not initialized), fallback to backend-only flow
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
            except Exception:
                tmp_path = None
        # Persist document to backend and attach to active session if present
        # If user is logged in, persist document to backend and attach to active session
        if st.session_state.get("token"):
            try:
                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                with open(tmp_path, "rb") as fh:
                    r = requests.post(f"{API_BASE_INTERNAL}/docs/upload", headers=headers, files={"file": (uploaded_file.name, fh, "application/pdf")})
                if r.ok:
                    doc_info = r.json()
                    doc_id = doc_info.get("doc_id")
                    # Attach to active session if present
                    sid = st.session_state.get("active_session_id")
                    if sid and doc_id:
                        try:
                            a = requests.post(f"{API_BASE_INTERNAL}/chat/{sid}/attach", headers=headers, json={"doc_id": doc_id})
                        except Exception:
                            pass
            except Exception:
                pass
        shown_uploads = st.session_state.setdefault("upload_notice_shown", set())
        if uploaded_file.name not in shown_uploads:
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": f"File {uploaded_file.name} uploaded. {num_chunks} chunks loaded and indexed.",
                    "kind": "upload_notice",
                }
            )
            shown_uploads.add(uploaded_file.name)

    input_cols = st.columns([0.92, 0.08])
    with input_cols[0]:
        prompt_text = st.text_area("Message DocGPT", key="composer_text", label_visibility="collapsed", height=60, placeholder="Ask anything")
    with input_cols[1]:
        send_clicked = st.form_submit_button("➤")

    # Composer options (single-line, small labels) inside the box
    opt1, opt2, opt3, opt4 = st.columns([1, 1, 1.4, 1.6])
    with opt1:
        st.markdown("<div class='composer-label'>Chain type</div>", unsafe_allow_html=True)
        chain_type = st.selectbox(
            "Chain type",
            ("stuff", "map_reduce", "refine"),
            index=0,
            label_visibility="collapsed",
            key="chain_type_dropdown",
            kwargs={"dropUp": True}  # Force dropdown to open upwards
        )
    with opt2:
        st.markdown("<div class='composer-label'>Mode</div>", unsafe_allow_html=True)
        mode = st.selectbox(
            "Mode",
            ("RAG", "ReAct Agent"),
            index=0,
            label_visibility="collapsed",
            key="mode_dropdown",
            kwargs={"dropUp": True}  # Force dropdown to open upwards
        )
    with opt3:
        st.markdown("<div class='composer-label'>Enable Chunk Compression</div>", unsafe_allow_html=True)
        use_compression = st.checkbox("Enable chunk compression", value=False, label_visibility="collapsed")
    with opt4:
        st.markdown("<div class='composer-label'>Enable Self-reflection</div>", unsafe_allow_html=True)
        enable_reflection = st.checkbox("Enable self-reflection", value=True, label_visibility="collapsed")

# Show intro screen when there are no messages yet (new chat)
if not st.session_state.get("messages"):
    st.markdown(
        """
        <style>
          .hero { padding: 1.25rem 1.25rem; border-radius: 18px; border: 1px solid rgba(255,255,255,0.12); background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(16,185,129,0.14)); }
          .tag { display: inline-block; padding: 0.25rem 0.6rem; margin-right: 0.4rem; border-radius: 999px; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.05); }
          .card { padding: 1.05rem 1.05rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.03); height: 100%; }
          .muted { opacity: 0.85; }
          .small { font-size: 0.92rem; }
          .divider { margin: 0.75rem 0; border-bottom: 1px solid rgba(255,255,255,0.10); }
          .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.85rem; padding: 0.15rem 0.45rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.06); }
        </style>

        <div class="hero">
          <h1 style="margin:0;">🤖📄 Welcome to <span style="letter-spacing:0.2px;">DocGPT</span></h1>
          <p class="muted" style="margin:0.35rem 0 0;">Ask questions to your PDFs with citations, quality checks, and optional web search. Built to feel like a product, not a one-off chatbot.</p>
          <div style="margin-top:0.8rem;">
            <span class="tag">RAG + Citations</span>
            <span class="tag">Self-Reflection</span>
            <span class="tag">Chunk Compression</span>
            <span class="tag">Stuff / Map-Reduce / Refine</span>
            <span class="tag">ReAct + Web Search</span>
            <span class="tag">Analytics</span>
          </div>
        </div>

        <br />

        <div class="card">
          <h3 style="margin-top:0;">🚀 Quick Start (60 seconds)</h3>
          <p class="muted small">1) Upload a PDF → the app splits it into chunks and builds semantic search.<br>2) Ask a question → DocGPT retrieves the best chunks and answers with sources.<br>3) Tune settings → use Compression / Chain types / ReAct based on your document and question.</p>
          <div class="divider"></div>
          <p class="small"><b>Tip:</b> Start with <b>Compression OFF</b>, <b>Chain = Stuff</b>, <b>Self-Reflection ON</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    chat_container = st.container()
    with chat_container:
        st.markdown("<div class='chat-scroll'>", unsafe_allow_html=True)
        for msg in st.session_state["messages"]:
            role = msg.get("role", "assistant")
            with st.chat_message(role):
                content = msg.get("content", "")
                if msg.get("kind") == "upload_notice":
                    st.markdown(
                        f"<div style='color:#1a7f37; font-weight:600;'>{html.escape(content)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(content)
                sources = msg.get("sources") or []
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- **{s.get('source', '?')}**: {s.get('snippet', '')}")
        st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        components.html(
            """
            <script>
            const root = window.parent.document;
            const bottom = root.getElementById('chat-bottom');
            if (bottom) {
              setTimeout(() => { bottom.scrollIntoView({behavior: 'auto', block: 'end'}); }, 50);
            }
            </script>
            """,
            height=0,
            width=0,
        )

prompt = (prompt_text or "").strip()
if send_clicked and prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    if mode == "ReAct Agent":
        # Keep existing local agent behavior for ReAct mode
        pipeline = get_pipeline(use_compression=use_compression, chain_type=chain_type)
        rag_tool = rag_query_tool(pipeline)
        web_tool = tavily_search_tool()
        llm_agent = initialize_agent(
            tools=[rag_tool, web_tool],
            llm=pipeline.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={
                "system_message": "Always use RAGSearch for any questions about the uploaded PDF. Use WebSearch only if PDF does not contain the answer."
            },
            verbose=True,
            handle_parsing_errors=True,
        )
        agent_response = llm_agent.invoke(prompt)
        answer_text = agent_response.get("output", "")
        st.session_state["messages"].append({"role": "assistant", "content": answer_text})
    else:
        # Use backend to persist and answer the question for the active session
        sid = st.session_state.get("active_session_id")
        if not sid and st.session_state.get("token"):
            # create a session on backend for logged-in users
            try:
                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                r = requests.post(f"{API_BASE_INTERNAL}/chat/session", headers=headers)
                if r.ok:
                    sid = r.json().get("session_id")
                    st.session_state["active_session_id"] = sid
                    st.session_state["active_session_is_new"] = True
            except Exception:
                sid = None

        if sid and st.session_state.get("token"):
            # Persisted backend flow for logged-in users
            try:
                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                payload = {"content": prompt, "chain_type": chain_type, "use_compression": use_compression}
                r = requests.post(f"{API_BASE_INTERNAL}/chat/{sid}/message", headers=headers, json=payload)
                if r.ok:
                    data = r.json()
                    ans = data.get("answer") or data.get("result") or ""
                    sources = data.get("sources", []) or []
                    st.session_state["messages"].append({"role": "assistant", "content": ans, "sources": sources})
                    # Mark session as used
                    st.session_state["active_session_is_new"] = False
                else:
                    # If backend call fails, fallback to local pipeline
                    raise Exception("backend call failed")
            except Exception:
                pipeline = get_pipeline(use_compression=use_compression, chain_type=chain_type)
                base_response = pipeline.ask(prompt)
                answer_text = (
                    base_response.get("answer")
                    or base_response.get("result")
                    or base_response.get("output")
                    or ""
                )
                sources = []
                for doc_obj in base_response.get("source_documents", []) or []:
                    sources.append({
                        "source": doc_obj.metadata.get("source", "?"),
                        "snippet": doc_obj.page_content[:200].replace(chr(10), " "),
                    })
                st.session_state["messages"].append({"role": "assistant", "content": answer_text, "sources": sources})
        else:
            # Anonymous/local-only flow: keep messages in session_state only
            pipeline = get_pipeline(use_compression=use_compression, chain_type=chain_type)
            base_response = pipeline.ask(prompt)
            answer_text = (
                base_response.get("answer")
                or base_response.get("result")
                or base_response.get("output")
                or ""
            )
            sources = []
            for doc_obj in base_response.get("source_documents", []) or []:
                sources.append({
                    "source": doc_obj.metadata.get("source", "?"),
                    "snippet": doc_obj.page_content[:200].replace(chr(10), " "),
                })
            st.session_state["messages"].append({"role": "assistant", "content": answer_text, "sources": sources})

    st.rerun()

