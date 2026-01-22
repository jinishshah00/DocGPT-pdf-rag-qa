import os
import sys
import tempfile, uuid
import warnings
import logging
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv

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
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
FRONTEND_BASE = os.getenv("FRONTEND_BASE", "http://localhost:8501")

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
from self_reflection import self_reflect_and_retry

# Ensure user/session IDs are persistent
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

user_id = st.session_state["user_id"]
session_id = st.session_state["session_id"]

st.set_page_config(page_title="PDF QA")
st.title("PDF Question Answering")

# Auth
if "token" not in st.session_state:
    st.session_state["token"] = None

with st.sidebar:
    st.subheader("Auth")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        r = requests.post(f"{API_BASE}/auth/register", json={"email": email, "password": password})
        if r.ok:
            st.session_state["token"] = r.json()["access_token"]
            st.success("Registered and logged in")
        else:
            st.error(r.text)
    if st.button("Login"):
        r = requests.post(f"{API_BASE}/auth/login", json={"email": email, "password": password})
        if r.ok:
            st.session_state["token"] = r.json()["access_token"]
            st.success("Logged in")
        else:
            st.error(r.text)
    google_link = f"{API_BASE}/auth/google/start?redirect_to={quote(FRONTEND_BASE)}"
    st.link_button("Sign in with Google", google_link)

# Handle token returned via callback (Streamlit >= 1.30)
token_param = st.query_params.get("token")
if token_param and not st.session_state["token"]:
    st.session_state["token"] = token_param
    st.success("Logged in with Google")
    try:
        st.query_params.pop("token")
    except Exception:
        pass

if not st.session_state["token"]:
    st.info("Login to continue.")
    st.stop()

use_compression = st.checkbox("Enable chunk compression (LLM compression)", value=False)
chain_type = st.selectbox("Choose chain combination type:", ("stuff", "map_reduce", "refine"))
enable_reflection = st.checkbox("Enable Self-Reflection", value=True)
client = OpenAI()

# Only create the pipeline once per session
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = RAGPipeline(use_compression=use_compression, chain_type=chain_type)
pipeline = st.session_state["pipeline"]

if st.button("Clear Chat History"):
    pipeline.chat_history.clear()
    st.success("Chat history cleared.")

uploaded_file = st.file_uploader("Upload a PDF for analysis", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    num_chunks = pipeline.add_pdf(tmp_path, orig_filename=uploaded_file.name)
    st.success(f"File {uploaded_file.name} uploaded. {num_chunks} chunks loaded and indexed.")

st.markdown("---")
query = st.text_input("Enter your question:", "What information do you have?")

if st.button("Get Answer") and query:
    with st.spinner(f"Working ({chain_type})..."):
        base_response = pipeline.ask(query)
        if enable_reflection:
            answer, reflection_comment = self_reflect_and_retry(client, query)
            result = {
                "result": answer,
                "source_documents": base_response["source_documents"]
            }
        else:
            result = {
                "result": base_response["result"],
                "source_documents": base_response["source_documents"]
            }

        doc_objects = result["source_documents"]
        sources = [doc.metadata.get('source', '?') for doc in doc_objects]

        langfuse_trace_span(
            user_id=user_id,
            session_id=session_id,
            query=query,
            chain_type=chain_type,
            use_compression=use_compression,
            result=result,
            sources=sources
        )

        st.success(result["result"]['answer'])

        if enable_reflection and reflection_comment:
            st.markdown(f"**Self-Reflection Retry Applied**")
            st.markdown(f"*LLM Evaluation Comment:* {reflection_comment}")

        st.markdown("---")
        st.markdown("### Sources:")
        for doc in doc_objects:
            st.markdown(f"- **{doc.metadata.get('source', '?')}**: {doc.page_content[:200].replace(chr(10), ' ')} ...")

        faithful, llm_feedback = custom_self_eval(query, result["result"], doc_objects)
        st.markdown(f"**Self-Eval (semantic faithfulness):** {'✅ Faithful' if faithful else '❌ Hallucination'}")
        if llm_feedback:
            st.markdown(f"*LLM comment: {llm_feedback}*")

        save_eval_row(query, result["result"], doc_objects, faithful, llm_feedback)
        save_llm_metrics(query, result["result"], doc_objects)

st.markdown("---")
st.markdown("### ReAct Agent Demo")

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
    handle_parsing_errors=True
)

user_query = st.text_input("Ask the ReAct agent:", "What information do you have?", key="react_agent_input")
if st.button("Get ReAct Agent Answer"):
    with st.spinner("Agent is reasoning..."):
        agent_response = llm_agent.invoke(user_query)
        st.success(agent_response["output"])

if st.checkbox("Show analytics dashboard"):
    analytics.analytics_dashboard("eval_results.csv")

if st.checkbox("Show LLM eval metrics"):
    analytics.llm_eval_dashboard("llm_eval_metrics.csv")
