# DocGPT (RAG PDF QA)

DocGPT is a small product-style Retrieval-Augmented Generation (RAG) application that lets you upload PDFs and ask questions with citations, optional web search (ReAct), chunk compression, and quality checks (self-reflection and evaluation). It includes a Streamlit-based frontend and a FastAPI backend with Postgres for persistence and Chroma for vector search.

---

## Features

- Upload PDFs and build a semantic index (local Chroma vector store).
- Ask questions with sources/citations (RAG): `stuff`, `map_reduce`, `refine` chain strategies.
- Optional chunk compression (reduces token usage and focuses context).
- ReAct Agent mode: combines RAG search with external web search tools when the PDF doesn't contain the answer.
- Conversational memory (chat sessions saved per authenticated user).
- Anonymous mode: complete, usable session stored only in the browser session (not persisted to the backend or DB).
- Built-in evaluation and analytic endpoints for saving LLM metrics and evaluation rows.
- PgAdmin included for DB browsing when running locally with Docker.

---

## High-level architecture

- Frontend: Streamlit app at `frontend/app.py`. Provides the UI for uploading PDFs, composing queries, configuring options (chain type, compression, self-reflection, ReAct), and viewing chat history and sources.
- Backend: FastAPI app at `backend/main.py` which exposes endpoints to upload documents, create/load chat sessions, and post chat messages. The backend persists users, documents, sessions, and messages to Postgres via SQLAlchemy.
- Vector DB: Chroma (file-backed SQLite store) used to persist embeddings; persisted under the Docker volume `backend_chroma` (mapped to `/app/chroma_db` in the backend container).
- Retrieval / LLM stack: LangChain wrappers using `langchain-openai` and `langchain-chroma`, with `ChatOpenAI` for LLMs (OpenAI API key required). The RAG pipeline is implemented in `backend/rag_pipeline.py` and the query tools for agents in `backend/rag_tools.py`.
- Authentication: Basic JWT flows + Google OAuth support found under `backend/google_oauth.py` and `backend/auth.py`.

---

## Tech stack

- Python 3.11
- Streamlit (frontend)
- FastAPI + Uvicorn (backend)
- PostgreSQL (persistent DB)
- Chroma (vector store)
- LangChain (chains, retrievers)
- OpenAI (LLMs) via `langchain-openai`
- Docker + docker-compose for local deployment

Key Python dependency versions (as used in repo):
- `langchain==0.1.20`
- `langchain-openai==0.1.7`
- `langchain-chroma==0.1.2`
- `streamlit` (see `frontend/requirements.txt`)

---

## Important files

- `frontend/app.py` — Streamlit UI and client logic.
- `backend/main.py` — FastAPI app and endpoints.
- `backend/rag_pipeline.py` — RAG pipeline: Chroma client initialization, text splitting, embedding, retriever, and ConversationalRetrievalChain wiring.
- `backend/rag_service.py` — Orchestration glue calling the pipeline and persisting messages.
- `backend/rag_tools.py` — Tools exposed to ReAct agents (safe error handling included).
- `infra/docker-compose.yml` — Docker Compose stack (Postgres, backend, frontend, pgadmin).
- `backend/requirements.txt`, `frontend/requirements.txt` — Python dependencies.

---

## Environment variables

Set these before running (or provide via an `.env` file):

- `OPENAI_API_KEY` — OpenAI API key for LLMs.
- `POSTGRES_USER` (default: `postgres`)
- `POSTGRES_PASSWORD` (default: `postgres`)
- `POSTGRES_DB` (default: `ragqa`)
- `JWT_SECRET` — secret to sign user JWTs (default `change-me`).
- Google OAuth (optional): `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`.
- `TAVILY_API_KEY` — optional for ReAct web search integration.

- Langfuse / tracing keys (optional): `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
