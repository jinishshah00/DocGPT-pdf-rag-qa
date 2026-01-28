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

Note: Docker Compose in `infra/docker-compose.yml` maps some of these variables; you can provide them via an `.env` file in the project root or your shell environment.

---

## Local development — Docker (recommended)

This repository is set up to run everything via Docker Compose. From the repo root run:

```bash
# Build and start services (frontend, backend, Postgres, pgadmin)
docker compose -f infra/docker-compose.yml up --build
```

Services and ports:
- Frontend (Streamlit): http://localhost:8501
- Backend (FastAPI): http://localhost:8000
- Postgres: localhost:5432
- PgAdmin: http://localhost:5050 (default: `admin@example.com` / `admin`)

Volumes are configured for persistence:
- Postgres data: `pgdata` volume
- Chroma DB: `backend_chroma` volume (persisted under `/app/chroma_db` in the backend container)
- Uploaded documents / storage: `backend_storage` volume

Troubleshooting notes:
- Chroma: if you see errors like `OperationalError: no such table: collections` the pipeline will attempt to recreate the Chroma schema and reinitialize the client. This project contains code to proactively reinitialize a broken Chroma sqlite file.
- If you upgrade `langchain` versions, some classes/arguments may have changed; the code pins a set of compatible versions in `requirements.txt`.

---

## Running locally without Docker (optional)

1. Create and activate a Python virtualenv (Python 3.11 recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

2. Start Postgres locally (or use a hosted DB) and set `DATABASE_URL` accordingly.
3. Start the backend:

```bash
cd backend
uvicorn backend.main:app --reload --port 8000
```

4. Start the frontend (Streamlit):

```bash
cd frontend
streamlit run app.py
```

This mode is useful for quick iteration, but Docker gives a more reproducible environment.

---

## Usage (quick)

1. Open the Streamlit app (`http://localhost:8501`).
2. Upload a PDF using the composer upload control.
3. Choose chain strategy: `Stuff` (fast), `Map-Reduce` (comprehensive), or `Refine` (iterative refinement).
4. Toggle `Chunk Compression` to compress retrieved chunks (useful for large PDFs).
5. Toggle `ReAct Agent` mode if you want the app to use a web search tool when the PDF lacks the answer.
6. Ask a question in the message box and submit.

Anonymous mode: if you do not sign in, you can still upload and query PDFs; the session is kept only in the Streamlit session and is not persisted to the backend. If you sign in with Google, chat sessions, documents, and messages will be saved to Postgres.

---

## Developer notes & tips

- The RAG pipeline is implemented in `backend/rag_pipeline.py`. It uses Chroma for vector storage; embeddings are created via `OpenAIEmbeddings`.
- The ReAct agent uses `backend/rag_tools.rag_query_tool` which now returns safe error messages instead of raising, so agent execution won't crash if the RAG step fails.
- To reset Chroma (when corrupted), the pipeline will remove the local `chroma.sqlite3` and its -wal/-shm files and recreate a fresh DB on demand.
- If you want to run experiments with different LLM providers or LangChain versions, update the pinned versions in `requirements.txt` and test.

---

## Contributing

- Fork, branch, and send PRs.
- Keep dependency changes minimal and test running with Docker Compose.

---

If you want, I can also:
- Add example env file (`.env.example`) with recommended variables.
- Add a short GIF walkthrough or screenshots for the README.
- Provide a `Makefile` or simple `./dev.sh` script to run the stack with one command.

