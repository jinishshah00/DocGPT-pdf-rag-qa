<div align="center">

# DocGPT

### Intelligent PDF Question-Answering with RAG, Agentic Search & Self-Reflection

Upload any PDF. Ask anything. Get cited, verified answers — powered by retrieval-augmented generation, a ReAct agent with web search fallback, and LLM self-reflection for answer quality.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.20-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--Turbo-412991?logo=openai&logoColor=white)](https://openai.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)](https://trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[**Live Demo**](https://docgpt-pdf-qa-rag.up.railway.app/)

</div>

---

## The Problem

Most LLMs hallucinate when asked about documents they haven't seen. Copy-pasting PDFs into ChatGPT hits token limits, loses page-level sources, and provides no way to verify answer quality. Teams need a system that can **ground every answer in the actual document**, tell you exactly where it found the information, and self-correct when the answer isn't good enough.

**DocGPT** solves this by combining retrieval-augmented generation with per-document scoping, multi-strategy retrieval chains, an agentic fallback to web search, and a built-in self-reflection loop that evaluates and improves answers automatically.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                        │
│  Upload PDFs · Chat UI · Chain/Compression/Agent controls        │
│  Analytics dashboards · Google OAuth / JWT auth flows             │
└──────────────────────┬───────────────────────────────────────────┘
                       │  REST API (HTTP)
┌──────────────────────▼───────────────────────────────────────────┐
│                        FastAPI Backend                            │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Auth Layer │  │ RAG Service  │  │ Agent Service (ReAct)    │  │
│  │ JWT+OAuth  │  │ Ingest/Query │  │ RAGSearch + WebSearch    │  │
│  └────────────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│                         │                       │                │
│  ┌──────────────────────▼───────────────────────▼─────────────┐  │
│  │                    RAG Pipeline                             │  │
│  │  PyMuPDF → Chunking → OpenAI Embeddings → ChromaDB         │  │
│  │  ConversationalRetrievalChain (stuff|map_reduce|refine)     │  │
│  │  Optional: ContextualCompressionRetriever                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ Reflection  │  │ Analytics   │  │ Observability            │  │
│  │ Score+Retry │  │ LLM Eval   │  │ Langfuse Tracing         │  │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
        │                    │                    │
   ┌────▼────┐        ┌─────▼─────┐       ┌─────▼──────┐
   │PostgreSQL│        │ ChromaDB  │       │ OpenAI API │
   │ Users,   │        │ Vectors,  │       │ GPT-3.5    │
   │ Sessions,│        │ Embeddings│       │ Embeddings │
   │ Messages │        │ (per-doc) │       └────────────┘
   └──────────┘        └───────────┘
```

### Data Flow

1. **Upload** — PDF sent to FastAPI → SHA-256 dedup → saved to disk → `Document` row in Postgres
2. **Index** — PyMuPDF extracts text → `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) → OpenAI Embeddings → ChromaDB with `doc_id` metadata
3. **Query** — User message → retriever scoped to session's document via `doc_id` filter → `ConversationalRetrievalChain` with chat history
4. **Respond** — Answer + source citations + latency metrics → persisted as `Message` → Langfuse trace logged
5. **Reflect** *(optional)* — LLM scores the answer (1–5), and if below threshold, generates an improved response

---

## Features

### RAG Pipeline
- **PDF ingestion** with PyMuPDF extraction and recursive character splitting (1000-char chunks, 200-char overlap)
- **Three chain strategies** — `stuff` (concatenate context), `map_reduce` (per-chunk reasoning + reduction), `refine` (iterative improvement)
- **Per-document vector scoping** — ChromaDB metadata filtering ensures queries only retrieve chunks from the session's attached PDF, preventing cross-document leakage
- **MMR retrieval** (k=5) for diversity-aware document selection
- **Pipeline caching** by `(doc_id, compression, chain_type)` tuple to avoid redundant initialization

### Chunk Compression
- Optional `ContextualCompressionRetriever` with `LLMChainExtractor` — compresses retrieved chunks before feeding them to the QA chain
- Reduces token consumption while preserving relevant information

### ReAct Agent (Agentic RAG)
- LangChain `ZERO_SHOT_REACT_DESCRIPTION` agent with two tools:
  - **RAGSearch** — queries the document's vector store (gracefully handles errors so the agent keeps reasoning)
  - **WebSearch** — Tavily web search for questions beyond the PDF's scope
- System prompt enforces: *"Always try RAGSearch first; use WebSearch only if the PDF doesn't contain the answer"*

### Self-Reflection & Answer Improvement
- After generating an answer, the reflection service scores it 1–5 with justification
- If `retry_needed`, an improved answer is generated automatically at a higher temperature
- Full trace logged to Langfuse for observability

### Authentication & Authorization
- **JWT register/login** — email + bcrypt-hashed password → HS256 JWT with configurable expiry
- **Google OAuth 2.0** — full authorization-code flow with CSRF state parameter
- **Guest mode** — shared `guest@docgpt.local` account for anonymous demo access
- **Per-user message quota** — database-tracked `message_count` with server-side enforcement (15 messages)
- **Document & session ownership** — all operations verify the resource belongs to the requesting user

### Analytics & Evaluation
- **LLM-as-judge evaluation** — faithfulness checking and multi-metric scoring (faithfulness, relevance, conciseness on a 1–5 scale)
- **CSV persistence** for evaluation results and LLM metrics
- **Frontend dashboards** — pie charts (faithful vs. hallucinated), histograms (answer length distribution), metric trend lines, tabular drill-down

### Observability
- **Langfuse tracing** — captures prompts, completions, token usage, latency, and retrieval decisions
- Structured logging throughout the backend

### Extras
- **Conversational memory** — multi-turn chat history passed to the retrieval chain
- **Auto-generated session titles** — LLM creates a concise 3–8 word title from the document name after the first question
- **Chroma self-healing** — detects and recovers from corrupted SQLite schemas automatically
- **Dual-mode frontend** — runs a local RAG pipeline for anonymous sessions AND calls the backend API for authenticated sessions
- **User feedback** — thumbs up/down ratings with optional notes, stored per-message

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Streamlit, Custom CSS, `streamlit-cookies-manager`, Matplotlib, Pandas |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **AI / LLM** | LangChain 0.1.20, OpenAI GPT-3.5-Turbo, LangChain Agents (ReAct) |
| **Embeddings** | OpenAI Embeddings via `langchain-openai` |
| **Vector Store** | ChromaDB (persistent, SQLite-backed), MMR retrieval |
| **Database** | PostgreSQL 15, SQLAlchemy 2.0 ORM |
| **Auth** | JWT (python-jose), bcrypt (passlib), Google OAuth 2.0 |
| **Document Processing** | PyMuPDF, RecursiveCharacterTextSplitter |
| **Web Search** | Tavily (langchain-community integration) |
| **Observability** | Langfuse (REST API tracing) |
| **Deployment** | Docker, Docker Compose, Railway |

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone the repository

```bash
git clone https://github.com/jinishshah00/rag-pdf-qa.git
cd rag-pdf-qa
```

### 2. Create an `.env` file

Create a `.env` file in the project root:

```ini
# ── Required ──────────────────────────────────
OPENAI_API_KEY=sk-your-key-here
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ragqa
JWT_SECRET=your-secret-key

# ── Optional ──────────────────────────────────
# Google OAuth
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

# Tavily web search (enables ReAct agent web tool)
TAVILY_API_KEY=...

# Langfuse observability
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

### 3. Start the stack

```bash
docker compose -f infra/docker-compose.yml up --build
```

### 4. Access the application

| Service | URL |
|---|---|
| **Streamlit UI** | [http://localhost:8501](http://localhost:8501) |
| **FastAPI Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **PgAdmin** | [http://localhost:5050](http://localhost:5050) (login: `admin@example.com` / `admin`) |

---

## Project Structure

```
rag-pdf-qa/
├── backend/
│   ├── main.py                 # FastAPI app — all REST endpoints
│   ├── rag_pipeline.py         # Core RAG: Chroma, LangChain chains, retriever
│   ├── rag_service.py          # Service layer: ingest, query, pipeline caching
│   ├── rag_tools.py            # LangChain tools for ReAct agent (RAGSearch)
│   ├── web_tools.py            # Tavily web search tool wrapper
│   ├── agent_service.py        # ReAct agent orchestration
│   ├── reflection_service.py   # LLM answer scoring & improvement
│   ├── self_reflection.py      # Self-reflect-and-retry loop with Langfuse
│   ├── analytics_utils.py      # LLM-as-judge evaluation & CSV metrics
│   ├── evaluation.py           # Faithfulness evaluation (frontend path)
│   ├── auth.py                 # JWT utilities (create/verify tokens)
│   ├── google_oauth.py         # Google OAuth 2.0 flow
│   ├── models.py               # SQLAlchemy ORM models (6 tables)
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── db.py                   # Database engine & session factory
│   ├── config.py               # Environment config loader
│   ├── langfuse_utils.py       # Langfuse REST API tracing
│   └── observability/
│       └── langfuse_client.py  # No-op / configurable Langfuse client
├── frontend/
│   ├── app.py                  # Streamlit UI (chat, upload, auth, dashboards)
│   └── analytics.py            # Analytics dashboard rendering
├── infra/
│   ├── docker-compose.yml      # 4-service stack (db, backend, frontend, pgadmin)
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── migrations/
│       ├── 001_init.sql         # Schema: users, documents, sessions, messages, feedback, eval_runs
│       └── 002_add_user_message_count.sql
├── .env                        # Environment variables (not committed)
├── LICENSE
└── README.md
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Per-document Chroma filtering** | Each document's chunks are tagged with a `doc_id` metadata field. Queries filter by `doc_id` so cross-session PDF leakage is impossible — users only get answers from their session's document. |
| **Pipeline caching by `(doc_id, compression, chain_type)`** | Avoids re-initializing Chroma connections and LangChain chains on every request. Same config → reuse the existing pipeline. |
| **Graceful RAG tool for agents** | The `RAGSearch` tool catches all exceptions and returns error strings instead of raising. This lets the ReAct agent continue reasoning even if retrieval fails, falling back to web search. |
| **Dual-mode frontend** | The Streamlit app runs its own local RAG pipeline for anonymous users AND calls the backend API for authenticated sessions. Anonymous users get a full demo experience without requiring sign-up or backend persistence. |
| **Chroma self-healing** | On startup, the pipeline checks the SQLite schema for expected tables. If corrupted, it deletes and recreates the database automatically — zero manual intervention. |
| **Database-tracked message quota** | The `message_count` column on the User table is incremented server-side. Logged-in users are checked against the DB counter; guests fall back to server-side session counting. |
| **LLM-as-judge evaluation** | Two parallel evaluation paths — custom faithfulness checking and multi-metric scoring — provide both binary (faithful/hallucinated) and granular (1–5 scale) quality signals, persisted to CSV for dashboard visualizations. |
| **SHA-256 file deduplication** | Uploaded PDFs are hashed before storage. Same file uploaded twice → same hash → skip re-indexing. Saves compute and storage. |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-3.5-Turbo and embeddings |
| `POSTGRES_USER` | Yes | PostgreSQL username |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password |
| `POSTGRES_DB` | Yes | PostgreSQL database name |
| `JWT_SECRET` | Yes | Secret key for JWT signing (HS256) |
| `DATABASE_URL` | No | Full Postgres connection string (overrides individual PG vars) |
| `GOOGLE_CLIENT_ID` | No | Google OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | No | Google OAuth 2.0 client secret |
| `GOOGLE_REDIRECT_URI` | No | Google OAuth callback URL |
| `TAVILY_API_KEY` | No | Tavily API key (enables web search in ReAct agent) |
| `LANGFUSE_HOST` | No | Langfuse instance URL |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `FRONTEND_BASE` | No | Frontend URL (for OAuth redirects) |
| `API_BASE_INTERNAL` | No | Backend URL for service-to-service calls |
| `API_BASE_PUBLIC` | No | Public-facing backend URL |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built by [Jinish Shah](mailto:jinishshah00@gmail.com)

</div>
