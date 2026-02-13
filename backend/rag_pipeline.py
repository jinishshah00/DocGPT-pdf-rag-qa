import os
import logging
import sqlite3

# Disable Chroma telemetry to avoid noisy posthog errors
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("CHROMA_ENABLE_TELEMETRY", "FALSE")

# Silence telemetry loggers
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

try:
    from chromadb.telemetry import posthog

    def _noop(*_args, **_kwargs):
        return None

    if hasattr(posthog, "capture"):
        posthog.capture = _noop
    if hasattr(posthog, "posthog"):
        try:
            posthog.posthog.disabled = True
        except Exception:
            pass
except Exception:
    pass

from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from backend.config import PERSIST_DIR

class RAGPipeline:
    def __init__(self, use_compression=False, chain_type="stuff"):
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        # Initialize Chroma and retriever via helper so we can recreate on schema errors
        self._ensure_chroma_schema(PERSIST_DIR)
        self._init_chroma_client()

        if use_compression:
            compressor = LLMChainExtractor.from_llm(self.llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.base_retriever
            )
        else:
            self.retriever = self.base_retriever

        self.chat_history = []
        self.chain_type = chain_type
        if chain_type == "map_reduce":
            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=True,
            )
        else:
            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=True,
                chain_type=chain_type
            )

    def _init_chroma_client(self):
        """Initialize or reinitialize the Chroma client and retriever."""
        client_settings = Settings(anonymized_telemetry=False)
        self.db = Chroma(
            embedding_function=self.embedding,
            persist_directory=PERSIST_DIR,
            client_settings=client_settings,
        )
        self.base_retriever = self.db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "score_threshold": 0.7})

        # Wrap base retriever with compression if requested earlier
        # Note: when reinitializing we keep the same retriever type as before
        try:
            # If self.retriever exists and was a compression retriever, preserve that
            if isinstance(getattr(self, "retriever", None), ContextualCompressionRetriever):
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=LLMChainExtractor.from_llm(self.llm),
                    base_retriever=self.base_retriever,
                )
            else:
                self.retriever = self.base_retriever
        except Exception:
            # Fallback: use base retriever
            self.retriever = self.base_retriever

    @staticmethod
    def _ensure_chroma_schema(persist_dir: str) -> None:
        """Reset broken/empty chroma sqlite db to avoid 'no such table: collections'."""
        os.makedirs(persist_dir, exist_ok=True)
        db_path = os.path.join(persist_dir, "chroma.sqlite3")
        if not os.path.exists(db_path):
            return
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
            required = {"collections", "tenants", "databases"}
            has_required = required.issubset(tables)
        except Exception:
            has_required = False
        finally:
            try:
                conn.close()
            except Exception:
                pass
        if not has_required:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(db_path + suffix)
                except FileNotFoundError:
                    pass

    def add_pdf(self, pdf_path, orig_filename=None, doc_id=None):
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        # Set metadata to original filename
        if orig_filename is not None:
            for d in docs:
                d.metadata["source"] = orig_filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        # Propagate metadata to chunks (if necessary)
        for c in chunks:
            c.metadata["source"] = orig_filename if orig_filename else pdf_path
            if doc_id is not None:
                c.metadata["doc_id"] = str(doc_id)

        try:
            self.db.add_documents(chunks)
        except Exception as e:
            msg = str(e)
            if "no such table" in msg or "collection" in msg or "tenant" in msg:
                # Recreate schema and retry once
                self._ensure_chroma_schema(PERSIST_DIR)
                # Reinitialize client and retriever then retry
                self._init_chroma_client()
                self.db.add_documents(chunks)
            else:
                raise
        return len(chunks)

    def ask(self, query, doc_id=None):
        # Apply per-query metadata filter to scope retrieval to the session's document
        if doc_id is not None:
            doc_filter = {"doc_id": str(doc_id)}
            self.base_retriever.search_kwargs["filter"] = doc_filter
            if isinstance(self.retriever, ContextualCompressionRetriever):
                self.retriever.base_retriever.search_kwargs["filter"] = doc_filter
        else:
            self.base_retriever.search_kwargs.pop("filter", None)
            if isinstance(self.retriever, ContextualCompressionRetriever):
                self.retriever.base_retriever.search_kwargs.pop("filter", None)

        # Try invoking QA chain; on DB schema errors, attempt to recreate Chroma and retry once
        try:
            response = self.qa.invoke({"question": query, "chat_history": self.chat_history})
        except Exception as e:
            msg = str(e)
            if "no such table" in msg or "collection" in msg or "tenant" in msg:
                # Recreate schema and reinitialize client/retriever and QA chain
                try:
                    self._ensure_chroma_schema(PERSIST_DIR)
                    self._init_chroma_client()
                    # Rebuild QA chain with same chain_type
                    self.qa = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.retriever,
                        return_source_documents=True,
                        chain_type=self.chain_type,
                    )
                    response = self.qa.invoke({"question": query, "chat_history": self.chat_history})
                except Exception:
                    raise
            else:
                raise

        # Append to history if answer present
        try:
            self.chat_history.append((query, response.get("answer") or response.get("result") or ""))
        except Exception:
            pass
        return response
