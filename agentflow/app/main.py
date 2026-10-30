"""AgentFlow FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AgentDeps, build_agent, run_agent
from .config import Settings
from .crawler import research_web
from .embeddings import get_embedder
from .knowledge import KnowledgeBase
from .loaders import UnsupportedFileType
from .models import get_chat_model

log = logging.getLogger("agentflow")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(settings: Settings | None = None, llm=None, web=None) -> FastAPI:
    """`llm` and `web` can be injected (used by the tests); otherwise built from settings."""
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        embedder = get_embedder(cfg.embedding_backend, cfg.embedding_model)
        kb = KnowledgeBase(cfg.data_dir, embedder, cfg.chunk_size, cfg.chunk_overlap)
        added = kb.preload(cfg.kb_dir)
        chat_model = llm if llm is not None else get_chat_model(cfg)
        app.state.kb, app.state.llm_ready = kb, chat_model is not None
        app.state.provider = cfg.resolved_provider() if llm is None else "injected"
        app.state.graph = None
        if chat_model is not None:
            deps = AgentDeps(
                llm=chat_model, kb=kb,
                web=web or (lambda q: research_web(q, cfg.crawl_max_pages, cfg.allow_private_urls)),
                web_enabled=cfg.web_enabled, top_k=cfg.top_k, max_iterations=cfg.max_iterations,
            )
            app.state.graph = build_agent(deps)
        log.info("Ready: llm=%s, kb_docs=%d (+%d preloaded)", app.state.provider, len(kb.documents()), added)
        yield

    app = FastAPI(title="AgentFlow", description="Supervisor-based research agent (LangGraph)",
                  version="1.0.0", lifespan=lifespan)

    @app.get("/api/health")
    def health(request: Request):
        s = request.app.state
        return {"status": "ok", "llm_provider": s.provider, "agent_ready": s.graph is not None,
                "documents": len(s.kb.documents()), "web_enabled": cfg.web_enabled}

    @app.post("/api/ask")
    def ask(request: Request, body: AskRequest):
        graph = request.app.state.graph
        if graph is None:
            raise HTTPException(503, "No LLM configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY and restart.")
        try:
            return run_agent(graph, body.question.strip())
        except Exception as exc:  # provider/network failures
            log.exception("agent run failed")
            raise HTTPException(502, f"Agent run failed: {exc}") from exc

    @app.get("/api/graph")
    def graph_definition(request: Request):
        graph = request.app.state.graph
        if graph is None:
            raise HTTPException(503, "Agent not initialised (no LLM configured)")
        return {"mermaid": graph.get_graph().draw_mermaid()}

    @app.get("/api/documents")
    def list_documents(request: Request):
        return {"documents": request.app.state.kb.documents()}

    @app.post("/api/documents")
    def upload_document(request: Request, file: UploadFile = File(...)):
        name = Path(file.filename or "upload").name
        data = file.file.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "File exceeds the 20 MB limit")
        try:
            return request.app.state.kb.add(name, data)
        except UnsupportedFileType as exc:
            raise HTTPException(415, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(422, f"Could not read '{name}': {exc}") from exc

    @app.delete("/api/documents/{doc_id}")
    def delete_document(request: Request, doc_id: str):
        if not request.app.state.kb.delete(doc_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": doc_id}

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
