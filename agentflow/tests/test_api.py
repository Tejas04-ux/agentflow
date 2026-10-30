from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import J, fake_web
from .fakes import ScriptedChatModel, make_handler

KB = Path(__file__).resolve().parent.parent / "knowledge_base"


def client_for(tmp_path, llm=None, preload=False):
    cfg = Settings(data_dir=tmp_path / "d", kb_dir=KB if preload else tmp_path / "none",
                   embedding_backend="hash", llm_provider="none")
    return TestClient(create_app(cfg, llm=llm, web=fake_web))


def test_ask_without_llm_returns_503(tmp_path):
    with client_for(tmp_path) as c:
        assert c.get("/api/health").json()["agent_ready"] is False
        assert c.post("/api/ask", json={"question": "hi"}).status_code == 503


def test_kb_is_preloaded_and_agent_answers(tmp_path):
    llm = ScriptedChatModel(handler=make_handler(router=[J(next="rag", reason="r")],
                                                 validator=[J(verdict="pass", feedback="")]))
    with client_for(tmp_path, llm, preload=True) as c:
        assert [d["name"] for d in c.get("/api/documents").json()["documents"]] == ["nimbus_product_faq.md"]
        r = c.post("/api/ask", json={"question": "Refund policy?"}).json()
        assert r["validated"] and r["route"] == ["rag"]
        assert "flowchart" in c.get("/api/graph").json()["mermaid"] or "graph" in c.get("/api/graph").json()["mermaid"]


def test_upload_and_delete_documents(tmp_path):
    with client_for(tmp_path) as c:
        r = c.post("/api/documents", files={"file": ("a.txt", b"Hello docs")})
        assert r.status_code == 200
        assert c.post("/api/documents", files={"file": ("a.exe", b"x")}).status_code == 415
        assert c.delete(f"/api/documents/{r.json()['id']}").status_code == 200
        assert c.delete("/api/documents/nope").status_code == 404


def test_preload_is_idempotent(tmp_path):
    with client_for(tmp_path, preload=True):
        pass
    with client_for(tmp_path, preload=True) as c:
        assert len(c.get("/api/documents").json()["documents"]) == 1


def test_ui(tmp_path):
    with client_for(tmp_path) as c:
        assert "AgentFlow" in c.get("/").text
