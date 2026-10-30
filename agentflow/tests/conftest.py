from pathlib import Path

import pytest

from app.agent import AgentDeps, build_agent
from app.config import Settings
from app.crawler import Page
from app.embeddings import HashBackend
from app.knowledge import KnowledgeBase

from .fakes import ScriptedChatModel, make_handler

KB_FILE = Path(__file__).resolve().parent.parent / "knowledge_base" / "nimbus_product_faq.md"


@pytest.fixture()
def kb(tmp_path):
    base = KnowledgeBase(tmp_path, HashBackend())
    base.add(KB_FILE.name, KB_FILE.read_bytes())
    return base


@pytest.fixture()
def empty_kb(tmp_path):
    return KnowledgeBase(tmp_path / "empty", HashBackend())


def fake_web(question):
    return [Page(url="https://example.com/news", title="Example News", text="Nimbus launched Nimbus Vault in 2026.")]


def agent(kb, **handler_kwargs):
    llm = ScriptedChatModel(handler=make_handler(**handler_kwargs))
    return build_agent(AgentDeps(llm=llm, kb=kb, web=fake_web, max_iterations=3)), llm


def J(**kw):
    import json
    return json.dumps(kw)


@pytest.fixture()
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "d", kb_dir=tmp_path / "none", embedding_backend="hash", llm_provider="none")
