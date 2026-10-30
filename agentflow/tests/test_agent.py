from app.agent import run_agent

from .conftest import J, agent

PASS = J(verdict="pass", feedback="ok")
FAIL = J(verdict="fail", feedback="not specific enough")


def test_happy_path_rag(kb):
    g, _ = agent(kb, router=[J(next="rag", reason="docs")], validator=[PASS])
    r = run_agent(g, "What is the refund policy for annual plans?")
    assert r["validated"] and r["tool"] == "rag" and r["route"] == ["rag"] and r["iterations"] == 1
    assert r["answer"] == "From docs [1]." and r["sources"]
    assert [t["node"] for t in r["trace"]] == ["supervisor", "rag", "validator"]


def test_validator_failure_loops_back_to_a_different_tool(kb):
    g, _ = agent(kb, router=[J(next="rag", reason="docs"), J(next="web", reason="try web")], validator=[FAIL, PASS])
    r = run_agent(g, "Did Nimbus launch anything new?")
    assert r["route"] == ["rag", "web"] and r["tool"] == "web" and r["validated"] and r["iterations"] == 2
    assert r["sources"][0]["url"] == "https://example.com/news"


def test_never_repeats_a_tool_even_if_router_asks(kb):
    g, _ = agent(kb, router=[J(next="rag", reason="x")], validator=[FAIL, FAIL, FAIL])
    r = run_agent(g, "question")
    assert r["route"] == ["rag", "web", "llm"]  # router kept saying rag; heuristic picked untried tools
    assert not r["validated"] and r["note"] and r["iterations"] == 3


def test_stops_at_max_iterations_with_best_unvalidated_answer(kb):
    llm_model_answer = "From model."
    g, _ = agent(kb, router=[J(next="llm", reason="x"), J(next="rag", reason="y")], validator=[FAIL])
    r = run_agent(g, "question")
    assert not r["validated"]
    assert r["answer"] != llm_model_answer or r["tool"] == "llm"  # grounded candidate preferred over bare LLM
    assert r["tool"] == "rag"


def test_garbage_router_output_uses_heuristic(kb):
    g, _ = agent(kb, router=["I think maybe docs?"], validator=[PASS])
    r = run_agent(g, "anything")
    assert r["route"] == ["rag"] and "heuristic" in r["trace"][0]["reason"]


def test_url_in_question_prefers_web_when_router_fails(kb):
    g, _ = agent(kb, router=["???"], validator=[PASS])
    assert run_agent(g, "Summarise https://example.com/news please")["route"] == ["web"]


def test_empty_kb_excludes_rag(empty_kb):
    g, _ = agent(empty_kb, router=[J(next="rag", reason="x")], validator=[PASS])
    r = run_agent(g, "What is 2+2?")
    assert "rag" not in r["route"]


def test_not_found_from_tool_counts_as_failure(kb):
    g, _ = agent(kb, rag="NOT_FOUND", router=[J(next="rag", reason="x"), J(next="llm", reason="fallback")], validator=[PASS])
    r = run_agent(g, "Who won the 1998 World Cup?")
    assert r["route"] == ["rag", "llm"] and r["tool"] == "llm" and r["validated"]
    assert r["trace"][2]["decision"] == "fail"


def test_unparseable_validator_is_accepted_by_default(kb):
    g, _ = agent(kb, router=[J(next="rag", reason="x")], validator=["looks good to me"])
    r = run_agent(g, "q")
    assert r["validated"] and "accepted by default" in r["trace"][-1]["reason"]
