"""AgentFlow: a LangGraph supervisor agent.

    START -> supervisor -> { rag | web | llm } -> validator -> END
                 ^                                   |
                 +----------- (fail, retry) ---------+

* supervisor: an LLM picks the next tool (falls back to heuristics if its output is unusable)
* rag / web / llm: each produces a candidate answer (with sources where applicable)
* validator: an LLM judges the candidate; on failure control returns to the supervisor,
  which must pick a tool that has not been tried yet. Bounded by MAX_ITERATIONS.
"""
from __future__ import annotations

import json
import operator
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .crawler import Page, find_urls

TOOLS = ("rag", "web", "llm")

TOOL_DESCRIPTIONS = {
    "rag": "search the user's own indexed documents (private knowledge base)",
    "web": "crawl web pages (URLs in the question, or live search results) for current / external information",
    "llm": "answer from the model's own general knowledge, without external sources",
}

ROUTER_SYSTEM = """You are the supervisor of a research agent. Pick the single best tool to answer the user's question next.

Respond with JSON only, no prose: {"next": "<tool>", "reason": "<one short sentence>"}
Choose <tool> only from the available tools listed in the message."""

RAG_SYSTEM = """You answer questions using ONLY the numbered document excerpts provided.
Cite the excerpts you use like [1] or [2][3]. Mention conditions and exceptions when present.
If the excerpts do not contain the answer, reply with exactly: NOT_FOUND"""

WEB_SYSTEM = """You answer questions using ONLY the numbered web page extracts provided.
Cite the pages you use like [1] or [2][3]. Do not use outside knowledge.
If the pages do not contain the answer, reply with exactly: NOT_FOUND"""

LLM_SYSTEM = """You are a knowledgeable research assistant. Answer the question accurately and concisely
from your own knowledge. If you are unsure, say so rather than guessing."""

VALIDATOR_SYSTEM = """You are the validator of a research agent. Judge whether the candidate answer is acceptable.

Pass only if ALL hold:
1. It directly answers the question that was asked.
2. When evidence is provided, every key claim is supported by that evidence (no invented details).
3. It is specific and not a refusal or a vague hedge.

Respond with JSON only, no prose: {"verdict": "pass" | "fail", "feedback": "<one short sentence on what is missing or wrong>"}"""


class AgentState(TypedDict, total=False):
    question: str
    next: str
    tried: list[str]
    feedback: str
    candidate: str
    candidate_tool: str
    sources: list[dict]
    best: dict | None
    verdict: str
    iterations: int
    trace: Annotated[list[dict], operator.add]


@dataclass
class AgentDeps:
    llm: Any                                    # LangChain chat model
    kb: Any                                     # KnowledgeBase-like: is_empty(), documents(), search(q, k)
    web: Callable[[str], list[Page]]            # question -> crawled pages
    web_enabled: bool = True
    top_k: int = 4
    max_iterations: int = 3


# ---------------------------------------------------------------- helpers
def _text(content: Any) -> str:
    """Normalise LangChain message content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _ask(llm, system: str, user: str) -> str:
    return _text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content).strip()


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("JSON output is not an object")
    return data


def _format_sources(sources: list[dict]) -> str:
    return "\n\n".join(f"[{s['n']}] {s['title']}\n{s['text']}" for s in sources)


def _available(deps: AgentDeps, tried: list[str]) -> list[str]:
    out = []
    for tool in TOOLS:
        if tool in tried:
            continue
        if tool == "rag" and deps.kb.is_empty():
            continue
        if tool == "web" and not deps.web_enabled:
            continue
        out.append(tool)
    return out


def _heuristic(question: str, available: list[str]) -> str:
    if "web" in available and find_urls(question):
        return "web"
    if "rag" in available:
        return "rag"
    if "web" in available and re.search(r"\b(latest|current|today|news|recent|20\d\d)\b", question.lower()):
        return "web"
    return available[0]


def _timed(fn):
    def wrapper(state):
        t0 = time.perf_counter()
        out = fn(state)
        for entry in out.get("trace", []):
            entry["ms"] = round((time.perf_counter() - t0) * 1000)
        return out

    return wrapper


# ---------------------------------------------------------------- graph
def build_agent(deps: AgentDeps):
    def supervisor(state: AgentState) -> dict:
        tried = state.get("tried", [])
        available = _available(deps, tried)
        if not available:
            return {"next": "finish",
                    "trace": [{"node": "supervisor", "decision": "finish", "reason": "no untried tools left"}]}

        docs = ", ".join(d["name"] for d in deps.kb.documents()) or "none"
        prompt = (
            f"Question: {state['question']}\n\n"
            "Available tools:\n" + "\n".join(f"- {t}: {TOOL_DESCRIPTIONS[t]}" for t in available) +
            f"\n\nIndexed documents: {docs}\nTools already tried: {', '.join(tried) or 'none'}\n"
            f"Validator feedback on the last attempt: {state.get('feedback') or 'n/a'}"
        )
        choice, reason = None, ""
        try:
            data = _extract_json(_ask(deps.llm, ROUTER_SYSTEM, prompt))
            reason = str(data.get("reason", ""))
            if str(data.get("next", "")).strip().lower() in available:
                choice = str(data["next"]).strip().lower()
        except ValueError:
            pass
        if choice is None:
            choice = _heuristic(state["question"], available)
            reason = "heuristic fallback (router output unusable)"
        return {"next": choice, "trace": [{"node": "supervisor", "decision": choice, "reason": reason}]}

    def rag(state: AgentState) -> dict:
        sources = deps.kb.search(state["question"], deps.top_k)
        candidate = ""
        if sources:
            candidate = _ask(deps.llm, RAG_SYSTEM,
                             f"Document excerpts:\n\n{_format_sources(sources)}\n\nQuestion: {state['question']}")
        return _tool_result(state, "rag", candidate, sources, f"retrieved {len(sources)} chunks")

    def web(state: AgentState) -> dict:
        pages = deps.web(state["question"])
        sources = [{"n": i, "title": p.title, "url": p.url, "text": p.text} for i, p in enumerate(pages, 1)]
        candidate = ""
        if sources:
            candidate = _ask(deps.llm, WEB_SYSTEM,
                             f"Web page extracts:\n\n{_format_sources(sources)}\n\nQuestion: {state['question']}")
        return _tool_result(state, "web", candidate, sources, f"crawled {len(sources)} pages")

    def llm_node(state: AgentState) -> dict:
        return _tool_result(state, "llm", _ask(deps.llm, LLM_SYSTEM, state["question"]), [], "answered from model knowledge")

    def _tool_result(state, tool, candidate, sources, detail) -> dict:
        if candidate.strip().upper().startswith("NOT_FOUND"):
            candidate = ""
        return {
            "candidate": candidate, "candidate_tool": tool, "sources": sources,
            "tried": [*state.get("tried", []), tool],
            "trace": [{"node": tool, "detail": detail + ("" if candidate else " - nothing usable")}],
        }

    def validator(state: AgentState) -> dict:
        candidate, tool = state.get("candidate", "").strip(), state["candidate_tool"]
        if not candidate:
            verdict, feedback = "fail", f"The {tool} tool found nothing relevant."
        else:
            evidence = "\n".join(f"[{s['n']}] {s['text'][:400]}" for s in state.get("sources", [])[:4]) or "none (general knowledge)"
            prompt = (f"Question: {state['question']}\n\nTool used: {tool}\n\nEvidence:\n{evidence}\n\n"
                      f"Candidate answer:\n{candidate}")
            try:
                data = _extract_json(_ask(deps.llm, VALIDATOR_SYSTEM, prompt))
                verdict = "pass" if str(data.get("verdict", "")).lower() == "pass" else "fail"
                feedback = str(data.get("feedback", ""))
            except ValueError:
                verdict, feedback = "pass", "validator output unparseable; accepted by default"

        best = state.get("best")
        if candidate and (verdict == "pass" or best is None or (best["tool"] == "llm" and tool != "llm")):
            best = {"answer": candidate, "tool": tool, "sources": state.get("sources", [])}
        return {
            "verdict": verdict, "feedback": feedback, "best": best,
            "iterations": state.get("iterations", 0) + 1,
            "trace": [{"node": "validator", "decision": verdict, "reason": feedback}],
        }

    def after_validator(state: AgentState) -> str:
        if state["verdict"] == "pass" or state["iterations"] >= deps.max_iterations:
            return END
        return "supervisor" if _available(deps, state.get("tried", [])) else END

    g = StateGraph(AgentState)
    g.add_node("supervisor", _timed(supervisor))
    g.add_node("rag", _timed(rag))
    g.add_node("web", _timed(web))
    g.add_node("llm", _timed(llm_node))
    g.add_node("validator", _timed(validator))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"],
                            {"rag": "rag", "web": "web", "llm": "llm", "finish": END})
    for tool in TOOLS:
        g.add_edge(tool, "validator")
    g.add_conditional_edges("validator", after_validator, {"supervisor": "supervisor", END: END})
    return g.compile()


def run_agent(graph, question: str) -> dict:
    state = graph.invoke(
        {"question": question, "tried": [], "iterations": 0, "trace": [], "feedback": "", "best": None},
        config={"recursion_limit": 40},
    )
    validated = state.get("verdict") == "pass"
    best = state.get("best")
    if best:
        answer, tool, sources = best["answer"], best["tool"], best["sources"]
        note = None if validated else f"Could not fully verify this answer. Validator: {state.get('feedback') or 'n/a'}"
    else:
        answer, tool, sources = "I couldn't find an answer with the available tools.", None, []
        note = state.get("feedback") or None
    return {
        "answer": answer, "tool": tool, "validated": validated, "note": note,
        "iterations": state.get("iterations", 0),
        "route": [e["node"] for e in state["trace"] if e["node"] in TOOLS],
        "sources": [{**s, "text": s["text"][:500]} for s in sources],
        "trace": state["trace"],
    }
