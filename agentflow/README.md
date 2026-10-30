# AgentFlow – AI Research Agent

A supervisor-based multi-tool agent built with **LangGraph + LangChain**. A supervisor LLM routes each
question to the best tool, and a validator LLM checks the answer before it is returned. If validation fails, control
loops back to the supervisor, which must try a different tool.

```
START ─► supervisor ─┬─► rag ──────┐
            ▲        ├─► web ──────┼─► validator ─┬─ pass ─► END
            │        └─► llm ──────┘              │
            └──────────── fail (retry, untried tool) ┘   (bounded by MAX_ITERATIONS)
```

| Tool | What it does |
|---|---|
| `rag` | Retrieves chunks from your documents (FAISS + bge-small embeddings) and answers with `[n]` citations |
| `web` | Crawls URLs in the question (following same-site links 1 level), or live search results, and answers from the pages |
| `llm` | Answers from the model's own knowledge |

**Stack:** Python · LangGraph · LangChain (Anthropic / OpenAI) · FastAPI · FAISS · FastEmbed · BeautifulSoup · Docker

## Quick start (Docker)

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY or OPENAI_API_KEY
docker compose up --build
```

Open <http://localhost:8000>. Try:

- *"What is the refund policy for annual plans?"* → routes to **rag** (the sample `knowledge_base/nimbus_product_faq.md`)
- *"Summarise https://example.com"* → routes to **web**
- *"Explain what a vector database is"* → routes to **llm**

Files placed in `./knowledge_base/` are indexed at startup; you can also upload from the UI. Each answer shows the
route taken, validation status, sources and a step-by-step trace with timings.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export ANTHROPIC_API_KEY=sk-...
uvicorn app.main:app --reload       # docs at /docs
pytest                              # 20 tests, no API key needed
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/ask` | `{"question": "..."}` → `{answer, tool, validated, note, iterations, route, sources, trace}` |
| `GET` | `/api/graph` | Mermaid definition of the compiled LangGraph |
| `GET/POST/DELETE` | `/api/documents` | Manage the knowledge base |
| `GET` | `/api/health` | Status |

## Design notes

- **Router robustness:** if the supervisor's JSON is unusable, or it names a tool that is unavailable or already tried,
  a heuristic picks an untried tool. No tool runs twice per question.
- **Honest failure:** when the iteration budget runs out, the best grounded candidate is returned with
  `validated: false` and a note, never presented as verified.
- **Validator fail-open:** if the validator's output cannot be parsed, the answer is accepted and the trace says so.
  Provider/network errors are *not* swallowed (the API returns 502).
- **Crawler safety:** only http(s); every redirect hop is checked and hosts resolving to private, loopback or
  link-local addresses are blocked (SSRF guard); 1.5 MB / 5k-char limits per page.
  Set `ALLOW_PRIVATE_URLS=true` only if you really need internal sites.
- **Web search** uses DuckDuckGo's HTML endpoint without an API key. That endpoint is unofficial and can rate-limit or
  change; if it returns nothing, the web tool reports "nothing usable" and the agent falls back to another tool.
  Swap `search_web()` in `app/crawler.py` for a search API (Tavily, Brave, SerpAPI) for production use.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | – | Required for the agent |
| `LLM_PROVIDER` | `auto` | `auto`, `anthropic`, `openai` |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` | `claude-sonnet-5-5` / `gpt-4o-mini` | |
| `MAX_ITERATIONS` | `3` | Supervisor→tool→validator rounds |
| `WEB_ENABLED` | `true` | Disable the crawler tool |
| `CRAWL_MAX_PAGES` | `3` | Pages per web query |
| `TOP_K` | `4` | Chunks per RAG query |
| `KB_DIR` | `./knowledge_base` | Folder indexed at startup |

Changing the embedding model needs a fresh `/data` volume.
