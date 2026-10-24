"""REST API for PersonaRAG.

    uvicorn api:app --reload
    -> http://127.0.0.1:8000/docs

Streaming endpoints return NDJSON: one JSON object per line.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from personarag import config
from personarag.engine import PersonaEngine, grounding_report
from personarag.llm import complete
from personarag.panel import run_panel
from personarag.personas import list_personas, load_persona
from personarag.prompts import MODERATOR_SYSTEM, MODES, moderator_turn

app = FastAPI(title="PersonaRAG", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = PersonaEngine()


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    persona: str
    question: str = Field(min_length=1, max_length=2000)
    mode: str = "ask"
    history: list[Turn] = []


class PanelRequest(BaseModel):
    personas: list[str] = Field(min_length=2, max_length=5)
    question: str = Field(min_length=1, max_length=2000)
    mode: str = "ask"


def _hit_json(i, h):
    m = h.metadata
    return {
        "n": i,
        "label": h.label,
        "title": m.get("title") or m.get("source_id"),
        "kind": m.get("kind"),
        "date": m.get("date"),
        "url": m.get("url"),
        "score": round(h.score, 3),
        "rerank": None if h.rerank is None else round(h.rerank, 2),
        "text": h.text,
    }


def _line(obj) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _check(persona_id: str):
    try:
        load_persona(persona_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Unknown persona '{persona_id}'")


@app.get("/config")
def get_config():
    return {
        "llm": engine.llm.name,
        "embedding_model": config.EMBED_MODEL,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "top_k": config.TOP_K,
    }


@app.get("/personas")
def personas():
    return [
        {
            "id": p.id,
            "name": p.name,
            "tagline": p.tagline,
            "avatar": p.avatar,
            "domains": p.domains,
            "chunks": engine.store.count(p.id),
        }
        for p in list_personas()
    ]


@app.get("/personas/{persona_id}/sources")
def persona_sources(persona_id: str):
    _check(persona_id)
    return [
        {k: s.get(k) for k in ("source_id", "title", "kind", "date", "url", "chunks")}
        for s in engine.store.sources(persona_id)
    ]


@app.get("/modes")
def modes():
    return MODES


@app.post("/chat")
def chat(req: ChatRequest):
    _check(req.persona)
    ans = engine.ask(req.persona, req.question, [t.model_dump() for t in req.history], req.mode)
    return {
        "answer": ans.text,
        "grounding": ans.grounding,
        "sources": [_hit_json(i, h) for i, h in enumerate(ans.hits, 1)],
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """NDJSON: {"type":"sources"} first, then {"type":"token"}..., then {"type":"done"} or {"type":"error"}."""
    _check(req.persona)
    hits, tokens = engine.stream(req.persona, req.question, [t.model_dump() for t in req.history], req.mode)

    def events():
        yield _line({"type": "sources", "sources": [_hit_json(i, h) for i, h in enumerate(hits, 1)]})
        text = ""
        try:
            for tok in tokens:
                text += tok
                yield _line({"type": "token", "text": tok})
        except Exception as exc:  # noqa: BLE001 - report LLM failures to the client
            yield _line({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            return
        yield _line({"type": "done", "grounding": grounding_report(text, hits)})

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/panel")
def panel(req: PanelRequest):
    for pid in req.personas:
        _check(pid)
    result = run_panel(engine, req.personas, req.question, req.mode)
    return {
        "summary": result.summary,
        "answers": [
            {"persona": a.persona.id, "answer": a.text, "grounding": a.grounding,
             "sources": [_hit_json(i, h) for i, h in enumerate(a.hits, 1)]}
            for a in result.answers
        ],
    }


@app.post("/panel/stream")
def panel_stream(req: PanelRequest):
    """NDJSON: one {"type":"answer"} per panelist as each finishes, then {"type":"summary"}."""
    for pid in req.personas:
        _check(pid)

    def events():
        engine.store.model  # load the embedding model once before threads share it
        answers = {}
        try:
            with ThreadPoolExecutor(max_workers=len(req.personas)) as pool:
                futures = {pool.submit(engine.ask, pid, req.question, None, req.mode): pid for pid in req.personas}
                for fut in as_completed(futures):
                    a = fut.result()
                    answers[a.persona.id] = a
                    yield _line({
                        "type": "answer", "persona": a.persona.id, "answer": a.text, "grounding": a.grounding,
                        "sources": [_hit_json(i, h) for i, h in enumerate(a.hits, 1)],
                    })
            ordered = [answers[pid] for pid in req.personas]
            summary = complete(
                engine.llm, MODERATOR_SYSTEM,
                [{"role": "user", "content": moderator_turn(req.question, [(a.persona.name, a.text) for a in ordered])}],
                max_tokens=1500,
            )
            yield _line({"type": "summary", "text": summary})
        except Exception as exc:  # noqa: BLE001
            yield _line({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(events(), media_type="application/x-ndjson")


# Serve the built React app (frontend/dist, from `npm run build`) at /, so one server runs everything.
# Registered last: the API routes above take precedence over the static files.
_DIST = config.ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
