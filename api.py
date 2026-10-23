"""REST API for PersonaRAG.

    uvicorn api:app --reload
    -> http://127.0.0.1:8000/docs
"""

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from personarag.engine import PersonaEngine, grounding_report
from personarag.panel import run_panel
from personarag.personas import list_personas, load_persona
from personarag.prompts import MODES

app = FastAPI(title="PersonaRAG", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = PersonaEngine()


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    persona: str
    question: str
    mode: str = "ask"
    history: list[Turn] = []


class PanelRequest(BaseModel):
    personas: list[str] = Field(min_length=2, max_length=5)
    question: str
    mode: str = "ask"


def _hit_json(i, h):
    return {"n": i, "label": h.label, "score": round(h.score, 3), "url": h.metadata.get("url"), "text": h.text}


def _check(persona_id: str):
    try:
        load_persona(persona_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Unknown persona '{persona_id}'")


@app.get("/personas")
def personas():
    return [
        {"id": p.id, "name": p.name, "tagline": p.tagline, "avatar": p.avatar, "chunks": engine.store.count(p.id)}
        for p in list_personas()
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
    """NDJSON stream: {"type":"sources"} first, then {"type":"token"}..., then {"type":"done"}."""
    _check(req.persona)
    hits, tokens = engine.stream(req.persona, req.question, [t.model_dump() for t in req.history], req.mode)

    def events():
        yield json.dumps({"type": "sources", "sources": [_hit_json(i, h) for i, h in enumerate(hits, 1)]}) + "\n"
        text = ""
        for tok in tokens:
            text += tok
            yield json.dumps({"type": "token", "text": tok}) + "\n"
        yield json.dumps({"type": "done", "grounding": grounding_report(text, hits)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/panel")
def panel(req: PanelRequest):
    for pid in req.personas:
        _check(pid)
    result = run_panel(engine, req.personas, req.question, req.mode)
    return {
        "summary": result.summary,
        "answers": [
            {"persona": a.persona.name, "answer": a.text, "grounding": a.grounding,
             "sources": [_hit_json(i, h) for i, h in enumerate(a.hits, 1)]}
            for a in result.answers
        ],
    }
