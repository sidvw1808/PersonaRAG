"""The persona engine: retrieve evidence -> build prompt -> generate -> score grounding."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import config
from .llm import LLM, complete, get_llm
from .personas import Persona, load_persona
from .prompts import system_prompt, user_turn
from .store import Hit, VectorStore

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TRAILING_CITES = re.compile(r"([.!?])\s*((?:\[\d+\]\s*)+)")


@dataclass
class Answer:
    persona: Persona
    question: str
    text: str
    hits: list[Hit]
    grounding: dict = field(default_factory=dict)

    @property
    def cited_hits(self) -> list[tuple[int, Hit]]:
        nums = self.grounding.get("cited", [])
        return [(n, self.hits[n - 1]) for n in nums if 0 < n <= len(self.hits)]


def grounding_report(text: str, hits: list[Hit]) -> dict:
    """Cheap, model-free grounding signals shown in the UI and used by the evaluator."""
    cited = sorted({int(n) for n in _CITATION.findall(text)})
    valid = [n for n in cited if 0 < n <= len(hits)]
    # Models often write "...vehicles. [1]"; move such trailing citations inside the sentence
    # they belong to, so the split below doesn't strand them in a fragment of their own.
    normalized = _TRAILING_CITES.sub(r" \2\1", text.strip())
    sentences = [s for s in _SENTENCE.split(normalized) if len(s.split()) >= 4]
    with_cite = sum(1 for s in sentences if _CITATION.search(s))
    coverage = with_cite / len(sentences) if sentences else 0.0
    top = max((h.score for h in hits), default=0.0)

    if not hits or not valid:
        level = "ungrounded"
    elif coverage >= 0.5 and top >= 0.6:
        level = "strong"
    elif coverage >= 0.25:
        level = "partial"
    else:
        level = "weak"
    return {
        "level": level,
        "cited": valid,
        "invalid_citations": [n for n in cited if n not in valid],
        "citation_coverage": round(coverage, 2),
        "top_similarity": round(top, 3),
        "sources_retrieved": len(hits),
    }


class PersonaEngine:
    def __init__(self, store: VectorStore | None = None, llm: LLM | None = None):
        self.store = store or VectorStore()
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def retrieve(
        self, persona_id: str, question: str, history: list[dict] | None = None, mode: str | None = None
    ) -> list[Hit]:
        query = question
        # Short follow-ups ("why?", "what about Mars?") lack context; borrow the previous question.
        prev = [m["content"] for m in (history or []) if m["role"] == "user"]
        if prev and len(question.split()) < 8:
            query = f"{prev[-1]} {question}"
        return self.store.search(persona_id, query, mode=mode)

    def _messages(self, question: str, hits: list[Hit], history: list[dict] | None, mode: str):
        # History holds plain text turns; only the newest user turn carries the sources block.
        recent = (history or [])[-config.HISTORY_TURNS * 2 :]
        msgs = [{"role": m["role"], "content": m["content"]} for m in recent]
        msgs.append({"role": "user", "content": user_turn(question, hits, mode)})
        return msgs

    def stream(
        self, persona_id: str, question: str, history: list[dict] | None = None, mode: str = "ask"
    ) -> tuple[list[Hit], Iterator[str]]:
        """Returns the retrieved sources immediately, plus a token stream for the answer."""
        persona = load_persona(persona_id)
        hits = self.retrieve(persona_id, question, history)
        tokens = self.llm.stream(system_prompt(persona), self._messages(question, hits, history, mode))
        return hits, tokens

    def ask(
        self, persona_id: str, question: str, history: list[dict] | None = None, mode: str = "ask"
    ) -> Answer:
        persona = load_persona(persona_id)
        hits = self.retrieve(persona_id, question, history)
        text = complete(self.llm, system_prompt(persona), self._messages(question, hits, history, mode))
        return Answer(persona, question, text, hits, grounding_report(text, hits))
