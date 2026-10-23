"""Panel mode: several personas answer the same question, then a moderator compares them."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .engine import Answer, PersonaEngine
from .llm import complete
from .prompts import MODERATOR_SYSTEM, moderator_turn


@dataclass
class PanelResult:
    question: str
    answers: list[Answer]
    summary: str


def run_panel(engine: PersonaEngine, persona_ids: list[str], question: str, mode: str = "ask") -> PanelResult:
    # Embedding model loads lazily; touch it once before fanning out to threads.
    engine.store.model
    with ThreadPoolExecutor(max_workers=len(persona_ids)) as pool:
        answers = list(pool.map(lambda pid: engine.ask(pid, question, mode=mode), persona_ids))
    summary = complete(
        engine.llm,
        MODERATOR_SYSTEM,
        [{"role": "user", "content": moderator_turn(question, [(a.persona.name, a.text) for a in answers])}],
        max_tokens=1500,
    )
    return PanelResult(question, answers, summary)
