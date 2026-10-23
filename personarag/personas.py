"""Persona profiles: who the simulation speaks as, and how."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import PERSONAS_DIR


@dataclass
class Persona:
    id: str
    name: str
    tagline: str = ""
    avatar: str = "🧑"
    background: str = ""
    domains: list[str] = field(default_factory=list)
    communication_style: str = ""
    known_positions: list[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return PERSONAS_DIR / self.id


def load_persona(persona_id: str) -> Persona:
    path = PERSONAS_DIR / persona_id / "persona.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No persona profile at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["id"] = persona_id
    known = Persona.__dataclass_fields__.keys()
    return Persona(**{k: v for k, v in data.items() if k in known})


def list_personas() -> list[Persona]:
    return [
        load_persona(p.name)
        for p in sorted(PERSONAS_DIR.iterdir())
        if (p / "persona.yaml").exists()
    ]
