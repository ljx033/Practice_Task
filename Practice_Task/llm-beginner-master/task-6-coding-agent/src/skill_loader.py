"""Tiny progressive-disclosure Skill loader.

Layout:

    src/skills/
      code-review/SKILL.md
      test-runner/SKILL.md
      pr-description-writer/SKILL.md

Each SKILL.md starts with YAML front-matter:

    ---
    name: code-review
    description: Use when ...
    ---
    <body>

The loader reads *only* metadata up front and defers the body until
`load(name)` is called. `match(task)` picks the top-K skills by a cheap
keyword-overlap score, so the agent never pays for loading a body it can't use.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover
    yaml = None

_FRONT_MATTER_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")


def _parse_front_matter(text: str) -> Dict[str, str]:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}
    raw = m.group(1)
    if yaml is not None:
        try:
            data = yaml.safe_load(raw) or {}
            if isinstance(data, dict):
                return {str(k): ("" if v is None else str(v)) for k, v in data.items()}
        except Exception:
            pass
    # Minimal fallback parser if PyYAML is missing / fails.
    out: Dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class SkillLoader:
    """Scans a skills directory and exposes metadata + on-demand bodies."""

    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, Dict[str, str]] = {}
        self._scan()

    def _scan(self) -> None:
        if not self.skills_dir.exists():
            return
        for entry in sorted(self.skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            md = entry / "SKILL.md"
            if not md.is_file():
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            meta = _parse_front_matter(text)
            name = (meta.get("name") or entry.name).strip()
            description = (meta.get("description") or "").strip()
            body = _FRONT_MATTER_RE.sub("", text, count=1).strip()
            self._skills[name] = {
                "name": name,
                "description": description,
                "path": str(md),
                "body": body,
            }

    # ---- public API ------------------------------------------------------- #

    def list_skills(self) -> List[Dict[str, str]]:
        """Metadata-only listing (no bodies) — used by `eval/run.py`."""
        return [
            {"name": v["name"], "description": v["description"], "path": v["path"]}
            for v in self._skills.values()
        ]

    def load(self, name: str) -> str:
        """Return the full body of a skill. Raises KeyError if unknown."""
        v = self._skills.get(name)
        if v is None:
            raise KeyError(f"unknown skill: {name}")
        return v["body"]

    def match(self, task: str, top_k: int = 2) -> List[str]:
        """Return names of the most relevant skills by keyword overlap."""
        if not task:
            return []
        task_tokens = {t.lower() for t in _TOKEN_RE.findall(task)}
        if not task_tokens:
            return []
        scored = []
        for name, v in self._skills.items():
            desc_tokens = {t.lower() for t in _TOKEN_RE.findall(v["description"])}
            name_tokens = {t.lower() for t in _TOKEN_RE.findall(name)}
            overlap = len(task_tokens & (desc_tokens | name_tokens))
            if overlap:
                scored.append((overlap, name))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [n for _, n in scored[:top_k]]