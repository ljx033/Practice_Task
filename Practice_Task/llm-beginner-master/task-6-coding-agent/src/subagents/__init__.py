"""Fresh-context subagents for the Mini Coding Agent.

Each subagent gets its own message list, its own step budget, and a restricted
tool subset — so triage work cannot leak state (or blow up the context) of the
main agent. The main loop only ever sees the subagent's *summary string*.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type


class _Subagent:
    """Base class: own message list, own step budget, restricted tools."""

    kind: str = "base"
    max_steps: int = 4
    allowed_tools: tuple = ()

    def __init__(self, repo: Path):
        self.repo = Path(repo)
        self.messages: List[Dict[str, str]] = []  # isolated from the caller

    def run(self, task: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class SearchSubagent(_Subagent):
    """Grep-like scanner. Returns `path:line: snippet` hits for task keywords."""

    kind = "search"
    allowed_tools = ("read_file",)

    def run(self, task: str) -> str:
        tokens = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", task)}
        if not tokens:
            return "no keywords extracted from task"
        hits: List[str] = []
        for py in sorted(self.repo.rglob("*.py")):
            if ".git" in py.parts:
                continue
            try:
                lines = py.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                low = line.lower()
                if any(tok in low for tok in tokens):
                    rel = py.relative_to(self.repo)
                    hits.append(f"{rel}:{i}: {line.strip()[:120]}")
                    if len(hits) >= 30:
                        break
            if len(hits) >= 30:
                break
        return "\n".join(hits) if hits else "no matches"


class TestDoctorSubagent(_Subagent):
    """Runs pytest, returns a compact failure summary for the main agent."""

    kind = "test-doctor"
    allowed_tools = ("run_tests",)

    def run(self, task: str) -> str:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short"],
                cwd=str(self.repo),
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return "pytest timed out after 30s"
        out = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0:
            return "ALL TESTS PASSED\n" + out[-500:]
        lines = out.splitlines()
        interesting = [l for l in lines
                       if "FAIL" in l or "Error" in l or "assert" in l.lower()]
        tail = out[-500:] if len(out) > 500 else ""
        return "TEST FAILURE\n" + "\n".join(interesting[:20]) + \
               (f"\n...\n{tail}" if tail else "")


_REGISTRY: Dict[str, Type[_Subagent]] = {
    "search": SearchSubagent,
    "test-doctor": TestDoctorSubagent,
}


def dispatch(kind: str, repo: str, task: str) -> str:
    """Spawn a fresh-context subagent of the given kind and return its summary."""
    cls = _REGISTRY.get(kind)
    if cls is None:
        return f"unknown subagent kind: {kind} (available: {list(_REGISTRY)})"
    return cls(Path(repo)).run(task)


def list_kinds() -> List[str]:
    return sorted(_REGISTRY)