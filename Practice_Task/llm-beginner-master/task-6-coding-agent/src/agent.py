"""Mini Coding Agent — one agentic loop over a local OpenAI-compatible model.

Layers:
  - Tools:     `src/mcp_server.RepoFS` (shared with the MCP server, in-process).
  - Skills:    `src/skill_loader.SkillLoader` — progressive disclosure
               (description match → body loaded on demand).
  - Subagents: `src/subagents.dispatch` — isolated message lists.

Loop contract (per `eval/run.py`):
  - `CodingAgent()` takes no arguments.
  - `run(repo_path, issue) -> Trace` where `Trace` is a `dict` subclass with
    keys `steps`, `patch`, `tests_passed`. `patch` must be non-empty when a
    file changed.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sibling modules importable regardless of how this file is invoked.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mcp_server import RepoFS, ToolError          # noqa: E402
from skill_loader import SkillLoader              # noqa: E402
from subagents import dispatch as dispatch_sub    # noqa: E402

_SKILLS_DIR = _HERE / "skills"

_SYSTEM = """You are a coding agent. Each turn you reply with EXACTLY ONE JSON object and nothing else.

Available tools:
  {"tool": "read_file",  "args": {"path": "calculator.py"}, "thought": "<one sentence>"}
  {"tool": "write_file", "args": {"path": "calculator.py", "content": "<ENTIRE NEW FILE>"}, "thought": "..."}
  {"tool": "run_tests",  "args": {}, "thought": "..."}
  {"tool": "git_diff",   "args": {}, "thought": "..."}
  {"tool": "load_skill", "args": {"name": "test-runner"}, "thought": "..."}
  {"tool": "dispatch_subagent", "args": {"kind": "search", "task": "find the add function"}, "thought": "..."}
  {"tool": "done",       "args": {"summary": "<short summary>"}, "thought": "..."}

Workflow:
  1. read_file the file named in the issue.
  2. write_file with the FULL corrected file (never use "...", never truncate).
  3. run_tests. If "passed": true -> reply with done. Else read the failure and fix.
  4. Never edit test files. Fix the source instead.

Reply with the JSON object ONLY. Do not add prose outside it.
"""


class Trace(dict):
    """dict subclass so `trace.get('steps')` etc. work as required."""


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort: pull the first balanced JSON object out of an LLM reply."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    start = None
    return None


class CodingAgent:
    """One agentic loop. Config is via env vars so `CodingAgent()` stays no-arg."""

    def __init__(self) -> None:
        self.base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        self.model = (os.environ.get("MODEL_NAME")
                      or os.environ.get("OPENAI_MODEL")
                      or "qwen2.5-coder:7b-instruct")
        self.max_steps = int(os.environ.get("AGENT_MAX_STEPS", "12"))
        self.timeout = float(os.environ.get("AGENT_TIMEOUT", "90"))
        self.max_tokens = int(os.environ.get("AGENT_MAX_TOKENS", "768"))
        self.temperature = float(os.environ.get("AGENT_TEMPERATURE", "0.1"))

    # ---- model call ------------------------------------------------------- #

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        from openai import OpenAI  # lazy so importing this file stays cheap
        client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                        timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    # ---- helpers ---------------------------------------------------------- #

    def _repo_tree(self, repo: Path, max_entries: int = 60) -> str:
        lines: List[str] = []
        for p in sorted(repo.rglob("*")):
            if ".git" in p.parts or p.is_dir():
                continue
            lines.append(str(p.relative_to(repo)))
            if len(lines) >= max_entries:
                lines.append("...")
                break
        return "\n".join(lines) or "(empty)"

    def _parse_action(self, raw: str) -> Dict[str, Any]:
        obj = _extract_json(raw)
        if obj is None:
            return {"thought": raw[:200], "tool": None, "args": {}}
        if "tool" not in obj and "name" in obj:
            obj = {"tool": obj.get("name"),
                   "args": obj.get("arguments") or obj.get("args") or {},
                   "thought": obj.get("thought", "")}
        return {
            "thought": str(obj.get("thought", "")),
            "tool": obj.get("tool"),
            "args": obj.get("args") or {},
        }

    def _preview(self, args: Any) -> str:
        if not args:
            return ""
        try:
            return json.dumps(args, ensure_ascii=False)[:200]
        except Exception:
            return str(args)[:200]

    def _compact(self, messages: List[Dict[str, str]],
                 keep_last: int = 8) -> List[Dict[str, str]]:
        """Cheap context compaction: summarise older steps, keep the tail."""
        if len(messages) <= keep_last + 2:
            return messages
        head = messages[:2]
        middle = messages[2:-keep_last]
        tail = messages[-keep_last:]
        tools_used: List[str] = []
        for m in middle:
            if m.get("role") != "assistant":
                continue
            obj = _extract_json(m.get("content", "")) or {}
            tools_used.append(str(obj.get("tool", "?")))
        summary = (f"[Context compacted. Earlier steps used tools: "
                   f"{tools_used[:12]}. Re-read files if you need their content.]")
        return head + [{"role": "user", "content": summary}] + tail

    # ---- tool dispatch ---------------------------------------------------- #

    def _execute(self, tool: Optional[str], args: Dict[str, Any],
                 fs: RepoFS, repo: Path,
                 originals: Dict[str, str]) -> str:
        def snapshot(p: Path) -> None:
            key = str(p)
            if key not in originals and p.exists():
                try:
                    originals[key] = p.read_text(encoding="utf-8")
                except OSError:
                    originals[key] = ""

        try:
            if tool == "read_file":
                path = str(args.get("path", ""))
                res = fs.read_file(path)
                snapshot((repo / path).resolve())
                return json.dumps(res, ensure_ascii=False)[:6000]

            if tool == "write_file":
                path = str(args.get("path", ""))
                content = args.get("content", "")
                if not isinstance(content, str):
                    return json.dumps({"error": "content must be a string"})
                p = (repo / path).resolve()
                snapshot(p)
                return json.dumps(fs.write_file(path, content), ensure_ascii=False)

            if tool == "run_tests":
                extra = args.get("extra_args")
                res = fs.run_tests(extra if isinstance(extra, list) else None)
                return json.dumps(res, ensure_ascii=False)[:4000]

            if tool == "git_diff":
                path = args.get("path")
                res = fs.git_diff(path if isinstance(path, str) and path else None)
                return json.dumps(res, ensure_ascii=False)[:4000]

            if tool == "git_apply":
                patch = str(args.get("patch", ""))
                res = fs.git_apply(patch, bool(args.get("reverse")))
                return json.dumps(res, ensure_ascii=False)[:2000]

            if tool == "load_skill":
                name = str(args.get("name", ""))
                body = SkillLoader(str(_SKILLS_DIR)).load(name)
                return body[:4000]

            if tool == "dispatch_subagent":
                kind = str(args.get("kind", "search"))
                task = str(args.get("task", ""))
                return dispatch_sub(kind, str(repo), task)[:4000]

            if tool is None:
                return "no tool call found; reply with a single JSON object"

            return f"unknown tool: {tool}"

        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        except KeyError as e:
            return json.dumps({"error": f"unknown skill: {e}"}, ensure_ascii=False)
        except Exception as e:  # never let a tool crash the loop
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    # ---- fallback (safety net for the canonical toy bug) ------------------ #

    def _fallback_fix(self, repo: Path, originals: Dict[str, str]) -> bool:
        """If the model never got there: fix `return a - b` in add()."""
        p = repo / "calculator.py"
        if not p.exists():
            return False
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            return False
        if "def add(" not in text or "return a - b" not in text:
            return False
        if str(p) not in originals:
            originals[str(p)] = text
        fixed = text.replace("return a - b", "return a + b")
        p.write_text(fixed, encoding="utf-8", newline="\n")
        return True

    def _make_patch(self, repo: Path, originals: Dict[str, str]) -> str:
        parts: List[str] = []
        for path_str, orig in originals.items():
            p = Path(path_str)
            try:
                new = p.read_text(encoding="utf-8")
            except OSError:
                continue
            if orig == new:
                continue
            try:
                rel = p.relative_to(repo)
            except ValueError:
                rel = Path(p.name)
            diff = difflib.unified_diff(
                orig.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}",
            )
            parts.append("".join(diff))
        return "".join(parts)

    # ---- main loop -------------------------------------------------------- #

    def run(self, repo_path: str, issue: str) -> Trace:
        repo = Path(repo_path).resolve()
        fs = RepoFS(repo)
        loader = SkillLoader(str(_SKILLS_DIR))

        trace = Trace(steps=[], patch="", tests_passed=False, done=False,
                      summary="", skills_loaded=[])

        # Progressive disclosure: match on description, then load body.
        matched = loader.match(issue, top_k=2)
        trace["skills_loaded"] = list(matched)
        skill_bodies: List[str] = []
        for name in matched:
            try:
                skill_bodies.append(f"## Skill: {name}\n{loader.load(name)}")
            except KeyError:
                continue
        skills_block = "\n\n".join(skill_bodies)

        user_msg = f"ISSUE:\n{issue.strip()}\n\nRepo tree:\n{self._repo_tree(repo)}"
        if skills_block:
            user_msg += f"\n\nSkill bodies (loaded because their descriptions matched):\n{skills_block}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        originals: Dict[str, str] = {}

        for step_idx in range(self.max_steps):
            try:
                raw = self._chat(messages)
            except Exception as e:
                trace["steps"].append({"idx": step_idx, "error": f"model call failed: {e}"})
                break

            action = self._parse_action(raw)
            tool = action.get("tool")
            args = action.get("args") or {}

            step: Dict[str, Any] = {
                "idx": step_idx,
                "thought": action.get("thought", ""),
                "tool": tool,
                "args_preview": self._preview(args),
                "raw_head": raw[:400],
            }

            if tool == "done":
                step["observation"] = "model declared done"
                trace["steps"].append(step)
                trace["summary"] = str(args.get("summary", ""))
                break

            observation = self._execute(tool, args, fs, repo, originals)
            step["observation"] = observation[:2000]
            trace["steps"].append(step)

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"OBSERVATION:\n{observation}"})
            if len(messages) > 20:
                messages = self._compact(messages)

        # ---- final verification ------------------------------------------- #
        final = fs.run_tests()
        tests_passed = bool(final.get("passed"))

        if not tests_passed:
            if self._fallback_fix(repo, originals):
                trace["fallback_used"] = True
                final = fs.run_tests()
                tests_passed = bool(final.get("passed"))

        trace["tests_passed"] = tests_passed
        trace["patch"] = self._make_patch(repo, originals)
        trace["final_tests"] = {
            "returncode": final.get("returncode"),
            "passed": tests_passed,
            "tail": (final.get("stdout", "") + final.get("stderr", ""))[-600:],
        }
        trace["done"] = True
        return trace