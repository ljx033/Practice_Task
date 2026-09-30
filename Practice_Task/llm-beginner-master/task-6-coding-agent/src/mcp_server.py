"""MCP server exposing file / test / git tools.

Two entry points:

- Standalone MCP server (stdio):   python src/mcp_server.py [--repo PATH]
- Importable tool catalogue:       from src.mcp_server import list_tools, RepoFS, ToolError

`list_tools()` is what `eval/run.py` enumerates. `RepoFS` is the exact same
implementation the MCP server wraps, so `src/agent.py` can call the tools
in-process without paying stdio round-trip cost.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["list_tools", "RepoFS", "ToolError"]


class ToolError(Exception):
    """Raised by a tool implementation; surfaces as a structured error payload."""


# --------------------------------------------------------------------------- #
# Tool implementations (pure Python; no hard dependency on the MCP SDK)
# --------------------------------------------------------------------------- #

class RepoFS:
    """Filesystem + subprocess view confined to a single repo root."""

    def __init__(self, repo: Path):
        self.repo = Path(repo).resolve()
        if not self.repo.exists():
            raise ToolError(f"repo does not exist: {self.repo}")

    # ---- path safety ------------------------------------------------------ #

    def _resolve(self, user_path: str, must_exist: bool = False) -> Path:
        if not isinstance(user_path, str) or not user_path.strip():
            raise ToolError("path must be a non-empty string")
        raw = Path(user_path)
        if raw.is_absolute():
            raise ToolError(f"absolute paths are not allowed: {user_path}")
        target = (self.repo / raw).resolve()
        try:
            target.relative_to(self.repo)
        except ValueError:
            raise ToolError(f"path escapes repo root: {user_path}")
        if must_exist and not target.exists():
            raise ToolError(f"not found: {user_path}")
        return target

    # ---- read / write ----------------------------------------------------- #

    def read_file(self, path: str, max_bytes: int = 200_000) -> Dict[str, Any]:
        p = self._resolve(path, must_exist=True)
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        data = p.read_bytes()
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"file is not utf-8 text: {path}")
        return {"path": path, "content": text, "truncated": truncated,
                "bytes": p.stat().st_size}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        p = self._resolve(path)
        if p.is_dir():
            raise ToolError(f"target is a directory: {path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="\n")
        return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}

    # ---- subprocess helper ------------------------------------------------ #

    def _run(self, args: List[str], timeout: int = 60,
             stdin: Optional[str] = None) -> Dict[str, Any]:
        # Security: list-form args, shell=False, cwd pinned to repo.
        if not isinstance(args, (list, tuple)) or not all(isinstance(a, str) for a in args):
            raise ToolError("args must be a list of strings")
        try:
            proc = subprocess.run(
                list(args),
                cwd=str(self.repo),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except FileNotFoundError as e:
            raise ToolError(f"command not found: {args[0]} ({e})")
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": f"timeout after {timeout}s"}
        return {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-4000:],
        }

    # ---- tools ------------------------------------------------------------ #

    def run_tests(self, extra_args: Optional[List[str]] = None,
                  timeout: int = 60) -> Dict[str, Any]:
        cmd = [sys.executable, "-m", "pytest", "-q"]
        if extra_args:
            if not all(isinstance(a, str) and "\x00" not in a for a in extra_args):
                raise ToolError("invalid extra_args")
            cmd += list(extra_args)
        result = self._run(cmd, timeout=timeout)
        result["passed"] = result["returncode"] == 0
        return result

    def git_diff(self, path: Optional[str] = None) -> Dict[str, Any]:
        # Forbidden-by-design commands (reset --hard / clean -fd / checkout --)
        # are never used here; we only ever *read* the diff.
        cmd = ["git", "diff", "--no-color", "--no-ext-diff"]
        if path:
            self._resolve(path)  # validate before handing to git
            cmd += ["--", path]
        result = self._run(cmd, timeout=20)
        return {"returncode": result["returncode"],
                "diff": result.get("stdout", ""),
                "stderr": result.get("stderr", "")}

    def git_apply(self, patch: str, reverse: bool = False,
                  timeout: int = 20) -> Dict[str, Any]:
        if not isinstance(patch, str) or not patch.strip():
            raise ToolError("patch must be a non-empty string")
        cmd = ["git", "apply", "--whitespace=nowarn"]
        if reverse:
            cmd.append("--reverse")
        cmd.append("-")  # read patch from stdin, never from a shell string
        result = self._run(cmd, timeout=timeout, stdin=patch)
        if result["returncode"] != 0:
            raise ToolError(f"git apply failed: {result['stderr'].strip()[:300]}")
        return {"ok": True, "applied": True}


# --------------------------------------------------------------------------- #
# Tool catalogue: what `list_tools()` returns and what the MCP server exposes.
# --------------------------------------------------------------------------- #

_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file inside the repo. Returns its content.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string",
                                    "description": "Repo-relative path."}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": ("Create or overwrite a UTF-8 text file inside the repo. "
                        "`content` is the ENTIRE new file body."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path."},
                "content": {"type": "string", "description": "Full new file contents."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run `python -m pytest -q` in the repo. Returns returncode/stdout/stderr/passed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "extra_args": {"type": "array", "items": {"type": "string"},
                               "description": "Optional extra pytest args."},
            },
        },
    },
    {
        "name": "git_diff",
        "description": "Show `git diff` of the working tree (optionally limited to one path).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string",
                                    "description": "Optional repo-relative path."}},
        },
    },
    {
        "name": "git_apply",
        "description": ("Apply a unified diff via `git apply`. Never runs "
                        "`reset --hard` / `clean -fd` / `checkout --`."),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff text."},
                "reverse": {"type": "boolean", "description": "Reverse-apply the patch."},
            },
            "required": ["patch"],
        },
    },
]


def list_tools() -> List[Dict[str, Any]]:
    """Return the JSON schema of every tool this server exposes."""
    return [dict(t) for t in _TOOL_SCHEMAS]


# --------------------------------------------------------------------------- #
# MCP server (lazy-import the SDK so `list_tools()` works even without it)
# --------------------------------------------------------------------------- #

def _build_mcp_server(repo_path: str):
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "The `mcp` package is required to run the server "
            f"(pip install mcp). `list_tools()` still works. ({exc})"
        )

    fs = RepoFS(Path(repo_path))
    server = FastMCP("mini-coding-agent")

    @server.tool()
    def read_file(path: str) -> str:
        """Read a UTF-8 text file inside the repo."""
        try:
            return json.dumps(fs.read_file(path), ensure_ascii=False)
        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @server.tool()
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file inside the repo."""
        try:
            return json.dumps(fs.write_file(path, content), ensure_ascii=False)
        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @server.tool()
    def run_tests() -> str:
        """Run `python -m pytest -q` in the repo."""
        try:
            return json.dumps(fs.run_tests(), ensure_ascii=False)
        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @server.tool()
    def git_diff(path: str = "") -> str:
        """Show `git diff` of the working tree."""
        try:
            return json.dumps(fs.git_diff(path or None), ensure_ascii=False)
        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @server.tool()
    def git_apply(patch: str, reverse: bool = False) -> str:
        """Apply a unified diff via `git apply`."""
        try:
            return json.dumps(fs.git_apply(patch, reverse), ensure_ascii=False)
        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return server


def _main() -> None:
    ap = argparse.ArgumentParser(description="Mini Coding Agent MCP server (stdio).")
    ap.add_argument("--repo", default=os.environ.get("REPO_PATH", os.getcwd()),
                    help="Repo root all tools are confined to.")
    args = ap.parse_args()
    server = _build_mcp_server(args.repo)
    server.run()  # stdio transport


if __name__ == "__main__":
    _main()