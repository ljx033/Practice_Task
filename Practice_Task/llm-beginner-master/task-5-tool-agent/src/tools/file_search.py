"""file_search: 本地文件按文件名或内容检索，返回路径 + 内容片段。"""
import fnmatch
from pathlib import Path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "在本地目录下按文件名或文件内容搜索文件。"
            "支持通配符（*.md），也支持内容关键字（TODO）。"
            "返回匹配文件路径；对内容命中或命中同名文件会附带内容片段。"
            "也可用于直接读取某个文件的开头内容。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "搜索模式：文件名（支持 * 通配符）、扩展名（如 .md）或内容关键字。",
                },
                "dir": {
                    "type": "string",
                    "description": "搜索根目录，例如 'data/agent-fixtures'。",
                },
            },
            "required": ["pattern", "dir"],
        },
    },
}

# 项目根：src/tools/file_search.py → parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_ROOTS = [_PROJECT_ROOT]
try:
    _cwd = Path.cwd().resolve()
    if _cwd not in _ALLOWED_ROOTS:
        _ALLOWED_ROOTS.append(_cwd)
except Exception:
    pass

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
              ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"}
_MAX_FILES = 5000
_MAX_HITS = 20


def _is_under(path: Path, roots) -> bool:
    try:
        rp = path.resolve()
    except Exception:
        return False
    for r in roots:
        try:
            rp.relative_to(Path(r).resolve())
            return True
        except ValueError:
            continue
    return False


def _resolve_dir(dir_arg: str) -> Path:
    p = Path(dir_arg)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((_PROJECT_ROOT / p).resolve())
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return candidates[0]


def _snippet(text: str, pattern: str, width: int = 200) -> str:
    low, p = text.lower(), pattern.lower()
    idx = low.find(p)
    if idx < 0:
        return text[:width].replace("\n", " ")
    start = max(0, idx - 80)
    end = min(len(text), idx + width)
    s = text[start:end].replace("\n", " ")
    if start > 0:
        s = "..." + s
    if end < len(text):
        s = s + "..."
    return s


def run(args: dict) -> str:
    if not isinstance(args, dict) or "pattern" not in args:
        raise KeyError("file_search 需要参数 {'pattern': str, 'dir': str}")
    pattern = str(args.get("pattern", "")).strip()
    dir_arg = str(args.get("dir", ".") or ".")
    if not pattern:
        raise ValueError("pattern 不能为空")

    base = _resolve_dir(dir_arg)
    if not base.exists():
        raise FileNotFoundError(f"目录不存在：{base}")
    if not _is_under(base, _ALLOWED_ROOTS):
        raise PermissionError(f"目录越界（不在允许根内）：{base}")

    # ① pattern 直接就是文件 → 读取返回
    maybe = Path(pattern)
    direct = None
    if maybe.is_absolute():
        if maybe.exists() and maybe.is_file():
            direct = maybe.resolve()
    else:
        for root in [base, _PROJECT_ROOT, Path.cwd()]:
            c = (root / maybe)
            try:
                cr = c.resolve()
            except Exception:
                continue
            if cr.exists() and cr.is_file() and _is_under(cr, _ALLOWED_ROOTS):
                direct = cr
                break
    if direct is not None:
        text = direct.read_text(encoding="utf-8", errors="ignore")
        try:
            rel = direct.relative_to(_PROJECT_ROOT)
        except Exception:
            rel = direct
        return f"文件：{rel}\n内容（前 800 字符）：\n{text[:800]}"

    # ② 遍历检索
    pat_lower = pattern.lower()
    use_glob = ("*" in pattern) or ("?" in pattern)
    matches = []
    count = 0

    for path in sorted(base.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        count += 1
        if count > _MAX_FILES:
            break
        if not path.is_file():
            continue

        name = path.name
        if use_glob:
            name_match = fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.lower(), pat_lower)
        else:
            name_match = pat_lower in name.lower()

        content_match = False
        snippet = ""
        try:
            if path.stat().st_size <= 1_000_000:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if pat_lower in text.lower():
                    content_match = True
                    snippet = _snippet(text, pattern)
        except Exception:
            pass

        if name_match or content_match:
            matches.append((path, name_match, content_match, snippet))

    if not matches:
        return f"在 {base} 下未找到匹配 '{pattern}' 的文件。"

    lines = [f"在 {base} 下找到 {len(matches)} 个匹配 '{pattern}' 的文件："]
    for i, (path, nm, cm, snip) in enumerate(matches[:_MAX_HITS], 1):
        try:
            rel = path.relative_to(_PROJECT_ROOT)
        except Exception:
            rel = path
        lines.append(f"{i}. {rel}")
        if snip:
            lines.append(f"   匹配片段：{snip}")
        else:
            try:
                head = path.read_text(encoding="utf-8", errors="ignore")[:200].replace("\n", " ")
                if head:
                    lines.append(f"   开头：{head}")
            except Exception:
                pass
    return "\n".join(lines)