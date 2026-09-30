"""python_sandbox: 教学级受限 exec。
⚠️ 黑白名单 + 超时只是教学防护，并非真正隔离，切勿用于不可信输入。
"""
import contextlib
import io
import signal
import traceback

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "在受限沙箱中执行 Python 代码并返回 print 的标准输出。"
            "适合计算、循环、函数定义、字符串处理。不支持 input()、文件读写、网络。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码，用 print 输出结果，例如 'print(sum(range(10)))'。",
                }
            },
            "required": ["code"],
        },
    },
}

_ALLOWED_MODULES = {
    "math", "cmath", "random", "statistics", "itertools", "functools",
    "collections", "re", "json", "datetime", "string", "decimal",
    "fractions", "heapq", "bisect", "operator", "textwrap",
}

# 只把这些放行到 builtins —— 不暴露 open / eval / exec / __import__ 原版
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"沙箱不允许导入模块：{name}")
    import importlib
    return importlib.import_module(name)


_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
    "bool": bool, "bytearray": bytearray, "bytes": bytes,
    "callable": callable, "chr": chr, "complex": complex,
    "dict": dict, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "hash": hash, "hex": hex,
    "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "oct": oct,
    "ord": ord, "pow": pow, "print": print, "range": range,
    "repr": repr, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "__import__": _safe_import,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "AssertionError": AssertionError, "NameError": NameError,
    "AttributeError": AttributeError, "ArithmeticError": ArithmeticError,
}

_TIMEOUT_SEC = 5

# 源码级黑名单（教学级，只拦最常见逃逸串）
_FORBIDDEN_SUBSTRINGS = (
    "__subclasses__", "__globals__", "__bases__", "__mro__",
    "__code__", "__closure__", "__builtins__",
)


def _run_with_timeout(code: str, g: dict, timeout: int):
    def _handler(signum, frame):
        raise TimeoutError(f"沙箱执行超过 {timeout} 秒")

    armed = False
    old = None
    try:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        armed = True
    except (ValueError, AttributeError, OSError):
        armed = False
    try:
        exec(code, g, g)
    finally:
        if armed:
            try:
                signal.alarm(0)
                if old is not None:
                    signal.signal(signal.SIGALRM, old)
            except Exception:
                pass


def run(args: dict) -> str:
    if not isinstance(args, dict) or "code" not in args:
        raise KeyError("python_sandbox 需要参数 {'code': str}")
    code = str(args["code"])
    if not code.strip():
        raise ValueError("代码为空")
    if len(code) > 8000:
        raise ValueError("代码过长（>8000 字符）")

    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in code:
            raise ValueError(f"沙箱禁止使用：{bad}")

    g = {"__builtins__": _SAFE_BUILTINS, "__name__": "__sandbox__"}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            _run_with_timeout(code, g, _TIMEOUT_SEC)
    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        out = buf.getvalue()
        return f"执行出错：{err}\n标准输出：\n{out}" if out.strip() else f"执行出错：{err}"

    out = buf.getvalue()
    if not out.strip():
        # 没有 print：从命名空间挑出用户定义的变量做兜底
        shown = []
        for k, v in g.items():
            if k.startswith("_") or callable(v):
                continue
            shown.append(f"{k} = {v!r}")
        out = "\n".join(shown) if shown else "(无输出)"
    return out.strip()