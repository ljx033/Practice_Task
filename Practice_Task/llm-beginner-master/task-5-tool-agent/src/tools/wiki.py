"""wiki: 维基百科 API 查询（中英文），返回条目简介纯文本。"""
import re
import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "查询维基百科条目，返回条目简介纯文本。支持中英文；"
            "中文查询优先走中文维基，英文/混合查询优先走英文维基。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要查询的词条，例如 '图灵机'、'Geoffrey Hinton'、'Transformer machine learning model'。",
                }
            },
            "required": ["query"],
        },
    },
}

_UA = "ReActAgentEval/1.0 (educational; contact: local)"
_TIMEOUT = 12
_CACHE = {}


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def _api(lang: str, params: dict) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    p = dict(params)
    p.setdefault("format", "json")
    p.setdefault("formatversion", "2")
    r = requests.get(url, params=p, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _search_and_fetch(query: str, lang: str):
    data = _api(lang, {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
    })
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    title = hits[0]["title"]
    data2 = _api(lang, {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    })
    pages = data2.get("query", {}).get("pages", [])
    if not pages:
        return None
    extract = pages[0].get("extract", "")
    if not extract:
        return None
    return title, extract


def run(args: dict) -> str:
    if not isinstance(args, dict) or "query" not in args:
        raise KeyError("wiki 需要参数 {'query': str}")
    query = str(args["query"]).strip()
    if not query:
        raise ValueError("query 不能为空")
    if query in _CACHE:
        return _CACHE[query]

    clean = re.sub(r"[（(].*?[)）]", "", query).strip() or query

    if _has_cjk(query):
        tries = [("zh", query)]
        if clean != query:
            tries.append(("en", clean))
        tries.append(("en", query))
    else:
        tries = [("en", query)]
        if clean != query:
            tries.append(("en", clean))

    last_err = None
    for lang, q in tries:
        try:
            res = _search_and_fetch(q, lang)
        except Exception as e:
            last_err = e
            continue
        if res:
            title, extract = res
            out = f"维基百科({lang}) 条目：{title}\n{extract[:2500]}"
            _CACHE[query] = out
            return out

    if last_err is not None:
        raise RuntimeError(f"维基百科查询失败：{last_err}")
    raise RuntimeError(f"维基百科未找到 '{query}' 的条目")