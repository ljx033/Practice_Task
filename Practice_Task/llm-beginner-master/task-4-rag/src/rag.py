"""端到端 RAG：检索 -> rerank -> 去重截断 -> 生成。

接口约定：
    answer(query: str) -> {"answer": str, "sources": List[dict]}
"""
from __future__ import annotations

from typing import List, Optional

# 建议召回数 >> 最终返回数
RECALL_K = 20
RERANK_TOP = 5

_retriever = None
_reranker = None
_generator = None


def _ensure_modules():
    global _retriever, _reranker, _generator
    if _retriever is None:
        from src.retriever import Retriever
        _retriever = Retriever()
    if _reranker is None:
        try:
            from src.reranker import Reranker
            _reranker = Reranker()
        except Exception as e:
            print(f"[rag] reranker 加载失败，退化为纯向量召回：{e}")
            _reranker = None
    if _generator is None:
        try:
            from src.generator import Generator
            _generator = Generator()
        except Exception as e:
            print(f"[rag] generator 初始化失败：{e}")
            _generator = None


def _dedup(contexts: List[str], sim_prefix: int = 64) -> List[str]:
    """按前 64 字符去重（相邻 chunk overlap 会重复）。"""
    seen = set()
    out = []
    for c in contexts:
        key = c[:sim_prefix]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def answer(query: str,
           recall_k: int = RECALL_K,
           rerank_top: int = RERANK_TOP) -> dict:
    """检索 + 生成，返回 {answer, sources}。"""
    _ensure_modules()

    # --- 1. 召回 ---
    # if _retriever is None:
    #     from src.retriever import Retriever
    #     _retriever = Retriever()
    candidates = _retriever.retrieve(query, k=recall_k)

    # --- 2. 精排 ---
    if _reranker is not None and candidates:
        try:
            ranked = _reranker.rerank(query, candidates, top_k=rerank_top)
        except Exception as e:
            print(f"[rag] rerank 出错，退化为原顺序：{e}")
            ranked = candidates[:rerank_top]
    else:
        ranked = candidates[:rerank_top]

    # --- 3. 去重 + 组装 ---
    texts = _dedup([r.get("text", "") for r in ranked if r.get("text")])
    sources: List[dict] = []
    for r in ranked:
        sources.append({
            "text": r.get("text", ""),
            "score": float(r.get("rerank_score", r.get("score", 0.0))),
            "source": r.get("source", "kb.pdf"),
        })
    # 若去重后没内容也要保证 sources 非空
    if not sources and candidates:
        sources = [{
            "text": candidates[0].get("text", ""),
            "score": float(candidates[0].get("score", 0.0)),
            "source": candidates[0].get("source", "kb.pdf"),
        }]
        texts = [candidates[0].get("text", "")]

    # --- 4. 生成（失败时给出可解释的兜底） ---
    ans = ""
    if _generator is not None and texts:
        try:
            ans = _generator.generate(query, texts)
        except Exception as e:
            print(f"[rag] 生成失败（{e}）；返回兜底答案。")
    if not ans:
        if texts:
            head = texts[0].strip().replace("\n", " ")
            ans = f"（LLM 不可用，展示检索到的最相关片段）{head[:220]}"
        else:
            ans = "根据提供的资料无法回答该问题。"

    return {"answer": ans, "sources": sources}