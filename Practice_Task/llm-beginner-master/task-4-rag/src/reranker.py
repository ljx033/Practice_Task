"""BGE reranker：对 [query, doc] 文本对做精排。

接口约定（任务未强制，但 rag.py 会用到）：
    class Reranker:
        rerank(query: str, docs: List[dict], top_k: Optional[int]) -> List[dict]
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RERANK_MODEL_DIR = MODELS_DIR / "bge-reranker-base"


class Reranker:
    def __init__(self,
                 model_dir: Optional[Path] = None,
                 device: Optional[str] = None,
                 max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir or RERANK_MODEL_DIR)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"未找到 reranker 模型：{model_dir}，先跑 python data/download.py"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        ).to(self.device).eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, query: str, docs: List[str], batch_size: int = 32) -> List[float]:
        """对 (query, doc) 打分。"""
        scores: List[float] = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            pairs = [[query, d] for d in batch]
            enc = self.tok(pairs, padding=True, truncation=True,
                           max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.view(-1).float()
            scores.extend(logits.detach().cpu().tolist())
        return scores

    def rerank(self, query: str, docs: List[dict],
               top_k: Optional[int] = None) -> List[dict]:
        """对 docs 重新排序；写回 rerank_score，返回新的 list（不改原 list 顺序）。"""
        if not docs:
            return []
        texts = [d.get("text", "") for d in docs]
        scores = self.score(query, texts)
        out = []
        for d, s in zip(docs, scores):
            nd = dict(d)
            nd["rerank_score"] = float(s)
            out.append(nd)
        out.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            out = out[:int(top_k)]
        return out