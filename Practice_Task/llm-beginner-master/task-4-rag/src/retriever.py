"""BGE 查询编码 + FAISS 召回。

接口约定：
    class Retriever:
        retrieve(query: str, k: int) -> List[dict]  # dict 含 text/score/source
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.indexer import (
    DEFAULT_INDEX_DIR,
    QUERY_PREFIX,
    embed_texts,
    load_embed_model,
)


class Retriever:
    """FAISS 内积检索（= cosine，因为 embedding 已 L2 normalize）。"""

    def __init__(self,
                 index_dir: Optional[Path] = None,
                 device: Optional[str] = None,
                 auto_build: bool = True):
        import faiss

        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        index_file = self.index_dir / "faiss.index"
        chunks_file = self.index_dir / "chunks.jsonl"

        if not (index_file.exists() and chunks_file.exists()):
            if not auto_build:
                raise FileNotFoundError(
                    f"未找到索引：{self.index_dir}，先运行 python -m src.indexer"
                )
            print(f"[retriever] 索引不存在，自动构建到 {self.index_dir} ...")
            from src.indexer import build_index
            build_index(index_dir=self.index_dir)

        self.index = faiss.read_index(str(index_file))
        self.chunks: List[dict] = []
        with chunks_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        self.tok, self.model, self.device = load_embed_model(device=device)

    # -------- 对外接口 --------
    def retrieve(self, query: str, k: int = 10) -> List[dict]:
        """返回 top-k 结果，每个 dict 含 text/score/source。

        注意：查询侧要加 BGE 检索前缀，文档侧不加。
        """
        if not query:
            return []
        k = int(k)
        if k <= 0:
            return []
        k = min(k, len(self.chunks))

        q = QUERY_PREFIX + query
        qv = embed_texts([q], self.tok, self.model, self.device,
                         batch_size=1)
        qv = np.ascontiguousarray(qv, dtype="float32")

        # FAISS search
        scores, idxs = self.index.search(qv, k)
        results: List[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[idx]
            results.append({
                "text": c["text"],
                "score": float(score),
                "source": c.get("source", "kb.pdf"),
            })
        return results