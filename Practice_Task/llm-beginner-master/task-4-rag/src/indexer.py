"""BGE embedding + FAISS 索引构建。

用法：
    python -m src.indexer                 # 默认参数建索引
    python -m src.indexer --chunk-size 512 --overlap 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.chunker import chunk_text, extract_pdf_text

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DEFAULT_INDEX_DIR = ROOT / "index"

EMBED_MODEL_DIR = MODELS_DIR / "bge-small-zh-v1.5"
# BGE 官方建议：查询侧加前缀，文档侧不加
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embed_model(model_dir: Optional[Path] = None,
                     device: Optional[str] = None):
    """加载 BGE 模型。返回 (tokenizer, model, device)。"""
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir or EMBED_MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"未找到 BGE 模型：{model_dir}，先跑 python data/download.py"
        )
    dev = _resolve_device(device)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(dev).eval()
    return tok, model, dev


@torch.no_grad()
def embed_texts(texts: List[str], tokenizer, model, device: str,
                batch_size: int = 64, max_length: int = 512) -> np.ndarray:
    """对一批文本做 BGE 编码（CLS pooling + L2 normalize）。

    IndexFlatIP 检索等价 cosine，前提是 embedding 已归一化。
    """
    if not texts:
        hidden = getattr(model.config, "hidden_size", 512)
        return np.zeros((0, hidden), dtype="float32")

    out_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc)
        emb = out.last_hidden_state[:, 0]                 # CLS
        emb = F.normalize(emb, p=2, dim=1)                # L2
        out_vecs.append(emb.detach().cpu().numpy().astype("float32"))
    return np.vstack(out_vecs)


def build_index(pdf_path: Optional[Path] = None,
                index_dir: Optional[Path] = None,
                chunk_size: int = 512,
                overlap: int = 64,
                batch_size: int = 64,
                device: Optional[str] = None,
                verbose: bool = True) -> Path:
    """从 PDF 建 FAISS 索引，落盘到 index_dir。"""
    import faiss  # 延迟 import，避免无索引时也加载

    pdf_path = Path(pdf_path or DATA_DIR / "kb.pdf")
    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[indexer] 抽取 PDF 文本: {pdf_path}")
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        raise RuntimeError(f"从 {pdf_path} 抽出的文本为空，检查 PDF 是否有效")

    if verbose:
        print(f"[indexer] 切分：chunk_size={chunk_size}, overlap={overlap}")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) < 10:
        raise RuntimeError(f"chunk 数过少：{len(chunks)}，检查 PDF 或切分参数")
    if verbose:
        avg = sum(len(c) for c in chunks) / len(chunks)
        print(f"[indexer] 得到 {len(chunks)} 个 chunk，平均长度 {avg:.1f}")

    tok, model, dev = load_embed_model(device=device)
    if verbose:
        print(f"[indexer] BGE 编码（device={dev}, batch={batch_size}）...")
    emb = embed_texts(chunks, tok, model, dev, batch_size=batch_size)
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)   # 内积 == cosine（已归一化）
    index.add(emb)
    faiss.write_index(index, str(index_dir / "faiss.index"))

    # 保存 chunk 元数据
    meta_path = index_dir / "chunks.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps(
                {"id": i, "text": c, "source": pdf_path.name},
                ensure_ascii=False,
            ) + "\n")

    with (index_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "dim": dim,
            "n_chunks": len(chunks),
            "pdf": str(pdf_path),
            "embed_model": str(EMBED_MODEL_DIR),
        }, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[indexer] 完成 -> {index_dir}（{len(chunks)} 向量，dim={dim}）")
    return index_dir


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    build_index(pdf_path=args.pdf, index_dir=args.index_dir,
                chunk_size=args.chunk_size, overlap=args.overlap,
                batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    _cli()
