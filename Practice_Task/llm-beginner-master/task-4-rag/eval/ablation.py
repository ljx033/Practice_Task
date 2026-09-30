"""统一消融脚本：chunk_size 扫描 / reranker 对比 / query 改写对比。

用法（PyCharm Terminal 或系统 CMD 均可）：
    python eval/ablation.py                      # 跑全部三种消融
    python eval/ablation.py --modes chunk        # 只跑 chunk_size 扫描
    python eval/ablation.py --modes rerank rewrite
    python eval/ablation.py --modes chunk --sizes 256 512
    python eval/ablation.py --modes rerank --rerank-cs 256 --rerank-top 5

结果写入 eval/ablation.json，终端同时打印表格。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "eval" / ".ablation_cache"

KS = (1, 3, 5, 10)


# ---------- 公共工具 ----------

def normalize_text(text) -> str:
    return re.sub(r"\s+", "", str(text))


def load_gold() -> List[dict]:
    path = ROOT / "data" / "gold_qa.jsonl"
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def eval_recall(retrieve_fn: Callable[[str, int], List[dict]],
                gold: List[dict],
                ks=KS,
                top_n: Optional[int] = None) -> Dict[str, float]:
    """对每个问题调 retrieve_fn(question, k=max(ks)) 拿结果，统计 Recall/MRR。

    命中口径与 eval/run.py 一致：chunk 文本去掉所有空白后包含任一 gold_anchor。
    """
    top_n = top_n or max(ks)
    hit = {k: 0 for k in ks}
    rr_sum = 0.0
    for item in gold:
        anchors = [normalize_text(a) for a in item.get("gold_anchors", [])]
        results = retrieve_fn(item["question"], top_n)
        matched_rank = None
        for rank, r in enumerate(results, 1):
            t = normalize_text(r.get("text", ""))
            if any(a and a in t for a in anchors):
                matched_rank = rank
                break
        if matched_rank:
            for k in ks:
                if matched_rank <= k:
                    hit[k] += 1
            rr_sum += 1.0 / matched_rank
    n = len(gold)
    metrics = {f"recall@{k}": round(hit[k] / n, 3) for k in ks}
    metrics["mrr"] = round(rr_sum / n, 3)
    return metrics


def print_row(tag: str, extra: Dict, m: Dict[str, float], sec: float):
    print(f"{tag:<26} {extra.get('n_chunks','-'):>8} "
          f"{m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
          f"{m['recall@5']:>6.3f} {m['recall@10']:>6.3f} "
          f"{m['mrr']:>7.3f} {sec:>7.1f}")


def print_header(title: str):
    print(f"\n===== {title} =====")
    header = (f"{'variant':<26} {'n_chunk':>8} "
              f"{'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} "
              f"{'MRR':>7} {'sec':>7}")
    print(header)
    print("-" * len(header))


# ---------- 消融 1：chunk_size 扫描 ----------

def run_chunk_ablation(sizes, overlap_ratio, batch_size, device) -> List[dict]:
    from src.indexer import build_index
    from src.retriever import Retriever

    gold = load_gold()
    print_header(f"chunk_size 扫描（gold={len(gold)}）")

    rows = []
    for cs in sizes:
        ov = max(1, cs // overlap_ratio)
        index_dir = ROOT / f"index_cs{cs}"
        t0 = time.time()
        if not (index_dir / "faiss.index").exists():
            build_index(index_dir=index_dir, chunk_size=cs, overlap=ov,
                        batch_size=batch_size, device=device, verbose=False)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        retriever = Retriever(index_dir=index_dir, device=device)

        m = eval_recall(retriever.retrieve, gold)
        elapsed = time.time() - t0
        print_row(f"chunk={cs} overlap={ov}", meta, m, elapsed)
        rows.append({"mode": "chunk", "chunk_size": cs, "overlap": ov,
                     "n_chunks": meta["n_chunks"], **m,
                     "build_sec": round(elapsed, 1)})
    return rows


# ---------- 消融 2：reranker ----------

def run_rerank_ablation(cs, overlap_ratio, batch_size, device,
                        recall_k, rerank_top) -> List[dict]:
    from src.indexer import build_index
    from src.retriever import Retriever
    from src.reranker import Reranker

    gold = load_gold()

    index_dir = ROOT / f"index_cs{cs}"
    if not (index_dir / "faiss.index").exists():
        ov = max(1, cs // overlap_ratio)
        build_index(index_dir=index_dir, chunk_size=cs, overlap=ov,
                    batch_size=batch_size, device=device, verbose=False)
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    retriever = Retriever(index_dir=index_dir, device=device)

    print_header(f"reranker 对比（chunk={cs} recall_k={recall_k} top={rerank_top}）")

    rows = []

    # baseline：不加 reranker，直接取 retrieve top-k
    def retrieve_no_rerank(q, k):
        return retriever.retrieve(q, k=k)[:k]

    t0 = time.time()
    m0 = eval_recall(retrieve_no_rerank, gold)
    sec0 = time.time() - t0
    print_row("baseline (no rerank)", meta, m0, sec0)
    rows.append({"mode": "rerank", "variant": "no_rerank",
                 "chunk_size": cs, "n_chunks": meta["n_chunks"],
                 **m0, "sec": round(sec0, 1)})

    # with reranker：召回 recall_k，rerank 后取 top rerank_top
    try:
        reranker = Reranker(device=device)
    except Exception as e:
        print(f"[warn] reranker 加载失败，跳过：{e}")
        return rows

    def retrieve_with_rerank(q, k):
        cand = retriever.retrieve(q, k=max(recall_k, k))
        ranked = reranker.rerank(q, cand, top_k=max(k, rerank_top))
        return ranked[:k]

    t0 = time.time()
    m1 = eval_recall(retrieve_with_rerank, gold)
    sec1 = time.time() - t0
    print_row(f"with rerank (top={rerank_top})", meta, m1, sec1)
    rows.append({"mode": "rerank", "variant": "with_rerank",
                 "chunk_size": cs, "recall_k": recall_k,
                 "rerank_top": rerank_top,
                 "n_chunks": meta["n_chunks"],
                 **m1, "sec": round(sec1, 1)})

    d10 = m1["recall@10"] - m0["recall@10"]
    dmrr = m1["mrr"] - m0["mrr"]
    print(f"\nΔ recall@10 = {d10:+.3f}   Δ mrr = {dmrr:+.3f}")
    return rows


# ---------- 消融 3：query 改写 ----------

REWRITE_PROMPT = (
    "你是一个检索查询改写器。请把下面的问题改写成一条更适合向量检索的查询，"
    "保留关键术语和实体，不要回答问题，不要解释，只输出改写后的查询本身。\n\n"
    "原问题：{q}\n改写："
)

HYDE_PROMPT = (
    "请针对下面的问题，写一段 120–200 字、事实性的中文段落，"
    "像教材或百科里的一段正文那样直接陈述答案要点，"
    "不要写“我不知道”“无法回答”，不要提问，只输出这一段正文。\n\n"
    "问题：{q}\n段落："
)


def _cache_path(kind: str, question: str) -> Path:
    h = hashlib.md5(question.encode("utf-8")).hexdigest()[:12]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{kind}_{h}.txt"


def _llm_call(generator, prompt: str, kind: str, question: str,
              max_tokens: int = 256) -> str:
    cp = _cache_path(kind, question)
    if cp.exists():
        return cp.read_text(encoding="utf-8")

    from openai import OpenAI
    resp = generator.client.chat.completions.create(
        model=generator.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    cp.write_text(text, encoding="utf-8")
    return text


def run_rewrite_ablation(cs, overlap_ratio, batch_size, device,
                         recall_k) -> List[dict]:
    from src.indexer import build_index
    from src.retriever import Retriever
    from src.generator import Generator

    gold = load_gold()

    index_dir = ROOT / f"index_cs{cs}"
    if not (index_dir / "faiss.index").exists():
        ov = max(1, cs // overlap_ratio)
        build_index(index_dir=index_dir, chunk_size=cs, overlap=ov,
                    batch_size=batch_size, device=device, verbose=False)
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    retriever = Retriever(index_dir=index_dir, device=device)

    try:
        generator = Generator()
    except Exception as e:
        print(f"[warn] generator 初始化失败，跳过 query 改写消融：{e}")
        return []

    print_header(f"query 改写对比（chunk={cs} recall_k={recall_k}）")
    rows = []

    # (a) raw
    def retrieve_raw(q, k):
        return retriever.retrieve(q, k=k)

    t0 = time.time()
    m_raw = eval_recall(retrieve_raw, gold)
    print_row("raw query", meta, m_raw, time.time() - t0)
    rows.append({"mode": "rewrite", "variant": "raw",
                 "chunk_size": cs, "n_chunks": meta["n_chunks"],
                 **m_raw, "sec": round(time.time() - t0, 1)})

    # (b) LLM rewrite
    def retrieve_rewrite(q, k):
        new_q = _llm_call(generator, REWRITE_PROMPT.format(q=q),
                          "rewrite", q, max_tokens=128)
        return retriever.retrieve(new_q or q, k=k)

    t0 = time.time()
    m_rw = eval_recall(retrieve_rewrite, gold)
    print_row("llm rewrite", meta, m_rw, time.time() - t0)
    rows.append({"mode": "rewrite", "variant": "llm_rewrite",
                 "chunk_size": cs, "n_chunks": meta["n_chunks"],
                 **m_rw, "sec": round(time.time() - t0, 1)})

    # (c) HyDE
    def retrieve_hyde(q, k):
        passage = _llm_call(generator, HYDE_PROMPT.format(q=q),
                            "hyde", q, max_tokens=384)
        return retriever.retrieve(passage or q, k=k)

    t0 = time.time()
    m_hyde = eval_recall(retrieve_hyde, gold)
    print_row("hyde", meta, m_hyde, time.time() - t0)
    rows.append({"mode": "rewrite", "variant": "hyde",
                 "chunk_size": cs, "n_chunks": meta["n_chunks"],
                 **m_hyde, "sec": round(time.time() - t0, 1)})

    for name, m in [("llm_rewrite", m_rw), ("hyde", m_hyde)]:
        print(f"\nΔ {name}: recall@10 {m['recall@10']-m_raw['recall@10']:+.3f}  "
              f"mrr {m['mrr']-m_raw['mrr']:+.3f}")
    return rows


# ---------- 入口 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+",
                    default=["chunk", "rerank", "rewrite"],
                    choices=["chunk", "rerank", "rewrite"])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[128, 256, 512, 1024])
    ap.add_argument("--overlap-ratio", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--rerank-cs", type=int, default=512,
                    help="reranker/rewrite 消融用的 chunk_size")
    ap.add_argument("--recall-k", type=int, default=20,
                    help="reranker/rewrite 阶段的召回数")
    ap.add_argument("--rerank-top", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "ablation.json")
    args = ap.parse_args()

    all_rows: List[dict] = []

    if "chunk" in args.modes:
        all_rows += run_chunk_ablation(args.sizes, args.overlap_ratio,
                                       args.batch_size, args.device)

    if "rerank" in args.modes:
        all_rows += run_rerank_ablation(
            cs=args.rerank_cs, overlap_ratio=args.overlap_ratio,
            batch_size=args.batch_size, device=args.device,
            recall_k=args.recall_k, rerank_top=args.rerank_top)

    if "rewrite" in args.modes:
        all_rows += run_rewrite_ablation(
            cs=args.rerank_cs, overlap_ratio=args.overlap_ratio,
            batch_size=args.batch_size, device=args.device,
            recall_k=args.recall_k)

    args.out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n结果写入 {args.out}")


if __name__ == "__main__":
    main()