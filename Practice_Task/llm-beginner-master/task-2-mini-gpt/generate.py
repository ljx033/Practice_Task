"""加载训练好的模型，用四种采样策略生成样例。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import load_for_eval


PROMPTS_DEFAULT = ["床前明月光", "春眠不觉晓", "白日依山尽"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="ckpt/best.pt")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, tok = load_for_eval(str(ROOT / args.ckpt))
    model.eval()

    prompts = [args.prompt] if args.prompt else PROMPTS_DEFAULT

    configs = [
        ("greedy       ", dict(top_k=None, top_p=None, temperature=0.0)),
        ("temp=0.8     ", dict(top_k=None, top_p=None, temperature=0.8)),
        ("top-k=20     ", dict(top_k=20,  top_p=None, temperature=0.8)),
        ("top-p=0.9    ", dict(top_k=None, top_p=0.9, temperature=0.8)),
        ("top-k+top-p  ", dict(top_k=40,  top_p=0.9, temperature=0.8)),
    ]

    for prompt in prompts:
        print(f"\n===== prompt: {prompt!r} =====")
        for name, cfg in configs:
            torch.manual_seed(args.seed)
            ids = tok.encode(prompt)
            out = model.generate(
                torch.tensor([ids], dtype=torch.long),
                max_new_tokens=args.max_new_tokens,
                top_k=cfg["top_k"],
                top_p=cfg["top_p"],
                temperature=cfg["temperature"],
            )
            text = tok.decode(out[0].tolist())
            print(f"[{name}] {text}")


if __name__ == "__main__":
    main()