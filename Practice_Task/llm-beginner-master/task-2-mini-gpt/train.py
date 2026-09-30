"""MiniGPT 预训练脚本。

用法示例（唐诗，CPU，i5-13500H 约 3–6 分钟跑完）：
  python train.py --device cpu --vocab_size 1024 --max_steps 1200 \
      --d_model 128 --n_heads 4 --n_layers 4 --block_size 128 \
      --batch_size 16 --lr 3e-4 --dropout 0.1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import MiniGPT
from src.tokenizer import BPETokenizer


def get_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_dev_ppl(model: MiniGPT, ids, block_size: int, device, max_tokens: int = 4096) -> float:
    """与 eval/run.py 的 perplexity_on_dev 保持一致的算法。"""
    model.eval()
    ids = ids[:max_tokens]
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(ids) - 1), block_size):
        window = ids[i:i + block_size + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(chunk)
        nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        n_tok += chunk.size(1) - 1
    if n_tok == 0:
        return float("inf")
    return math.exp(nll / n_tok)


def get_batch(ids, block_size: int, batch_size: int, device):
    n = len(ids) - block_size - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([torch.tensor(ids[i:i + block_size], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(ids[i + 1:i + 1 + block_size], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)


def get_lr(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--reuse_tokenizer", action="store_true")
    ap.add_argument("--out_dir", type=str, default="ckpt")
    ap.add_argument("--train_file", type=str, default="data/train.txt")
    ap.add_argument("--dev_file", type=str, default="data/dev.txt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"[config] device={device}")

    train_text = (ROOT / args.train_file).read_text(encoding="utf-8")
    dev_text = (ROOT / args.dev_file).read_text(encoding="utf-8")
    print(f"[data] train chars={len(train_text)}, dev chars={len(dev_text)}")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"

    # ---------- tokenizer ----------
    if args.reuse_tokenizer and tok_path.exists():
        print(f"[tokenizer] reuse {tok_path}")
        tok = BPETokenizer.from_pretrained(str(tok_path))
    else:
        print(f"[tokenizer] training BPE, vocab_size={args.vocab_size} ...")
        t0 = time.time()
        tok = BPETokenizer()
        tok.train(train_text, vocab_size=args.vocab_size, verbose=True)
        tok.save(str(tok_path))
        print(f"[tokenizer] done in {time.time() - t0:.1f}s, vocab={tok.vocab_size}")

    train_ids = tok.encode(train_text)
    dev_ids = tok.encode(dev_text)
    print(f"[tokenize] train_tokens={len(train_ids)}, dev_tokens={len(dev_ids)}")

    # ---------- model ----------
    model = MiniGPT(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params / 1e6:.3f}M")

    # 权重衰减：LayerNorm 与 bias 不做
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    best_ppl = float("inf")
    best_path = out_dir / "best.pt"
    log = []

    t0 = time.time()
    for step in range(args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        model.train()
        x, y = get_batch(train_ids, args.block_size, args.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if step % 50 == 0 or step == args.max_steps - 1:
            print(f"[step {step:4d}] loss={loss.item():.4f} lr={lr:.2e} "
                  f"elapsed={time.time() - t0:.1f}s")

        if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
            ppl = compute_dev_ppl(model, dev_ids, args.block_size, device)
            print(f"[eval step={step + 1}] dev_ppl={ppl:.3f}")
            log.append({"step": step + 1, "dev_ppl": ppl, "train_loss": loss.item()})
            if ppl < best_ppl:
                best_ppl = ppl
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": model.config,
                    "tokenizer_path": str(tok_path),
                    "step": step + 1,
                    "dev_ppl": ppl,
                }, str(best_path))
                print(f"  -> saved {best_path} (best so far)")

    print(f"[done] best_dev_ppl={best_ppl:.3f}, total {time.time() - t0:.1f}s")
    log_path = out_dir / "train_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()