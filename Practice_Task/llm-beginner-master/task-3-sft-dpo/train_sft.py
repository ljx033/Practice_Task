"""用 MOSS-003-sft 数据在 Qwen2.5-0.5B + 手写 LoRA 上跑 SFT。

用法示例见 README。产物：ckpt/sft/lora.pt + lora_config.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.lora import inject_lora, lora_state_dict  # noqa: E402
from src.chat import format_messages, build_labels  # noqa: E402


# ---------------- 数据读取 ----------------

def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_zip(path: Path) -> Iterator[dict]:
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".jsonl"):
                continue
            with zf.open(name) as fh:
                for raw in fh:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def iter_records(path: Path) -> Iterator[dict]:
    if path.is_dir():
        for sub in sorted(path.iterdir()):
            yield from iter_records(sub)
    elif path.suffix == ".zip":
        yield from _iter_zip(path)
    elif path.suffix in (".jsonl", ".json"):
        yield from _iter_jsonl(path)


# def _normalize_turns(turns) -> List[dict]:
#     out: List[dict] = []
#     for t in turns:
#         if not isinstance(t, dict):
#             continue
#         role = t.get("role") or t.get("from") or t.get("sender")
#         content = t.get("content") or t.get("value") or t.get("text")
#         if role is None or content is None:
#             continue
#         role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
#         if role not in ("user", "assistant", "system"):
#             continue
#         content = str(content)
#         if not content.strip():
#             continue
#         out.append({"role": role, "content": content})
#     return out
def _normalize_turns(turns) -> List[dict]:
    """把多种常见对话回合格式统一成 [{"role","content"}, ...]。

    支持的回合写法：
      1) MOSS 003：{"human": "...", "assistant": "..."}  （一个 dict 含一对）
      2) 通用：    {"role": "user"/"assistant"/"system", "content": "..."}
      3) ShareGPT：{"from": "human"/"gpt", "value": "..."}
      4) 变体：    {"sender": ..., "text": ...}
    """
    out: List[dict] = []
    for t in turns:
        if not isinstance(t, dict):
            continue

        # --- 格式 1：MOSS 的 {"human": ..., "assistant": ...} ---
        has_human = isinstance(t.get("human"), str)
        has_assistant = isinstance(t.get("assistant"), str)
        if has_human or has_assistant:
            if has_human and t["human"].strip():
                out.append({"role": "user", "content": str(t["human"])})
            if has_assistant and t["assistant"].strip():
                out.append({"role": "assistant", "content": str(t["assistant"])})
            # 若同一 dict 里还带 user/assistant 之外的 role，可继续走下面通用分支
            if has_human or has_assistant:
                continue

        # --- 格式 2/3/4：通用 role/content 系 ---
        role = t.get("role") or t.get("from") or t.get("sender")
        content = t.get("content") or t.get("value") or t.get("text")
        if role is None or content is None:
            continue
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
        if role not in ("user", "assistant", "system"):
            continue
        content = str(content)
        if not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out

def parse_record(rec: dict) -> Optional[List[dict]]:
    # MOSS 常见：dialogue / conversations / messages
    for key in ("dialogue", "conversations", "conversation", "messages"):
        if key in rec and isinstance(rec[key], list):
            msgs = _normalize_turns(rec[key])
            if msgs and any(m["role"] == "assistant" for m in msgs):
                return msgs
    # 兜底：instruction / input / output
    if "instruction" in rec and "output" in rec:
        user = str(rec.get("instruction", ""))
        if rec.get("input"):
            user = user + "\n" + str(rec["input"])
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(rec["output"])},
        ]
    return None


def load_sft_samples(path: str, limit: Optional[int] = None) -> List[List[dict]]:
    samples: List[List[dict]] = []
    for rec in iter_records(Path(path)):
        msgs = parse_record(rec)
        if msgs is None:
            continue
        samples.append(msgs)
        if limit is not None and len(samples) >= limit:
            break
    return samples


# ---------------- Dataset / Collate ----------------

class SFTDataset(Dataset):
    def __init__(self, samples: List[List[dict]], tok, max_length: int = 1024):
        self.samples = samples
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        msgs = self.samples[idx]
        text = format_messages(msgs)
        ids = self.tok(text, return_tensors="pt").input_ids[0]
        labels = build_labels(ids, msgs, tokenizer=self.tok)
        if ids.size(0) > self.max_length:
            ids = ids[: self.max_length]
            labels = labels[: self.max_length]
        return {"input_ids": ids, "labels": labels}


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        B = len(batch)
        ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        lab = torch.full((B, max_len), -100, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.long)
        for i, x in enumerate(batch):
            n = x["input_ids"].size(0)
            ids[i, :n] = x["input_ids"]
            lab[i, :n] = x["labels"]
            attn[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": attn}
    return collate


# ---------------- 训练 ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    p.add_argument("--data_path", default=str(ROOT / "data" / "moss-sft"))
    p.add_argument("--out_dir", default=str(ROOT / "ckpt" / "sft"))
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[info] device={device} dtype={torch_dtype}")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # data
    print(f"[info] loading SFT data from {args.data_path} ...")
    samples = load_sft_samples(args.data_path, limit=args.max_samples)
    print(f"[info] loaded {len(samples)} samples")
    if not samples:
        sys.exit("[错误] 没有读到样本，检查 --data_path")
    random.shuffle(samples)

    ds = SFTDataset(samples, tok, max_length=args.max_length)
    collate = make_collate(tok.pad_token_id)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, collate_fn=collate)

    # model + LoRA
    print("[info] loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model.config.use_cache = False
    inject_lora(model, target_modules=args.target_modules,
                r=args.lora_r, alpha=args.lora_alpha)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[info] trainable {n_train} / total {n_total} = {n_train / n_total:.4%}")

    # optimizer / scheduler
    optim = torch.optim.AdamW(trainable, lr=args.lr,
                              weight_decay=args.weight_decay,
                              betas=(0.9, 0.95))
    steps_per_epoch = max(1, (len(dl) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # train
    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_cnt = 0
    optim.zero_grad()

    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            logits = out.logits[:, :-1, :]
            labels = batch["labels"][:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            running_cnt += 1
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0:
                    print(f"[sft] epoch={epoch} step={global_step} "
                          f"loss={running_loss / running_cnt:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}")
                    running_loss, running_cnt = 0.0, 0

    # 保存
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), out_dir / "lora.pt")
    (out_dir / "lora_config.json").write_text(
        json.dumps({
            "r": args.lora_r, "alpha": args.lora_alpha,
            "target_modules": args.target_modules,
        }, indent=2), encoding="utf-8",
    )
    print(f"[done] SFT LoRA saved to {out_dir}")


if __name__ == "__main__":
    main()