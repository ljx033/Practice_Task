"""在 SFT LoRA 之上用 DPO 继续训练。reference model 冻结，只 forward。

产物：ckpt/dpo/lora.pt + lora_config.json
"""
from __future__ import annotations

import argparse
import copy
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


def iter_records(path: Path):
    if path.is_dir():
        for sub in sorted(path.iterdir()):
            yield from iter_records(sub)
    elif path.suffix == ".zip":
        yield from _iter_zip(path)
    elif path.suffix in (".jsonl", ".json"):
        yield from _iter_jsonl(path)


# def _turns_to_msgs(turns) -> List[dict]:
#     out = []
#     for t in turns:
#         if not isinstance(t, dict):
#             continue
#         role = t.get("role") or t.get("from")
#         content = t.get("content") or t.get("value")
#         if role is None or content is None:
#             continue
#         role = {"human": "user", "gpt": "assistant"}.get(role, role)
#         if role not in ("user", "assistant", "system"):
#             continue
#         out.append({"role": role, "content": str(content)})
#     return out
def _turns_to_msgs(turns) -> List[dict]:
    out: List[dict] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        # MOSS 风格 {"human":..., "assistant":...}
        if isinstance(t.get("human"), str) or isinstance(t.get("assistant"), str):
            if isinstance(t.get("human"), str) and t["human"].strip():
                out.append({"role": "user", "content": str(t["human"])})
            if isinstance(t.get("assistant"), str) and t["assistant"].strip():
                out.append({"role": "assistant", "content": str(t["assistant"])})
            continue
        role = t.get("role") or t.get("from") or t.get("sender")
        content = t.get("content") or t.get("value") or t.get("text")
        if role is None or content is None:
            continue
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
        if role not in ("user", "assistant", "system"):
            continue
        if not str(content).strip():
            continue
        out.append({"role": role, "content": str(content)})
    return out

def parse_dpo_record(rec: dict) -> Optional[dict]:
    """兼容多种 DPO 数据 schema。"""
    # 1) prompt / chosen / rejected 直接给出
    if "prompt" in rec and "chosen" in rec and "rejected" in rec:
        prompt = rec["prompt"]
        if isinstance(prompt, str):
            prompt_msgs = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            prompt_msgs = _turns_to_msgs(prompt)
        else:
            return None
        chosen = rec["chosen"]
        rejected = rec["rejected"]
        if isinstance(chosen, dict):
            chosen = chosen.get("content") or chosen.get("value") or ""
        if isinstance(rejected, dict):
            rejected = rejected.get("content") or rejected.get("value") or ""
        if not prompt_msgs or not chosen or not rejected:
            return None
        return {"prompt": prompt_msgs, "chosen": str(chosen), "rejected": str(rejected)}

    # 2) question / response_chosen / response_rejected（hiyouga / argilla 风格）
    if "question" in rec:
        chosen = rec.get("response_chosen") or rec.get("chosen")
        rejected = rec.get("response_rejected") or rec.get("rejected")
        if chosen and rejected:
            prompt = []
            if rec.get("system"):
                prompt.append({"role": "system", "content": str(rec["system"])})
            prompt.append({"role": "user", "content": str(rec["question"])})
            return {"prompt": prompt, "chosen": str(chosen), "rejected": str(rejected)}

    # 3) messages + chosen/rejected
    for key in ("conversations", "conversation", "messages"):
        if key in rec and isinstance(rec[key], list):
            msgs = _turns_to_msgs(rec[key])
            chosen = rec.get("chosen")
            rejected = rec.get("rejected")
            if not chosen or not rejected:
                continue
            if msgs and msgs[-1]["role"] == "assistant":
                prompt_msgs = msgs[:-1]
            else:
                prompt_msgs = msgs
            if not prompt_msgs:
                continue
            return {"prompt": prompt_msgs,
                    "chosen": str(chosen), "rejected": str(rejected)}
    return None


def load_dpo_samples(path: str, limit: Optional[int] = None) -> List[dict]:
    samples: List[dict] = []
    for rec in iter_records(Path(path)):
        s = parse_dpo_record(rec)
        if s is None:
            continue
        samples.append(s)
        if limit is not None and len(samples) >= limit:
            break
    return samples


# ---------------- Dataset ----------------

class DPODataset(Dataset):
    def __init__(self, samples: List[dict], tok, max_length: int = 1024):
        self.samples = samples
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def _encode(self, msgs):
        text = format_messages(msgs)
        ids = self.tok(text, return_tensors="pt").input_ids[0]
        labels = build_labels(ids, msgs, tokenizer=self.tok)
        if ids.size(0) > self.max_length:
            ids = ids[: self.max_length]
            labels = labels[: self.max_length]
        return ids, labels

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        c_msgs = s["prompt"] + [{"role": "assistant", "content": s["chosen"]}]
        r_msgs = s["prompt"] + [{"role": "assistant", "content": s["rejected"]}]
        c_ids, c_lab = self._encode(c_msgs)
        r_ids, r_lab = self._encode(r_msgs)
        return {"chosen_ids": c_ids, "chosen_labels": c_lab,
                "rejected_ids": r_ids, "rejected_labels": r_lab}


def make_collate(pad_id: int):
    def collate(batch):
        out = {}
        for side in ("chosen", "rejected"):
            key_ids, key_lab = f"{side}_ids", f"{side}_labels"
            max_len = max(x[key_ids].size(0) for x in batch)
            B = len(batch)
            ids = torch.full((B, max_len), pad_id, dtype=torch.long)
            lab = torch.full((B, max_len), -100, dtype=torch.long)
            attn = torch.zeros((B, max_len), dtype=torch.long)
            for i, x in enumerate(batch):
                n = x[key_ids].size(0)
                ids[i, :n] = x[key_ids]
                lab[i, :n] = x[key_lab]
                attn[i, :n] = 1
            out[key_ids] = ids
            out[key_lab] = lab
            out[f"{side}_attn"] = attn
        return out
    return collate


# ---------------- DPO 计算 ----------------

def sequence_logprob(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """返回每条样本的 Σ log π(y_t | y_<t)，只累计 labels != -100 的位置。"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logp = F.log_softmax(shift_logits, dim=-1)
    mask = (shift_labels != -100).float()
    safe = shift_labels.clone()
    safe[safe == -100] = 0
    per_tok = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (per_tok * mask).sum(dim=-1)


# ---------------- 训练 ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    p.add_argument("--sft_ckpt", default=str(ROOT / "ckpt" / "sft" / "lora.pt"))
    p.add_argument("--data_path", default=str(ROOT / "data" / "dpo"))
    p.add_argument("--out_dir", default=str(ROOT / "ckpt" / "dpo"))
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=10000)
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

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # data
    print(f"[info] loading DPO data from {args.data_path} ...")
    samples = load_dpo_samples(args.data_path, limit=args.max_samples)
    print(f"[info] loaded {len(samples)} DPO pairs")
    if not samples:
        sys.exit("[错误] 没有读到 DPO 样本，检查 --data_path（jsonl 字段需含 chosen/rejected）")
    random.shuffle(samples)

    ds = DPODataset(samples, tok, max_length=args.max_length)
    collate = make_collate(tok.pad_token_id)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, collate_fn=collate)

    # policy = base + SFT LoRA (继续训练)
    print("[info] loading base + SFT LoRA ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, trust_remote_code=True
    )
    base.config.use_cache = False
    inject_lora(base, target_modules=args.target_modules,
                r=args.lora_r, alpha=args.lora_alpha)
    sft_sd = torch.load(args.sft_ckpt, map_location="cpu")
    if isinstance(sft_sd, dict) and "state_dict" in sft_sd:
        sft_sd = sft_sd["state_dict"]
    missing, unexpected = base.load_state_dict(sft_sd, strict=False)
    print(f"[info] SFT load: missing={len(missing)} unexpected={len(unexpected)}")

    # reference 冻结副本
    print("[info] building frozen reference model ...")
    ref = copy.deepcopy(base)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    policy = base
    policy.to(device)
    ref.to(device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    print(f"[info] policy trainable params: {sum(p.numel() for p in trainable)}")

    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    steps_per_epoch = max(1, (len(dl) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    policy.train()
    global_step = 0
    micro_step = 0
    optim.zero_grad()
    log_loss = log_acc = log_margin = 0.0
    cnt = 0

    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            # ---------- reference: 只 forward ----------
            with torch.no_grad():
                ref_c = sequence_logprob(ref, batch["chosen_ids"],
                                         batch["chosen_attn"], batch["chosen_labels"])
                ref_r = sequence_logprob(ref, batch["rejected_ids"],
                                         batch["rejected_attn"], batch["rejected_labels"])

            # ---------- policy ----------
            pi_c = sequence_logprob(policy, batch["chosen_ids"],
                                    batch["chosen_attn"], batch["chosen_labels"])
            pi_r = sequence_logprob(policy, batch["rejected_ids"],
                                    batch["rejected_attn"], batch["rejected_labels"])

            # ---------- DPO loss ----------
            inside = args.beta * ((pi_c - pi_r) - (ref_c - ref_r))
            loss = -F.logsigmoid(inside).mean()

            (loss / args.grad_accum).backward()

            with torch.no_grad():
                log_loss += loss.item()
                log_acc += (inside > 0).float().mean().item()
                log_margin += inside.mean().item()
                cnt += 1
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0:
                    print(f"[dpo] epoch={epoch} step={global_step} "
                          f"loss={log_loss / cnt:.4f} "
                          f"acc={log_acc / cnt:.3f} "
                          f"margin={log_margin / cnt:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}")
                    log_loss = log_acc = log_margin = 0.0
                    cnt = 0

    # 保存
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(policy), out_dir / "lora.pt")
    (out_dir / "lora_config.json").write_text(
        json.dumps({
            "r": args.lora_r, "alpha": args.lora_alpha,
            "target_modules": args.target_modules, "beta": args.beta,
        }, indent=2), encoding="utf-8",
    )
    print(f"[done] DPO LoRA saved to {out_dir}")


if __name__ == "__main__":
    main()