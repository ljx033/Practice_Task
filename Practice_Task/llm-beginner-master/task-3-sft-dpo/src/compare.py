"""在同一批指令上对比 base / SFT / DPO 输出。

用法：
    python -m src.compare
或
    python src/compare.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lora import inject_lora, load_lora_state_dict  # noqa: E402
from src.chat import format_messages  # noqa: E402


DEFAULT_PROMPTS: List[str] = [
    "你好，请用一句话介绍你自己。",
    "什么是过拟合？应该怎么缓解？",
    "把下面这句话翻译成英文：今天天气很好，我们一起去公园吧。",
    "请写一个 Python 函数，判断一个整数是否为素数。",
    "给我讲一个短笑话，要求不超过两句话。",
    "请列出三条学习大模型的建议，每条一句话。",
]


def _build(model_path: str, lora_path: Optional[str], tok, args, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    if lora_path is not None:
        inject_lora(model, target_modules=args.target_modules,
                    r=args.lora_r, alpha=args.lora_alpha)
        sd = torch.load(lora_path, map_location="cpu")
        load_lora_state_dict(model, sd)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _generate(model, tok, prompt: str, args) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = format_messages(msgs, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[0][ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def _run_one(tag: str, model_path: str, lora_path: Optional[str],
             tok, prompts, args, device, dtype):
    print(f"\n========== [{tag}] ==========")
    if lora_path is not None and not Path(lora_path).exists():
        print(f"[skip] {lora_path} 不存在，跳过 {tag}")
        return [None] * len(prompts)
    model = _build(model_path, lora_path, tok, args, device, dtype)
    outs = []
    for p in prompts:
        text = _generate(model, tok, p, args)
        outs.append(text)
        print(f"\n--- prompt ---\n{p}\n--- {tag} ---\n{text}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    ap.add_argument("--sft_ckpt", default=str(ROOT / "ckpt" / "sft" / "lora.pt"))
    ap.add_argument("--dpo_ckpt", default=str(ROOT / "ckpt" / "dpo" / "lora.pt"))
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--prompts_file", default=None,
                    help="每行一条 prompt 的文本文件；不给则用内置 prompts")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--out", default=str(ROOT / "eval" / "compare_outputs.json"))
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[info] device={device} dtype={dtype}")

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [l.strip() for l in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if l.strip()]

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_outs = _run_one("base", args.model_path, None, tok, prompts, args, device, dtype)
    sft_outs = _run_one("sft", args.model_path, args.sft_ckpt, tok, prompts, args, device, dtype)
    dpo_outs = _run_one("dpo", args.model_path, args.dpo_ckpt, tok, prompts, args, device, dtype)

    results = []
    for p, b, s, d in zip(prompts, base_outs, sft_outs, dpo_outs):
        results.append({"prompt": p, "base": b, "sft": s, "dpo": d})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[info] 结果已写入 {out_path}")


if __name__ == "__main__":
    main()