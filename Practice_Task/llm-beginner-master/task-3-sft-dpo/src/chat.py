"""Qwen chat template + loss masking。

约定（与 eval/run.py 对齐）：
    format_messages(messages: List[dict]) -> str
    build_labels(input_ids, messages, tokenizer=None) -> Tensor(同形状)
labels 中 assistant content 之外的 token（user/system/模板控制符/首尾特殊 token）全部 -100。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

_TOKENIZER = None


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B"


def _get_default_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(str(_default_model_dir()), trust_remote_code=True)
    return _TOKENIZER


# ---------- chat template ----------

def format_messages(messages: List[Dict], add_generation_prompt: bool = False) -> str:
    """套 Qwen 官方 chat template：<|im_start|>role\\ncontent<|im_end|>\\n"""
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role is None:
            raise ValueError(f"format_messages: message 缺 role: {msg}")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ---------- label masking ----------

def _tok_ids(tok, text: str) -> List[int]:
    return tok(text, add_special_tokens=False).input_ids


def build_labels(
    input_ids,
    messages: List[Dict],
    tokenizer=None,
) -> torch.Tensor:
    """给 user/system/模板控制符打 -100，只对 assistant content 算 loss。

    - 支持 torch.Tensor 或 list 形式 input_ids
    - 返回 Tensor，与 input_ids 同形状/同 dtype/同 device
    - 先尝试「token-id 增量匹配」（最精确），失败则回退到「字符偏移匹配」
    """
    if tokenizer is None:
        tokenizer = _get_default_tokenizer()

    is_tensor = isinstance(input_ids, torch.Tensor)
    ids_list: List[int] = input_ids.tolist() if is_tensor else list(input_ids)
    labels = [-100] * len(ids_list)

    ok, cur = _fill_by_token_match(ids_list, messages, tokenizer, labels)
    if not ok or cur != len(ids_list):
        labels = _fill_by_offset_match(ids_list, messages, tokenizer)

    if is_tensor:
        return torch.tensor(labels, dtype=input_ids.dtype, device=input_ids.device)
    return torch.tensor(labels, dtype=torch.long)


def _fill_by_token_match(ids_list, messages, tok, labels) -> (bool, int):
    cur = 0
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        prefix_ids = _tok_ids(tok, f"<|im_start|>{role}\n")
        content_ids = _tok_ids(tok, content)
        suffix_ids = _tok_ids(tok, "<|im_end|>\n")

        # prefix
        if ids_list[cur:cur + len(prefix_ids)] != prefix_ids:
            return False, cur
        cur += len(prefix_ids)
        # content
        if ids_list[cur:cur + len(content_ids)] != content_ids:
            return False, cur
        if role == "assistant":
            for j in range(cur, cur + len(content_ids)):
                labels[j] = ids_list[j]
        cur += len(content_ids)
        # suffix
        if ids_list[cur:cur + len(suffix_ids)] != suffix_ids:
            return False, cur
        cur += len(suffix_ids)
    return True, cur


def _fill_by_offset_match(ids_list, messages, tok):
    """回退方案：按字符偏移把 token 归到对应 span。"""
    text = format_messages(messages)
    spans = []
    pos = 0
    for msg in messages:
        pos += len(f"<|im_start|>{msg['role']}\n")
        cs = pos
        pos += len(msg.get("content", ""))
        ce = pos
        if msg["role"] == "assistant":
            spans.append((cs, ce))
        pos += len("<|im_end|>\n")

    labels = [-100] * len(ids_list)
    try:
        enc = tok(text, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except Exception:
        return labels

    n = min(len(offsets), len(ids_list))
    for i in range(n):
        s, e = offsets[i]
        if e <= s:
            continue
        for cs, ce in spans:
            if s >= cs and e <= ce:
                labels[i] = ids_list[i]
                break
    return labels