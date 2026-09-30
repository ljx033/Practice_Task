"""采样策略：greedy / temperature / top-k / top-p（可叠加）。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_next_token(logits: torch.Tensor,
                      top_k: Optional[int] = None,
                      top_p: Optional[float] = None,
                      temperature: float = 1.0) -> torch.Tensor:
    """从 logits 采样下一个 token。

    logits: (..., vocab_size)
    返回:   (...) 与输入去掉最后一维一致
    temperature == 0 或 None 时退化为 greedy（不除零）。
    """
    original_dim = logits.dim()
    if original_dim == 1:
        logits = logits.unsqueeze(0)

    # greedy
    if temperature is None or float(temperature) == 0.0:
        ids = logits.argmax(dim=-1)
        return ids.squeeze(0) if original_dim == 1 else ids

    logits = logits / float(temperature)

    # top-k
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1, None]
        logits = torch.where(logits < threshold,
                             torch.full_like(logits, float("-inf")),
                             logits)

    # top-p
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # 需要移除的位置：cum[i-1] > top_p 表示前面已累计够
        to_remove = cum > float(top_p)
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered = filtered.scatter(-1, sorted_idx, sorted_logits)
        logits = filtered

    probs = F.softmax(logits, dim=-1)
    ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return ids.squeeze(0) if original_dim == 1 else ids