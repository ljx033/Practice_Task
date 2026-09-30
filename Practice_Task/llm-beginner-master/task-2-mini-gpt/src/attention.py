"""因果多头自注意力，支持增量解码时传入/返回 KV cache。

KV cache 结构（供 MiniGPT 跨层使用）：
  cache = (k, v)
  k: (B, H, T_cached, head_dim)
  v: (B, H, T_cached, head_dim)
每层一份，MiniGPT.forward 里外层用 list 组织。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                position_offset: int = 0
                ) -> Tuple[torch.Tensor, KVCache]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE 只加在 Q/K
        q = apply_rope(q, cos, sin, position_offset)
        k = apply_rope(k, cos, sin, position_offset)

        # 拼 KV cache（沿序列长度维 dim=2）
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache: KVCache = (k, v)

        T_total = k.size(2)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = (q @ k.transpose(-2, -1)) * scale   # (B, H, T, T_total)

        # 因果 mask：query 绝对位置 < key 绝对位置则屏蔽
        if T > 1 or position_offset > 0:
            q_pos = torch.arange(position_offset, position_offset + T, device=x.device).view(T, 1)
            k_pos = torch.arange(0, T_total, device=x.device).view(1, T_total)
            mask = (q_pos < k_pos).view(1, 1, T, T_total)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                   # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.resid_dropout(self.o_proj(out)), new_cache