"""旋转位置编码（RoPE），按「前后折半」配对（GPT-NeoX 风格）。

约定：
- head_dim 必须为偶数；
- 配对为 (x[i], x[i + head_dim/2])，i in [0, head_dim/2)；
- 频率基底 base=10000；
- 只作用在 Q 和 K 上，不作用于 V。
"""
from __future__ import annotations

import torch


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0):
    """返回 (cos, sin)，形状均为 (max_seq_len, dim // 2)。"""
    assert dim % 2 == 0, "RoPE 需要偶数 head_dim"
    half = dim // 2
    # theta_i = base^{-2i/dim}, i = 0..half-1
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)   # (T, half)
    return torch.cos(angles), torch.sin(angles)


# def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
#                position_offset: int = 0) -> torch.Tensor:
#     """对 x 应用 RoPE。
#
#     x: (B, H, T, D)
#     cos, sin: (max_seq_len, D//2)
#     position_offset: 本段第一个 token 的绝对位置（增量解码时等于历史长度）
#     """
#     B, H, T, D = x.shape
#     half = D // 2
#     # 取本段对应位置的 cos/sin，形状 (T, half)
#     cos_ = cos[position_offset:position_offset + T].unsqueeze(0).unsqueeze(0)
#     sin_ = sin[position_offset:position_offset + T].unsqueeze(0).unsqueeze(0)
#
#     x1 = x[..., :half]
#     x2 = x[..., half:]
#     out1 = x1 * cos_ - x2 * sin_
#     out2 = x1 * sin_ + x2 * cos_
#     return torch.cat([out1, out2], dim=-1)
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               position_offset: int = 0, base: float = 10000.0) -> torch.Tensor:
    """对 x 应用 RoPE。

    x: (B, H, T, D)
    cos, sin: (max_seq_len, D//2)
    position_offset: 本段第一个 token 的绝对位置（增量解码时等于历史长度）
    base: 频率基底（与训练时一致）；cos/sin 长度不够时会用该 base 动态补算。
    """
    B, H, T, D = x.shape
    half = D // 2
    need = position_offset + T

    # 保险：若现有 cos/sin 不够长，动态补算到 need
    if need > cos.size(0):
        extra_cos, extra_sin = precompute_freqs_cis(D, need, base=base)
        cos = extra_cos.to(x.device, dtype=x.dtype)
        sin = extra_sin.to(x.device, dtype=x.dtype)

    cos_ = cos[position_offset:position_offset + T].unsqueeze(0).unsqueeze(0)
    sin_ = sin[position_offset:position_offset + T].unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half]
    x2 = x[..., half:]
    out1 = x1 * cos_ - x2 * sin_
    out2 = x1 * sin_ + x2 * cos_
    return torch.cat([out1, out2], dim=-1)