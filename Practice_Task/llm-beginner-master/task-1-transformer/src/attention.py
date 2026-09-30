import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, return_attn=False):
    """
    手写缩放点积注意力。
    Q/K/V: (B, H, T, D)
    mask: 可广播到 (B, H, T, T)，True 表示被屏蔽
    返回:
      out: (B, H, T, D)
      若 return_attn=True，额外返回 attn: (B, H, T, T)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)

    if return_attn:
        return out, attn
    return out


class MultiHeadAttention(nn.Module):
    """
    手写多头注意力。
    输入 x: (B, T, d_model)
    key_padding_mask: (B, T)，True 表示 padding 位置，需要屏蔽
    attn_mask: 可广播到 (B, H, T, T)，True 表示屏蔽，例如 causal mask
    """

    def __init__(self, d_model, n_heads, dropout=0.1, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None, need_weights=False):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        mask = None
        if key_padding_mask is not None:
            # (B, T) -> (B, 1, 1, T)
            mask = key_padding_mask[:, None, None, :]

        if attn_mask is not None:
            if mask is None:
                mask = attn_mask
            else:
                mask = mask | attn_mask

        out, attn = scaled_dot_product_attention(q, k, v, mask=mask, return_attn=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        if need_weights:
            return out, attn
        return out