import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block:
      x = x + Attention(LN(x))
      x = x + FFN(LN(x))
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None, causal=False, need_weights=False):
        T = x.size(1)
        attn_mask = None
        if causal:
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )

        if need_weights:
            attn_out, attn = self.attn(
                self.ln1(x),
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=True,
            )
        else:
            attn_out = self.attn(
                self.ln1(x),
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            attn = None

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))

        if need_weights:
            return x, attn
        return x