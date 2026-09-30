"""MiniGPT：decoder-only 语言模型（RoPE + KV cache）。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, KVCache
from .rope import precompute_freqs_cis
from .tokenizer import BPETokenizer


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden_mult * d_model
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x, cos, sin,
                kv_cache: Optional[KVCache] = None,
                position_offset: int = 0):
        h, new_cache = self.attn(self.ln1(x), cos, sin, kv_cache, position_offset)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 block_size: int = 128,
                 dropout: float = 0.0,
                 rope_base: float = 10000.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.max_seq_len = block_size        # 自检读 block_size 或 max_seq_len
        self.rope_base = rope_base

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            Block(d_model, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 预计算 RoPE cos/sin（形状 (block_size, head_dim/2)）
        head_dim = d_model // n_heads
        rope_max = block_size + 1
        cos, sin = precompute_freqs_cis(head_dim, block_size, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 权重绑定（节省参数；初始化已完成，绑定到 tok_emb 的权重）
        self.lm_head.weight = self.tok_emb.weight

        self.config = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            block_size=block_size,
            dropout=dropout,
            rope_base=rope_base,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---------------- forward ----------------
    def forward(self,
                ids: torch.Tensor,
                kv_cache: Optional[List[KVCache]] = None,
                return_cache: bool = False):
        """ids: (B, T) 的 int64。

        若 return_cache=True 返回 (logits, new_cache)；否则只返回 logits。
        kv_cache: list[KVCache]，长度为 n_layers，每项 (k, v)。
        """
        B, T = ids.shape
        x = self.drop(self.tok_emb(ids))

        position_offset = 0
        if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
            position_offset = kv_cache[0][0].size(2)

        new_cache: List[KVCache] = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, c = layer(x, self.rope_cos, self.rope_sin, layer_cache, position_offset)
            new_cache.append(c)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        if return_cache:
            return logits, new_cache
        return logits

    # ---------------- 生成 ----------------
    @torch.no_grad()
    def generate(self,
                 prompt_ids,
                 max_new_tokens: int = 64,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 temperature: float = 1.0,
                 use_cache: bool = True) -> torch.Tensor:
        from .sampling import sample_next_token

        self.eval()
        device = next(self.parameters()).device

        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(device)

        generated = prompt_ids

        if use_cache:
            logits, cache = self(prompt_ids, return_cache=True)
            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :]
                nid = sample_next_token(next_logits, top_k, top_p, temperature)
                generated = torch.cat([generated, nid.view(-1, 1)], dim=1)
                logits, cache = self(nid.view(-1, 1), kv_cache=cache, return_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = self(generated)
                next_logits = logits[:, -1, :]
                nid = sample_next_token(next_logits, top_k, top_p, temperature)
                generated = torch.cat([generated, nid.view(-1, 1)], dim=1)

        return generated


# ---------------- 便捷加载 ----------------
def load_for_eval(ckpt_path: str):
    """加载训练好的 MiniGPT 与配套 tokenizer，返回 (model, tokenizer)。"""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = MiniGPT(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # tokenizer 默认与 ckpt 同目录
    tok_path = ckpt_path.parent / "tokenizer.json"
    if not tok_path.exists():
        alt = ckpt.get("tokenizer_path")
        if alt and Path(alt).exists():
            tok_path = Path(alt)
    tok = BPETokenizer.from_pretrained(str(tok_path))
    return model, tok