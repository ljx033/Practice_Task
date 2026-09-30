import torch
import torch.nn as nn

from src.attention import scaled_dot_product_attention


class ToyLM(nn.Module):
    """
    极简 next-token prediction 语言模型，用于验证 causal mask。
    """

    def __init__(self, vocab_size=20, d_model=32, n_heads=4, max_len=16):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.token_emb(ids) + self.pos_emb(pos)

        q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        causal_mask = torch.triu(
            torch.ones(T, T, device=ids.device, dtype=torch.bool),
            diagonal=1,
        )

        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.out(out)


def check_causal():
    torch.manual_seed(0)
    model = ToyLM().eval()

    ids = torch.randint(0, 20, (1, 8))
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % 20

    with torch.no_grad():
        logits1 = model(ids)
        logits2 = model(ids2)

    diff = (logits1[:, :-1] - logits2[:, :-1]).abs().max().item()
    print(f"[causal_mask] 未来 token 改动后过去位置最大差异 = {diff:.2e}")

    assert diff < 1e-6, "causal mask 失败：未来信息泄漏"
    print("[causal_mask] 通过")


if __name__ == "__main__":
    check_causal()