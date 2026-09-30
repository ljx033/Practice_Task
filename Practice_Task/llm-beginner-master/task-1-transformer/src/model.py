import torch
import torch.nn as nn

from .block import TransformerBlock


class TransformerClassifier(nn.Module):
    """
    字符级 Transformer 文本分类器。
    forward(ids) 默认返回 logits: (B, num_classes)
    forward(ids, return_attn=True) 返回 (logits, attns)
    """

    def __init__(
        self,
        vocab_size,
        num_classes=2,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_len=256,
        dropout=0.1,
        pooling="cls",
        pad_id=0,
        use_pos_emb=True,    #新加
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.pooling = pooling
        self.use_pos_emb = use_pos_emb   #新加

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ids, return_attn=False):
        B, T = ids.shape
        if T > self.max_len:
            ids = ids[:, : self.max_len]
            T = self.max_len

        padding_mask = ids == self.pad_id  # True 表示 pad

        # pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        # x = self.token_emb(ids) + self.pos_emb(pos)
        # x = self.dropout(x)原来的三行代码新加实验修改的
        if self.use_pos_emb:
            pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
            x = self.token_emb(ids) + self.pos_emb(pos)
        else:
            x = self.token_emb(ids)
        x = self.dropout(x)    #这六行新加

        attns = []
        for block in self.blocks:
            if return_attn:
                x, attn = block(
                    x,
                    key_padding_mask=padding_mask,
                    need_weights=True,
                )
                attns.append(attn)
            else:
                x = block(x, key_padding_mask=padding_mask)

        if self.pooling == "cls":
            pooled = x[:, 0]
        elif self.pooling == "mean":
            mask = (~padding_mask).unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError(f"未知 pooling: {self.pooling}")

        pooled = self.ln_final(pooled)
        logits = self.classifier(pooled)

        if return_attn:
            return logits, attns
        return logits


def make_tokenize_fn(vocab, max_len):
    """
    vocab: dict[str, int]
    返回 tokenize_fn(text) -> LongTensor (T,)
    特殊 token: <pad>=0, <unk>=1, <cls>=2, <sep>=3
    """
    pad_id = vocab.get("<pad>", 0)
    unk_id = vocab.get("<unk>", 1)
    cls_id = vocab.get("<cls>", 2)
    sep_id = vocab.get("<sep>", 3)

    def tokenize_fn(text):
        text = str(text)
        ids = [cls_id] + [vocab.get(ch, unk_id) for ch in text.strip()] + [sep_id]
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [sep_id]
        return torch.tensor(ids, dtype=torch.long)

    return tokenize_fn


def load_for_eval(ckpt_path):
    """
    加载训练保存的 checkpoint，返回 (model, tokenize_fn)。
    注意 PyTorch 2.6+ 默认 weights_only=True，这里显式设为 False。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        config = ckpt.get("config", {})
        vocab = ckpt.get("vocab", None)
        max_len = ckpt.get("max_len", config.get("max_len", 256))
    else:
        # 兼容纯 state_dict，但缺少 vocab 时无法恢复 tokenizer
        state_dict = ckpt
        config = {}
        vocab = None
        max_len = 256

    if vocab is None:
        raise ValueError(
            "checkpoint 缺少 vocab，无法构建 tokenizer。请使用 train.py 保存的 best.pt。"
        )

    model = TransformerClassifier(**config)
    model.load_state_dict(state_dict)
    model.eval()

    tokenize_fn = make_tokenize_fn(vocab, max_len)
    return model, tokenize_fn