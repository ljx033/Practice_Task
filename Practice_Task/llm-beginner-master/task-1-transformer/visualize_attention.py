import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.model import load_for_eval

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def get_vocab_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("vocab", {})


def ids_to_tokens(ids, inv_vocab):
    tokens = []
    for i in ids.tolist():
        tok = inv_vocab.get(i, "<unk>")
        if tok == "<pad>":
            tok = ""
        elif tok == "<cls>":
            tok = "[CLS]"
        elif tok == "<sep>":
            tok = "[SEP]"
        tokens.append(tok)
    return tokens


def plot_attention(text, model, tokenize_fn, vocab, out_path, layer=-1, head=0):
    model.eval()
    ids = tokenize_fn(text).unsqueeze(0)

    with torch.no_grad():
        logits, attns = model(ids, return_attn=True)

    attn = attns[layer][0, head].cpu().numpy()

    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = ids_to_tokens(ids[0], inv_vocab)
    T = len(tokens)

    plt.figure(figsize=(max(8, T * 0.35), max(6, T * 0.35)))
    plt.imshow(attn, aspect="auto", cmap="YlOrRd")  #viridis
    plt.colorbar()

    plt.xticks(range(T), tokens, rotation=90, fontsize=7)
    plt.yticks(range(T), tokens, fontsize=7)

    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title(f"Attention (layer={layer}, head={head})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"保存 {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt/best.pt")
    parser.add_argument("--out_dir", type=str, default="figures")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenize_fn = load_for_eval(args.ckpt)
    vocab = get_vocab_from_ckpt(args.ckpt)

    dev_path = Path(args.data_dir) / "validation.parquet"

    if dev_path.exists():
        dev = pd.read_parquet(dev_path)

        pos = dev[dev["label"] == 1].iloc[0]["text"]
        neg = dev[dev["label"] == 0].iloc[0]["text"]

        long_idx = dev["text"].astype(str).str.len().idxmax()
        long_text = dev.loc[long_idx, "text"]
    else:
        pos = "这家酒店位置很好，服务也不错，下次还会来。"
        neg = "非常失望，房间很脏，服务态度也很差，不会再来了。"
        long_text = pos * 3

    plot_attention(
        pos,
        model,
        tokenize_fn,
        vocab,
        out_dir / "attention_positive.png",
        args.layer,
        args.head,
    )
    plot_attention(
        neg,
        model,
        tokenize_fn,
        vocab,
        out_dir / "attention_negative.png",
        args.layer,
        args.head,
    )
    plot_attention(
        long_text,
        model,
        tokenize_fn,
        vocab,
        out_dir / "attention_long.png",
        args.layer,
        args.head,
    )


if __name__ == "__main__":
    main()