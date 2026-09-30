import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 解决 OMP Error #15

import matplotlib
matplotlib.use("Agg")   # 解决 QThreadStorage 警告，图片只写盘不弹窗
import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.model import TransformerClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab(texts, min_freq=2):
    from collections import Counter

    counter = Counter()
    for text in texts:
        counter.update(str(text))

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<cls>": 2,
        "<sep>": 3,
    }

    for ch, freq in counter.most_common():
        if freq >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)

    return vocab


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

        self.unk_id = vocab["<unk>"]
        self.cls_id = vocab["<cls>"]
        self.sep_id = vocab["<sep>"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        ids = [self.cls_id] + [
            self.vocab.get(ch, self.unk_id) for ch in text.strip()
        ] + [self.sep_id]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len - 1] + [self.sep_id]

        return torch.tensor(ids, dtype=torch.long), int(self.labels[idx])


def collate_fn(batch):
    ids_list, labels = zip(*batch)
    ids = pad_sequence(ids_list, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return ids, labels


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for ids, labels in loader:
            ids = ids.to(device)
            labels = labels.to(device)

            logits = model(ids)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * labels.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / total, loss_sum / total


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    fig_dir = Path(args.fig_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    dev_df = pd.read_parquet(data_dir / "validation.parquet")

    print(f"训练集: {len(train_df)}，验证集: {len(dev_df)}")

    vocab = build_vocab(train_df["text"].astype(str).tolist(), args.min_freq)
    print(f"词表大小: {len(vocab)}")

    train_ds = SentimentDataset(
        train_df["text"], train_df["label"], vocab, args.max_len
    )
    dev_ds = SentimentDataset(
        dev_df["text"], dev_df["label"], vocab, args.max_len
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    config = dict(
        vocab_size=len(vocab),
        num_classes=2,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        pooling=args.pooling,
        pad_id=0,
        use_pos_emb=not args.no_pos_emb, #这一行新加
    )

    model = TransformerClassifier(**config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_epoch = -1
    bad_epochs = 0

    train_losses = []
    dev_accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for ids, labels in train_loader:
            ids = ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_ds)
        dev_acc, dev_loss = evaluate(model, dev_loader, device)

        train_losses.append(train_loss)
        dev_accs.append(dev_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | "
            f"dev_acc={dev_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            bad_epochs = 0

            ckpt = {
                "state_dict": model.state_dict(),
                "config": config,
                "vocab": vocab,
                "max_len": args.max_len,
                "best_acc": best_acc,
                "epoch": epoch,
            }
            # torch.save(ckpt, ckpt_dir / "best.pt")  #这一行替换为下一行新加
            torch.save(ckpt, ckpt_dir / f"best{args.tag}.pt")
            print(f"  -> 保存 best.pt (dev_acc={best_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, best acc {best_acc:.4f})"
                )
                break

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    plt.plot(range(1, len(dev_accs) + 1), dev_accs, label="dev_acc")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(fig_dir / "loss_curve.png", dpi=150)#这一行替换为下一行新加
    plt.savefig(fig_dir / f"loss_curve{args.tag}.png", dpi=150)
    plt.close()

    print(f"完成。best dev acc = {best_acc:.4f}")
    print(f"模型保存在: {ckpt_dir / 'best.pt'}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--fig_dir", type=str, default="figures")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_freq", type=int, default=2)

    parser.add_argument("--num_threads", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no_pos_emb", action="store_true",
                        help="消融：去掉位置编码")
    parser.add_argument("--tag", type=str, default="",
                        help="实验标签，用于区分 ckpt / 图片文件名")  #以上四行新加
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)