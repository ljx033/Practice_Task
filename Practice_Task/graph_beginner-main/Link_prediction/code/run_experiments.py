"""批量实验脚本"""
import subprocess
import sys
import os

PYTHON = sys.executable
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")


def run_one(dataset, model, mode, layers=2, lr=None):
    cmd = [
        PYTHON, SCRIPT,
        "--dataset", dataset,
        "--model", model,
        "--mode", mode,
        "--layers", str(layers),
        "--epochs", "100",
        "--batch_size", "512",
        "--device", "cuda:0",
    ]
    if lr is not None:
        cmd.extend(["--lr", str(lr)])

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result


def main():
    datasets = ["cora", "citeseer", "flickr"]
    models = ["gcn", "gat", "graphsage", "gin"]

    # 实验1：主对比（全图 vs 小批次）
    print("\n" + "=" * 70)
    print("Experiment 1: Full-graph vs Mini-batch")
    print("=" * 70)
    for ds in datasets:
        for model in models:
            # Flickr 跳过 GAT/GIN（显存友好）
            if ds == "flickr" and model in ("gat", "gin"):
                print(f"Skipping {ds} + {model} (显存优化)")
                continue
            run_one(ds, model, "both", lr=0.001)
            print("-" * 50)

    # 实验2：学习率对比
    print("\n" + "=" * 70)
    print("Experiment 2: Learning Rate (Cora, GCN)")
    print("=" * 70)
    for lr in [0.001, 0.005, 0.01]:
        run_one("cora", "gcn", "both", lr=lr)

    # 实验3：层数对比
    print("\n" + "=" * 70)
    print("Experiment 3: Layer Depth (Cora, GCN)")
    print("=" * 70)
    for layers in [2, 3]:
        run_one("cora", "gcn", "both", layers=layers, lr=0.001)


if __name__ == "__main__":
    main()