"""主入口：单次实验"""
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASETS, TRAIN_CONFIG, MODELS
from data_utils import load_dataset, split_edges, set_seed, describe_split
from models import build_model
from train import train_full_graph, train_minibatch
from evaluate import format_time


def run_single_experiment(dataset_name, model_name, mode="both",
                          num_layers=2, lr=None, config_override=None):
    """
    运行单次实验。
    参数:
        dataset_name: 'cora' / 'citeseer' / 'flickr'
        model_name: 'gcn' / 'gat' / 'graphsage' / 'gin'
        mode: 'full' / 'minibatch' / 'both'
    """
    config = {**TRAIN_CONFIG}
    if lr is not None:
        config["lr"] = lr
    if config_override:
        config.update(config_override)

    set_seed(config["seed"])

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name} | Model: {model_name} | "
          f"Layers: {num_layers} | LR: {config['lr']}")
    print(f"{'=' * 60}")

    # 加载数据
    data = load_dataset(dataset_name)
    in_dim = data.num_node_features
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"feature dim: {in_dim}")

    # 划分链路预测数据
    train_data, val_data, test_data = split_edges(data, seed=config["seed"])
    describe_split(train_data, val_data, test_data)

    # 把 val/test 挂到 train_data 上，方便训练函数访问
    train_data.val_data = val_data
    train_data.test_data = test_data

    results = {}

    # 全图训练
    if mode in ("full", "both"):
        print(f"\n--- Full Graph Training ---")
        model = build_model(model_name, in_dim,
                            DATASETS[dataset_name]["hidden_dim"], num_layers)
        val_auc, test_auc, total_time, epoch_times = train_full_graph(
            model, train_data, config
        )
        results["full"] = {
            "val_auc": val_auc,
            "test_auc": test_auc,
            "total_time": total_time,
            "avg_epoch_time": float(np.mean(epoch_times)),
            "epoch_times": epoch_times,
        }
        print(f"  Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | "
              f"Time: {format_time(total_time)} | "
              f"Avg epoch: {format_time(np.mean(epoch_times))}")

    # 小批次训练
    if mode in ("minibatch", "both"):
        print(f"\n--- Mini-batch Training ---")
        model = build_model(model_name, in_dim,
                            DATASETS[dataset_name]["hidden_dim"], num_layers)
        val_auc, test_auc, total_time, epoch_times = train_minibatch(
            model, train_data, config
        )
        results["minibatch"] = {
            "val_auc": val_auc,
            "test_auc": test_auc,
            "total_time": total_time,
            "avg_epoch_time": float(np.mean(epoch_times)),
            "epoch_times": epoch_times,
        }
        print(f"  Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | "
              f"Time: {format_time(total_time)} | "
              f"Avg epoch: {format_time(np.mean(epoch_times))}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PyG Link Prediction")
    parser.add_argument("--dataset", type=str, default="cora",
                        choices=["cora", "citeseer", "flickr"])
    parser.add_argument("--model", type=str, default="gcn",
                        choices=["gcn", "gat", "graphsage", "gin"])
    parser.add_argument("--mode", type=str, default="both",
                        choices=["full", "minibatch", "both"])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config_override = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
    }

    results = run_single_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
        mode=args.mode,
        num_layers=args.layers,
        lr=args.lr,
        config_override=config_override,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()