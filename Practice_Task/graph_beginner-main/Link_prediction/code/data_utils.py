"""数据加载与链路预测数据集划分（完全手动实现，不依赖 RandomLinkSplit）"""
import os
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.data import Data


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_name: str, data_dir: str = "../data"):
    """
    加载数据集。
    Cora/Citeseer 来自 Planetoid，Flickr 来自 GraphSAINT 数据集。
    返回单个 Data 对象，边已转为无向。
    """
    if dataset_name == "cora":
        dataset = Planetoid(root=data_dir, name="Cora",
                            transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name == "citeseer":
        dataset = Planetoid(root=data_dir, name="Citeseer",
                            transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name == "flickr":
        dataset = Flickr(root=r"E:/Aljx/graph_beginner-main/Link_prediction/data/Flickr",
                         transform=T.NormalizeFeatures())
        data = dataset[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 确保边为无向
    if not data.is_undirected():
        data = T.ToUndirected()(data)

    return data


def split_edges(data, val_ratio=0.05, test_ratio=0.10, seed=42):
    """
    完全手动实现链路预测数据集划分。
    返回 (train_data, val_data, test_data)，每个 Data 包含:
        - x, num_nodes: 节点信息
        - edge_index: 用于消息传递的训练边（去除了验证/测试边，避免标签泄漏）
        - pos_edge_index: 正样本边 [2, P]
        - neg_edge_index: 负样本边 [2, N]
    """
    set_seed(seed)

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 去重：无向边只保留一份 (u < v)
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    num_edges = row.size(0)

    # 构建邻接集合，用于负采样查重
    adj_set = set()
    for i in range(num_edges):
        u, v = int(row[i]), int(col[i])
        adj_set.add((u, v))
        adj_set.add((v, u))

    # 随机打乱
    perm = torch.randperm(num_edges)
    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)
    train_size = num_edges - val_size - test_size

    train_perm = perm[:train_size]
    val_perm = perm[train_size:train_size + val_size]
    test_perm = perm[train_size + val_size:]

    train_pos = torch.stack([row[train_perm], col[train_perm]], dim=0)
    val_pos = torch.stack([row[val_perm], col[val_perm]], dim=0)
    test_pos = torch.stack([row[test_perm], col[test_perm]], dim=0)

    # 负采样：随机生成不存在的边
    def sample_negative(num_samples):
        neg_u = []
        neg_v = []
        attempts = 0
        max_attempts = num_samples * 100
        while len(neg_u) < num_samples and attempts < max_attempts:
            u = int(np.random.randint(0, num_nodes))
            v = int(np.random.randint(0, num_nodes))
            attempts += 1
            if u == v:
                continue
            if (u, v) in adj_set:
                continue
            neg_u.append(u)
            neg_v.append(v)
        return torch.tensor([neg_u, neg_v], dtype=torch.long)

    train_neg = sample_negative(train_size)
    val_neg = sample_negative(val_size)
    test_neg = sample_negative(test_size)

    # 消息传递时只能用训练边（恢复为无向）
    train_edge_index = torch.cat([train_pos, train_pos.flip(0)], dim=1)

    def _make_data(pos, neg):
        d = Data()
        d.x = data.x
        d.num_nodes = num_nodes
        d.edge_index = train_edge_index   # 消息传递边
        d.pos_edge_index = pos
        d.neg_edge_index = neg
        return d

    train_data = _make_data(train_pos, train_neg)
    val_data = _make_data(val_pos, val_neg)
    test_data = _make_data(test_pos, test_neg)

    return train_data, val_data, test_data


def describe_split(train_data, val_data, test_data):
    """打印划分统计信息"""
    def _count(d):
        pos = d.pos_edge_index.size(1)
        neg = d.neg_edge_index.size(1)
        return pos, neg

    tr_pos, tr_neg = _count(train_data)
    va_pos, va_neg = _count(val_data)
    te_pos, te_neg = _count(test_data)

    print(f"  Train: {tr_pos} pos + {tr_neg} neg = {tr_pos + tr_neg} edges")
    print(f"  Val:   {va_pos} pos + {va_neg} neg = {va_pos + va_neg} edges")
    print(f"  Test:  {te_pos} pos + {te_neg} neg = {te_pos + te_neg} edges")