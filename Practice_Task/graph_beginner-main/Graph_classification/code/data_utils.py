"""数据集加载与划分工具。"""
import os
import torch
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree


def load_tu_dataset(name='MUTAG', root='../data/TUDataset'):
    """加载 TUDataset。返回 dataset, train_loader, val_loader, test_loader, info"""
    dataset = TUDataset(root=root, name=name)
    dataset = dataset.shuffle()
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]

    info = {
        'num_features': dataset.num_node_features,
        'num_classes': dataset.num_classes,
        'task': 'classification',
    }
    return train_dataset, val_dataset, test_dataset, info


def load_zinc_dataset(root='../data/ZINC', subset=True):
    """加载 ZINC。返回 train/val/test 数据集及信息。"""
    from torch_geometric.transforms import Compose
    train_dataset = ZINC(root=root, subset=subset, split='train')
    val_dataset = ZINC(root=root, subset=subset, split='val')
    test_dataset = ZINC(root=root, subset=subset, split='test')
    info = {
        'num_features': train_dataset.num_node_features,
        'num_classes': 1,          # 回归任务
        'task': 'regression',
    }
    return train_dataset, val_dataset, test_dataset, info


def make_loaders(train_dataset, val_dataset, test_dataset,
                 batch_size=32, num_workers=0):
    """构建 PyG DataLoader。"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def make_full_batch_loader(dataset):
    """全图训练：把整个数据集作为一个 batch。"""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return loader