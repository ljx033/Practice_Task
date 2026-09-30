import json
import sys
import torch
import pandas as pd
from train_full import train_full_graph
from train_sampling import train_sampling

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASETS = ['Cora', 'Citeseer', 'Flickr']
MODELS = ['GCN', 'GAT', 'GraphSAGE', 'GIN']


def experiment_model_comparison():
    """实验1：四个模型在三个数据集上的全图 vs 采样对比"""
    records = []
    for ds in DATASETS:
        for mn in MODELS:
            print(f'  [全图] {ds} / {mn}')
            r = train_full_graph(ds, mn, epochs=5, device=DEVICE)
            r['mode'] = 'Full-Graph'
            records.append(r)

            print(f'  [采样] {ds} / {mn}')
            r = train_sampling(ds, mn, epochs=5, device=DEVICE)
            r['mode'] = 'Sampling'
            records.append(r)

    df = pd.DataFrame(records)
    df.to_csv('../results_model_comparison.csv', index=False)
    print(df.to_string())
    return df


def experiment_lr():
    """实验2：学习率影响（Cora + GCN）"""
    records = []
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        print(f'  [LR] lr={lr}')
        r = train_full_graph('Cora', 'GCN', lr=lr, epochs=5, device=DEVICE)
        records.append(r)
    df = pd.DataFrame(records)
    df.to_csv('../results_lr.csv', index=False)
    print(df.to_string())
    return df


def experiment_layers():
    """实验3：网络层数影响（Cora + GCN）"""
    records = []
    for nl in [1, 2, 3, 4, 5]:
        print(f'  [Layers] layers={nl}')
        r = train_full_graph('Cora', 'GCN', layers=nl, epochs=5, device=DEVICE)
        records.append(r)
    df = pd.DataFrame(records)
    df.to_csv('../results_layers.csv', index=False)
    print(df.to_string())
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['all', 'models', 'lr', 'layers'],
                        default='all')
    args = parser.parse_args()

    if args.exp in ('all', 'models'):
        print('\n' + '=' * 60)
        print('实验1：模型对比（全图 vs 采样）')
        print('=' * 60)
        experiment_model_comparison()

    if args.exp in ('all', 'lr'):
        print('\n' + '=' * 60)
        print('实验2：学习率影响')
        print('=' * 60)
        experiment_lr()

    if args.exp in ('all', 'layers'):
        print('\n' + '=' * 60)
        print('实验3：网络层数影响')
        print('=' * 60)
        experiment_layers()