"""GNN 图分类/回归模型，支持 GCN / GAT / GraphSAGE / GIN
   以及 AvgPooling / MaxPooling / MinPooling。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, GATConv, SAGEConv, GINConv,
                                global_mean_pool, global_max_pool)
from torch_geometric.nn import aggr


# ---------- 池化封装 ----------
def get_pool(pool_name):
    """返回 (pool_fn, readout_dim_scale)"""
    if pool_name == 'avg':
        return global_mean_pool
    elif pool_name == 'max':
        return global_max_pool
    elif pool_name == 'min':
        def global_min_pool(x, batch, size=None):
            if size is None:
                size = int(batch.max()) + 1
            return aggr.MinAggregation()(x, batch, dim_size=size)
        return global_min_pool
    else:
        raise ValueError(f"Unknown pool: {pool_name}")


# ---------- 通用分类模型 ----------
class GNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, num_classes,
                 model_name='GCN', pool_name='avg', dropout=0.5,
                 heads=4, task='classification'):
        super().__init__()
        self.model_name = model_name
        self.pool_name = pool_name
        self.task = task
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            dim_in = in_dim if i == 0 else hidden
            if model_name == 'GCN':
                conv = GCNConv(dim_in, hidden)
            elif model_name == 'GAT':
                conv = GATConv(dim_in, hidden // heads, heads=heads,
                               concat=True, dropout=dropout)
            elif model_name == 'SAGE':
                conv = SAGEConv(dim_in, hidden)
            elif model_name == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(dim_in, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden))

        out_dim = 1 if task == 'regression' else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pool_fn = get_pool(self.pool_name)
        x = pool_fn(x, batch)
        out = self.classifier(x)
        if self.task == 'regression':
            return out.squeeze(-1)
        return out