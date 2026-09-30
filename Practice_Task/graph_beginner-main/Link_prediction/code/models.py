"""四种 GNN 模型的链路预测实现（PyG）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class GNNEncoder(nn.Module):
    """
    通用 GNN 编码器：支持 GCN / GAT / GraphSAGE / GIN。
    输出节点嵌入，链路预测通过点积解码。
    """

    def __init__(self, in_dim, hidden_dim, num_layers=2,
                 model_name="gcn", num_heads=4):
        super().__init__()
        self.model_name = model_name
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        if model_name == "gcn":
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.out_dim = hidden_dim

        elif model_name == "graphsage":
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.out_dim = hidden_dim

        elif model_name == "gat":
            self.convs.append(
                GATConv(in_dim, hidden_dim, heads=num_heads,
                        concat=True, dropout=0.1)
            )
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_dim * num_heads, hidden_dim,
                            heads=num_heads, concat=True, dropout=0.1)
                )
            self.out_dim = hidden_dim * num_heads

        elif model_name == "gin":
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ))
            for _ in range(num_layers - 1):
                self.convs.append(GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                ))
            self.out_dim = hidden_dim

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x, edge_index):
        """返回节点嵌入 [num_nodes, out_dim]"""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.1, training=self.training)
        return h


class LinkPredictor(nn.Module):
    """点积链路预测解码器"""

    def __init__(self, hidden_dim):
        super().__init__()

    def forward(self, h_src, h_dst):
        """返回边分数（logits）"""
        return (h_src * h_dst).sum(dim=-1)


class LinkPredModel(nn.Module):
    """完整的链路预测模型：Encoder + Decoder"""

    def __init__(self, in_dim, hidden_dim, num_layers=2,
                 model_name="gcn", num_heads=4):
        super().__init__()
        self.encoder = GNNEncoder(
            in_dim, hidden_dim, num_layers, model_name, num_heads
        )
        self.predictor = LinkPredictor(self.encoder.out_dim)

    def forward(self, x, edge_index, edge_label_index):
        h = self.encoder(x, edge_index)
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        return self.predictor(h_src, h_dst)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)


def build_model(model_name, in_dim, hidden_dim, num_layers=2, num_heads=4):
    return LinkPredModel(in_dim, hidden_dim, num_layers, model_name, num_heads)