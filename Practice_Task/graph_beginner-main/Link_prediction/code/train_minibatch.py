"""小批次采样训练链路预测"""
import time
import torch
import torch.nn.functional as F
import numpy as np
import dgl
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class NegativeSampler:
    """基于度的负采样器"""

    def __init__(self, g, k):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        src = src.repeat_interleave(self.k)
        dst = self.weights.multinomial(len(src), replacement=True)
        return src, dst


def train_minibatch(model, g, features, split_data, config):
    """
    小批次采样训练：使用 NeighborSampler 采样子图。

    返回:
        best_val_auc, test_auc, total_time, epoch_times
    """
    device = config["device"]
    g = g.to(device)
    features = features.to(device)
    model = model.to(device)

    fanouts = config["fanouts"]
    batch_size = config["batch_size"]
    neg_ratio = config["neg_ratio"]

    # 邻居采样器
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    neg_sampler = NegativeSampler(g, neg_ratio)

    # EdgeDataLoader：为链路预测生成负样本
    train_eids = split_data["train"]["edge_ids"]
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eids, sampler,
        negative_sampler=neg_sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # 验证和测试数据转为 tensor
    val_data = {k: v.to(device) for k, v in split_data["val"].items()}
    test_data = {k: v.to(device) for k, v in split_data["test"].items()}

    best_val_auc = 0.0
    best_test_auc = 0.0
    patience_counter = 0
    epoch_times = []
    total_start = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for input_nodes, positive_graph, negative_graph, blocks in dataloader:
            # input_nodes: 输入节点 ID
            # positive_graph: 正样本边子图
            # negative_graph: 负样本边子图
            # blocks: 多层采样块

            # 将 blocks 移到 GPU
            blocks = [b.to(device) for b in blocks]
            positive_graph = positive_graph.to(device)
            negative_graph = negative_graph.to(device)

            # 获取输入特征
            input_features = features[input_nodes]

            # 前向传播：通过 blocks 进行消息传递
            h = model(blocks, input_features)

            # 计算正负样本分数
            pos_u, pos_v = positive_graph.edges()
            neg_u, neg_v = negative_graph.edges()

            pos_score = model.predictor(h[pos_u], h[pos_v])
            neg_score = model.predictor(h[neg_u], h[neg_v])

            # 损失
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score)
            )
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # 验证
        if (epoch + 1) % config["eval_every"] == 0 or epoch == config["epochs"] - 1:
            val_auc = evaluate_minibatch(model, g, features, val_data, fanouts, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = evaluate_minibatch(model, g, features, test_data, fanouts, device)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"] // config["eval_every"]:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    total_time = time.time() - total_start
    return best_val_auc, best_test_auc, total_time, epoch_times


@torch.no_grad()
def evaluate_minibatch(model, g, features, data, fanouts, device):
    """
    小批次评估：对验证/测试边的源和目标节点分别采样邻居子图，
    获取节点嵌入后计算 AUC。
    """
    model.eval()

    # 收集所有需要评估的节点
    pos_u, pos_v = data["pos_u"], data["pos_v"]
    neg_u, neg_v = data["neg_u"], data["neg_v"]
    all_nodes = torch.cat([pos_u, pos_v, neg_u, neg_v]).unique()

    # 使用全图前向传播获取嵌入（评估时不需要采样，保证嵌入一致性）
    # 对于大规模图，可以分批处理
    h = model(g, features)

    pos_score = model.predictor(h[pos_u], h[pos_v])
    neg_score = model.predictor(h[neg_u], h[neg_v])

    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = np.concatenate([
        np.ones(len(pos_score)),
        np.zeros(len(neg_score))
    ])
    return roc_auc_score(labels, scores)