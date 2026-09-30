"""全图训练与小批次采样训练（手动划分版本）"""
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data


# ============================================================
#  通用评估
# ============================================================

@torch.no_grad()
def evaluate_split(model, x, edge_index, split_data, device):
    """在给定边上评估 AUC"""
    model.eval()
    h = model.encoder(x, edge_index)

    pos = split_data.pos_edge_index.to(device)
    neg = split_data.neg_edge_index.to(device)

    pos_logits = model.predictor(h[pos[0]], h[pos[1]])
    neg_logits = model.predictor(h[neg[0]], h[neg[1]])

    scores = torch.cat([pos_logits, neg_logits]).sigmoid().cpu().numpy()
    labels = np.concatenate([
        np.ones(pos.size(1)), np.zeros(neg.size(1))
    ])
    return roc_auc_score(labels, scores)


# ============================================================
#  全图训练
# ============================================================

def train_full_graph(model, data, config):
    """
    全图训练：每个 epoch 在完整图上进行消息传递和损失计算。
    """
    device = torch.device(config["device"])
    model = model.to(device)

    x = data.x.to(device)
    train_edge_index = data.edge_index.to(device)
    train_pos = data.pos_edge_index.to(device)
    train_neg = data.neg_edge_index.to(device)

    val_data = data.val_data
    test_data = data.test_data

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    best_val_auc = 0.0
    best_test_auc = 0.0
    patience_counter = 0
    epoch_times = []
    total_start = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()

        h = model.encoder(x, train_edge_index)

        pos_logits = model.predictor(h[train_pos[0]], h[train_pos[1]])
        neg_logits = model.predictor(h[train_neg[0]], h[train_neg[1]])

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits)
        )
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        epoch_times.append(time.time() - epoch_start)

        # 验证
        if (epoch + 1) % config["eval_every"] == 0 or epoch == config["epochs"] - 1:
            val_auc = evaluate_split(model, x, train_edge_index, val_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = evaluate_split(
                    model, x, train_edge_index, test_data, device
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"] // config["eval_every"]:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    total_time = time.time() - total_start
    return best_val_auc, best_test_auc, total_time, epoch_times


# ============================================================
#  小批次采样训练（NeighborLoader + 手动正负样本匹配）
# ============================================================

def train_minibatch(model, data, config):
    """
    小批次训练：使用 NeighborLoader 对源节点采样子图，
    批内使用训练边做消息传递，手动构造正负样本对。
    """
    from torch_geometric.loader import NeighborLoader

    device = torch.device(config["device"])
    model = model.to(device)

    x = data.x.to(device)
    train_edge_index = data.edge_index.to(device)
    train_pos = data.pos_edge_index
    train_neg = data.neg_edge_index
    val_data = data.val_data
    test_data = data.test_data

    # 用于 NeighborLoader 的临时 Data
    temp = Data()
    temp.x = data.x
    temp.edge_index = data.edge_index
    temp.num_nodes = data.num_nodes

    # 采样源节点集合 = 正负样本中出现的所有节点
    all_src = torch.cat([train_pos[0], train_neg[0]])
    input_nodes = all_src.unique()

    loader = NeighborLoader(
        temp,
        num_neighbors=config["num_neighbors"],
        batch_size=config["batch_size"],
        input_nodes=input_nodes,
        shuffle=True,
        num_workers=0,          # Windows 必须为 0
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # 预构建源节点 -> 目标节点的映射
    src_to_pos = {}
    for i in range(train_pos.size(1)):
        u = int(train_pos[0, i]); v = int(train_pos[1, i])
        src_to_pos.setdefault(u, []).append(v)

    src_to_neg = {}
    for i in range(train_neg.size(1)):
        u = int(train_neg[0, i]); v = int(train_neg[1, i])
        src_to_neg.setdefault(u, []).append(v)

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

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            h = model.encoder(batch.x, batch.edge_index)

            # 全局 ID -> batch 内 ID 映射
            n_id = batch.n_id.cpu().numpy()
            global_to_local = {int(g): i for i, g in enumerate(n_id)}

            pos_u_list, pos_v_list = [], []
            neg_u_list, neg_v_list = [], []

            for g_src in n_id:
                g_src = int(g_src)
                if g_src in src_to_pos:
                    for g_dst in src_to_pos[g_src]:
                        if g_dst in global_to_local:
                            pos_u_list.append(global_to_local[g_src])
                            pos_v_list.append(global_to_local[g_dst])
                if g_src in src_to_neg:
                    for g_dst in src_to_neg[g_src]:
                        if g_dst in global_to_local:
                            neg_u_list.append(global_to_local[g_src])
                            neg_v_list.append(global_to_local[g_dst])

            if len(pos_u_list) == 0 or len(neg_u_list) == 0:
                continue

            pos_u = torch.tensor(pos_u_list, device=device)
            pos_v = torch.tensor(pos_v_list, device=device)
            neg_u = torch.tensor(neg_u_list, device=device)
            neg_v = torch.tensor(neg_v_list, device=device)

            pos_logits = model.predictor(h[pos_u], h[pos_v])
            neg_logits = model.predictor(h[neg_u], h[neg_v])

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )
            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        epoch_times.append(time.time() - epoch_start)

        if (epoch + 1) % config["eval_every"] == 0 or epoch == config["epochs"] - 1:
            val_auc = evaluate_split(model, x, train_edge_index, val_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = evaluate_split(
                    model, x, train_edge_index, test_data, device
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"] // config["eval_every"]:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    total_time = time.time() - total_start
    return best_val_auc, best_test_auc, total_time, epoch_times