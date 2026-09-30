"""全图训练链路预测"""
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_full_graph(model, g, features, split_data, config):
    """
    全图训练：每个 epoch 在整个图上进行消息传递。

    返回:
        best_val_auc: 最佳验证 AUC
        test_auc: 测试 AUC
        total_time: 总训练时间（秒）
        epoch_times: 每个 epoch 的时间列表
    """
    device = config["device"]
    g = g.to(device)
    features = features.to(device)
    model = model.to(device)

    train_data = {k: v.to(device) for k, v in split_data["train"].items()}
    val_data = {k: v.to(device) for k, v in split_data["val"].items()}
    test_data = {k: v.to(device) for k, v in split_data["test"].items()}

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
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

        # 全图前向传播
        h = model(g, features)

        pos_score, neg_score = model.predict(
            h,
            train_data["pos_u"], train_data["pos_v"],
            train_data["neg_u"], train_data["neg_v"]
        )

        # 二分类损失
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

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # 验证
        if (epoch + 1) % config["eval_every"] == 0 or epoch == config["epochs"] - 1:
            val_auc = evaluate_full(model, g, features, val_data)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = evaluate_full(model, g, features, test_data)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"] // config["eval_every"]:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    total_time = time.time() - total_start
    return best_val_auc, best_test_auc, total_time, epoch_times


@torch.no_grad()
def evaluate_full(model, g, features, data):
    """全图评估：在完整图上计算节点嵌入，然后计算 AUC"""
    model.eval()
    h = model(g, features)
    pos_score, neg_score = model.predict(
        h, data["pos_u"], data["pos_v"], data["neg_u"], data["neg_v"]
    )
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = np.concatenate([
        np.ones(len(pos_score)),
        np.zeros(len(neg_score))
    ])
    return roc_auc_score(labels, scores)