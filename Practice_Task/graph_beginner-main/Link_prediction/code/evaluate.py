"""评估工具"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(pos_score, neg_score):
    """根据正负样本分数计算 AUC"""
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = np.concatenate([
        np.ones(len(pos_score)),
        np.zeros(len(neg_score))
    ])
    return roc_auc_score(labels, scores)


def format_time(seconds):
    """格式化时间输出"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}min"
    else:
        return f"{seconds / 3600:.2f}h"