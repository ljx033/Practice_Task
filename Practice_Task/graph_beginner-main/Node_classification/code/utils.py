import time
import torch


class Timer:
    """上下文管理器，精确计时"""
    def __init__(self, name=''):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


@torch.no_grad()
def evaluate(model, data, mask):
    """评估节点分类准确率"""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = int((pred[mask] == data.y[mask]).sum())
    total = int(mask.sum())
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_sampling(model, data, mask, loader):
    """子图采样模式下的评估：使用 NeighborLoader 进行推理"""
    model.eval()
    total_correct = 0
    total_nodes = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index)
        pred = out[:batch.batch_size].argmax(dim=-1)
        true = batch.y[:batch.batch_size]
        total_correct += int((pred == true).sum())
        total_nodes += batch.batch_size
    return total_correct / total_nodes if total_nodes > 0 else 0.0