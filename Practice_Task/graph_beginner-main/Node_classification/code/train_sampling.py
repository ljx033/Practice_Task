import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from models import MODEL_REGISTRY
from dataset import load_dataset
from utils import Timer, evaluate_sampling


def train_sampling(dataset_name='Cora', model_name='GraphSAGE',
                   hidden=64, layers=2, lr=0.01, weight_decay=5e-4,
                   dropout=0.5, epochs=200, batch_size=64,
                   num_neighbors=None, device='cuda', seed=42):
    """
    子图采样训练。
    num_neighbors: 每层采样邻居数列表，如 [10, 10]。
    默认根据数据集规模自动设置。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data, in_feat, n_class, n_nodes = load_dataset(dataset_name)

    # 自动设置采样邻居数
    if num_neighbors is None:
        if dataset_name == 'Flickr':
            num_neighbors = [15, 10]
            batch_size = 256
        else:
            num_neighbors = [10, 10]
            batch_size = 64

    # 构建 NeighborLoader
    train_loader = NeighborLoader(
        data, input_nodes=data.train_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size, shuffle=True,
        num_workers=0,
    )
    val_loader = NeighborLoader(
        data, input_nodes=data.val_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    test_loader = NeighborLoader(
        data, input_nodes=data.test_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(in_feat, hidden, n_class, num_layers=layers,
                      dropout=dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    with Timer() as t:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            total_examples = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                # 只对种子节点（前 batch_size 个）计算损失
                loss = F.cross_entropy(
                    out[:batch.batch_size],
                    batch.y[:batch.batch_size]
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.batch_size
                total_examples += batch.batch_size

            # 验证（使用子图采样推理）
            val_correct = 0
            val_total = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index)
                    pred = out[:batch.batch_size].argmax(dim=-1)
                    val_correct += int(
                        (pred == batch.y[:batch.batch_size]).sum())
                    val_total += batch.batch_size
            val_acc = val_correct / val_total if val_total > 0 else 0
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 测试
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index)
                        pred = out[:batch.batch_size].argmax(dim=-1)
                        test_correct += int(
                            (pred == batch.y[:batch.batch_size]).sum())
                        test_total += batch.batch_size
                best_test_acc = test_correct / \
                    test_total if test_total > 0 else 0
                best_epoch = epoch

    total_time = t.elapsed
    avg_epoch = total_time / epochs

    return {
        'model': model_name,
        'dataset': dataset_name,
        'hidden': hidden,
        'layers': layers,
        'lr': lr,
        'batch_size': batch_size,
        'num_neighbors': num_neighbors,
        'best_val_acc': round(best_val_acc * 100, 2),
        'best_test_acc': round(best_test_acc * 100, 2),
        'best_epoch': best_epoch,
        'total_time_s': round(total_time, 2),
        'avg_epoch_ms': round(avg_epoch * 1000, 2),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--model', default='GraphSAGE')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_neighbors', nargs='+', type=int, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = train_sampling(
        dataset_name=args.dataset, model_name=args.model,
        hidden=args.hidden, layers=args.layers,
        lr=args.lr, weight_decay=args.weight_decay,
        dropout=args.dropout, epochs=args.epochs,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbors, device=device,
    )
    print(result)