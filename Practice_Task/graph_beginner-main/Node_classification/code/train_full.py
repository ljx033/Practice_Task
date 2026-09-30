import argparse
import copy
import sys
import torch
import torch.nn.functional as F

from models import MODEL_REGISTRY
from dataset import load_dataset
from utils import Timer, evaluate


def train_full_graph(dataset_name='Cora', model_name='GCN',
                     hidden=64, layers=2, lr=0.01, weight_decay=5e-4,
                     dropout=0.5, epochs=200, device='cuda', seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data, in_feat, n_class, n_nodes = load_dataset(dataset_name)
    data = data.to(device)

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
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            val_acc = evaluate(model, data, data.val_mask)
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = evaluate(model, data, data.test_mask)
                best_epoch = epoch

    total_time = t.elapsed
    avg_epoch = total_time / epochs
    return {
        'model': model_name,
        'dataset': dataset_name,
        'hidden': hidden,
        'layers': layers,
        'lr': lr,
        'best_val_acc': round(best_val_acc * 100, 2),
        'best_test_acc': round(best_test_acc * 100, 2),
        'best_epoch': best_epoch,
        'total_time_s': round(total_time, 2),
        'avg_epoch_ms': round(avg_epoch * 1000, 2),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--model', default='GCN')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = train_full_graph(
        dataset_name=args.dataset, model_name=args.model,
        hidden=args.hidden, layers=args.layers,
        lr=args.lr, weight_decay=args.weight_decay,
        dropout=args.dropout, epochs=args.epochs, device=device,
    )
    print(result)