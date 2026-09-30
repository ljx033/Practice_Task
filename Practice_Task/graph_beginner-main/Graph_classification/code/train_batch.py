"""分批次（Mini-batch）训练脚本。"""
import argparse, time, json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from data_utils import load_tu_dataset, load_zinc_dataset
from models import GNNClassifier
from train_full import evaluate


def train_batch(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Mini-batch] Device: {device}")

    if args.dataset.lower() == 'zinc':
        train_ds, val_ds, test_ds, info = load_zinc_dataset(subset=True)
    else:
        train_ds, val_ds, test_ds, info = load_tu_dataset(name=args.dataset)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    model = GNNClassifier(
        in_dim=info['num_features'], hidden=args.hidden,
        num_layers=args.layers, num_classes=info['num_classes'],
        model_name=args.model, pool_name=args.pool,
        dropout=args.dropout, task=info['task'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                                gamma=0.5)

    best_val, best_test, best_epoch = 0, 0, 0
    history = {'train_loss': [], 'val_loss': [], 'val_metric': []}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_samples = 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            if info['task'] == 'classification':
                loss = F.cross_entropy(out, data.y)
            else:
                loss = F.mse_loss(out, data.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item() * data.num_graphs
            epoch_samples += data.num_graphs
        scheduler.step()

        train_loss = epoch_loss / max(epoch_samples, 1)
        val_loss, val_metric, _, _ = evaluate(model, val_loader,
                                              info['task'], device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metric'].append(val_metric)

        if val_metric > best_val:
            best_val = val_metric
            best_epoch = epoch
            _, best_test, _, _ = evaluate(model, test_loader,
                                          info['task'], device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | TrainLoss {train_loss:.4f} | "
                  f"ValLoss {val_loss:.4f} | ValMetric {val_metric:.4f} | "
                  f"BestTest {best_test:.4f}")

    total_time = time.time() - t_start
    print(f"[Mini-batch] Best epoch {best_epoch}, "
          f"Test metric {best_test:.4f}, Total time {total_time:.1f}s")

    return {
        'best_val': best_val, 'best_test': best_test,
        'best_epoch': best_epoch, 'total_time': total_time,
        'history': history,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='MUTAG')
    p.add_argument('--model', type=str, default='GIN',
                   choices=['GCN', 'GAT', 'SAGE', 'GIN'])
    p.add_argument('--pool', type=str, default='avg',
                   choices=['avg', 'max', 'min'])
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    args = p.parse_args()
    result = train_batch(args)
    print(json.dumps({k: v for k, v in result.items()
                      if k != 'history'}, indent=2))