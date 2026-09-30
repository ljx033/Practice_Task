import argparse
import os
import sys
import json
import torch
import numpy as np
from data_loader import KGDataset
from models import TransE, RotatE, ConvE
from train import train
from evaluate import evaluate


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='FB15k-237')
    parser.add_argument('--model', type=str, default='TransE',
                        choices=['TransE', 'RotatE', 'ConvE'])
    parser.add_argument('--dim', type=int, default=200,
                        help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_step', type=int, default=50)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--save_history', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    print("Loading dataset...")
    dataset = KGDataset(args.data_dir, args.dataset)
    print(f"Entities: {dataset.num_entities}, Relations: {dataset.num_relations}")
    print(f"Train: {len(dataset.train_triples)}, "
          f"Valid: {len(dataset.valid_triples)}, "
          f"Test: {len(dataset.test_triples)}")

    # 构建模型
    if args.model == 'TransE':
        model = TransE(dataset.num_entities, dataset.num_relations,
                       dim=args.dim, margin=args.margin)
    elif args.model == 'RotatE':
        model = RotatE(dataset.num_entities, dataset.num_relations,
                       dim=args.dim, margin=args.margin)
    elif args.model == 'ConvE':
        model = ConvE(dataset.num_entities, dataset.num_relations, dim=args.dim)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Parameters: {total_params:,}")

    # 训练
    best_state, history, total_time = train(model, dataset, args, device)

    # 在测试集上评估最佳模型
    model.load_state_dict(best_state)
    test_loader = dataset.get_eval_loader(
        dataset.test_triples, batch_size=args.eval_batch_size
    )
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, dataset.all_triples,
        dataset.num_entities, dataset, device
    )
    print("Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'model': args.model,
        'dataset': args.dataset,
        'dim': args.dim,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'total_time_seconds': total_time,
        'test_metrics': test_metrics,
        'history': history if args.save_history else None,
    }
    result_path = os.path.join(
        args.output_dir,
        f'{args.model}_{args.dataset}_dim{args.dim}_lr{args.lr}.json'
    )
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    main()