import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from evaluate import evaluate


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for pos, neg in tqdm(train_loader, desc='Training', leave=False):
        pos = pos.to(device)
        neg = neg.to(device)
        h_pos, r_pos, t_pos = pos[:, 0], pos[:, 1], pos[:, 2]
        h_neg, r_neg, t_neg = neg[:, 0], neg[:, 1], neg[:, 2]

        pos_score = model(h_pos, r_pos, t_pos)
        neg_score = model(h_neg, r_neg, t_neg)

        loss = model.loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def train(model, dataset, args, device):
    train_loader = dataset.get_train_loader(
        batch_size=args.batch_size, num_workers=0
    )
    valid_loader = dataset.get_eval_loader(
        dataset.valid_triples, batch_size=args.eval_batch_size
    )

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )

    best_mrr = 0.0
    best_state = None
    history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # 验证
        metrics = evaluate(
            model, valid_loader, dataset.all_triples,
            dataset.num_entities, dataset, device
        )
        epoch_time = time.time() - epoch_start

        history.append({
            'epoch': epoch,
            'loss': loss,
            **metrics,
            'time': epoch_time
        })

        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"MRR: {metrics['MRR']:.4f} | "
              f"Hits@1: {metrics['Hits@1']:.4f} | "
              f"Hits@10: {metrics['Hits@10']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        if metrics['MRR'] > best_mrr:
            best_mrr = metrics['MRR']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s")
    print(f"Best validation MRR: {best_mrr:.4f}")

    return best_state, history, total_time