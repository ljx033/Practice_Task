import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, eval_loader, all_triples, num_entities,
             dataset, device='cuda', filtered=True):
    """
    计算 MRR, Hits@1, Hits@3, Hits@10
    对每个测试三元组，同时进行头实体和尾实体预测
    """
    model.eval()
    ranks = []

    # 将 all_triples 转为 set 以便快速查询
    triple_set = all_triples

    for batch in tqdm(eval_loader, desc='Evaluating', leave=False):
        h, r, t = batch[:, 0].to(device), batch[:, 1].to(device), batch[:, 2].to(device)
        B = h.size(0)

        # 尾实体预测：固定 h, r，对所有实体打分
        all_e = torch.arange(num_entities, device=device)
        h_exp = h.unsqueeze(1).expand(B, num_entities).reshape(-1)
        r_exp = r.unsqueeze(1).expand(B, num_entities).reshape(-1)
        t_exp = all_e.unsqueeze(0).expand(B, num_entities).reshape(-1)

        scores = model(h_exp, r_exp, t_exp).view(B, num_entities)

        # 过滤：将已知的正确三元组的分数设为 -inf（除当前正样本外）
        if filtered:
            for i in range(B):
                h_i, r_i, t_i = h[i].item(), r[i].item(), t[i].item()
                # 找出所有 (h_i, r_i, *) 的实体
                for e in range(num_entities):
                    if (h_i, r_i, e) in triple_set and e != t_i:
                        scores[i, e] = -1e9

        # 计算排名
        target_scores = scores[torch.arange(B), t]
        rank = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
        ranks.append(rank.cpu())

        # 头实体预测：固定 r, t，对所有实体打分
        h_exp = all_e.unsqueeze(0).expand(B, num_entities).reshape(-1)
        r_exp = r.unsqueeze(1).expand(B, num_entities).reshape(-1)
        t_exp = t.unsqueeze(1).expand(B, num_entities).reshape(-1)

        scores_h = model(h_exp, r_exp, t_exp).view(B, num_entities)

        if filtered:
            for i in range(B):
                h_i, r_i, t_i = h[i].item(), r[i].item(), t[i].item()
                for e in range(num_entities):
                    if (e, r_i, t_i) in triple_set and e != h_i:
                        scores_h[i, e] = -1e9

        target_scores_h = scores_h[torch.arange(B), h]
        rank_h = (scores_h > target_scores_h.unsqueeze(1)).sum(dim=1) + 1
        ranks.append(rank_h.cpu())

    ranks = torch.cat(ranks).float()
    mrr = (1.0 / ranks).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()

    return {
        'MRR': mrr,
        'Hits@1': hits1,
        'Hits@3': hits3,
        'Hits@10': hits10
    }