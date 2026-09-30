import os
import torch
from torch.utils.data import Dataset, DataLoader


class KGDataset:
    """FB15k-237 数据加载器，支持实体/关系 ID 映射和负采样"""

    def __init__(self, data_dir, dataset_name='FB15k-237'):
        self.data_dir = os.path.join(data_dir, dataset_name)
        self.entity2id = {}
        self.relation2id = {}

        self.train_triples = self._load_triples('train.txt')
        self.valid_triples = self._load_triples('valid.txt')
        self.test_triples = self._load_triples('test.txt')

        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

        # 构建所有已知三元组集合（用于 filtered ranking）
        self.all_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def _load_triples(self, filename):
        path = os.path.join(self.data_dir, filename)
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                h, r, t = parts
                if h not in self.entity2id:
                    self.entity2id[h] = len(self.entity2id)
                if t not in self.entity2id:
                    self.entity2id[t] = len(self.entity2id)
                if r not in self.relation2id:
                    self.relation2id[r] = len(self.relation2id)
                triples.append(
                    (self.entity2id[h], self.relation2id[r], self.entity2id[t])
                )
        return triples

    def get_train_loader(self, batch_size=1024, num_workers=0, shuffle=True):
        dataset = TripleDataset(
            self.train_triples, self.num_entities, self.num_relations
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_eval_loader(self, triples, batch_size=512, num_workers=0):
        dataset = TripleDataset(
            triples, self.num_entities, self.num_relations, is_eval=True
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


class TripleDataset(Dataset):
    """三元组 Dataset，训练时动态生成负样本"""

    def __init__(self, triples, num_entities, num_relations, is_eval=False):
        self.triples = triples
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.is_eval = is_eval

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        if self.is_eval:
            return torch.tensor([h, r, t], dtype=torch.long)

        # 训练时：随机替换头实体或尾实体生成负样本
        if torch.rand(1).item() < 0.5:
            neg_h = torch.randint(0, self.num_entities, (1,)).item()
            return (
                torch.tensor([h, r, t], dtype=torch.long),
                torch.tensor([neg_h, r, t], dtype=torch.long),
            )
        else:
            neg_t = torch.randint(0, self.num_entities, (1,)).item()
            return (
                torch.tensor([h, r, t], dtype=torch.long),
                torch.tensor([h, r, neg_t], dtype=torch.long),
            )