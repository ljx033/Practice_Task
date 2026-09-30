"""
数据集加载模块（修正版）
- 优先使用本地数据，绝不静默下载
- 正确处理 Planetoid 与 Flickr 的 root 语义差异
"""
import os
import torch
from torch_geometric.datasets import Planetoid, Flickr


# ============================================================
# 1. 数据根目录自动探测
# ============================================================
def _find_data_root():
    here = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        # ★ 若自动探测不对，请把你实际路径写在这一行
        # r'E:\Aljx\graph_beginner-main\newnewNode_classification\data',
        os.path.abspath(os.path.join(here, '..', 'data')),
        os.path.abspath(os.path.join(here, '..', 'dataset')),
        os.path.abspath(os.path.join(here, '..', 'datasets')),
        os.path.abspath(os.path.join(here, 'data')),
        os.path.abspath('data'),
    ]

    for path in candidates:
        if os.path.isdir(path):
            if any(os.path.isdir(os.path.join(path, n))
                   for n in ('Cora', 'Citeseer', 'Flickr')):
                return path
    return candidates[0]


DATA_ROOT = _find_data_root()
print(f'[dataset] DATA_ROOT = {DATA_ROOT}')

if os.path.isdir(DATA_ROOT):
    existing = [d for d in os.listdir(DATA_ROOT)
                if os.path.isdir(os.path.join(DATA_ROOT, d))]
    print(f'[dataset] 检测到的数据集目录: {existing}')
else:
    print(f'[dataset] 警告: 目录不存在 {DATA_ROOT}')


# ============================================================
# 2. 判断某个数据集本地文件是否存在
# ============================================================
def _local_ready(name):
    ds_dir  = os.path.join(DATA_ROOT, name)
    raw_ok  = os.path.exists(os.path.join(ds_dir, 'raw'))
    proc_ok = os.path.exists(os.path.join(ds_dir, 'processed', 'data.pt'))
    return ds_dir, raw_ok, proc_ok


# ============================================================
# 3. 加载函数
# ============================================================
def _load_planetoid(name):
    """
    正确用法: Planetoid(root=DATA_ROOT, name=name)
    它内部会拼成 DATA_ROOT/name/raw 和 DATA_ROOT/name/processed
    """
    ds_dir, raw_ok, proc_ok = _local_ready(name)

    if not raw_ok and not proc_ok:
        raise FileNotFoundError(
            f'\n[dataset] 找不到 {name} 的本地数据！\n'
            f'  期望位置: {ds_dir}\n'
            f'  其中应有 raw/ 或 processed/data.pt\n'
            f'  如需添加其他候选路径，请修改 _find_data_root()。\n'
        )

    return Planetoid(root=DATA_ROOT, name=name)   # ★ root 传父目录


def _load_flickr():
    """
    Flickr 的 root 直接指向 Flickr 目录本身
    """
    ds_dir, raw_ok, proc_ok = _local_ready('Flickr')

    if not raw_ok and not proc_ok:
        raise FileNotFoundError(
            f'\n[dataset] 找不到 Flickr 的本地数据！\n'
            f'  期望位置: {ds_dir}\n'
            f'  其中应有 raw/ 或 processed/data.pt\n'
        )

    return Flickr(root=ds_dir)                    # ★ root 就是 Flickr 目录


DATASET_REGISTRY = {
    'Cora':     lambda: _load_planetoid('Cora'),
    'Citeseer': lambda: _load_planetoid('Citeseer'),
    'Flickr':   lambda: _load_flickr(),
}


# ============================================================
# 4. 统一入口
# ============================================================
def load_dataset(name, root=None):
    if name not in DATASET_REGISTRY:
        raise KeyError(f'未知数据集: {name}, 可选: {list(DATASET_REGISTRY)}')

    dataset = DATASET_REGISTRY[name]()
    data = dataset[0]

    if name == 'Flickr':
        if data.train_mask.dim() > 1:
            data.train_mask = data.train_mask.view(-1)
        if data.val_mask.dim() > 1:
            data.val_mask = data.val_mask.view(-1)
        if data.test_mask.dim() > 1:
            data.test_mask = data.test_mask.view(-1)

    return data, dataset.num_features, dataset.num_classes, data.num_nodes


# ============================================================
# 5. 自测
# ============================================================
if __name__ == '__main__':
    import time
    for name in ['Cora', 'Citeseer', 'Flickr']:
        print(f'\n===== {name} =====')
        t0 = time.time()
        try:
            data, nf, nc, nn = load_dataset(name)
            print(f'  加载耗时: {time.time() - t0:.2f}s')
            print(f'  节点数={nn}  边数={data.edge_index.size(1)}  '
                  f'特征={nf}  类别={nc}')
            print(f'  train/val/test = '
                  f'{int(data.train_mask.sum())}/'
                  f'{int(data.val_mask.sum())}/'
                  f'{int(data.test_mask.sum())}')
        except Exception as e:
            print(f'  [ERROR] {e}')