"""批量运行所有对比实验，输出 JSON 结果。"""
import subprocess, json, os, itertools, sys

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'MUTAG'
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

MODELS = ['GCN', 'GAT', 'SAGE', 'GIN']
POOLS = ['avg', 'max', 'min']
BATCH_MODES = ['full', 'batch']
BATCH_SIZES = [16, 32, 64, 128]
LRS = [0.0001, 0.001, 0.01]
LAYERS_LIST = [2, 3, 4]

os.makedirs('../results', exist_ok=True)

results = []

# ---- 实验 1：模型 × 池化（固定其他超参）----
for model, pool in itertools.product(MODELS, POOLS):
    for mode in BATCH_MODES:
        script = 'train_full.py' if mode == 'full' else 'train_batch.py'
        cmd = [sys.executable, script,
               '--dataset', DATASET, '--model', model, '--pool', pool,
               '--epochs', str(EPOCHS), '--hidden', '64', '--layers', '3',
               '--lr', '0.001']
        if mode == 'batch':
            cmd += ['--batch_size', '32']
        print('>>>', ' '.join(cmd))
        subprocess.run(cmd)
        results.append({'exp': 'model_pool', 'model': model, 'pool': pool,
                        'mode': mode})

# ---- 实验 2：学习率影响（固定 GIN + avg）----
for lr in LRS:
    for mode in BATCH_MODES:
        script = 'train_full.py' if mode == 'full' else 'train_batch.py'
        cmd = [sys.executable, script,
               '--dataset', DATASET, '--model', 'GIN', '--pool', 'avg',
               '--epochs', str(EPOCHS), '--lr', str(lr)]
        if mode == 'batch':
            cmd += ['--batch_size', '32']
        subprocess.run(cmd)

# ---- 实验 3：层数影响 ----
for n_layer in LAYERS_LIST:
    for mode in BATCH_MODES:
        script = 'train_full.py' if mode == 'full' else 'train_batch.py'
        cmd = [sys.executable, script,
               '--dataset', DATASET, '--model', 'GIN', '--pool', 'avg',
               '--epochs', str(EPOCHS), '--layers', str(n_layer)]
        if mode == 'batch':
            cmd += ['--batch_size', '32']
        subprocess.run(cmd)

# ---- 实验 4：batch size 影响（仅 mini-batch）----
for bs in BATCH_SIZES:
    cmd = [sys.executable, 'train_batch.py',
           '--dataset', DATASET, '--model', 'GIN', '--pool', 'avg',
           '--epochs', str(EPOCHS), '--batch_size', str(bs)]
    subprocess.run(cmd)

print("\nAll experiments finished.")