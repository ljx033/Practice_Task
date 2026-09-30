"""全局配置参数"""

# 数据集配置
DATASETS = {
    "cora": {
        "name": "Cora",
        "num_hops": 2,
        "hidden_dim": 128,
    },
    "citeseer": {
        "name": "Citeseer",
        "num_hops": 2,
        "hidden_dim": 128,
    },
    "flickr": {
        "name": "Flickr",
        "num_hops": 2,
        "hidden_dim": 128,
        "root": "E:/Aljx/graph_beginner-main/Link_prediction/data/Flickr"
    },
}

# 模型名称列表
MODELS = ["gcn", "gat", "graphsage", "gin"]

# 训练配置
TRAIN_CONFIG = {
    "lr": 0.001,
    "weight_decay": 5e-4,
    "epochs": 100,
    "patience": 20,
    "eval_every": 5,
    "seed": 42,
    "device": "cuda:0",

    # 小批次训练参数
    "batch_size": 512,
    "num_neighbors": [5, 5],       # 每层采样邻居数
    "neg_sampling_ratio": 1.0,     # 负采样比例
}

# 参数对比实验配置
EXPERIMENT_CONFIGS = {
    "lr": [0.001, 0.005, 0.01],
    "num_layers": [2, 3],
}