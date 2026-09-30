"""根据手动记录的结果绘制对比图（可选）。"""
import matplotlib.pyplot as plt
import numpy as np

# 示例数据（请替换为实际运行结果）
models = ['GCN', 'GAT', 'SAGE', 'GIN']
avg_pool = [0.72, 0.74, 0.73, 0.78]
max_pool = [0.70, 0.72, 0.71, 0.76]
min_pool = [0.65, 0.67, 0.66, 0.70]

x = np.arange(len(models))
w = 0.25
plt.figure(figsize=(9, 5))
plt.bar(x - w, avg_pool, w, label='AvgPool')
plt.bar(x, max_pool, w, label='MaxPool')
plt.bar(x + w, min_pool, w, label='MinPool')
plt.xticks(x, models)
plt.ylabel('Test Accuracy')
plt.title('不同池化方法 × 不同模型在 MUTAG 上的表现')
plt.legend()
plt.tight_layout()
plt.savefig('../results/pooling_comparison.png', dpi=150)
plt.show()