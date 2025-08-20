import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义 token 和 metric
tokens = ["-Sub"] + [f"-mod{i}" for i in range(1, 13)]
metrics = ["1-SIIS", "1-STS", "1-SS", "1-PS"]

# 原始 similarity 数据
data = np.array([
    [0.71, 0.83, 0.46, 0.32],
    [0.96, 0.99, 0.76, 0.91],
    [0.81, 0.98, 0.71, 0.60],
    [0.96, 0.98, 0.79, 0.89],
    [0.95, 0.99, 0.86, 0.89],
    [0.96, 0.99, 0.86, 0.92],
    [0.89, 0.99, 0.76, 0.77],
    [0.93, 0.98, 0.80, 0.85],
    [0.95, 0.98, 0.71, 0.87],
    [0.95, 0.99, 0.75, 0.88],
    [0.94, 0.99, 0.73, 0.86],
    [0.93, 0.88, 0.65, 0.75],
    [0.91, 0.95, 0.83, 0.78]
])

# 计算贡献度：1 - similarity
impact = 1 - data

# 创建 DataFrame
df = pd.DataFrame(impact, index=tokens, columns=metrics)

# 转置 DataFrame 以便 metrics 为 Y 轴，tokens 为 X 轴
df_transposed = df.T

# 绘图
fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow(df_transposed.values, cmap="YlOrRd")

# 设置坐标轴标签
ax.set_xticks(np.arange(len(df_transposed.columns)))
ax.set_yticks(np.arange(len(df_transposed.index)))
ax.set_xticklabels(df_transposed.columns)
ax.set_yticklabels(df_transposed.index)

# 旋转 X 轴标签并设置字体大小
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

# 标注每个格子的数值
for i in range(len(df_transposed.index)):
    for j in range(len(df_transposed.columns)):
        value = df_transposed.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=16)

# 添加颜色条
cbar = plt.colorbar(im)
# cbar.set_label("1 - Similarity (Token Impact)", fontsize=12)
cbar.ax.tick_params(labelsize=16)
# 标题
# plt.title("Metric-wise Token Impact (1 - Similarity)")
plt.tight_layout()
plt.savefig("heat.png", dpi=300)
