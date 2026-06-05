import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label

grid_size = 80

# 假设你已经有 attractor_strength（可以直接从 v1.4 复制生成部分）

# -------------------------
# ⭐ 局部极大检测
# -------------------------
neighborhood = maximum_filter(attractor_strength, size=3)
peaks = (attractor_strength == neighborhood)

# 阈值过滤（避免噪声）
threshold = np.mean(attractor_strength) * 1.5
peaks = peaks & (attractor_strength > threshold)


# -------------------------
# ⭐ 连通区域 = 实体
# -------------------------
labeled, num_entities = label(peaks)

print(f"Detected entities: {num_entities}")


# -------------------------
# 可视化
# -------------------------
plt.figure(figsize=(6, 6))
plt.imshow(attractor_strength.T, origin="lower", cmap="plasma")

# 标出实体中心
for i in range(1, num_entities + 1):
    ys, xs = np.where(labeled == i)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    plt.scatter(cx, cy, c='white', s=40)

plt.title("ACUE v1.5 — Entities")
plt.savefig("acue_v1_5.png", dpi=200)

print("Saved: acue_v1_5.png")
