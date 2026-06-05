import numpy as np
import matplotlib.pyplot as plt

grid_size = 80

visit = np.zeros((grid_size, grid_size))
importance = np.zeros((grid_size, grid_size))
variance = np.zeros((grid_size, grid_size))
return_count = np.zeros((grid_size, grid_size))  # ⭐关键


def to_index(x, y):
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size - 1), np.clip(iy, 0, grid_size - 1)


def V(ix, iy):
    threshold = np.mean(importance) * 1.2
    return importance[ix, iy] > threshold


def step(x, y):
    dx, dy = np.random.normal(0, 0.22, 2)
    return x + dx, y + dy


num_agents = 6

agents = []
for _ in range(num_agents):
    agents.append({
        "x": np.random.uniform(-3, 3),
        "y": np.random.uniform(-3, 3),
        "history": []
    })


for t in range(7000):

    for a in agents:

        old_ix, old_iy = to_index(a["x"], a["y"])

        nx, ny = step(a["x"], a["y"])
        ix, iy = to_index(nx, ny)

        if V(ix, iy):
            a["x"], a["y"] = nx, ny

        a["history"].append((ix, iy))

        visit[ix, iy] += 1

        importance[ix, iy] += 0.01 * visit[ix, iy] / (1 + t * 0.002)

        move = np.sqrt((nx - a["x"])**2 + (ny - a["y"])**2)
        variance[ix, iy] += move

        # ⭐ 回流统计：如果最近来过又回来
        if len(a["history"]) > 20:
            if (ix, iy) in a["history"][-20:-1]:
                return_count[ix, iy] += 1

    importance *= 0.998
    variance *= 0.995
    return_count *= 0.999


# -------------------------
# ⭐ 吸引子强度
# -------------------------
stability = visit / (variance + 1e-5)
attractor_strength = stability * (1 + return_count)

# 归一化
attractor_strength /= np.max(attractor_strength)


# -------------------------
# 可视化
# -------------------------
plt.figure(figsize=(6, 6))
plt.imshow(attractor_strength.T, origin="lower", cmap="plasma")

plt.title("ACUE v1.4 — Attractor Field")

plt.savefig("acue_v1_4.png", dpi=200)
print("Saved: acue_v1_4.png")

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

# === v1.5 实体检测 ===
from scipy.ndimage import maximum_filter, label

neighborhood = maximum_filter(attractor_strength, size=3)
peaks = (attractor_strength == neighborhood)

threshold = np.mean(attractor_strength) * 1.5
peaks = peaks & (attractor_strength > threshold)

labeled, num_entities = label(peaks)

print("Detected entities:", num_entities)

