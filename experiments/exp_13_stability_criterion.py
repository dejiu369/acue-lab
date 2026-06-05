import numpy as np
import matplotlib.pyplot as plt

grid_size = 80

visit = np.zeros((grid_size, grid_size))
importance = np.zeros((grid_size, grid_size))
variance = np.zeros((grid_size, grid_size))


def to_index(x, y):
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size - 1), np.clip(iy, 0, grid_size - 1)


def V(ix, iy):
    threshold = np.mean(importance) * 1.2
    return importance[ix, iy] > threshold


def step(x, y):
    dx, dy = np.random.normal(0, 0.25, 2)
    return x + dx, y + dy


num_agents = 6

agents = []
for _ in range(num_agents):
    agents.append({
        "x": np.random.uniform(-2, 2),
        "y": np.random.uniform(-2, 2),
        "traj": []
    })


for t in range(6000):

    for a in agents:

        old_x, old_y = a["x"], a["y"]

        nx, ny = step(old_x, old_y)
        ix, iy = to_index(nx, ny)

        if V(ix, iy):
            a["x"], a["y"] = nx, ny

        a["traj"].append((a["x"], a["y"]))

        ix, iy = to_index(a["x"], a["y"])

        visit[ix, iy] += 1

        # importance
        importance[ix, iy] += 0.01 * visit[ix, iy] / (1 + t * 0.002)

        # ⭐ variance（局部波动）
        move = np.sqrt((a["x"] - old_x)**2 + (a["y"] - old_y)**2)
        variance[ix, iy] += move


    importance *= 0.998
    variance *= 0.995


# -------------------------
# ⭐ 稳定性定义
# -------------------------
stability = visit / (variance + 1e-5)

# 归一化
stability = stability / np.max(stability)


# -------------------------
# entropy（验证用）
# -------------------------
prob = stability / (np.sum(stability) + 1e-8)
entropy = -np.sum(prob * np.log(prob + 1e-10))

print(f"Stability entropy: {entropy:.4f}")


# -------------------------
# 可视化
# -------------------------
plt.figure(figsize=(6, 6))
plt.imshow(stability.T, origin="lower", cmap="viridis")

plt.title("ACUE v1.3 — Stability Field")

plt.savefig("acue_v1_3.png", dpi=200)
print("Saved: acue_v1_3.png")
