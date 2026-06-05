import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 系统空间
# -------------------------
grid_size = 80

visit = np.zeros((grid_size, grid_size))
importance = np.zeros((grid_size, grid_size))


def to_index(x, y):
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size - 1), np.clip(iy, 0, grid_size - 1)


# -------------------------
# 2. admissibility（更严格）
# -------------------------
def V(ix, iy):
    threshold = np.mean(importance) * 1.2
    return importance[ix, iy] > threshold


# -------------------------
# 3. 动力系统（减弱随机性）
# -------------------------
def step(x, y):
    dx, dy = np.random.normal(0, 0.3, 2)
    return x + dx, y + dy


# -------------------------
# 4. agents
# -------------------------
num_agents = 6

agents = []
for _ in range(num_agents):
    agents.append({
        "x": np.random.uniform(-2, 2),
        "y": np.random.uniform(-2, 2),
        "traj": []
    })


# -------------------------
# 5. 主循环（收敛机制）
# -------------------------
for t in range(5000):

    for a in agents:

        nx, ny = step(a["x"], a["y"])
        ix, iy = to_index(nx, ny)

        if V(ix, iy):
            a["x"], a["y"] = nx, ny

        a["traj"].append((a["x"], a["y"]))

        ix, iy = to_index(a["x"], a["y"])

        visit[ix, iy] += 1

        # ⭐重要性更新（逐渐减弱）
        importance[ix, iy] += 0.01 * visit[ix, iy] / (1 + t * 0.002)

    # ⭐强裁剪 + 衰减（推动收敛）
    importance *= 0.998


# -------------------------
# 6. 复杂度测量（关键）
# -------------------------
prob = importance / (np.sum(importance) + 1e-8)
entropy = -np.sum(prob * np.log(prob + 1e-10))

print(f"Final entropy: {entropy:.4f}")


# -------------------------
# 7. 可视化
# -------------------------
plt.figure(figsize=(6, 6))

plt.imshow(importance.T, origin="lower", cmap="hot")

colors = ["cyan", "blue", "green", "yellow", "white", "magenta"]

for i, a in enumerate(agents):
    traj = np.array(a["traj"])
    plt.plot(
        traj[:, 0] * 6 + 40,
        traj[:, 1] * 6 + 40,
        linewidth=0.6,
        color=colors[i % len(colors)]
    )

plt.title("ACUE v1.2 — Minimal Ontology")

plt.savefig("acue_v1_2.png", dpi=200)
print("Saved: acue_v1_2.png")
