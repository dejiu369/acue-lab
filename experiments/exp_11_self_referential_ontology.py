import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 世界结构
# -------------------------
grid_size = 80

visit = np.zeros((grid_size, grid_size))
importance = np.zeros((grid_size, grid_size))


def to_index(x, y):
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size - 1), np.clip(iy, 0, grid_size - 1)


# -------------------------
# 2. ⭐自指 admissibility（关键）
# -------------------------
def V(ix, iy):
    # 系统判断：这个区域“是否值得继续存在”
    return importance[ix, iy] > np.mean(importance) * 0.8


# -------------------------
# 3. 动力系统
# -------------------------
def step(x, y):
    dx, dy = np.random.normal(0, 0.5, 2)
    return x + dx, y + dy


# -------------------------
# 4. 多 agent
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
# 5. 主循环（自我裁剪）
# -------------------------
for t in range(3500):

    for a in agents:

        nx, ny = step(a["x"], a["y"])
        ix, iy = to_index(nx, ny)

        # ⭐系统开始“自我审查”
        if V(ix, iy):
            a["x"], a["y"] = nx, ny

        a["traj"].append((a["x"], a["y"]))

        ix, iy = to_index(a["x"], a["y"])

        visit[ix, iy] += 1

        # ⭐重要性更新（结构价值）
        importance[ix, iy] += 0.02 * visit[ix, iy] / (1 + t * 0.001)

    # ⭐系统自裁剪（关键）
    importance *= 0.9995


# -------------------------
# 6. 可视化
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

plt.title("ACUE v1.1 — Self-Referential Ontology")

plt.savefig("acue_v1_1.png", dpi=200)
print("Saved: acue_v1_1.png")
