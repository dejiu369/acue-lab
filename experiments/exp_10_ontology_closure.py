import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 系统空间
# -------------------------
grid_size = 80

visit = np.zeros((grid_size, grid_size))
state = np.zeros((grid_size, grid_size))


def to_index(x, y):
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size - 1), np.clip(iy, 0, grid_size - 1)


# -------------------------
# 2. ⭐关键：V 不再是规则，而是统计结果
# -------------------------
def V(ix, iy):
    # admissibility = “未被过度占用 + 有结构潜力”
    return visit[ix, iy] < np.mean(visit) + 0.5


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
# 5. 主循环
# -------------------------
for t in range(3500):

    for a in agents:

        nx, ny = step(a["x"], a["y"])
        ix, iy = to_index(nx, ny)

        # ⭐ V 变成“系统内部涌现判据”
        if V(ix, iy):
            a["x"], a["y"] = nx, ny

        a["traj"].append((a["x"], a["y"]))

        ix, iy = to_index(a["x"], a["y"])

        visit[ix, iy] += 1
        state[ix, iy] += 0.01

    # ⭐扩散 + 退火（结构稳定）
    visit *= 0.999


# -------------------------
# 6. 可视化
# -------------------------
plt.figure(figsize=(6, 6))

plt.imshow(visit.T, origin="lower", cmap="hot")

colors = ["cyan", "blue", "green", "yellow", "white", "magenta"]

for i, a in enumerate(agents):
    traj = np.array(a["traj"])
    plt.plot(
        traj[:, 0] * 6 + 40,
        traj[:, 1] * 6 + 40,
        linewidth=0.6,
        color=colors[i % len(colors)]
    )

plt.title("ACUE v1.0 — Ontology Closure")

plt.savefig("acue_v1_0.png", dpi=200)
print("Saved: acue_v1_0.png")
