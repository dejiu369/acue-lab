import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 世界状态
# -------------------------
grid_size = 80
field = np.zeros((grid_size, grid_size))
visit_count = np.zeros((grid_size, grid_size))


def to_index(x, y):
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size - 1), np.clip(iy, 0, grid_size - 1)


# -------------------------
# 2. ⭐反身约束（关键）
# -------------------------
def learning_rate(ix, iy):
    # ⭐访问越多 → 学习率越低（饱和）
    return 1.0 / (1.0 + visit_count[ix, iy])


def V(x, y):
    ix, iy = to_index(x, y)
    return field[ix, iy] > -0.4


# -------------------------
# 3. 动力系统
# -------------------------
def step(x, y):
    dx, dy = np.random.normal(0, 0.5, 2)
    return x + dx, y + dy


# -------------------------
# 4. 多 agent
# -------------------------
num_agents = 5

agents = []
for _ in range(num_agents):
    agents.append({
        "x": np.random.uniform(-2, 2),
        "y": np.random.uniform(-2, 2),
        "traj": []
    })


# -------------------------
# 5. 主循环（反身更新）
# -------------------------
for t in range(3000):

    for agent in agents:

        nx, ny = step(agent["x"], agent["y"])

        if V(nx, ny):
            agent["x"], agent["y"] = nx, ny

        agent["traj"].append((agent["x"], agent["y"]))

        ix, iy = to_index(agent["x"], agent["y"])

        # -------------------------
        # ⭐反身机制：规则被“经验调节”
        # -------------------------
        lr = learning_rate(ix, iy)

        field[ix, iy] += 0.02 * lr
        visit_count[ix, iy] += 1

    # 扩散（保持系统稳定）
    field *= 0.9995


# -------------------------
# 6. 可视化
# -------------------------
plt.figure(figsize=(6, 6))

plt.imshow(field.T, origin="lower", cmap="hot")

colors = ["cyan", "blue", "green", "yellow", "white"]

for i, agent in enumerate(agents):
    traj = np.array(agent["traj"])
    plt.plot(
        traj[:, 0] * 6 + 40,
        traj[:, 1] * 6 + 40,
        linewidth=0.6,
        color=colors[i % len(colors)]
    )

plt.title("ACUE v0.9 — Reflexive System")

plt.savefig("acue_v0_9.png", dpi=200)
print("Saved: acue_v0_9.png")
