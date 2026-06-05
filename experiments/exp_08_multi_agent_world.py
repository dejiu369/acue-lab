import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 世界（共享约束场）
# -------------------------
grid_size = 80
field = np.zeros((grid_size, grid_size))


def to_index(x, y):
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size - 1), np.clip(iy, 0, grid_size - 1)


def V(x, y):
    ix, iy = to_index(x, y)
    return field[ix, iy] > -0.4


# -------------------------
# 2. 多 agent 初始化
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
# 3. 动力学
# -------------------------
def step(x, y):
    dx, dy = np.random.normal(0, 0.5, 2)
    return x + dx, y + dy


# -------------------------
# 4. 主循环
# -------------------------
for t in range(3000):

    for agent in agents:

        nx, ny = step(agent["x"], agent["y"])

        if V(nx, ny):
            agent["x"], agent["y"] = nx, ny

        agent["traj"].append((agent["x"], agent["y"]))

        ix, iy = to_index(agent["x"], agent["y"])

        # ⭐共享空间被所有 agent 改写
        field[ix, iy] += 0.015

    # ⭐扩散（防止单点垄断）
    field *= 0.9995


# -------------------------
# 5. 可视化
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

plt.title("ACUE v0.8 — Multi-Agent Constraint World")

plt.savefig("acue_v0_8.png", dpi=200)
print("Saved: acue_v0_8.png")
