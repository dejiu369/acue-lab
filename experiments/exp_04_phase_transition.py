import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 参数：控制“宇宙大小”
# -------------------------
r_values = np.linspace(2, 6, 6)  # 半径逐渐变大


fig, axes = plt.subplots(2, 3, figsize=(10, 6))

axes = axes.flatten()

for i, r in enumerate(r_values):

    # -------------------------
    # 定义约束（随参数变化）
    # -------------------------
    def V(x, y):
        return x**2 + y**2 < r**2

    # -------------------------
    # 动力系统
    # -------------------------
    def step(x, y):
        dx, dy = np.random.normal(0, 0.4, 2)
        return x + dx, y + dy

    x, y = 0.0, 0.0
    trajectory = []

    for _ in range(1500):
        nx, ny = step(x, y)

        if V(nx, ny):
            x, y = nx, ny

        trajectory.append((x, y))

    trajectory = np.array(trajectory)

    # 活动度（平均位移）
    diffs = np.diff(trajectory, axis=0)
    activity = np.mean(np.linalg.norm(diffs, axis=1))

    print(f"r={r:.2f}, activity={activity:.3f}")


    # -------------------------
    # 可视化
    # -------------------------
    grid_x = np.linspace(-6, 6, 200)
    grid_y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = np.vectorize(V)(X, Y)

    ax = axes[i]
    ax.contourf(X, Y, Z, levels=1, colors=["black", "white"])
    ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.5)

    ax.set_title(f"r = {round(r,2)}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("ACUE v0.4 — Phase Transition")
plt.tight_layout()

plt.savefig("acue_v0_4.png", dpi=200)
print("Saved: acue_v0_4.png")
