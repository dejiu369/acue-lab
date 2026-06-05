import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. Constraint（存在条件）
# -------------------------
def V(x, y):
    # 圆形存在域
    return x**2 + y**2 < 25


# -------------------------
# 2. Dynamics（演化规则）
# -------------------------
def step(x, y):
    # 小随机扰动（模拟“世界演化”）
    dx, dy = np.random.normal(0, 0.3, 2)
    return x + dx, y + dy


# -------------------------
# 3. Simulation（核心循环）
# -------------------------
x, y = 0.0, 0.0  # 初始点（存在域内部）

trajectory = []

for _ in range(2000):
    nx, ny = step(x, y)

    # ⭐关键：约束裁决（ACUE核心）
    if V(nx, ny):
        x, y = nx, ny
    # else: 拒绝移动（系统被“困住”）

    trajectory.append((x, y))


# -------------------------
# 4. 可视化
# -------------------------
trajectory = np.array(trajectory)

# 背景：存在区域
grid_x = np.linspace(-6, 6, 300)
grid_y = np.linspace(-6, 6, 300)
X, Y = np.meshgrid(grid_x, grid_y)
Z = np.vectorize(V)(X, Y)

plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, levels=1, colors=["black", "white"])

# ⭐轨迹（涌现行为）
plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.5)

plt.title("ACUE v0.3 — Emergent Trajectory")
plt.xlabel("x")
plt.ylabel("y")

plt.savefig("acue_v0_3.png", dpi=200)

print("Saved: acue_v0_3.png")
