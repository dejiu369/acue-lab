import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 多个约束（竞争结构）
# -------------------------
def V1(x, y):
    # 圆形约束
    return x**2 + y**2 < 25


def V2(x, y):
    # 右半空间约束
    return x > -1


def V3(x, y):
    # 上半空间约束
    return y > -1


# -------------------------
# 2. 竞争规则（关键）
# -------------------------
def V(x, y):
    # ⭐ 这里是“竞争逻辑”
    return V1(x, y) and V2(x, y) and V3(x, y)


# -------------------------
# 3. 动力系统
# -------------------------
def step(x, y):
    dx, dy = np.random.normal(0, 0.4, 2)
    return x + dx, y + dy


x, y = 0.0, 0.0
trajectory = []

for _ in range(2500):
    nx, ny = step(x, y)

    # 约束裁决（竞争后的结果）
    if V(nx, ny):
        x, y = nx, ny

    trajectory.append((x, y))


trajectory = np.array(trajectory)

# -------------------------
# 4. 可视化存在空间
# -------------------------
grid = np.linspace(-6, 6, 300)
X, Y = np.meshgrid(grid, grid)

Z = np.vectorize(V)(X, Y)

plt.figure(figsize=(6, 6))

plt.contourf(X, Y, Z, levels=1, colors=["black", "white"])
plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.5)

plt.title("ACUE v0.5 — Constraint Competition")

plt.savefig("acue_v0_5.png", dpi=200)
print("Saved: acue_v0_5.png")
