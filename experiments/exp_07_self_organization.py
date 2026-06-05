import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 环境初始化
# -------------------------
grid_size = 80
field = np.zeros((grid_size, grid_size))


def to_index(x, y):
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size - 1), np.clip(iy, 0, grid_size - 1)


# -------------------------
# 2. 动态约束（soft V）
# -------------------------
def V(x, y):
    ix, iy = to_index(x, y)
    return field[ix, iy] > -0.3


# -------------------------
# 3. 动力系统
# -------------------------
def step(x, y):
    dx, dy = np.random.normal(0, 0.5, 2)
    return x + dx, y + dy


# -------------------------
# 4. 主循环
# -------------------------
x, y = 0.0, 0.0
trajectory = []

for t in range(4000):

    nx, ny = step(x, y)

    if V(nx, ny):
        x, y = nx, ny

    trajectory.append((x, y))

    ix, iy = to_index(x, y)

    # ⭐强化（访问）
    field[ix, iy] += 0.02

    # ⭐扩散（关键：打破局部过强）
    field = field * 0.999


trajectory = np.array(trajectory)

# -------------------------
# 5. 可视化
# -------------------------
plt.figure(figsize=(6, 6))

plt.imshow(field.T, origin="lower", cmap="hot")

plt.plot(
    trajectory[:, 0] * 6 + 40,
    trajectory[:, 1] * 6 + 40,
    linewidth=0.5,
    color="cyan",
)

plt.title("ACUE v0.7 — Self-Organization")

plt.savefig("acue_v0_7.png", dpi=200)
print("Saved: acue_v0_7.png")
