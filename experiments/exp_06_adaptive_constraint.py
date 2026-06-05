import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 初始“软约束场”
# -------------------------
grid_size = 60
field = np.zeros((grid_size, grid_size))

def to_index(x, y):
    # 映射 [-6,6] → grid
    ix = int((x + 6) / 12 * (grid_size - 1))
    iy = int((y + 6) / 12 * (grid_size - 1))
    return np.clip(ix, 0, grid_size-1), np.clip(iy, 0, grid_size-1)


# -------------------------
# 当前约束（动态）
# -------------------------
def V(x, y):
    ix, iy = to_index(x, y)
    # ⭐访问频率越高 → 越“允许”
    return field[ix, iy] > -0.5


# -------------------------
# 动力系统
# -------------------------
def step(x, y):
    dx, dy = np.random.normal(0, 0.5, 2)
    return x + dx, y + dy


x, y = 0.0, 0.0
trajectory = []

# -------------------------
# 主循环
# -------------------------
for t in range(3000):

    nx, ny = step(x, y)

    if V(nx, ny):
        x, y = nx, ny

    trajectory.append((x, y))

    # ⭐关键：更新“存在记忆”
    ix, iy = to_index(x, y)
    field[ix, iy] += 0.02   # 被访问 → 更“存在”


trajectory = np.array(trajectory)

# -------------------------
# 可视化：存在结构自组织
# -------------------------
plt.figure(figsize=(6, 6))

plt.imshow(field.T, origin='lower', cmap='hot')
plt.plot(trajectory[:, 0] * 5 + 30, trajectory[:, 1] * 5 + 30,
         linewidth=0.5, color='cyan')

plt.title("ACUE v0.6 — Adaptive Constraints")

plt.savefig("acue_v0_6.png", dpi=200)
print("Saved: acue_v0_6.png")
