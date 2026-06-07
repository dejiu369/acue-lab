import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================
# ACUE-2D 人工约束宇宙引擎 v2.0
# 能性哲学：二维显相维 + 一维时间维 = 三维物质宇宙
# ==============================================

# -------------------------
# 1. 二维能态空间 (对应2维循环显相维)
# -------------------------
GRID_SIZE = 50
X = np.arange(GRID_SIZE)
Y = np.arange(GRID_SIZE)

# -------------------------
# 2. 世界动力学 (解纹力自发涨落)
# -------------------------
def O(x, y):
    """二维能流的自发运动，解纹力的数学表达"""
    # 八个方向的随机移动，步长为奇数(对应解纹力的不对称性)
    dx = random.choice([-3, -1, 0, 1, 3])
    dy = random.choice([-3, -1, 0, 1, 3])
    # 边界循环(对应二维显相维的循环性)
    new_x = (x + dx) % GRID_SIZE
    new_y = (y + dy) % GRID_SIZE
    return new_x, new_y

# -------------------------
# 3. 二维约束族 (凝纹约束结构，对应物理定律原型)
# -------------------------
class Constraint2D:
    def __init__(self, bias_x, bias_y):
        self.bias_x = bias_x  # X方向约束偏置
        self.bias_y = bias_y  # Y方向约束偏置
        self.history = []     # 约束演化历史
    
    def V(self, x, y):
        """二维约束筛选函数：返回1表示允许存在，0表示湮灭"""
        # 约束对能流的坐标特征敏感(对应物理定律的选择性)
        score_x = (x % 7) / 7 + self.bias_x
        score_y = (y % 7) / 7 + self.bias_y
        total_score = (score_x + score_y) / 2
        return 1 if total_score > 0.5 else 0
    
    def update(self, x, y):
        """约束自演化规则：能流反过来塑造约束"""
        # 约束会逐渐偏好那些曾经通过它的能流特征
        self.bias_x += 0.003 * ((x % 2) - 0.5)
        self.bias_y += 0.003 * ((y % 2) - 0.5)
        # 限制偏置范围，防止约束极端化
        self.bias_x = max(-0.4, min(0.4, self.bias_x))
        self.bias_y = max(-0.4, min(0.4, self.bias_y))
        self.history.append((self.bias_x, self.bias_y))

# -------------------------
# 4. R值计算 (能性平衡度：解纹力/凝纹力)
# -------------------------
def calculate_R_2d(constraints):
    """计算二维宇宙的当前R值"""
    allowed = 0
    total = GRID_SIZE * GRID_SIZE
    for x in X:
        for y in Y:
            if any(V.V(x, y) for V in constraints):
                allowed += 1
    return total / allowed if allowed > 0 else float('inf')

# -------------------------
# 5. 二维宇宙模拟主循环
# -------------------------
def run_simulation_2d(steps=300, num_constraints=9):
    # 初始化9个约束(对应3×3的基础约束矩阵)
    constraints = [
        Constraint2D(random.random()-0.5, random.random()-0.5)
        for _ in range(num_constraints)
    ]
    
    # 初始能流位置(随机)
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    
    # 记录数据
    trajectory = []
    R_values = []
    density_map = np.zeros((GRID_SIZE, GRID_SIZE))
    
    print("=== ACUE-2D 二维能态宇宙启动 ===")
    print(f"网格大小: {GRID_SIZE}×{GRID_SIZE} | 约束数量: {num_constraints}\n")
    
    for t in range(steps):
        # 1. 能流自发涨落(解纹)
        x, y = O(x, y)
        
        # 2. 约束筛选(凝纹)
        allowed = any(V.V(x, y) for V in constraints)
        
        if allowed:
            trajectory.append((x, y))
            density_map[y, x] += 1
        else:
            # 不被允许的能流湮灭，随机重生
            x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            trajectory.append((None, None))
        
        # 3. 约束自演化(自指)
        for V in constraints:
            V.update(x, y)
        
        # 4. 计算R值
        if t % 10 == 0:
            current_R = calculate_R_2d(constraints)
            R_values.append(current_R)
            print(f"时间步 {t:3d} | 能流位置: ({x:2d},{y:2d}) | R值: {current_R:.2f}")
    
    print("\n=== 模拟完成 ===")
    return trajectory, R_values, density_map, constraints

# -------------------------
# 6. 结果可视化
# -------------------------
def visualize_results(trajectory, R_values, density_map, constraints):
    fig = plt.figure(figsize=(16, 12))
    
    # 子图1：能流密度热力图(对应物质分布)
    plt.subplot(2, 2, 1)
    plt.imshow(density_map, cmap='viridis', origin='lower')
    plt.colorbar(label='能流密度')
    plt.title('能流密度分布(物质结构原型)')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    
    # 子图2：R值演化曲线
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(0, len(R_values)*10, 10), R_values, 'r-', linewidth=1)
    plt.axhline(y=1, color='g', linestyle='--', label='R=1 归源稳态')
    plt.title('系统R值演化')
    plt.xlabel('时间步')
    plt.ylabel('R值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：约束偏置演化
    plt.subplot(2, 2, 3)
    for i, V in enumerate(constraints[:5]):  # 显示前5个约束
        history = np.array(V.history)
        plt.plot(history[:, 0], label=f'约束{i+1}-X')
        plt.plot(history[:, 1], label=f'约束{i+1}-Y', linestyle='--')
    plt.title('约束偏置演化(物理定律形成)')
    plt.xlabel('时间步')
    plt.ylabel('约束偏置')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    # 子图4：能流轨迹(前200步)
    plt.subplot(2, 2, 4)
    valid_traj = [(x,y) for x,y in trajectory[:200] if x is not None]
    if valid_traj:
        xs, ys = zip(*valid_traj)
        plt.plot(xs, ys, 'b-', linewidth=0.5, alpha=0.7)
        plt.scatter(xs[0], ys[0], c='g', s=50, label='起点')
        plt.scatter(xs[-1], ys[-1], c='r', s=50, label='终点')
    plt.title('能流轨迹(前200步)')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acue_2d_results.png', dpi=300)
    print("\n结果已保存为 acue_2d_results.png")
    
    # 生成能流演化动画
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
    im = ax_anim.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap='viridis', origin='lower', vmin=0, vmax=10)
    plt.colorbar(im, label='能流密度')
    ax_anim.set_title('能流演化动画')
    
    def update(frame):
        current_density = np.zeros((GRID_SIZE, GRID_SIZE))
        for x, y in trajectory[:frame*3]:
            if x is not None:
                current_density[y, x] += 1
        im.set_data(current_density)
        ax_anim.set_xlabel(f'时间步: {frame*3}')
        return [im]
    
    anim = FuncAnimation(fig_anim, update, frames=len(trajectory)//3, interval=50, blit=True)
    anim.save('acue_2d_animation.gif', writer='pillow', fps=20)
    print("能流演化动画已保存为 acue_2d_animation.gif")

# -------------------------
# 运行模拟
# -------------------------
if __name__ == "__main__":
    trajectory, R_values, density_map, constraints = run_simulation_2d(steps=300, num_constraints=9)
    visualize_results(trajectory, R_values, density_map, constraints)
    
    # 打印最终约束状态
    print("\n最终约束偏置:")
    for i, V in enumerate(constraints):
        print(f"约束 {i+1}: X={V.bias_x:.4f}, Y={V.bias_y:.4f}")
