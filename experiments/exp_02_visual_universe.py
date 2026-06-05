import numpy as np
import matplotlib.pyplot as plt

# Constraint surface (2D admissibility field)
def V(x, y):
    # 圆形约束区域（存在域）
    return x**2 + y**2 < 25


# Generate grid
x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)

X, Y = np.meshgrid(x, y)

Z = np.vectorize(V)(X, Y)

# Plot admissibility field
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, levels=1, colors=["white", "black"])

plt.title("ACUE v0.2 — Admissibility Space")
plt.xlabel("x")
plt.ylabel("y")

plt.show()
