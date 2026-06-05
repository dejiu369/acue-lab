import numpy as np
import matplotlib.pyplot as plt

def V(x, y):
    return x**2 + y**2 < 25


x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)

X, Y = np.meshgrid(x, y)

Z = np.vectorize(V)(X, Y)

plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, levels=1, colors=["white", "black"])

plt.title("ACUE v0.2 — Admissibility Space")
plt.xlabel("x")
plt.ylabel("y")

# ⭐关键修改：保存图片，而不是 show
plt.savefig("acue_v0_2.png", dpi=200)

print("Visualization saved: acue_v0_2.png")
