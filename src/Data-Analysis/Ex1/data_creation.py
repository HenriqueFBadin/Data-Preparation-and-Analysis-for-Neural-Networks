import numpy as np

# https://numpy.org/doc/2.2/reference/random/generated/numpy.random.multivariate_normal.html

np.random.seed(2)


def create_data(N, mean, std_dev):
    cov = np.diag(std_dev)
    x, y = np.random.multivariate_normal(mean, cov, N).T
    return x, y


x1, y1 = create_data(100, [2, 3], [0.8, 2.5])
x2, y2 = create_data(100, [5, 6], [1.2, 1.9])
x3, y3 = create_data(100, [8, 1], [0.9, 0.9])
x4, y4 = create_data(100, [15, 4], [0.5, 2.0])

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(x1, y1, alpha=0.5, s=50, c="blue")
axes[0].scatter(x2, y2, alpha=0.5, s=50, c="red")
axes[0].scatter(x3, y3, alpha=0.5, s=50, c="green")
axes[0].scatter(x4, y4, alpha=0.5, s=50, c="yellow")
axes[0].set_title("Scatter Plot de dados gerados com gaussianas")
axes[0].set_xlabel("X values")
axes[0].set_ylabel("Y values")

axes[1].scatter(x1, y1, alpha=0.5, s=50, c="blue")
axes[1].scatter(x2, y2, alpha=0.5, s=50, c="red")
axes[1].scatter(x3, y3, alpha=0.5, s=50, c="green")
axes[1].scatter(x4, y4, alpha=0.5, s=50, c="yellow")

axes[1].plot([6.38, 0.56], [-2, 10], c="black", linewidth=2)
axes[1].plot([11.92, 2.5], [10, -2], c="black", linewidth=2)
axes[1].plot([13.4, 7.4], [-1.7, 10], c="black", linewidth=2)

axes[1].set_title("Scatter Plot + Linhas de Separação")
axes[1].set_xlabel("X values")
axes[1].set_ylabel("Y values")

plt.tight_layout()
plt.show()