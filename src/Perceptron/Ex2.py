import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(42)

mean_A = np.array([3, 3])

Sigma_A = np.array(
    [
        [1.5, 0],
        [0, 1.5],
    ]
)

mean_B = np.array([4, 4])

Sigma_B = np.array(
    [
        [1.5, 0],
        [0, 1.5],
    ]
)

ptsA = np.random.multivariate_normal(mean_A, Sigma_A, 1000)
ptsB = np.random.multivariate_normal(mean_B, Sigma_B, 1000)

plt.scatter(ptsA[:, 0], ptsA[:, 1], alpha=0.6, label="Class A", color="blue")
plt.scatter(ptsB[:, 0], ptsB[:, 1], alpha=0.6, label="Class B", color="red")
plt.legend()
plt.title("Generated Data — Exercise 2")
plt.show()
plt.close()


def activation_function(x):
    return 1 if x >= 0 else 0


Ya = np.zeros(len(ptsA), dtype=int)
Yb = np.ones(len(ptsB), dtype=int)

X = np.vstack([ptsA, ptsB])
Y = np.hstack([Ya, Yb])

w = np.array([1, 0])
b = 0
eta = 0.1
max_epochs = 100
hist = [(w.copy(), b)]
accuracies = []

for epoch in range(max_epochs):
    errors = 0
    for i, x in enumerate(X):
        y_pred = activation_function(float(w @ x + b))
        erro = Y[i] - y_pred
        if erro != 0:
            w = w + eta * erro * x
            b = b + eta * erro
            errors += 1
    hist.append((w.copy(), b))

    y_pred_epoch = (X @ w + b >= 0).astype(int)
    acc = (y_pred_epoch == Y).mean()
    accuracies.append(acc)

    if errors == 0:
        break
    print(f"Época: {epoch} | w_{epoch} = {w} | b_{epoch} = {b:.4f}")

print(f"Épocas: {len(hist)-1} | w final = {w} | b final = {b:.4f}")


# ---------------------
# Plot inicial
# ---------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(ptsA[:, 0], ptsA[:, 1], alpha=0.6, label="Class A", color="blue")
ax.scatter(ptsB[:, 0], ptsB[:, 1], alpha=0.6, label="Class B", color="red")
(line,) = ax.plot([], [], "k-", lw=2, label="Decision boundary")
mis = ax.scatter([], [], marker="x", s=80, label="Misclassified")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.legend()
ax.grid(True, alpha=0.3)

x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)


def update(k):
    w_k, b_k = hist[k]

    # reta: w1*x1 + w2*x2 + b = 0 => x2 = -(w1/w2)*x1 - b/w2
    if abs(w_k[1]) < 1e-12:
        line.set_data([], [])  # reta ~ vertical (simplificando, não plota)
    else:
        x2 = -(w_k[0] / w_k[1]) * x1 - b_k / w_k[1]
        line.set_data(x1, x2)

    # pontos mal classificados nessa época
    y_pred_k = (X @ w_k + b_k > 0).astype(int)
    miss = X[y_pred_k != Y]
    mis.set_offsets(miss if len(miss) else np.empty((0, 2)))

    ax.set_title(f"Época {k}/{len(hist)-1} | w_{k} = {w_k} | b_{k} = {b_k:.4f}")
    return line, mis


# ---------------------
# Criar animação
# ---------------------
ani = FuncAnimation(fig, update, frames=len(hist), interval=300, blit=False)
from matplotlib.animation import PillowWriter

ani.save("src/Perceptron/animacao_perceptron_2.gif", writer=PillowWriter(fps=2))
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
epochs = range(1, len(accuracies) + 1)

# linha preta + bolinhas vermelhas
plt.plot(epochs, accuracies, color="black", linewidth=1.5)
plt.scatter(epochs, accuracies, color="red", s=50, zorder=3)
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Evolução da Accuracy por Época")
plt.grid(True, alpha=0.3)
plt.show()
plt.close()

print(f"Mean Accuracy: {np.mean(accuracies)*100:.2f}%")
