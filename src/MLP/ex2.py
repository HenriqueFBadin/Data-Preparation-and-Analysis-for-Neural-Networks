import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Geração do dataset
# -------------------------------
n = 1000
X0, _ = make_classification(
    n_samples=n // 2,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    class_sep=1.3,
    random_state=42,
)
y0 = np.zeros(X0.shape[0], dtype=int)

X1a, _ = make_classification(
    n_samples=n // 4,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    class_sep=1.3,
    random_state=42,
)
X1b, _ = make_classification(
    n_samples=n - (n // 2 + n // 4),
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    class_sep=1.3,
    random_state=42,
)
y1a = np.ones(X1a.shape[0], dtype=int)
y1b = np.ones(X1b.shape[0], dtype=int)

X = np.vstack([X0, X1a, X1b])
y = np.hstack([y0, y1a, y1b])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 2. Inicialização dos pesos (24 neurônios)
# -------------------------------
rng = np.random.default_rng(42)
W1 = rng.normal(0, 1, (12, 2)) / np.sqrt(2)
b1 = np.zeros(12)
W2 = rng.normal(0, 1, (1, 12)) / np.sqrt(12)
b2 = np.zeros(1)


def tanh(x):
    return np.tanh(x)


eta = 0.05
epochs = 500
y_train = y_train.reshape(-1, 1).astype(float)

# -------------------------------
# 3. Setup da animação
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))


def forward(X_in):
    z1 = X_in @ W1.T + b1
    h1 = tanh(z1)
    z2 = h1 @ W2.T + b2
    return tanh(z2), h1


def update(frame):
    global W1, b1, W2, b2

    # treina algumas épocas por frame
    for _ in range(5):
        y_pred, h1 = forward(X_train)
        loss = np.mean((y_train - y_pred) ** 2)

        # backward
        grad_z2 = 2 * (y_pred - y_train) * (1 - y_pred**2) / len(X_train)
        grad_W2 = grad_z2.T @ h1
        grad_b2 = grad_z2.sum(axis=0)

        grad_h1 = grad_z2 @ W2
        grad_z1 = grad_h1 * (1 - h1**2)
        grad_W1 = grad_z1.T @ X_train
        grad_b1 = grad_z1.sum(axis=0)

        # update
        W2 -= eta * grad_W2
        b2 -= eta * grad_b2
        W1 -= eta * grad_W1
        b1 -= eta * grad_b1

    # recalcula a fronteira
    y_grid, _ = forward(np.c_[xx.ravel(), yy.ravel()])
    y_grid_class = (y_grid.ravel() >= 0).astype(int)

    # predições no treino
    y_pred_train, _ = forward(X_train)
    y_pred_class = (y_pred_train.ravel() >= 0).astype(int)

    # identifica erros e acertos
    errors = y_pred_class != y_train.ravel()
    correct = ~errors
    acc = (y_pred_class == y_train.ravel()).mean() * 100

    ax.clear()
    ax.contourf(xx, yy, y_grid_class.reshape(xx.shape), alpha=0.3, cmap=plt.cm.coolwarm)

    # pontos corretos
    ax.scatter(
        X_train[correct, 0],
        X_train[correct, 1],
        c=y_train.ravel()[correct],
        cmap=plt.cm.coolwarm,
        edgecolors="k",
        s=30,
        marker="o",
        label="Corretos",
    )

    # pontos errados (X)
    ax.scatter(
        X_train[errors, 0],
        X_train[errors, 1],
        c=y_train.ravel()[errors],
        cmap=plt.cm.coolwarm,
        edgecolors="k",
        s=60,
        marker="x",
        label="Errados",
    )

    ax.set_title(f"Época {frame*5} | Loss={loss:.4f} | Acc={acc:.2f}%")
    ax.legend(loc="upper right")
    return ax


ani = FuncAnimation(fig, update, frames=epochs // 5, interval=200, repeat=False)

# Para salvar:
ani.save("mlp_training.gif", writer="pillow")

plt.show()
