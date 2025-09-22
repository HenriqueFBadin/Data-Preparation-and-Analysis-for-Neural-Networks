import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Dataset (3 classes, 4 features, clusters diferentes)
# -------------------------------
n = 1500

X0, _ = make_classification(
    n_samples=n // 3,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=2,
    n_classes=3,
    random_state=42,
)
y0 = np.zeros(X0.shape[0], dtype=int)

X1, _ = make_classification(
    n_samples=n // 3,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=3,
    n_classes=3,
    random_state=43,
)
y1 = np.ones(X1.shape[0], dtype=int)

X2, _ = make_classification(
    n_samples=n - (n // 3) * 2,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=4,
    n_classes=3,
    random_state=44,
)
y2 = np.full(X2.shape[0], 2)

X = np.vstack([X0, X1, X2])
y = np.hstack([y0, y1, y2])

# one-hot encoding
y_onehot = np.zeros((y.size, y.max() + 1))
y_onehot[np.arange(y.size), y] = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 2. Rede (12 neurônios ocultos, 3 saídas)
# -------------------------------
rng = np.random.default_rng(42)
W1 = rng.normal(0, 1, (12, 4)) / np.sqrt(4)
b1 = np.zeros(12)
W2 = rng.normal(0, 1, (3, 12)) / np.sqrt(12)
b2 = np.zeros(3)


def tanh(x):
    return np.tanh(x)


eta = 0.05
epochs = 500

# -------------------------------
# 3. Animação
# -------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

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

    # treino por alguns steps
    for _ in range(5):
        y_pred, h1 = forward(X_train)
        loss = np.mean((y_train - y_pred) ** 2)

        grad_z2 = 2 * (y_pred - y_train) * (1 - y_pred**2) / len(X_train)
        grad_W2 = grad_z2.T @ h1
        grad_b2 = grad_z2.sum(axis=0)

        grad_h1 = grad_z2 @ W2
        grad_z1 = grad_h1 * (1 - h1**2)
        grad_W1 = grad_z1.T @ X_train
        grad_b1 = grad_z1.sum(axis=0)

        W2 -= eta * grad_W2
        b2 -= eta * grad_b2
        W1 -= eta * grad_W1
        b1 -= eta * grad_b1

    # fronteira (2 primeiras features, extras zeradas)
    grid = np.c_[
        xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())
    ]
    y_grid, _ = forward(grid)
    y_grid_class = np.argmax(y_grid, axis=1)

    # predições no treino
    y_pred_train, _ = forward(X_train)
    y_pred_class = np.argmax(y_pred_train, axis=1)
    y_true_class = np.argmax(y_train, axis=1)

    acc = (y_pred_class == y_true_class).mean() * 100
    errors = y_pred_class != y_true_class
    correct = ~errors

    ax.clear()
    ax.contourf(xx, yy, y_grid_class.reshape(xx.shape), alpha=0.3, cmap=plt.cm.Set1)

    # pontos corretos
    ax.scatter(
        X_train[correct, 0],
        X_train[correct, 1],
        c=y_true_class[correct],
        cmap=plt.cm.Set1,
        edgecolors="k",
        s=30,
        marker="o",
        label="Corretos",
    )

    # pontos errados
    ax.scatter(
        X_train[errors, 0],
        X_train[errors, 1],
        c=y_true_class[errors],
        cmap=plt.cm.Set1,
        edgecolors="k",
        s=60,
        marker="x",
        label="Errados",
    )

    # título dinâmico
    ax.set_title(f"Época {frame*5} | Loss={loss:.4f} | Acc={acc:.2f}%")

    # legenda de classes + corretos/errados
    class_labels = [
        Patch(color=plt.cm.Set1(i / 3), label=f"Classe {i}") for i in range(3)
    ]
    ax.legend(
        handles=class_labels
        + [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markeredgecolor="k",
                label="Corretos",
                markersize=8,
            ),
            plt.Line2D([0], [0], marker="x", color="k", label="Errados", markersize=8),
        ],
        loc="upper right",
    )
    return ax


ani = FuncAnimation(fig, update, frames=epochs // 5, interval=200, repeat=False)

ani.save("mlp_multiclass_legendas.gif", writer="pillow")
plt.show()
