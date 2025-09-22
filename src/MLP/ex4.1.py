import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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
    n_samples=n - 2 * (n // 3),
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

y_onehot = np.zeros((y.size, y.max() + 1))
y_onehot[np.arange(y.size), y] = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

rng = np.random.default_rng(42)

W1 = rng.normal(0, 1, (64, 4)) / np.sqrt(4)
b1 = np.zeros(64)

W2 = rng.normal(0, 1, (32, 64)) / np.sqrt(64)
b2 = np.zeros(32)

W3 = rng.normal(0, 1, (3, 32)) / np.sqrt(32)
b3 = np.zeros(3)


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    eps = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


eta = 0.05
epochs = 1000
y_train = y_train.astype(float)

for ep in range(1, epochs + 1):
    # forward
    z1 = X_train @ W1.T + b1
    h1 = tanh(z1)

    z2 = h1 @ W2.T + b2
    h2 = tanh(z2)

    z3 = h2 @ W3.T + b3
    y_pred = softmax(z3)

    # loss
    loss = cross_entropy(y_train, y_pred)

    # backward
    grad_z3 = (y_pred - y_train) / len(X_train)
    grad_W3 = grad_z3.T @ h2
    grad_b3 = grad_z3.sum(axis=0)

    grad_h2 = grad_z3 @ W3
    grad_z2 = grad_h2 * (1 - h2**2)
    grad_W2 = grad_z2.T @ h1
    grad_b2 = grad_z2.sum(axis=0)

    grad_h1 = grad_z2 @ W2
    grad_z1 = grad_h1 * (1 - h1**2)
    grad_W1 = grad_z1.T @ X_train
    grad_b1 = grad_z1.sum(axis=0)

    # updates
    W3 -= eta * grad_W3
    b3 -= eta * grad_b3
    W2 -= eta * grad_W2
    b2 -= eta * grad_b2
    W1 -= eta * grad_W1
    b1 -= eta * grad_b1

    if ep % 100 == 0 or ep == 1:
        print(f"época {ep} | loss {loss:.4f}")

z1 = X_test @ W1.T + b1
h1 = tanh(z1)

z2 = h1 @ W2.T + b2
h2 = tanh(z2)

z3 = h2 @ W3.T + b3
y_pred_test = softmax(z3)

ypred = np.argmax(y_pred_test, axis=1)
ytrue = np.argmax(y_test, axis=1)

acc = (ypred == ytrue).mean()
print(f"\nacurácia teste: {(acc*100):.4f}%")
