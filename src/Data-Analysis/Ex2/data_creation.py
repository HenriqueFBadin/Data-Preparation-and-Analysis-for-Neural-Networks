import numpy as np

mean_A = np.zeros(5)

Sigma_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]
)

mean_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])

Sigma_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ]
)

ptsA = np.random.multivariate_normal(mean_A, Sigma_A, 500)
ptsB = np.random.multivariate_normal(mean_B, Sigma_B, 500)

sub_a = ptsA - mean_A
sub_b = ptsB - mean_B
stdA = np.sqrt(np.diag(Sigma_A))
stdB = np.sqrt(np.diag(Sigma_B))
ptsA_ = sub_a / stdA
ptsB_ = sub_b / stdB


C_A = (1 / 500) * (ptsA_.T @ ptsA_)
C_B = (1 / 500) * (ptsB_.T @ ptsB_)

eigenvalues_A, eigenvectors_A = np.linalg.eig(C_A)
eigenvalues_B, eigenvectors_B = np.linalg.eig(C_B)

sorted_indices_A = np.argsort(eigenvalues_A)[::-1]
sorted_indices_B = np.argsort(eigenvalues_B)[::-1]

k = 2
W_A = eigenvectors_A[:, sorted_indices_A[:k]]
W_B = eigenvectors_B[:, sorted_indices_B[:k]]

YA = ptsA_ @ W_A
YB = ptsB_ @ W_B

import matplotlib.pyplot as plt

plt.scatter(YA[:, 0], YA[:, 1], alpha=0.5, s=50, c="black")
plt.scatter(YB[:, 0], YB[:, 1], alpha=0.5, s=50, c="red")
plt.title("Scatter Plot de dados gerados com gaussianas")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()
