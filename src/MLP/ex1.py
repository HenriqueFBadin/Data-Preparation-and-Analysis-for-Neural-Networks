import numpy as np

x = np.array([0.5, -0.2])
y = 1
W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([[0.5, -0.3]])
b2 = np.array([0.2])
eta = 0.3


def activation_function(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


print("Forward Pass:")
z1 = W1.dot(x) + b1
print("z1 =", np.round(z1, 4))

h1 = activation_function(z1)
print("h1 =", np.round(h1, 4))

u2 = W2.dot(h1) + b2
print("u2 =", np.round(u2, 4))

y_pred = activation_function(u2)
print("y_pred =", np.round(y_pred, 4))

L = (1 / (len(x))) * (y - y_pred) ** 2
print("MSE loss =", np.round(L, 4))

delL_dely_pred = -2 * (y - y_pred) / len(x)
dely_pred_du2 = 1 - y_pred**2
delL_delW2 = delL_dely_pred * dely_pred_du2 * h1
delL_delb2 = delL_dely_pred * dely_pred_du2
delL_delh1 = delL_dely_pred * dely_pred_du2 * W2
delh1_dz1 = 1 - h1**2
delL_delW1 = np.outer(delL_delh1 * delh1_dz1, x)
delL_delb1 = delL_delh1 * delh1_dz1


W2 = W2 - eta * delL_delW2
b2 = b2 - eta * delL_delb2
W1 = W1 - eta * delL_delW1
b1 = b1 - eta * delL_delb1

print("\nBackpropagation:")
print("delL/dely_pred =", delL_dely_pred)
print("dely_pred/du2  =", dely_pred_du2)
print("delL/delW2     =", delL_delW2)
print("delL/delb2     =", delL_delb2)
print("delL/delh1     =", delL_delh1)
print("delh1/dz1      =", delh1_dz1)
print("delL/delW1     =\n", delL_delW1)
print("delL/delb1     =", delL_delb1)

print("\nParameter Update:")
print("W2 =", W2)
print("b2 =", b2)
print("W1 =\n", W1)
print("b1 =", b1)
