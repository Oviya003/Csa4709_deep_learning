import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Vector Addition:", a + b)
print("Dot Product:", np.dot(a, b))

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix Addition:\n", A + B)
print("Matrix Multiplication:\n", np.dot(A, B))

plt.plot(a, label="Vector A")
plt.plot(b, label="Vector B")
plt.legend()
plt.title("Vector Operations")
plt.show()
