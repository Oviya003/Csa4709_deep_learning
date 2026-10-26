import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 4, 5, 4, 6]
})

print("Covariance Matrix:")
print(data.cov())

matrix = np.array([[2, 1], [1, 2]])

values, vectors = np.linalg.eig(matrix)

print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)

plt.scatter(data['X'], data['Y'])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Covariance Analysis")
plt.show()
