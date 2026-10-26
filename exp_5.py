import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X = np.linspace(0, 1, 30).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel()
y += np.random.normal(0, 0.1, len(y))

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test = np.sin(2 * np.pi * X_test).ravel()

for degree in [1, 4, 15]:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X, y)
    predictions = model.predict(X_test)

    error = mean_squared_error(y_test, predictions)
    print(f"Degree {degree} Test MSE: {error:.4f}")

    plt.plot(X_test, predictions, label=f"Degree {degree}")

plt.scatter(X, y, label="Training Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Bias-Variance Analysis")
plt.legend()
plt.show()
