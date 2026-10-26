import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 3 * X + 2

optimizers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.1),
    "Momentum": tf.keras.optimizers.SGD(
        learning_rate=0.1, momentum=0.9
    ),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.1)
}

for name, optimizer in optimizers.items():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=optimizer, loss="mse")

    history = model.fit(X, y, epochs=50, verbose=0)

    print(name, "Final Loss:",
          round(history.history["loss"][-1], 6))

    plt.plot(history.history["loss"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.legend()
plt.show()
