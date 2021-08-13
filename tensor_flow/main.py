import tensorflow as tf
import numpy as np
rg = np.random.default_rng(1)
import matplotlib.pyplot as plot


model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1,1)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1)
])


def squared(x):
    return x**2


def loss(x, y):
    return .5 * (x - y)**2


def evaluate():
    inputs = np.array([rg.random() * 2. - 1. for _ in range(100)]).reshape((100, 1, 1))
    targets = np.array([squared(i) for i in inputs]).reshape((100, 1, 1))
    outputs = model(inputs)

    plot.scatter(inputs[:, 0, 0], targets[:, 0, 0])
    plot.scatter(inputs[:, 0, 0], outputs[:, 0, 0])
    plot.show()


evaluate()

inputs = np.array([rg.random() * 2. - 1. for _ in range(100)]).reshape((100, 1, 1))
targets = np.array([squared(i) for i in inputs]).reshape((100, 1, 1))
model.compile(
    optimizer='adam',
    loss=loss
)
model.fit(inputs, targets, epochs=500)

evaluate()
