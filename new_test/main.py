import numpy as np

from new_test.network import Layer, Network
from new_test.training import train


def act_func(x):
    return np.array([max(0, xe[0]) for xe in x]).reshape(np.shape(x))


def act_func_der(x):
    return np.array([0 if xe[0] < 0 else 1 for xe in x]).reshape(np.shape(x))


layers = [
    Layer(1, 5, act_func, act_func_der),
    Layer(5, 5, act_func, act_func_der),
    Layer(5, 5, act_func, act_func_der),
    Layer(5, 1, act_func, act_func_der)
]


def loss_func(x, y):
    return 0.5 * (x - y)**2


def loss_func_der(x, y):
    return x - y


def squared(x):
    return x**2


def linear(x):
    return x


network = Network(layers, loss_func, loss_func_der)
train(network, squared, 0.05, 5000)
