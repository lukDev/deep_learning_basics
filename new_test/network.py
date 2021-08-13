import numpy as np


rg = np.random.default_rng(1)


class Network:
    def __init__(self, layers, loss_func, loss_func_der):
        self.layers = layers
        self.loss_func = loss_func
        self.loss_func_der = loss_func_der


class Layer:
    def __init__(self, input_size, output_size, act_func, act_func_der):
        self.weights = rg.random((output_size, input_size)) * 2 - 1.
        self.bias = np.zeros((output_size, 1))
        self.act_func = act_func
        self.act_func_der = act_func_der

        self.input_size = input_size
        self.output_size = output_size
        self.weights_der = None
        self.bias_der = None
        self.input = None
        self.output = None
        self.zero_grad()

    def zero_grad(self):
        self.weights_der = np.zeros((self.output_size, self.input_size))
        self.bias_der = np.zeros((self.output_size, 1))
        self.input = np.zeros((self.input_size, 1))
        self.output = np.zeros((self.output_size, 1))
