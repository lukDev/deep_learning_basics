import numpy as np
import matplotlib.pyplot as plot


rg = np.random.default_rng(1)


def evaluate(network, optimal_func):
    inputs = []
    targets = []
    outputs = []

    for i in range(1000):
        input = rg.random() * 2 - 1.
        inputs.append(input)

        target = optimal_func(input)
        targets.append(target)

        outputs.append(forward_pass(
            network,
            np.array(input).reshape((1, 1)),
            np.array(target).reshape((1, 1))
        )[1][0][0])

    plot.scatter(inputs, targets)
    plot.scatter(inputs, outputs)
    plot.show()


def train(network, optimal_func, alpha, gens):
    evaluate(network, optimal_func)

    for gen in range(gens):
        losses = []

        for i in range(100):
            input = rg.random() * 2 - 1.
            target = optimal_func(input)
            losses.append(forward_pass(
                network,
                np.array(input).reshape((1, 1)),
                np.array(target).reshape((1, 1))
            )[0])

            backward_pass(network, target)
            gradient_descent(network, alpha)

        avg_loss = np.average(losses)
        print(f"gen: {gen} \t loss: {avg_loss}")

    evaluate(network, optimal_func)


def forward_pass(network, input, target):
    layer_out = compute_layer(network.layers[0], input)
    for layer in network.layers[1:]:
        layer_out = compute_layer(layer, layer_out)

    loss = network.loss_func(layer_out, target)
    return loss, layer_out


def compute_layer(layer, input):
    layer.zero_grad()

    layer.input = input
    layer.output = layer.weights @ input + layer.bias
    return layer.act_func(layer.output)


def backward_pass(network, target):
    derivative = network.loss_func_der(network.layers[-1].output, target)

    for layer in network.layers[::-1]:
        derivative = compute_backwards(layer, derivative)


def compute_backwards(layer, derivative):
    derivative = np.multiply(derivative, layer.act_func_der(layer.output))

    layer.weights_der = derivative @ np.transpose(layer.input)
    layer.bias_der = derivative

    return np.transpose(layer.weights) @ derivative


def gradient_descent(network, alpha):
    for layer in network.layers:
        gradient_step(layer, alpha)


def gradient_step(layer, alpha):
    layer.weights -= layer.weights_der * alpha
    layer.bias -= layer.bias_der * alpha
