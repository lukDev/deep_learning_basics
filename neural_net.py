from backprop import *
import numpy as np


class NeuralNet:
    def __init__(self, layers, loss_func, loss_func_der, out_interpretation):
        self.layers = layers
        self.loss_func = loss_func
        self.loss_func_der = loss_func_der
        self.out_interpretation = out_interpretation

    def to_graph(self, with_loss):
        nodes = [Node(None, None, (self.input_dimension(), 1))]  # input node
        edges = []

        for layer in self.layers:
            prev_out_node = nodes[-1]

            def weights_func(w):
                return lambda _: w
            weight_node = Node(weights_func(layer.weights), None, (layer.output_dimension(), layer.input_dimension()))
            nodes.append(weight_node)

            def matmul(prev_activations):  # matrix multiplication
                return np.matmul(prev_activations[0], prev_activations[1])

            def matmul_grad(forward_grad, prev_activations):  # gradient for matrix multiplication
                return (
                    np.matmul(forward_grad, np.transpose(prev_activations[1])),
                    np.matmul(np.transpose(prev_activations[0]), forward_grad)
                )

            matmul_node = Node(matmul, matmul_grad, (layer.output_dimension(), 1))
            nodes.append(matmul_node)

            def bias_func(b):
                return lambda _: b
            bias_node = Node(bias_func(layer.biases), None, (layer.output_dimension(), 1))
            nodes.append(bias_node)

            def plus(prev_activations):
                return np.add(prev_activations[0], prev_activations[1])

            def plus_grad(forward_grad, _):
                return forward_grad, forward_grad

            plus_node = Node(plus, plus_grad, (layer.output_dimension(), 1))
            nodes.append(plus_node)

            def get_activation_func(act_func):
                def activation(prev_activations):
                    prev_result = prev_activations[0]
                    act_res = act_func(prev_result)
                    return act_res

                return activation

            def get_act_der(act_der):
                def activation_der(forward_grad, prev_activations):
                    prev_result = prev_activations[0]
                    derived_acts = act_der(prev_result)
                    new_grad = np.multiply(forward_grad, derived_acts)
                    return (new_grad,)

                return activation_der

            activation_node = Node(get_activation_func(layer.activation_func), get_act_der(layer.act_func_der), (layer.output_dimension(), 1))
            nodes.append(activation_node)

            edges.append((weight_node, matmul_node))
            edges.append((prev_out_node, matmul_node))
            edges.append((matmul_node, plus_node))
            edges.append((bias_node, plus_node))
            edges.append((plus_node, activation_node))

        if with_loss:
            y_head_node = nodes[-1]
            target_node = Node(None, None, (self.output_dimension(), 1))
            nodes.append(target_node)

            def loss(prev_activations):
                return self.loss_func(prev_activations[0], prev_activations[1])

            def loss_der(_, prev_activations):
                return (
                    self.loss_func_der(prev_activations[0], prev_activations[1]),
                    0  # w.r.t. target --> not needed
                )

            loss_node = Node(loss, loss_der, (1, 1))
            nodes.append(loss_node)

            edges.append((y_head_node, loss_node))
            edges.append((target_node, loss_node))

            pois_w = [i * 5 + 1 for i in range(len(self.layers))]
            pois_b = [i * 5 + 3 for i in range(len(self.layers))]
            pois = [None] * (2 * len(self.layers))
            pois[::2] = pois_w
            pois[1::2] = pois_b

            ins = [0, len(nodes) - 2]

            return Graph(nodes, edges, ins, pois)

        else:
            return Graph(nodes, edges, [0], [])

    def input_dimension(self):
        return self.layers[0].input_dimension()

    def output_dimension(self):
        return self.layers[-1].output_dimension()

    def evaluate(self, x):
        graph = self.to_graph(False)
        return self.out_interpretation(graph.evaluate([x])[graph.nodes[-1]])


class Layer:
    def __init__(self, weights, biases, activation_func, act_func_der):
        self.weights = weights
        self.biases = biases
        self.activation_func = activation_func
        self.act_func_der = act_func_der

    def input_dimension(self):
        if len(self.weights) == 0:
            return None

        return len(self.weights[0])

    def output_dimension(self):
        return len(self.biases)
