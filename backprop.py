import numpy as np


class Graph:
    def __init__(self, nodes, edges, ins, pois):
        self.nodes = nodes
        self.edges = edges
        self.ins = ins  # indices of the input nodes
        self.pois = pois  # "points of interest" - the indices of the input nodes for which the gradient should be computed

    def parents_of(self, node):
        parents = []

        for edge in self.edges:
            if edge[1] == node:
                parents.append(edge[0])

        return parents

    def children_of(self, node):
        children = []

        for edge in self.edges:
            if edge[0] == node:
                children.append(edge[1])

        return children

    def evaluate(self, inputs):
        node_values = dict()
        for i in range(len(inputs)):
            node_values[self.nodes[self.ins[i]]] = inputs[i]

        for i in range(len(self.nodes)):
            if i in self.ins:
                continue

            node = self.nodes[i]
            pars = self.parents_of(node)
            par_values = [node_values[p] for p in pars]
            node_values[node] = node.function(par_values)

        return node_values


class Node:
    def __init__(self, function, gradient, dimension):
        self.function = function
        self.gradient = gradient
        self.dimension = dimension


def backprop(graph, input):
    activations = graph.evaluate(input)

    grad_table = dict()
    grad_table[graph.nodes[-1]] = np.array([[1]])

    for i in range(len(graph.nodes) - 1)[::-1]:
        node = graph.nodes[i]
        grad_table[node] = np.zeros((node.dimension[0], node.dimension[1]))
        children = graph.children_of(node)

        for child in children:
            pars = graph.parents_of(child)
            deriv_index = pars.index(node)
            child_gradient = child.gradient(grad_table[child], [activations[p] for p in pars])
            grad_table[node] = np.add(grad_table[node], child_gradient[deriv_index])

    return [grad_table[graph.nodes[i]] for i in graph.pois]
