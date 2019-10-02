from learning import *
import numpy as np


def order_3_reg_1x1():
    def original_func(x):
        a3 = np.array([[1.74]])
        a2 = np.array([[2.902]])
        a1 = np.array([[-20.34]])
        a0 = np.array([[3.0001]])

        return a3 * np.power(x, 3) + a2 * np.power(x, 2) + a1 * x + a0

    def out_interpretation(x):
        return x

    layer_dimensions = [(1, 45), (45, 45), (45, 1)]

    regression_1x1(-5., 5., 5000, 0.0001, layer_dimensions, original_func, out_interpretation)


def cat_1x1():
    categories = np.array([5, 3, 6, 2, 1, 9, 8, 0, 4, 7])
    lb = -5.
    ub = 5.

    def original_func(x):
        [[x_inner]] = x
        res = np.full((10,), 0)
        i = int((x_inner - lb) / (ub - lb) * 10)
        res[i] = 1
        return res

    def out_interpretation(x):
        x = x.reshape((x.shape[0]))
        max_i = np.argmax(x)
        return categories[max_i]

    layer_dimensions = [(1, 70), (70, 70), (70, 10)]

    classification_1x1(lb, ub, 30000, 0.001, layer_dimensions, original_func, out_interpretation)


cat_1x1()
