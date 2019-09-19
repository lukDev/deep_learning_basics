from learning import *
import numpy as np


def order_3_reg_1x1():
    def original_func(x):
        a3 = np.array([[0.]])
        a2 = np.array([[2.]])
        a1 = np.array([[0.]])
        a0 = np.array([[1.]])

        return a3 * np.power(x, 3) + a2 * np.power(x, 2) + a1 * x + a0

    def out_interpretation(x):
        return x

    regression_1x1(-5., 5., 500000, 0.0001, original_func, out_interpretation)


def cat_3_1x1():
    categories = np.array([1, 2, 3])

    def original_func(x):
        [[x_inner]] = x
        if x_inner < -0.3333:
            return [1, 0, 0]
        elif -0.3333 <= x_inner < 0.3333:
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    def out_interpretation(x):
        x = x.reshape((x.shape[0]))
        max_i = np.argmax(x)
        return categories[max_i]

    classification_1x1(-1., 1., 10000, 0.01, original_func, out_interpretation)


order_3_reg_1x1()
