from neural_net import *
from matplotlib import pyplot as plt


def learn(nn, data, learning_rate):
    xs = data[0]
    ys = data[1]
    for i in range(len(xs)):
        graph = nn.to_graph(True)
        gradient = backprop(graph, [xs[i], ys[i]])
        # print(gradient)

        for j in range(len(nn.layers)):
            layer = nn.layers[j]
            layer.weights = np.subtract(layer.weights, np.multiply(learning_rate, gradient[2 * j]))
            layer.biases = np.subtract(layer.biases, np.multiply(learning_rate, gradient[2 * j + 1]))

    return nn


def get_data(original_func, lb, ub, size):
    xs = np.random.uniform(lb, ub, size)
    ys = np.array([original_func(x) for x in xs])

    return xs, ys


def get_nn_from_architecture(layer_dimensions, loss, loss_der, hidden_act, hidden_act_der, output_act, output_act_der, out_interpretation):
    weights = [np.random.uniform(-1., 1., layer_dimension[::-1]) for layer_dimension in layer_dimensions]  # dimension is reversed because a R^n -> R^m function needs a m x n matrix
    biases = [np.random.uniform(-0.1, 0.1, (layer_dimension[1], 1)) for layer_dimension in layer_dimensions]

    layers = [Layer(weights[i], biases[i], hidden_act, hidden_act_der) for i in range(len(layer_dimensions) - 1)]
    layers.append(Layer(weights[-1], biases[-1], output_act, output_act_der))

    return NeuralNet(layers, loss, loss_der, out_interpretation)


def regression_1x1(lb, ub, d_size, learning_rate, layer_dimensions, original_func, out_interpretation):
    def loss(y_head, y):
        return 0.5 * np.power(np.linalg.norm(np.subtract(y_head, y)), 2)

    def loss_der(y_head, y):  # only w.r.t. y_head, other derivative is not needed
        return np.subtract(y_head, y)

    def hidden_act(x):
        return np.array([[max(0, xv) for xv in xs] for xs in x])

    def hidden_act_der(x):
        def der(x):
            return 1. if x > 0. else 0.
        return np.array([[der(xv) for xv in xs] for xs in x])

    def output_act(x):
        return x

    def output_act_der(x):
        return np.ones(x.shape)

    nn = learn_regression_1x1(original_func, d_size, lb, ub, learning_rate, layer_dimensions, loss, loss_der, hidden_act, hidden_act_der, output_act, output_act_der, out_interpretation)
    test_regression_1x1(nn, original_func, 100, lb, ub)


def learn_regression_1x1(func, d_size, d_lb, d_ub, learning_rate, layer_dimensions, loss, loss_der, hidden_act, hidden_act_der, output_act, output_act_der, out_interpretation):
    nn = get_nn_from_architecture(layer_dimensions, loss, loss_der, hidden_act, hidden_act_der, output_act, output_act_der, out_interpretation)

    (ds_xs, ds_ys) = get_data(func, d_lb, d_ub, (d_size, 1, 1))
    d_xs = np.array([np.array(x).reshape(1, 1) for x in ds_xs])
    d_ys = np.array([np.array(y).reshape(1, 1) for y in ds_ys])

    new_nn = learn(nn, (d_xs, d_ys), learning_rate)
    return new_nn


def test_regression_1x1(nn, func, t_size, d_lb, d_ub):
    (ts_xs, t_ys) = get_data(func, d_lb, d_ub, (t_size, 1, 1))
    t_xs = np.array([x.reshape((1, 1)) for x in ts_xs])
    t_nn = np.array([nn.evaluate(x) for x in t_xs])

    for i in range(t_size):
        print("%s: %s - %s" % (ts_xs[i], t_ys[i], t_nn[i]))

    plot_x = [x for [[x]] in t_xs]
    plt.plot(plot_x, [y for [[y]] in t_ys], '.')
    plt.plot(plot_x, [y for [[y]] in t_nn], '.')
    plt.show()


def classification_1x1(lb, ub, d_size, learning_rate, layer_dimensions, original_func, out_interpretation):
    def hidden_act(x):
        return np.array([[max(0, xv) for xv in xs] for xs in x])

    def hidden_act_der(x):
        def der(x):
            return 1. if x > 0. else 0.

        return np.array([[der(xv) for xv in xs] for xs in x])

    def output_act(x):
        return x

    def output_act_der(x):
        return np.ones(x.shape)

    def loss(y_head, y):
        y_head = y_head.reshape((y_head.shape[0],))
        max_yh = max(y_head)
        exps = np.array([np.exp(y_h - max_yh) for y_h in y_head])
        return np.log(sum(exps)) - (y_head[y] - max_yh)

    def loss_der(y_head, y):
        y_head = y_head.reshape((y_head.shape[0],))
        max_yh = max(y_head)

        exps = np.array([np.exp(y_h - max_yh) for y_h in y_head])
        recip_sum = 1. / sum(exps)

        der = np.full(y_head.shape, recip_sum)
        for i in range(y_head.shape[0]):
            der[i] *= np.exp(y_head[i] - max_yh)
            if i == y:
                der[i] -= 1

        return der.reshape((y_head.shape[0], 1))

    nn = learn_classification_1x1(original_func, d_size, lb, ub, learning_rate, layer_dimensions, loss, loss_der, hidden_act, hidden_act_der, output_act, output_act_der, out_interpretation)
    test_classification_1x1(nn, original_func, 100, lb, ub, out_interpretation)


def learn_classification_1x1(original_func, d_size, d_lb, d_ub, learning_rate, layer_dimensions, loss, loss_der, hidden_act, hidden_act_der, output_act, output_act_der, out_interpretation):
    nn = get_nn_from_architecture(layer_dimensions, loss, loss_der, hidden_act, hidden_act_der, output_act, output_act_der, out_interpretation)

    (ds_xs, ds_ys) = get_data(original_func, d_lb, d_ub, (d_size, 1, 1))
    d_xs = np.array([np.array(x).reshape(1, 1) for x in ds_xs])
    d_ys = np.array([np.argmax(y) for y in ds_ys])

    new_nn = learn(nn, (d_xs, d_ys), learning_rate)
    return new_nn


def test_classification_1x1(nn, original_func, t_size, d_lb, d_ub, out_interpretation):
    (ts_xs, t_ys) = get_data(original_func, d_lb, d_ub, (t_size, 1, 1))
    ti_ys = np.array([out_interpretation(y) for y in t_ys])
    t_xs = np.array([x.reshape((1, 1)) for x in ts_xs])
    t_nn = np.array([nn.evaluate(x) for x in t_xs])

    for i in range(t_size):
        print("%s: %s - %s" % (ts_xs[i], ti_ys[i], t_nn[i]))

    plot_x = [x for [[x]] in t_xs]
    plt.plot(plot_x, ti_ys, '.')
    plt.plot(plot_x, t_nn, '.')
    plt.show()
