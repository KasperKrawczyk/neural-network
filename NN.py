import math
import numpy as np
import utils


class Network:
    network = list()
    n_iters: int = 0
    learn_rate: float = 0
    input: np.ndarray = None

    def __init__(self, dimensions: list[int], learn_rate: float = 0.5, n_iters: int = 40):
        self.cur_out: np.ndarray = None
        self.cur_err: np.ndarray = None
        self.dimensions = dimensions
        self.num_hidden = len(dimensions) - 2
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self._initialise()

    def _initialise(self):
        for i in range(len(self.dimensions)):
            if i > 0:
                self.network.append(Layer(self.dimensions[i], self.dimensions[i - 1]))

    def fit(self):
        for iteration in range(self.n_iters):
            iter_err = 0.0
            for input_row_index, input_row in enumerate(self.input):
                cur_in = input_row[:-1]
                exp_vec = self._get_exp_vec(input_row)
                self.cur_out = self.fwd_prop(cur_in)
                iter_err += np.sum((np.power((exp_vec - self.cur_out), 2)))
                self.bk_prop(exp_vec)
                self.update_w(cur_in)
            print('iteration={}, learn_rate={}, iter_err={}'.format(iteration, self.learn_rate, iter_err))

    def _get_exp_vec(self, input_row: np.ndarray):
        exp_class = int(input_row[-1])
        exp_vec = np.zeros(self.dimensions[-1])
        exp_vec[exp_class] = 1
        return exp_vec

    def fwd_prop(self, cur_in: np.ndarray):
        cur_out = cur_in
        for layer in self.network:
            cur_out = layer.fwd_prop(cur_out)
        return cur_out

    def bk_prop(self, expected: np.ndarray):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            if i != len(self.network) - 1:
                prev_layer = self.network[i + 1]
                layer.cur_err = np.dot(prev_layer.cur_delta.T, prev_layer.w[:, :-1])
            else:
                layer.cur_err = layer.cur_out - expected
            layer.cur_delta = layer.cur_err * activation_derivative(layer.cur_out)
            layer.cur_delta = layer.cur_delta.reshape(-1, 1)

    def update_w(self, cur_row: np.ndarray):
        for i in range(len(self.network)):
            cur_layer = self.network[i]
            if i != 0:
                ins = self.network[i - 1].cur_out
            else:
                ins = cur_row
            cur_layer.w[:, :-1] -= self.learn_rate * cur_layer.cur_delta * ins
            cur_layer.w[:, -1:] -= self.learn_rate * cur_layer.cur_delta


def activation(val):
    return 1.0 / (1.0 + math.exp(-val))


def activation_derivative(outs: np.ndarray):
    return outs * (1.0 - outs)


class Layer:

    def __init__(self, num_neurons: int, prev_layer_num_neurons: int):
        self.num_neurons = num_neurons
        self.prev_layer_num_neurons = prev_layer_num_neurons
        self.w = np.random.rand(num_neurons, prev_layer_num_neurons + 1)
        self.cur_out: np.ndarray = None
        self.cur_delta: np.ndarray = None
        self.cur_err: np.ndarray = None
        self.dim = self.w.shape
        self._vec_transfer = np.vectorize(activation)

    def get_cur_err(self):
        self.cur_err = np.dot(self.w.T, self.cur_delta)
        return self.cur_err

    def _activate(self, ins: np.ndarray):
        return (np.dot(self.w[:, :-1], ins) + self.w[:, -1:].T)[0]

    def _transfer(self, mat: np.ndarray):
        return self._vec_transfer(mat)

    def fwd_prop(self, ins: np.ndarray):
        act = self._activate(ins)
        self.cur_out = self._transfer(act)
        return self.cur_out


if __name__ == '__main__':
    n = Network(dimensions=[2, 5, 7, 2], n_iters=100, learn_rate=0.5)

    # inputs = np.array([[2.7810836,2.550537003,0],
    #                    [1.465489372,2.362125076,0],
    #                    [3.396561688,4.400293529,0],
    #                    [1.38807019,1.850220317,0],
    #                    [3.06407232,3.005305973,0],
    #                    [7.627531214,2.759262235,1],
    #                    [5.332441248,2.088626775,1],
    #                    [6.922596716,1.77106367,1],
    #                    [8.675418651,-0.242068655,1],
    #                    [7.673756466,3.508563011,1]])
    inputs = utils.make_blobs(n=100)

    n.input = inputs
    n.fit()
