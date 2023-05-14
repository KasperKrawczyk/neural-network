import numpy as np
import pandas as pd

from activations import sigmoid, sigmoid_derivative, relu, softmax, relu_derivative, softmax_derivative


class Network:
    network = list()
    n_iters: int = 0
    learn_rate: float = 0
    input: np.ndarray = None

    def __init__(self,
                 layer_list: list[(int, str)],
                 learn_rate: float = 5e-3,
                 n_iters: int = 50,
                 class_ind: str = 'first'):
        self.cur_out: np.ndarray = None
        self.cur_err: np.ndarray = None
        self.layer_list = layer_list
        self.num_hidden = len(layer_list) - 2
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.class_ind = class_ind
        self._initialise()

    def _initialise(self):
        for i in range(len(self.layer_list)):
            if i > 0:
                dimensions, activation = self.layer_list[i]
                self.network.append(Layer(dimensions, self.layer_list[i - 1][0], activation))

    def fit(self):
        for iteration in range(self.n_iters):
            iter_err = 0.0
            for input_row_index, input_row in enumerate(self.input):
                cur_in = input_row[1:] if self.class_ind == 'first' else input_row[:-1]
                exp_vec = self._get_exp_vec(input_row)
                self.cur_out = self.fwd_prop(cur_in)
                iter_err += np.sum((np.power((exp_vec - self.cur_out), 2)))
                self.bk_prop(exp_vec)
                self.update_w(cur_in)
            print('iteration={}, learn_rate={}, iter_err={}'.format(iteration, self.learn_rate, iter_err))

    def predict(self, test_set: np.ndarray):
        for input_row_index, input_row in enumerate(test_set):
            cur_in = input_row[1:] if self.class_ind == 'first' else input_row[:-1]
            exp_class = input_row[0] if self.class_ind == 'first' else input_row[-1]
            cur_out = self.fwd_prop(cur_in)
            print('actual={}, predicted={}'.format(exp_class, cur_out.argmax()))

    def _get_exp_vec(self, input_row: np.ndarray):
        exp_class = int(input_row[0]) if self.class_ind == 'first' else input_row[-1]
        exp_vec = np.zeros(self.layer_list[-1][0])
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
            layer.bk_prop()

    def update_w(self, cur_row: np.ndarray):
        for i in range(len(self.network)):
            cur_layer = self.network[i]
            if i != 0:
                ins = self.network[i - 1].cur_out
            else:
                ins = cur_row
            cur_layer.w[:, :-1] -= self.learn_rate * cur_layer.cur_delta * ins
            cur_layer.w[:, -1:] -= self.learn_rate * cur_layer.cur_delta


class Layer:

    def __init__(self, num_neurons: int, prev_layer_num_neurons: int, transfer_func_descr: str):
        self.num_neurons = num_neurons
        self.prev_layer_num_neurons = prev_layer_num_neurons
        self.w = np.random.rand(num_neurons, prev_layer_num_neurons + 1)
        self.cur_out: np.ndarray = None
        self.cur_delta: np.ndarray = None
        self.cur_err: np.ndarray = None
        self.dim = self.w.shape
        self.transfer_func_descr = transfer_func_descr
        self._set_activation_func()

    def _set_activation_func(self):
        if self.transfer_func_descr == 'relu':
            self._vec_transfer = np.vectorize(relu)
            self._vec_transfer_derivative = np.vectorize(relu_derivative)
        if self.transfer_func_descr == 'sigmoid':
            self._vec_transfer = np.vectorize(sigmoid)
            self._vec_transfer_derivative = np.vectorize(sigmoid_derivative)
        if self.transfer_func_descr == 'tanh':
            pass
        if self.transfer_func_descr == 'softmax':
            self._vec_transfer = softmax
            self._vec_transfer_derivative = softmax_derivative


    def get_cur_err(self):
        self.cur_err = np.dot(self.w.T, self.cur_delta)
        return self.cur_err

    def _activate(self, ins: np.ndarray):
        return (np.dot(self.w[:, :-1], ins) + self.w[:, -1:].T)[0]

    def _transfer(self, mat: np.ndarray):
        return self._vec_transfer(mat)

    def _transfer_derivative(self, mat: np.ndarray):
        return self._vec_transfer_derivative(mat)

    def fwd_prop(self, ins: np.ndarray):
        act = self._activate(ins)
        self.cur_out = self._transfer(act)
        return self.cur_out

    def bk_prop(self):
        if self.transfer_func_descr == 'softmax':
            self.cur_delta = self._vec_transfer_derivative(self.cur_err, self.cur_out)
            self.cur_delta = self.cur_delta.reshape(-1, 1)
        else:
            self.cur_delta = self.cur_err * self._vec_transfer_derivative(self.cur_out)
            self.cur_delta = self.cur_delta.reshape(-1, 1)


if __name__ == '__main__':
    n = Network(layer_list=[(784, ''), (100, 'relu'), (30, 'relu'), (10, 'softmax')])

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
    # inputs = utils.make_blobs(n=100)
    # inputs = np.genfromtxt('data/mnist/mnist_train.csv', delimiter=',')
    df_train = pd.read_csv('data/mnist/mnist_train.csv', sep=',', nrows=100)
    df_test = pd.read_csv('data/mnist/mnist_test.csv', sep=',', nrows=100)
    df_train.iloc[1:, 1:] = df_train.iloc[1:, 1:].astype(np.float32)
    df_train.iloc[1:, 1:] = df_train.iloc[1:, 1:] / 255.0
    df_test.iloc[1:, 1:] = df_test.iloc[1:, 1:].astype(np.float32)
    df_test.iloc[1:, 1:] = df_test.iloc[1:, 1:] / 255.0
    n.input = df_train.values
    n.fit()
    n.predict(df_test.values)
