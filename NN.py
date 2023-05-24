import numpy as np
import pandas as pd

from activations import *


class Network:
    network = list()
    n_iters: int = 0
    learn_rate: float = 0

    def __init__(self,
                 inputs: np.ndarray,
                 input_classes: np.ndarray,
                 layer_list: list[(int, str)],
                 learn_rate: float = 5e-3,
                 batch_size: int = 100,
                 n_iters: int = 10):
        self.cur_out: np.ndarray = None
        self.cur_err: np.ndarray = None
        self.layer_list = layer_list
        self.num_hidden = len(layer_list) - 2
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.n_iters = n_iters
        self._initialise()
        self.input = inputs
        self.input_classes = self._to_one_hot(input_classes)

    def _initialise(self):
        for i in range(len(self.layer_list)):
            if i > 0:
                dimensions, activation = self.layer_list[i]
                self.network.append(Layer(dimensions, self.layer_list[i - 1][0], activation))

    def _batch_generator(self):
        num_samples = self.input.shape[0]
        for i in np.arange(0, num_samples, self.batch_size):
            start, end = i, min(i + self.batch_size, num_samples)
            yield self.input[start:end], self.input_classes[start:end]

    def _loss_gradient(self, exp_vec: np.ndarray):
        clipped = np.clip(self.cur_out, 1e-15, 1 - 1e-15)
        return -(exp_vec / clipped) + (1 - exp_vec) / (1 - clipped)

    def _acc(self, input_classes_batch, predicted_classes_batch):
        return np.sum(input_classes_batch == predicted_classes_batch, axis=0) / len(input_classes_batch)

    def fit(self):
        predictions = []
        for iteration in range(self.n_iters):
            for input_batch, input_classes_batch in self._batch_generator():
                exp_vec = input_classes_batch
                self.cur_out = self.fwd_prop(input_batch)
                predictions.append(self._acc(np.argmax(self.cur_out, axis=1), np.argmax(exp_vec, axis=1)))
                error = self._loss_gradient(input_classes_batch)
                self.bk_prop(error)
            acc = np.mean(predictions) * 100
            print('iteration={0}, learn_rate={1}, acc={2:.2f}%'
                  .format(iteration, self.learn_rate, acc))

    def predict(self, test_set: np.ndarray, test_classes: np.ndarray):
        for input_row_index, input_row in enumerate(test_set):
            cur_in = input_row.reshape(1, input_row.size)
            exp_class = test_classes[input_row_index]
            cur_out = self.fwd_prop(cur_in)
            print('actual={}, predicted={}'.format(exp_class, cur_out.argmax()))

    def _to_one_hot(self, input_classes: np.ndarray):
        n_col = np.amax(input_classes) + 1
        one_hot = np.zeros((input_classes.shape[0], n_col))
        one_hot[np.arange(input_classes.shape[0]), input_classes] = 1
        return one_hot

    def fwd_prop(self, cur_in: np.ndarray):
        cur_out = cur_in
        for layer in self.network:
            cur_out = layer.fwd_prop(cur_out)
        return cur_out

    def bk_prop(self, output_err: np.ndarray):
        err = output_err
        for layer in reversed(self.network):
            err = layer.bk_prop(err, self.learn_rate)


class Layer:

    def __init__(self, num_neurons: int, prev_layer_num_neurons: int, transfer_func_descr: str):
        self.num_neurons = num_neurons
        self.prev_layer_num_neurons = prev_layer_num_neurons
        lim = 1 / np.sqrt(prev_layer_num_neurons)
        self.w = np.random.uniform(-lim, lim, (prev_layer_num_neurons, num_neurons))
        self.b = np.zeros((1, num_neurons))
        self.act: np.ndarray = None
        self.delta: np.ndarray = None
        self.input_err: np.ndarray = None
        self.cur_out: np.ndarray = None
        self.cur_in: np.ndarray = None
        self.transfer_func_descr = transfer_func_descr
        self._set_transfer_func()

    def _set_transfer_func(self):
        if self.transfer_func_descr == 'relu':
            self._vec_transfer = relu
            self._vec_transfer_derivative = relu_derivative
        if self.transfer_func_descr == 'leaky_relu':
            self._vec_transfer = leaky_relu_wrap(0.2)
            self._vec_transfer_derivative = leaky_relu_derivative_wrap(0.2)
        if self.transfer_func_descr == 'sigmoid':
            self._vec_transfer = sigmoid
            self._vec_transfer_derivative = sigmoid_derivative
        if self.transfer_func_descr == 'tanh':
            pass
        if self.transfer_func_descr == 'softmax':
            self._vec_transfer = softmax
            self._vec_transfer_derivative = softmax_derivative

    def fwd_prop(self, cur_in: np.ndarray):
        self.cur_in = cur_in
        self.act = np.dot(cur_in, self.w) + self.b
        self.cur_out = self._vec_transfer(self.act)
        return self.cur_out

    def bk_prop(self, output_err: np.ndarray, learn_rate: float):
        gradient = self._vec_transfer_derivative(self.cur_out) * output_err
        self.input_err = np.dot(gradient, self.w.T)
        self.delta = np.dot(self.cur_in.T, gradient)

        self.w -= learn_rate * self.delta
        self.b -= learn_rate * np.mean(gradient)

        return self.input_err


if __name__ == '__main__':

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
    df_train = pd.read_csv('data/mnist/mnist_train.csv', sep=',', nrows=1500)
    df_test = pd.read_csv('data/mnist/mnist_test.csv', sep=',', nrows=100)
    train_norm = df_train.values[:, 1:].astype(np.float32)
    train_norm = train_norm / 255.0
    test_norm = df_test.values[:, 1:].astype(np.float32)
    test_norm = test_norm / 255.0

    n = Network(train_norm, df_train.values[:, 0], layer_list=[(784, ''), (256, 'leaky_relu'), (128, 'leaky_relu'), (10, 'softmax')])

    n.fit()
    n.predict(test_norm, df_test.values[:, 0])
