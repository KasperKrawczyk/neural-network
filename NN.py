import math
import numpy as np


class Network:
    network = list()
    input: list[float] = None

    def __init__(self, dimensions: list[int]):
        self.cur_out: np.ndarray = None
        self.cur_err: np.ndarray = None
        self.dimensions = dimensions
        self.num_hidden = len(dimensions) - 2
        self._initialise()

    def _initialise(self):
        for i in range(len(self.dimensions)):
            if i > 0:
                self.network.append(Layer(self.dimensions[i], self.dimensions[i - 1]))
                
    def run(self):
        self.cur_out = self.fwd_prop()
        self.cur_err = self.get_error()

    def fwd_prop(self):
        cur_out = self.input
        for layer in self.network:
            cur_out = layer.fwd_prop(cur_out)
        return cur_out

    def get_error(self, y: np.ndarray):
        (self.cur_out - y) * transfer_derivative()


def transfer(val):
    return 1.0 / (1.0 + math.exp(-val))

def transfer_derivative(outs: np.ndarray):
    return outs * (1.0 - outs)

class Layer:

    def __init__(self, num_neurons: int, prev_layer_num_neurons: int):
        self.num_neurons = num_neurons
        self.prev_layer_num_neurons = prev_layer_num_neurons
        self.mat = np.random.rand(num_neurons, prev_layer_num_neurons + 1)
        self.dim = self.mat.shape
        self._vec_transfer = np.vectorize(transfer)

    def _activate(self, ins: np.ndarray):
        return (np.dot(self.mat[:, :-1], ins) + self.mat[:, -1:].T)[0]

    def _transfer(self, mat: np.ndarray):
        return self._vec_transfer(mat)

    def fwd_prop(self, ins: np.ndarray):
        activation = self._activate(ins)
        return self._transfer(activation)

if __name__ == '__main__':
    n = Network(dimensions=[3, 5, 5, 2])

    inputs = np.array([2, 2, 2])
    n.input = inputs
    n.fwd_prop()



