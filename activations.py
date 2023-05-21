import math

import numpy as np


def sigmoid(val):
    return 1.0 / (1.0 + math.exp(-val))


def sigmoid_derivative(outs: np.ndarray):
    return outs * (1.0 - outs)


def relu(val):
    return max(0.0, val)


def relu_derivative(val):
    if val < 0:
        return 0
    else:
        return 1


def softmax(val):
    val = val - np.max(val)
    return np.exp(val) / np.sum(np.exp(val))


def softmax_derivative(cur_out: np.ndarray):
    exps = np.exp(cur_out - cur_out.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
