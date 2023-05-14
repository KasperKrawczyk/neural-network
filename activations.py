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


def softmax_derivative(cur_err, cur_out):
    soft_max = np.reshape(cur_out, (1, -1))
    grad = np.reshape(cur_err, (1, -1))

    derivative_softmax = (soft_max * np.identity(soft_max.size) - soft_max.transpose() @ soft_max)
    return (grad @ derivative_softmax).ravel()
