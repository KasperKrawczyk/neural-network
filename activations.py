import math

import numpy as np


def sigmoid(val):
    return 1.0 / (1.0 + math.exp(-val))


def sigmoid_derivative(val: np.ndarray):
    return val * (1.0 - val)


def relu(val: np.ndarray):
    return np.maximum(0.0, val)


def relu_derivative(val: np.ndarray):
    return (val > 0) * 1.0


def softmax(val: np.ndarray):
    val = val - np.max(val)
    return np.exp(val) / np.sum(np.exp(val))


def softmax_derivative(cur_out: np.ndarray):
    exps = np.exp(cur_out - cur_out.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
