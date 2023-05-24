import math

import numpy as np


def sigmoid(val: np.ndarray):
    return 1.0 / (1.0 + math.exp(-val))


def sigmoid_derivative(val: np.ndarray):
    return val * (1.0 - val)


def relu(val: np.ndarray):
    return np.maximum(0.0, val)


def relu_derivative(val: np.ndarray):
    return (val > 0) * 1.0


def leaky_relu_wrap(alpha: float):
    def leaky_relu(val: np.ndarray):
        return np.where(val >= 0, val, alpha * val)
    return leaky_relu


def leaky_relu_derivative_wrap(alpha: float):
    def leaky_relu_derivative(val: np.ndarray):
        return np.where(val >= 0, 1, alpha)
    return leaky_relu_derivative


def softmax(val):
    exp_val = np.exp(val - np.max(val, axis=-1, keepdims=True))
    return exp_val / np.sum(exp_val, axis=-1, keepdims=True)


def softmax_derivative(val: np.ndarray):
    x = softmax(val)
    return x * (1 - x)
