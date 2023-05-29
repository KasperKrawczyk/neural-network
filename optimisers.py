import numpy as np


class Optimiser:
    def __init__(self, name: str, learn_rate: float):
        self.name = name
        self.learn_rate = learn_rate

    def update(self, w: np.ndarray, b: np.ndarray, gradient: np.ndarray):
        pass


class Adam(Optimiser):

    def __init__(self, learn_rate: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 e: float = 1e-15,
                 is_weight_decay: bool = False,
                 initial_gamma: float = 1e-5,
                 weight_decay_rate: float = 0.8,
                 demon: bool = False):
        super().__init__("Adam", learn_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = e
        self.is_weight_decay = is_weight_decay
        if is_weight_decay:
            self.initial_gamma = initial_gamma
            self.weight_decay_rate = weight_decay_rate
        self.demon = demon
