from typing import Iterable, Tuple
import numpy as np


class Optimizer:
    def step(self, params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray]]):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self, params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray]]):
        for i, (p, g) in enumerate(params_and_grads):
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p
            if self.momentum != 0.0:
                v = self.velocities.get(i)
                if v is None:
                    v = np.zeros_like(p)
                v = 0.9 * v + (1 - 0.9) * g
                self.velocities[i] = v
                p -= self.lr * v
            else:
                p -= self.lr * g


class Adam(Optimizer):
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray]]):
        self.t += 1
        for i, (p, g) in enumerate(params_and_grads):
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p
            m = self.m.get(i)
            v = self.v.get(i)
            if m is None:
                m = np.zeros_like(p)
                v = np.zeros_like(p)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.m[i] = m
            self.v[i] = v 