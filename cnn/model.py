from typing import List, Iterable, Tuple
import numpy as np
from .layers import Layer


class Sequential:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=self.training)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def params_and_grads(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        for layer in self.layers:
            for param, grad in layer.params_and_grads():
                yield param, grad 