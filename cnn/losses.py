import numpy as np
from typing import Tuple


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), targets] + 1e-12).mean()
    grad = probs
    grad[np.arange(N), targets] -= 1.0
    grad /= N
    return float(loss), grad.astype(np.float32) 