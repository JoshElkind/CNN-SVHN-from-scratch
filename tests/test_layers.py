import numpy as np
import pytest

from cnn.layers import Dense, ReLU, Conv2D, MaxPool2D, BatchNorm


def rel_error(a, b):
    return np.max(np.abs(a - b) / (np.maximum(1e-6, np.abs(a) + np.abs(b))))


def numeric_grad(f, x, eps=1e-3):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fx1 = f(x)
        x[idx] = old - eps
        fx2 = f(x)
        x[idx] = old
        grad[idx] = (fx1 - fx2) / (2 * eps)
        it.iternext()
    return grad


def test_dense_gradients():
    rng = np.random.default_rng(0)
    N, D, M = 4, 5, 3
    x = rng.standard_normal((N, D)).astype(np.float32)
    layer = Dense(D, M)
    upstream = rng.standard_normal((N, M)).astype(np.float32)

    # forward
    out = layer.forward(x)

    def loss_wrt_x(x_in):
        out = layer.forward(x_in)
        return float(np.sum(out * upstream))

    def loss_wrt_W(W_in):
        old = layer.W.copy()
        layer.W = W_in
        out = layer.forward(x)
        val = float(np.sum(out * upstream))
        layer.W = old
        return val

    def loss_wrt_b(b_in):
        old = layer.b.copy()
        layer.b = b_in
        out = layer.forward(x)
        val = float(np.sum(out * upstream))
        layer.b = old
        return val

    dx = layer.backward(upstream)

    num_dx = numeric_grad(loss_wrt_x, x.copy())
    assert rel_error(dx, num_dx) < 5e-2

    num_dW = numeric_grad(loss_wrt_W, layer.W.copy())
    assert rel_error(layer.dW, num_dW) < 5e-2

    if layer.b is not None:
        num_db = numeric_grad(loss_wrt_b, layer.b.copy())
        assert rel_error(layer.db, num_db) < 5e-2


def test_relu_gradients():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((3, 4)) * 0.5).astype(np.float32)
    layer = ReLU()
    upstream = rng.standard_normal((3, 4)).astype(np.float32)

    def loss_wrt_x(x_in):
        y = np.maximum(0, x_in)
        return float(np.sum(y * upstream))

    y = layer.forward(x)
    dx = layer.backward(upstream)
    num_dx = numeric_grad(loss_wrt_x, x.copy())
    assert rel_error(dx, num_dx) < 5e-2


def test_conv2d_gradients():
    rng = np.random.default_rng(2)
    N, C, H, W = 2, 3, 5, 5
    F, KH, KW = 4, 3, 3
    x = rng.standard_normal((N, C, H, W)).astype(np.float32)
    layer = Conv2D(C, F, (KH, KW), stride=1, padding=1)
    upstream = rng.standard_normal(layer.forward(x).shape).astype(np.float32)

    def loss_wrt_x(x_in):
        out = layer.forward(x_in)
        return float(np.sum(out * upstream))

    def loss_wrt_W(W_in):
        old = layer.W.copy()
        layer.W = W_in
        out = layer.forward(x)
        val = float(np.sum(out * upstream))
        layer.W = old
        return val

    def loss_wrt_b(b_in):
        old = layer.b.copy()
        layer.b = b_in
        out = layer.forward(x)
        val = float(np.sum(out * upstream))
        layer.b = old
        return val

    dx = layer.backward(upstream)
    num_dx = numeric_grad(loss_wrt_x, x.copy())
    assert np.allclose(dx, num_dx, rtol=1e-2, atol=1e-2)

    num_dW = numeric_grad(loss_wrt_W, layer.W.copy())
    assert rel_error(layer.dW, num_dW) < 8e-2

    if layer.b is not None:
        num_db = numeric_grad(loss_wrt_b, layer.b.copy())
        assert rel_error(layer.db, num_db) < 8e-2


def test_maxpool2d_backward_shapes():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
    pool = MaxPool2D(2, 2)
    out = pool.forward(x)
    upstream = np.ones_like(out)
    dx = pool.backward(upstream)
    assert dx.shape == x.shape


def test_batchnorm_gradients():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 8, 6, 6)).astype(np.float32)
    bn = BatchNorm(8)
    upstream = rng.standard_normal(x.shape).astype(np.float32)

    def loss_wrt_x(x_in):
        out = bn.forward(x_in, training=True)
        return float(np.sum(out * upstream))

    # cache by forward call once
    _ = bn.forward(x, training=True)
    dx = bn.backward(upstream)
    num_dx = numeric_grad(loss_wrt_x, x.copy())
    assert np.allclose(dx, num_dx, rtol=2e-2, atol=2e-2)  # BN numeric grad can be noisy 