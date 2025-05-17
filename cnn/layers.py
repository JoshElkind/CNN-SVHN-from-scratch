import numpy as np
from typing import Tuple, Optional


def get_im2col_indices(x_shape: Tuple[int, int, int, int], field_height: int, field_width: int, padding: int, stride: int):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k, i, j)


def im2col_indices(x: np.ndarray, field_height: int, field_width: int, padding: int, stride: int) -> np.ndarray:
    N, C, H, W = x.shape
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C_k = C * field_height * field_width
    cols = cols.transpose(1, 2, 0).reshape(C_k, -1)
    return cols


def col2im_indices(cols: np.ndarray, x_shape: Tuple[int, int, int, int], field_height: int, field_width: int, padding: int, stride: int) -> np.ndarray:
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Layer:
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params_and_grads(self):
        return []


class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | Tuple[int, int], stride: int = 1, padding: int = 0, bias: bool = True):
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kh
        self.kernel_w = kw
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kh * kw
        scale = np.sqrt(2.0 / fan_in)
        self.W = np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32) * scale
        self.b = np.zeros(out_channels, dtype=np.float32) if bias else None
        self.x_shape: Optional[Tuple[int, int, int, int]] = None
        self.x_cols: Optional[np.ndarray] = None
        self.W_col: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        N, C, H, W = x.shape
        assert C == self.in_channels
        out_h = (H + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_w) // self.stride + 1
        x_cols = im2col_indices(x, self.kernel_h, self.kernel_w, self.padding, self.stride)
        W_col = self.W.reshape(self.out_channels, -1)
        out = (W_col @ x_cols).reshape(self.out_channels, N, out_h, out_w).transpose(1, 0, 2, 3)
        if self.b is not None:
            out += self.b.reshape(1, -1, 1, 1)
        self.x_shape = x.shape
        self.x_cols = x_cols
        self.W_col = W_col
        return out.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.x_cols is not None and self.W_col is not None and self.x_shape is not None
        N, _, out_h, out_w = grad_output.shape
        grad_out_reshaped = grad_output.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)
        dW_col = grad_out_reshaped @ self.x_cols.T
        dW = dW_col.reshape(self.W.shape)
        db = grad_out_reshaped.sum(axis=1) if self.b is not None else None
        dx_cols = self.W_col.T @ grad_out_reshaped
        dx = col2im_indices(dx_cols, self.x_shape, self.kernel_h, self.kernel_w, self.padding, self.stride)
        self.dW = dW.astype(np.float32)
        if self.b is not None:
            self.db = db.astype(np.float32)
        self.x_cols = None
        self.W_col = None
        self.x_shape = None
        return dx.astype(np.float32)

    def params_and_grads(self):
        params = [(self.W, self.dW)]
        if self.b is not None:
            params.append((self.b, self.db))
        return params


class MaxPool2D(Layer):
    def __init__(self, kernel_size: int | Tuple[int, int] = 2, stride: Optional[int] = None):
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        self.kernel_h = kh
        self.kernel_w = kw
        self.stride = stride if stride is not None else kernel_size if isinstance(kernel_size, int) else kh
        self.x: Optional[np.ndarray] = None
        self.argmax: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        N, C, H, W = x.shape
        out_h = (H - self.kernel_h) // self.stride + 1
        out_w = (W - self.kernel_w) // self.stride + 1
        x_reshaped = x.reshape(N * C, 1, H, W)
        x_cols = im2col_indices(x_reshaped, self.kernel_h, self.kernel_w, padding=0, stride=self.stride)
        x_cols = x_cols.reshape(self.kernel_h * self.kernel_w, -1)
        argmax = np.argmax(x_cols, axis=0)
        out = x_cols[argmax, np.arange(argmax.size)]
        out = out.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)
        self.x = x
        self.argmax = argmax
        self.out_shape = (N, C, out_h, out_w)
        self.cols_shape = x_cols.shape
        return out.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.x is not None and self.argmax is not None
        N, C, H, W = self.x.shape
        out_h = (H - self.kernel_h) // self.stride + 1
        out_w = (W - self.kernel_w) // self.stride + 1
        dcols = np.zeros((self.kernel_h * self.kernel_w, out_h * out_w * N * C), dtype=np.float32)
        dcols[self.argmax, np.arange(self.argmax.size)] = grad_output.transpose(2, 3, 0, 1).ravel()
        dx = col2im_indices(dcols, (N * C, 1, H, W), self.kernel_h, self.kernel_w, padding=0, stride=self.stride)
        dx = dx.reshape(self.x.shape)
        self.x = None
        self.argmax = None
        return dx.astype(np.float32)


class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)
        self.W = (np.random.randn(in_features, out_features).astype(np.float32) * scale)
        self.b = np.zeros(out_features, dtype=np.float32) if bias else None
        self.x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.x = x
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.x is not None
        dW = self.x.T @ grad_output
        db = grad_output.sum(axis=0) if self.b is not None else None
        dx = grad_output @ self.W.T
        self.dW = dW.astype(np.float32)
        if self.b is not None:
            self.db = db.astype(np.float32)
        self.x = None
        return dx.astype(np.float32)

    def params_and_grads(self):
        params = [(self.W, self.dW)]
        if self.b is not None:
            params.append((self.b, self.db))
        return params


class ReLU(Layer):
    def __init__(self):
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.mask = x > 0
        return (x * self.mask).astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.mask is not None
        dx = grad_output * self.mask
        self.mask = None
        return dx.astype(np.float32)


class BatchNorm(Layer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9):
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        self.track_running = False
        self.x_centered: Optional[np.ndarray] = None
        self.inv_std: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim == 4:
            N, C, H, W = x.shape
            x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            x_reshaped = x
        xr = x_reshaped.astype(np.float64, copy=False)
        if training:
            mean = xr.mean(axis=0)
            var = xr.var(axis=0)
            if self.track_running:
                self.running_mean = (self.momentum * self.running_mean + (1 - self.momentum) * mean).astype(np.float32)
                self.running_var = (self.momentum * self.running_var + (1 - self.momentum) * var).astype(np.float32)
        else:
            mean = self.running_mean.astype(np.float64)
            var = self.running_var.astype(np.float64)
        x_centered = xr - mean
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = x_centered * inv_std
        out = self.gamma.astype(np.float64) * x_hat + self.beta.astype(np.float64)
        if x.ndim == 4:
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        self.x_centered = x_centered
        self.inv_std = inv_std
        self.shape_cache = x.shape
        return out.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.x_centered is not None and self.inv_std is not None
        if grad_output.ndim == 4:
            N, C, H, W = grad_output.shape
            go = grad_output.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            go = grad_output
        m = go.shape[0]
        x_hat = self.x_centered * self.inv_std
        dgamma = np.sum(go.astype(np.float64) * x_hat, axis=0)
        dbeta = np.sum(go.astype(np.float64), axis=0)
        dxhat = go.astype(np.float64) * self.gamma.astype(np.float64)
        sum_dxhat = np.sum(dxhat, axis=0)
        sum_dxhat_xhat = np.sum(dxhat * x_hat, axis=0)
        dx = (1.0 / m) * self.inv_std * (m * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
        self.dgamma = dgamma.astype(np.float32)
        self.dbeta = dbeta.astype(np.float32)
        if grad_output.ndim == 4:
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        self.x_centered = None
        self.inv_std = None
        return dx.astype(np.float32)

    def params_and_grads(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)] 