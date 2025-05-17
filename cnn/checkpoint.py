import os
from typing import Any, Dict, List, Tuple
import numpy as np
from .model import Sequential
from .layers import Conv2D, Dense, BatchNorm


def _layer_params(layer) -> Dict[str, np.ndarray]:
    params: Dict[str, np.ndarray] = {}
    if isinstance(layer, (Conv2D, Dense)):
        params['W'] = layer.W
        if layer.b is not None:
            params['b'] = layer.b
    if isinstance(layer, BatchNorm):
        params['gamma'] = layer.gamma
        params['beta'] = layer.beta
        params['running_mean'] = layer.running_mean
        params['running_var'] = layer.running_var
    return params


def _set_layer_params(layer, params: Dict[str, np.ndarray]):
    if isinstance(layer, (Conv2D, Dense)):
        if 'W' in params:
            layer.W[...] = params['W']
        if 'b' in params and layer.b is not None:
            layer.b[...] = params['b']
    if isinstance(layer, BatchNorm):
        if 'gamma' in params:
            layer.gamma[...] = params['gamma']
        if 'beta' in params:
            layer.beta[...] = params['beta']
        if 'running_mean' in params:
            layer.running_mean[...] = params['running_mean']
        if 'running_var' in params:
            layer.running_var[...] = params['running_var']


def state_dict(features: Sequential, classifier: Sequential, extra: Dict[str, np.ndarray] | None = None) -> Dict[str, np.ndarray]:
    state: Dict[str, np.ndarray] = {}
    for i, layer in enumerate(features.layers):
        lp = _layer_params(layer)
        for k, v in lp.items():
            state[f'features.{i}.{k}'] = v.astype(np.float32)
    for i, layer in enumerate(classifier.layers):
        lp = _layer_params(layer)
        for k, v in lp.items():
            state[f'classifier.{i}.{k}'] = v.astype(np.float32)
    if extra:
        for k, v in extra.items():
            state[k] = v.astype(np.float32)
    return state


def load_state(features: Sequential, classifier: Sequential, state: Dict[str, np.ndarray]):
    # Group by layer index
    feat_groups: Dict[int, Dict[str, np.ndarray]] = {}
    cls_groups: Dict[int, Dict[str, np.ndarray]] = {}
    for k, v in state.items():
        if k.startswith('features.'):
            _, idx, name = k.split('.', 2)
            feat_groups.setdefault(int(idx), {})[name] = v
        elif k.startswith('classifier.'):
            _, idx, name = k.split('.', 2)
            cls_groups.setdefault(int(idx), {})[name] = v
    for i, layer in enumerate(features.layers):
        if i in feat_groups:
            _set_layer_params(layer, feat_groups[i])
    for i, layer in enumerate(classifier.layers):
        if i in cls_groups:
            _set_layer_params(layer, cls_groups[i])


def save_checkpoint(features: Sequential, classifier: Sequential, path: str, extra: Dict[str, np.ndarray] | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = state_dict(features, classifier, extra)
    np.savez_compressed(path, **sd)


def load_checkpoint(features: Sequential, classifier: Sequential, path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    state = {k: data[k] for k in data.files}
    load_state(features, classifier, state)
    return state 