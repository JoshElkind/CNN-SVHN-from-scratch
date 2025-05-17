import os
import tarfile
import tempfile
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator

import numpy as np
import imageio.v2 as imageio
import requests

SVHN_TRAIN_URL = "https://ufldl.stanford.edu/housenumbers/train.tar.gz"
SVHN_TEST_URL = "https://ufldl.stanford.edu/housenumbers/test.tar.gz"

SVHN_TRAIN_URL_ALT = "http://ufldl.stanford.edu/housenumbers/train.tar.gz"
SVHN_TEST_URL_ALT = "http://ufldl.stanford.edu/housenumbers/test.tar.gz"


@dataclass
class SVHNConfig:
    root: str
    download: bool = True
    normalize: bool = True
    augment: bool = True
    rng_seed: int = 42


def _download(url: str, dest: str):
    print(f"Downloading from {url}...")
    
    urls_to_try = [url]
    if 'https://' in url:
        urls_to_try.append(url.replace('https://', 'http://'))
    elif 'http://' in url:
        urls_to_try.append(url.replace('http://', 'https://'))
    
    for try_url in urls_to_try:
        try:
            print(f"Trying {try_url}...")
            with requests.get(try_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownload progress: {percent:.1f}%", end='', flush=True)
                print(f"\nDownload completed: {downloaded} bytes")
                return
        except Exception as e:
            print(f"Failed to download from {try_url}: {e}")
            continue
    
    raise Exception(f"Failed to download from all URLs: {urls_to_try}")


def _extract_tar_gz(path: str, dest_dir: str):
    with tarfile.open(path, 'r:gz') as tar:
        tar.extractall(dest_dir)


def _discover_pngs_and_labels(folder: str) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith('.png'):
                continue
            fpath = os.path.join(root, name)
            parent = os.path.basename(root)
            label = None
            if parent.isdigit():
                label = int(parent)
            else:
                for ch in name:
                    if ch.isdigit():
                        label = int(ch)
                        break
            if label is None:
                continue
            img = imageio.imread(fpath)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            xs.append(img)
            ys.append(label)
    if not xs:
        raise FileNotFoundError("No PNGs discovered in extracted SVHN folder")
    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return X, y


def _maybe_download_and_prepare(root: str, split: str, download: bool) -> Tuple[np.ndarray, np.ndarray]:
    os.makedirs(root, exist_ok=True)
    split_dir = os.path.join(root, f"svhn_{split}")
    if os.path.isdir(split_dir):
        try:
            return _discover_pngs_and_labels(split_dir)
        except Exception:
            pass
    if not download:
        raise FileNotFoundError(f"SVHN {split} not found in {split_dir} and download=False")
    url = SVHN_TRAIN_URL if split == 'train' else SVHN_TEST_URL
    with tempfile.TemporaryDirectory() as td:
        archive = os.path.join(td, f"{split}.tar.gz")
        _download(url, archive)
        _extract_tar_gz(archive, td)
        png_root = td
        X, y = _discover_pngs_and_labels(png_root)
    os.makedirs(split_dir, exist_ok=True)
    np.savez_compressed(os.path.join(root, f"svhn_{split}.npz"), X=X, y=y)
    return X, y


def _normalize_per_channel(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.reshape(-1, 3).mean(axis=0)
    std = X.reshape(-1, 3).std(axis=0) + 1e-6
    Xn = (X - mean) / std
    return Xn.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def random_crop_and_flip(X: np.ndarray, out_size: int = 32, pad: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    N, H, W, C = X.shape
    padded = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    H_p, W_p = padded.shape[1:3]
    ys = rng.integers(0, H_p - out_size + 1, size=N)
    xs = rng.integers(0, W_p - out_size + 1, size=N)
    out = np.empty((N, out_size, out_size, C), dtype=X.dtype)
    for i in range(N):
        out[i] = padded[i, ys[i]:ys[i]+out_size, xs[i]:xs[i]+out_size, :]
        if rng.random() < 0.5:
            out[i] = out[i, :, ::-1, :]
    return out


def jitter_brightness_contrast(X: np.ndarray, strength: float = 0.2, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    N = X.shape[0]
    out = X.copy()
    for i in range(N):
        b = (rng.random() * 2 - 1) * strength
        c = 1.0 + (rng.random() * 2 - 1) * strength
        out[i] = (out[i] * c + b).astype(np.float32)
    return out


class SVHNDataset:
    def __init__(self, config: SVHNConfig, split: str = 'train'):
        self.config = config
        self.split = split
        self.rng = np.random.default_rng(config.rng_seed)
        npz_path = os.path.join(config.root, f"svhn_{split}.npz")
        if os.path.isfile(npz_path):
            data = np.load(npz_path)
            X, y = data['X'].astype(np.float32), data['y'].astype(np.int64)
        else:
            try:
                X, y = _maybe_download_and_prepare(config.root, split, config.download)
            except Exception as e:
                print(f"Failed to download real SVHN data: {e}")
                print("Using better synthetic data instead...")
                if split == 'train':
                    n = 10000
                else:
                    n = 2000
                from create_better_data import create_dataset
                X, y = create_dataset(n, split)
        y = y % 10
        if config.normalize:
            X, self.mean, self.std = _normalize_per_channel(X)
        else:
            self.mean = np.zeros(3, dtype=np.float32)
            self.std = np.ones(3, dtype=np.float32)
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def get_batches(self, batch_size: int, shuffle: bool = True, augment: Optional[bool] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if augment is None:
            augment = self.config.augment and self.split == 'train'
        idx = np.arange(len(self))
        if shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(self), batch_size):
            batch_idx = idx[start:start+batch_size]
            xb = self.X[batch_idx]
            yb = self.y[batch_idx]
            if augment:
                xb = random_crop_and_flip(xb, out_size=32, pad=2, rng=self.rng)
                xb = jitter_brightness_contrast(xb, strength=0.2, rng=self.rng)
            xb = xb.transpose(0, 3, 1, 2).astype(np.float32)
            yield xb, yb 