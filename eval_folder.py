import os
import sys
import argparse
import numpy as np
import imageio.v2 as imageio

from cnn.layers import Conv2D, MaxPool2D, Dense, ReLU, BatchNorm
from cnn.model import Sequential
from cnn.checkpoint import load_checkpoint


def build_model():
    feat = Sequential([
        Conv2D(3, 32, 3, stride=1, padding=1),
        BatchNorm(32),
        ReLU(),
        Conv2D(32, 32, 3, stride=1, padding=1),
        BatchNorm(32),
        ReLU(),
        MaxPool2D(2, 2),
        
        Conv2D(32, 64, 3, stride=1, padding=1),
        BatchNorm(64),
        ReLU(),
        Conv2D(64, 64, 3, stride=1, padding=1),
        BatchNorm(64),
        ReLU(),
        MaxPool2D(2, 2),
        
        Conv2D(64, 128, 3, stride=1, padding=1),
        BatchNorm(128),
        ReLU(),
        Conv2D(128, 128, 3, stride=1, padding=1),
        BatchNorm(128),
        ReLU(),
        MaxPool2D(2, 2),
    ])
    clf = Sequential([
        Dense(128 * 4 * 4, 256),
        ReLU(),
        Dense(256, 128),
        ReLU(),
        Dense(128, 10),
    ])
    return feat, clf


def preprocess(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img.astype(np.float32)
    img = (img - mean) / (std + 1e-6)
    img = img.transpose(2, 0, 1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='test-data', help='Folder with test PNG images')
    parser.add_argument('--ckpt', default='checkpoints/best_model.npz', help='Path to checkpoint npz')
    args = parser.parse_args()

    feat, clf = build_model()
    state = load_checkpoint(feat, clf, args.ckpt)
    mean = state.get('mean', np.zeros(3, dtype=np.float32))
    std = state.get('std', np.ones(3, dtype=np.float32))

    files = [f for f in sorted(os.listdir(args.folder)) if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
    if not files:
        print(f'No images found in {args.folder}. Put PNG/JPG files there.')
        return

    images = []
    for name in files:
        path = os.path.join(args.folder, name)
        img = imageio.imread(path)
        img = preprocess(img, mean, std)
        images.append(img)
    X = np.stack(images).astype(np.float32)

    feat.eval(); clf.eval()
    feats = feat.forward(X)
    N = feats.shape[0]
    logits = clf.forward(feats.reshape(N, -1))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    print(f"Predictions for {len(files)} images:")
    print("-" * 40)
    for name, p, prob in zip(files, preds, probs):
        confidence = prob[p] * 100
        print(f"{name}: predicted {int(p)} (confidence: {confidence:.1f}%)")


if __name__ == '__main__':
    main() 