import os
import time
from typing import Tuple

import numpy as np

from cnn.layers import Conv2D, MaxPool2D, Dense, ReLU, BatchNorm
from cnn.model import Sequential
from cnn.losses import softmax_cross_entropy
from cnn.optim import Adam
from cnn.dataset_svhn import SVHNDataset, SVHNConfig


def build_model(num_classes: int = 10) -> Sequential:
    layers = [
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
    ]
    layers += [
    ]
    model = Sequential(layers)
    model.num_flat = 128 * 4 * 4
    model.classifier = Sequential([
        Dense(model.num_flat, 256),
        ReLU(),
        Dense(256, 128),
        ReLU(),
        Dense(128, num_classes),
    ])
    return model


def forward_logits(model: Sequential, x: np.ndarray) -> np.ndarray:
    feats = model.forward(x)
    N = feats.shape[0]
    feats = feats.reshape(N, -1)
    logits = model.classifier.forward(feats)
    return logits


def backward_from_logits(model: Sequential, grad_logits: np.ndarray, feats_shape: Tuple[int, int, int, int]):
    N = feats_shape[0]
    grad_feats = model.classifier.backward(grad_logits)
    grad_feats = grad_feats.reshape((N, feats_shape[1], feats_shape[2], feats_shape[3]))
    grad_feats = grad_feats
    model.backward(grad_feats)


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    preds = logits.argmax(axis=1)
    return float((preds == y).mean())


def train():
    cfg = SVHNConfig(root=os.path.join(os.path.dirname(__file__), 'data'), download=True, augment=True)
    train_ds = SVHNDataset(cfg, split='train')
    test_ds = SVHNDataset(SVHNConfig(root=cfg.root, download=False, augment=False), split='test')

    model = build_model(num_classes=10)
    opt = Adam(lr=1e-4, beta1=0.9, beta2=0.999)

    epochs = 30
    batch_size = 16

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_test_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0
        t0 = time.time()
        
        for xb, yb in train_ds.get_batches(batch_size=batch_size, shuffle=True, augment=True):
            feats = model.forward(xb)
            N = feats.shape[0]
            feats_flat = feats.reshape(N, -1)
            logits = model.classifier.forward(feats_flat)
            loss, grad_logits = softmax_cross_entropy(logits, yb)
            running_loss += loss
            running_acc += accuracy(logits, yb)
            n_batches += 1
            grad_feats_flat = model.classifier.backward(grad_logits)
            grad_feats = grad_feats_flat.reshape(feats.shape)
            model.backward(grad_feats)
            opt.step(list(model.params_and_grads()) + list(model.classifier.params_and_grads()))
        
        dt = time.time() - t0
        train_loss = running_loss/n_batches
        train_acc = running_acc/n_batches
        
        print(f"Epoch {epoch}/{epochs} train: loss={train_loss:.4f} acc={train_acc:.4f} ({dt:.1f}s)")
        
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            total_acc = 0.0
            n_eval = 0
            for xb, yb in test_ds.get_batches(batch_size=256, shuffle=False, augment=False):
                logits = forward_logits(model, xb)
                total_acc += accuracy(logits, yb)
                n_eval += 1
            test_acc = total_acc / n_eval
            print(f"Epoch {epoch} test: acc={test_acc:.4f}")
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                from cnn.checkpoint import save_checkpoint
                extra = {"mean": getattr(train_ds, "mean", np.zeros(3, dtype=np.float32)),
                         "std": getattr(train_ds, "std", np.ones(3, dtype=np.float32))}
                save_checkpoint(model, model.classifier, os.path.join(ckpt_dir, "best_model.npz"), extra=extra)
                print(f"New best model saved with test accuracy: {test_acc:.4f}")
        
        if epoch % 10 == 0:
            from cnn.checkpoint import save_checkpoint
            extra = {"mean": getattr(train_ds, "mean", np.zeros(3, dtype=np.float32)),
                     "std": getattr(train_ds, "std", np.ones(3, dtype=np.float32))}
            save_checkpoint(model, model.classifier, os.path.join(ckpt_dir, f"epoch_{epoch}.npz"), extra=extra)
    
    print(f"Training completed! Best test accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    train() 