"""Minimal MNIST IDX loader (no torchvision)."""
import gzip, struct, numpy as np


def _read_images(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * r * c)
        return np.frombuffer(buf, dtype=np.uint8).reshape(n, r, c)


def _read_labels(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(n), dtype=np.uint8)


def load_mnist(d):
    Xtr = _read_images(f"{d}/train-images-idx3-ubyte")
    Ytr = _read_labels(f"{d}/train-labels-idx1-ubyte")
    Xte = _read_images(f"{d}/t10k-images-idx3-ubyte")
    Yte = _read_labels(f"{d}/t10k-labels-idx1-ubyte")
    return Xtr, Ytr, Xte, Yte
