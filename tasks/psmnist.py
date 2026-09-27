"""Permuted sequential MNIST (Le et al., 2015), tokenized.

The 28x28 image is fed one pixel per step (784 steps) in a fixed random order shared by all images; the
digit label is predicted at the last step, which is the only position in the loss. Each pixel intensity
(0-255) is a token, rather than a real-valued input as in the paper. The 60k training images are split
into train / val (val size = data.num_val, split by data.seed); the official 10k test set is evaluated
at the end of training.
MNIST is downloaded to <data.dir>/mnist on first use.
"""
import gzip
import math
import os
import urllib.request

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .base import Task

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz",
}


def load_mnist(root):
    os.makedirs(root, exist_ok=True)
    arrays = {}
    for key, name in MNIST_FILES.items():
        path = os.path.join(root, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(MNIST_URL + name, path + ".tmp")
            os.replace(path + ".tmp", path)
        with gzip.open(path, "rb") as f:
            raw = f.read()
        # idx format: 16-byte header for images, 8-byte header for labels
        arrays[key] = np.frombuffer(raw, np.uint8, offset=16 if key.endswith("x") else 8)
    return (torch.from_numpy(arrays["train_x"].reshape(-1, 784).copy()), torch.from_numpy(arrays["train_y"].copy()),
            torch.from_numpy(arrays["test_x"].reshape(-1, 784).copy()), torch.from_numpy(arrays["test_y"].copy()))


class PSMNIST(Task):
    input_size = 256
    output_size = 10
    is_dataset = True

    def __init__(self, permute=True, permutation_seed=0):
        self.permute = permute
        self.permutation_seed = permutation_seed

    def load_splits(self, data_dir, num_val, seed):
        train_x, train_y, test_x, test_y = load_mnist(os.path.join(data_dir, "mnist"))
        if self.permute:
            perm = torch.randperm(784, generator=torch.Generator().manual_seed(self.permutation_seed))
            train_x, test_x = train_x[:, perm], test_x[:, perm]
        order = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(seed))
        train_idx, val_idx = order[num_val:], order[:num_val]
        return {
            "train": TensorDataset(train_x[train_idx], train_y[train_idx]),
            "val": TensorDataset(train_x[val_idx], train_y[val_idx]),
            "test": TensorDataset(test_x, test_y),
        }

    def make_batch(self, pixels, labels):
        x = pixels.long()
        y = torch.zeros(pixels.shape, dtype=torch.long)
        y[:, -1] = labels.long()
        mask = torch.zeros(pixels.shape, dtype=torch.bool)
        mask[:, -1] = True
        return x, y, mask

    def metrics(self, logits, y, mask):
        return {"acc": (logits[:, -1].argmax(-1) == y[:, -1]).float().mean().item()}

    def format_sample(self, x, y, mask, logits):
        label, pred = int(y[-1]), int(logits[-1].argmax())
        return f"label {label}  │  pred [{'green' if label == pred else 'red'}]{pred}[/]"

    @property
    def baseline_loss(self):
        return math.log(10)
