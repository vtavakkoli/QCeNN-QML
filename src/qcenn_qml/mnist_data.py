"""Original MNIST IDX data, checksummed downloads, and seeded disjoint splits.

The mirror and archive MD5s are the ones used by torchvision's MNIST loader.
No PyTorch installation is needed to read the original 28x28 uint8 images.
"""
import gzip
import hashlib
from pathlib import Path
import struct
import urllib.request

import numpy as np


MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
ARCHIVES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def read_idx(raw):
    """Parse only the unsigned-byte, nonempty IDX arrays used by MNIST."""
    if len(raw) < 4 or raw[:3] != b"\x00\x00\x08" or raw[3] not in (1, 3):
        raise ValueError("Invalid MNIST IDX header")
    ndim = raw[3]
    offset = 4 + 4 * ndim
    if len(raw) < offset:
        raise ValueError("Truncated MNIST IDX dimensions")
    shape = struct.unpack(">" + "I" * ndim, raw[4:offset])
    if not all(shape) or len(raw) - offset != int(np.prod(shape)):
        raise ValueError("Invalid MNIST IDX payload length")
    return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape)


def load_mnist(root="data", *, train=True, download=True):
    directory = Path(root) / "MNIST" / "raw"
    prefix = "train" if train else "t10k"
    arrays = []
    for kind in ("images-idx3", "labels-idx1"):
        name = f"{prefix}-{kind}-ubyte.gz"
        path = directory / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; enable MNIST download")
            directory.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(MNIST_MIRROR + name, timeout=60) as response:
                content = response.read()
            if hashlib.md5(content).hexdigest() != ARCHIVES[name]:
                raise ValueError(f"MNIST archive checksum mismatch: {name}")
            temporary = path.with_suffix(".download")
            temporary.write_bytes(content)
            temporary.replace(path)
        content = path.read_bytes()
        if hashlib.md5(content).hexdigest() != ARCHIVES[name]:
            raise ValueError(f"MNIST archive checksum mismatch: {path}")
        arrays.append(read_idx(gzip.decompress(content)))
    images, labels = arrays
    expected = 60000 if train else 10000
    if images.shape != (expected, 28, 28) or labels.shape != (expected,):
        raise ValueError("Unexpected original MNIST dimensions")
    if np.any(labels > 9):
        raise ValueError("Invalid MNIST digit label")
    return images, labels


def load_dataset(dataset, root="data", *, download=True):
    if dataset == "mnist":
        return (*load_mnist(root, train=True, download=download),
                *load_mnist(root, train=False, download=download))
    if dataset != "digits-proxy":
        raise ValueError("Expected mnist or digits-proxy")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    images = np.asarray(digits.images * (255 / 16), dtype=np.uint8)
    train, test = train_test_split(np.arange(len(images)), test_size=.25,
                                   random_state=42, stratify=digits.target)
    return images[train], digits.target[train], images[test], digits.target[test]


def stratified_indices(train_labels, test_labels, train_per_class,
                       validation_per_class, test_per_class, seed):
    """Validation is held out of the training source; test has its own source."""
    counts = (train_per_class, validation_per_class, test_per_class)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError("All per-class split counts must be positive integers")
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for label in range(10):
        available = rng.permutation(np.flatnonzero(np.asarray(train_labels) == label))
        held_out = rng.permutation(np.flatnonzero(np.asarray(test_labels) == label))
        if len(available) < train_per_class + validation_per_class or len(held_out) < test_per_class:
            raise ValueError(f"Not enough examples for class {label}")
        train.extend(available[:train_per_class])
        validation.extend(available[train_per_class:train_per_class + validation_per_class])
        test.extend(held_out[:test_per_class])
    return tuple(rng.permutation(np.asarray(ids, dtype=int))
                 for ids in (train, validation, test))


def split_manifest(dataset, seed, train_ids, validation_ids, test_ids,
                   train_images, train_labels, test_images, test_labels):
    """Archive source IDs and hashes of actual selected bytes, before resizing."""
    result = {"dataset": dataset, "is_real_mnist": dataset == "mnist", "seed": seed,
              "original_image_shape": list(train_images.shape[1:]), "splits": {}}
    for name, ids, images, labels, source in (
        ("train", train_ids, train_images, train_labels, "training_source"),
        ("validation", validation_ids, train_images, train_labels, "training_source"),
        ("test", test_ids, test_images, test_labels, "test_source"),
    ):
        digest = hashlib.sha256()
        digest.update(np.asarray(ids, dtype="<i8").tobytes())
        digest.update(np.asarray(images[ids], dtype=np.uint8).tobytes())
        digest.update(np.asarray(labels[ids], dtype=np.uint8).tobytes())
        result["splits"][name] = {
            "source": source, "indices": ids.tolist(), "sha256": digest.hexdigest(),
            "class_counts": np.bincount(labels[ids], minlength=10).tolist(),
        }
    return result
