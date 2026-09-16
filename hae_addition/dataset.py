"""Paired MNIST addition dataset loader and generator in Python."""

import gzip
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "juku" / "data"


def load_mnist_0to9(filename):
    path = DATA_DIR / filename
    with gzip.open(path, "rb") as f:
        buf = f.read()
    rec = 145
    num = len(buf) // rec
    labels = np.zeros(num, dtype=np.int64)
    images = np.zeros((num, 144), dtype=np.float32)

    by_digit = {d: [] for d in range(10)}
    for i in range(num):
        offset = i * rec
        lbl = buf[offset]
        labels[i] = lbl
        by_digit[lbl].append(i)
        pixels = np.frombuffer(buf[offset + 1 : offset + rec], dtype=np.uint8).astype(np.float32) / 255.0
        images[i] = pixels

    return {"num": num, "labels": labels, "images": images, "by_digit": by_digit}


class AdditionDatasetPy:

    def __init__(self):
        self.train = load_mnist_0to9("mnist12_0to9_train.bin.gz")
        self.test = load_mnist_0to9("mnist12_0to9_test.bin.gz")
        self.all_pairs = []
        for l in range(10):
            for r in range(10):
                self.all_pairs.append({"left": l, "right": r, "key": f"{l}+{r}", "target": l + r})

    def sample(self, mode="train", allowed_pairs=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        split = self.train if mode == "train" else self.test

        if allowed_pairs is not None:
            pair_key = rng.choice(list(allowed_pairs))
            l, r = map(int, pair_key.split("+"))
            pair = {"left": l, "right": r, "key": pair_key, "target": l + r}
        else:
            idx = rng.integers(0, len(self.all_pairs))
            pair = self.all_pairs[idx]

        left_pool = split["by_digit"][pair["left"]]
        right_pool = split["by_digit"][pair["right"]]
        left_idx = rng.choice(left_pool)
        right_idx = rng.choice(right_pool)

        return {
            "left_image": split["images"][left_idx],
            "right_image": split["images"][right_idx],
            "left_digit": pair["left"],
            "right_digit": pair["right"],
            "target": pair["target"],
            "pair_key": pair["key"],
        }
