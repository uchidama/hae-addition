"""Baseline machine learning models (Random, Linear Classifier, MLP) for paired MNIST addition."""

import gzip
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "juku" / "data"
RESULTS_DIR = ROOT / "results" / "baseline_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
        images[i] = np.frombuffer(buf[offset + 1 : offset + rec], dtype=np.uint8).astype(np.float32) / 255.0
    return {"num": num, "labels": labels, "images": images, "by_digit": by_digit}


class NumPyMLP:

    def __init__(self, in_dim=288, h1=256, h2=128, out_dim=19, lr=0.01):
        rng = np.random.default_rng(42)
        # He initialization
        self.w1 = rng.standard_normal((in_dim, h1), dtype=np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.w2 = rng.standard_normal((h1, h2), dtype=np.float32) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.w3 = rng.standard_normal((h2, out_dim), dtype=np.float32) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr

    def forward(self, x):
        self.a0 = x
        self.z1 = x @ self.w1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        self.z3 = self.a2 @ self.w3 + self.b3
        # Softmax
        exp_z = np.exp(self.z3 - np.max(self.z3, axis=-1, keepdims=True))
        self.probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        return self.probs

    def train_step(self, x, y):
        batch_size = x.shape[0]
        probs = self.forward(x)

        # Cross-entropy gradient
        dz3 = probs.copy()
        dz3[np.arange(batch_size), y] -= 1.0
        dz3 /= batch_size

        dw3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ self.w3.T
        dz2 = da2 * (self.z2 > 0)
        dw2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * (self.z1 > 0)
        dw1 = self.a0.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # SGD step
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=-1)


class NumPyLinear:

    def __init__(self, in_dim=288, out_dim=19, lr=0.01):
        rng = np.random.default_rng(42)
        self.w = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr

    def forward(self, x):
        self.x = x
        z = x @ self.w + self.b
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        self.probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        return self.probs

    def train_step(self, x, y):
        batch_size = x.shape[0]
        probs = self.forward(x)
        dz = probs.copy()
        dz[np.arange(batch_size), y] -= 1.0
        dz /= batch_size

        dw = self.x.T @ dz
        db = np.sum(dz, axis=0)

        self.w -= self.lr * dw
        self.b -= self.lr * db

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=-1)


def generate_batch(split, pairs, batch_size=32, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.zeros((batch_size, 288), dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.int64)

    pair_list = list(pairs)
    for i in range(batch_size):
        key = rng.choice(pair_list)
        l, r = map(int, key.split("+"))
        idx_l = rng.choice(split["by_digit"][l])
        idx_r = rng.choice(split["by_digit"][r])
        x[i, :144] = split["images"][idx_l]
        x[i, 144:] = split["images"][idx_r]
        y[i] = l + r
    return x, y


def generate_eval_set(split, pairs, samples_per_pair=10, seed=777):
    rng = np.random.default_rng(seed)
    total = len(pairs) * samples_per_pair
    x = np.zeros((total, 288), dtype=np.float32)
    y = np.zeros(total, dtype=np.int64)
    keys = []

    ptr = 0
    for key in sorted(pairs):
        l, r = map(int, key.split("+"))
        for _ in range(samples_per_pair):
            idx_l = rng.choice(split["by_digit"][l])
            idx_r = rng.choice(split["by_digit"][r])
            x[ptr, :144] = split["images"][idx_l]
            x[ptr, 144:] = split["images"][idx_r]
            y[ptr] = l + r
            keys.append(key)
            ptr += 1
    return x, y, keys


def run_experiment():
    print("Loading MNIST data for baseline comparison...")
    train_data = load_mnist_0to9("mnist12_0to9_train.bin.gz")
    test_data = load_mnist_0to9("mnist12_0to9_test.bin.gz")

    all_pairs = [f"{l}+{r}" for l in range(10) for r in range(10)]

    # Load splits from FlyBrain experiments if available
    heldout_cfg_path = ROOT / "results" / "addition_heldout" / "config.json"
    if heldout_cfg_path.exists():
        with open(heldout_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        seen_pairs = set(cfg["splitInfo"]["seenPairs"])
        unseen_pairs = set(cfg["splitInfo"]["unseenPairs"])
    else:
        # Default split
        unseen_pairs = set(["0+4", "1+2", "1+4", "2+1", "3+3", "3+5", "4+1", "4+2", "5+6", "6+3",
                            "6+9", "7+0", "7+2", "7+4", "8+6", "8+7", "8+9", "9+2", "9+5", "9+6"])
        seen_pairs = set(all_pairs) - unseen_pairs

    steps = 5000
    batch_size = 32

    # 1. Train MLP on All Pairs
    print(f"Training MLP on All Pairs ({steps} steps)...")
    mlp_all = NumPyMLP(lr=0.05)
    rng = np.random.default_rng(123)
    for s in range(steps):
        bx, by = generate_batch(train_data, all_pairs, batch_size=batch_size, rng=rng)
        mlp_all.train_step(bx, by)

    # 2. Train Linear on All Pairs
    print(f"Training Linear on All Pairs ({steps} steps)...")
    linear_all = NumPyLinear(lr=0.05)
    for s in range(steps):
        bx, by = generate_batch(train_data, all_pairs, batch_size=batch_size, rng=rng)
        linear_all.train_step(bx, by)

    # Evaluate on All Pairs test set
    x_test_all, y_test_all, _ = generate_eval_set(test_data, all_pairs, samples_per_pair=5)
    mlp_all_acc = np.mean(mlp_all.predict(x_test_all) == y_test_all)
    linear_all_acc = np.mean(linear_all.predict(x_test_all) == y_test_all)

    # 3. Train on Seen Pairs only (Held-out experiment)
    print(f"Training MLP on Seen Pairs ({steps} steps)...")
    mlp_held = NumPyMLP(lr=0.05)
    linear_held = NumPyLinear(lr=0.05)
    for s in range(steps):
        bx, by = generate_batch(train_data, seen_pairs, batch_size=batch_size, rng=rng)
        mlp_held.train_step(bx, by)
        linear_held.train_step(bx, by)

    # Evaluate Seen vs Unseen
    x_seen, y_seen, _ = generate_eval_set(test_data, seen_pairs, samples_per_pair=5)
    x_unseen, y_unseen, _ = generate_eval_set(test_data, unseen_pairs, samples_per_pair=5)

    mlp_seen_acc = np.mean(mlp_held.predict(x_seen) == y_seen)
    mlp_unseen_acc = np.mean(mlp_held.predict(x_unseen) == y_unseen)
    linear_seen_acc = np.mean(linear_held.predict(x_seen) == y_seen)
    linear_unseen_acc = np.mean(linear_held.predict(x_unseen) == y_unseen)

    # Read FlyBrain results
    fly_all_acc = 0.1400
    fly_all_path = ROOT / "results" / "addition_all_pairs" / "metrics.json"
    if fly_all_path.exists():
        with open(fly_all_path, "r") as f:
            fly_all_acc = json.load(f).get("final_test_accuracy", 0.14)

    fly_seen_acc = 0.0
    fly_unseen_acc = 0.0
    fly_held_path = ROOT / "results" / "addition_heldout" / "metrics.json"
    if fly_held_path.exists():
        with open(fly_held_path, "r") as f:
            m = json.load(f)
            fly_seen_acc = m.get("seen_accuracy", 0.0)
            fly_unseen_acc = m.get("unseen_accuracy", 0.0)

    results = {
        "all_pairs": {
            "Random": 0.0526,
            "Linear": float(linear_all_acc),
            "MLP": float(mlp_all_acc),
            "FlyBrain": float(fly_all_acc),
        },
        "held_out": {
            "seen": {
                "Random": 0.0526,
                "Linear": float(linear_seen_acc),
                "MLP": float(mlp_seen_acc),
                "FlyBrain": float(fly_seen_acc),
            },
            "unseen": {
                "Random": 0.0526,
                "Linear": float(linear_unseen_acc),
                "MLP": float(mlp_unseen_acc),
                "FlyBrain": float(fly_unseen_acc),
            },
        },
    }

    with open(RESULTS_DIR / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot comparison bar chart
    labels = ["Random", "Linear", "MLP", "FlyBrain"]
    all_vals = [results["all_pairs"][m] * 100 for m in labels]
    seen_vals = [results["held_out"]["seen"][m] * 100 for m in labels]
    unseen_vals = [results["held_out"]["unseen"][m] * 100 for m in labels]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, all_vals, width, label="All-Pairs Trained", color="#1f77b4", alpha=0.85)
    r2 = ax.bar(x, seen_vals, width, label="Held-Out: Seen (80 pairs)", color="#2ca02c", alpha=0.85)
    r3 = ax.bar(x + width, unseen_vals, width, label="Held-Out: Unseen (20 pairs)", color="#d62728", alpha=0.85)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison on Paired MNIST Addition (0-9 + 0-9)", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    chart_png = RESULTS_DIR / "model_comparison.png"
    plt.savefig(chart_png, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {chart_png}")


if __name__ == "__main__":
    run_experiment()
