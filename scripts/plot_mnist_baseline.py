import json
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "mnist_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_DIR / "juku" / "state" / "suji" / "log.jsonl"
METRICS_FILE = RESULTS_DIR / "metrics.json"

# 1. Convert log.jsonl to accuracy_curve.csv and plot accuracy_curve.png
history = []
if LOG_FILE.exists():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))

csv_file = RESULTS_DIR / "accuracy_curve.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["practised", "test_acc", "run_acc", "timestamp"])
    for pt in history:
        writer.writerow([pt.get("n"), pt.get("test"), pt.get("run"), pt.get("t")])
print(f"Saved {len(history)} points to {csv_file}")

# Plot accuracy curve
if history:
    steps = [pt["n"] for pt in history]
    test_acc = [pt["test"] * 100 for pt in history]
    run_acc = [pt["run"] * 100 for pt in history]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, test_acc, label="Test Accuracy (450 held-out samples)", color="#1f77b4", linewidth=2)
    plt.plot(steps, run_acc, label="Running Practice Accuracy (window 500)", color="#ff7f0e", linestyle="--", alpha=0.7)
    plt.title("MNIST Digit Recognition Learning Curve (FlyBrain suji)", fontsize=14, pad=12)
    plt.xlabel("Practised Pictures", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.axhline(89.56, color="green", linestyle=":", label="Final Benchmark (89.56%)")
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    curve_png = RESULTS_DIR / "accuracy_curve.png"
    plt.savefig(curve_png, dpi=150)
    plt.close()
    print(f"Saved accuracy curve plot to {curve_png}")

# 2. Plot Confusion Matrix
if METRICS_FILE.exists():
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    labels = metrics.get("labels", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])
    cm = np.array(metrics["full_test"]["confusion_matrix"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=f"Confusion Matrix (Full Test: 9,020 samples, Acc: {metrics['full_test']['accuracy']*100:.2f}%)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    cm_png = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_png, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_png}")
