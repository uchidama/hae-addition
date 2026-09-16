"""Plot evaluation results for FlyBrain addition experiments."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR_ARG = sys.argv[1] if len(sys.argv) > 1 else "results/addition_all_pairs"
RESULTS_DIR = ROOT / DIR_ARG

METRICS_FILE = RESULTS_DIR / "metrics.json"
LOG_FILE = RESULTS_DIR / "log.jsonl"

if not METRICS_FILE.exists():
    print(f"Error: {METRICS_FILE} does not exist.")
    sys.exit(1)

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# 1. Plot Learning Curve
if LOG_FILE.exists():
    steps, test_accs, run_accs = [], [], []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pt = json.loads(line)
                steps.append(pt["step"])
                test_accs.append(pt["test_acc"] * 100)
                if pt.get("run_acc") is not None:
                    run_accs.append((pt["step"], pt["run_acc"] * 100))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, test_accs, label="Test Accuracy (Fixed Test Set)", color="#1f77b4", marker="o", linewidth=2)
    if run_accs:
        rx, ry = zip(*run_accs)
        plt.plot(rx, ry, label="Recent Practice Accuracy (Window 200)", color="#ff7f0e", linestyle="--", alpha=0.7)

    plt.axhline(100.0 / 19.0, color="gray", linestyle=":", label="Random Guess (5.26%)")
    plt.title(f"Addition Learning Curve (0-9 + 0-9, 19 classes)\nFinal Acc: {metrics['final_test_accuracy']*100:.2f}%", fontsize=14, pad=12)
    plt.xlabel("Training Steps (Pictures Practised)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    curve_png = RESULTS_DIR / "learning_curve.png"
    plt.savefig(curve_png, dpi=150)
    plt.close()
    print(f"Saved learning curve to {curve_png}")

# 2. Plot 19x19 Confusion Matrix
cm = np.array(metrics["confusion_matrix"])
row_sums = cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.zeros_like(cm, dtype=float)
np.divide(cm.astype(float), row_sums, out=cm_norm, where=row_sums != 0)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

labels = [str(i) for i in range(19)]
ax.set(
    xticks=np.arange(19),
    yticks=np.arange(19),
    xticklabels=labels,
    yticklabels=labels,
    title=f"Addition Confusion Matrix (0-18 Sums, Acc: {metrics['final_test_accuracy']*100:.2f}%)",
    ylabel="True Sum",
    xlabel="Predicted Sum",
)

thresh = 0.5
for i in range(19):
    for j in range(19):
        if cm[i, j] > 0:
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7,
            )

fig.tight_layout()
cm_png = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_png, dpi=150)
plt.close()
print(f"Saved confusion matrix to {cm_png}")

# 3. Plot 10x10 Operand Pair Accuracy Heatmap
pair_grid = np.zeros((10, 10), dtype=float)
pair_accs = metrics.get("pair_accuracies", {})
for l in range(10):
    for r in range(10):
        key = f"{l}+{r}"
        pair_grid[l, r] = pair_accs.get(key, 0.0) * 100

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(pair_grid, cmap="YlGnBu", vmin=0, vmax=100)
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy (%)")

ax.set(
    xticks=np.arange(10),
    yticks=np.arange(10),
    xticklabels=[str(i) for i in range(10)],
    yticklabels=[str(i) for i in range(10)],
    title="Operand Pair Accuracy Heatmap (Left Operand + Right Operand)",
    ylabel="Left Operand (A)",
    xlabel="Right Operand (B)",
)

for l in range(10):
    for r in range(10):
        val = pair_grid[l, r]
        ax.text(
            r, l, f"{val:.0f}%",
            ha="center", va="center",
            color="white" if val > 60 else "black",
            fontsize=8,
        )

fig.tight_layout()
pair_png = RESULTS_DIR / "pair_accuracy_heatmap.png"
plt.savefig(pair_png, dpi=150)
plt.close()
print(f"Saved pair accuracy heatmap to {pair_png}")

# 4. Plot Per-Sum Accuracy Bar Chart
per_sum = metrics.get("per_sum_accuracy", {})
sums = list(range(19))
acc_vals = [per_sum.get(str(s), 0.0) * 100 for s in sums]

plt.figure(figsize=(11, 5))
bars = plt.bar(sums, acc_vals, color="#2ca02c", alpha=0.85, edgecolor="black", width=0.7)
plt.title("Accuracy per Sum (0 to 18)", fontsize=14, pad=12)
plt.xlabel("Sum (Target Class)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xticks(sums)
plt.ylim(0, 100)
plt.grid(axis="y", linestyle=":", alpha=0.6)

for bar in bars:
    h = bar.get_height()
    if h > 0:
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
bar_png = RESULTS_DIR / "per_sum_accuracy.png"
plt.savefig(bar_png, dpi=150)
plt.close()
print(f"Saved per-sum accuracy bar chart to {bar_png}")
