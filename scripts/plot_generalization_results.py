"""Plot evaluation results for generalization experiments (held-out and commutativity)."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR_ARG = sys.argv[1] if len(sys.argv) > 1 else "results/addition_heldout"
RESULTS_DIR = ROOT / DIR_ARG

METRICS_FILE = RESULTS_DIR / "metrics.json"
LOG_FILE = RESULTS_DIR / "log.jsonl"

if not METRICS_FILE.exists():
    print(f"Error: {METRICS_FILE} does not exist.")
    sys.exit(1)

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    metrics = json.load(f)

mode = metrics.get("mode", "held-out")

# 1. Plot Learning Curves
if LOG_FILE.exists():
    steps, overall, seen, unseen, fwd, rev = [], [], [], [], [], []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pt = json.loads(line)
                steps.append(pt["step"])
                overall.append(pt["test_acc"] * 100)
                if pt.get("seen_acc") is not None:
                    seen.append(pt["seen_acc"] * 100)
                if pt.get("unseen_acc") is not None:
                    unseen.append(pt["unseen_acc"] * 100)
                if pt.get("comm_source_acc") is not None:
                    fwd.append(pt["comm_source_acc"] * 100)
                if pt.get("comm_gen_acc") is not None:
                    rev.append(pt["comm_gen_acc"] * 100)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, overall, label="Overall Test Acc", color="black", linestyle="--", alpha=0.5)

    if mode == "held-out":
        if seen:
            plt.plot(steps, seen, label="Seen Pairs Acc (80 pairs)", color="#1f77b4", marker="o", linewidth=2)
        if unseen:
            plt.plot(steps, unseen, label="Unseen Held-Out Pairs Acc (20 pairs)", color="#d62728", marker="s", linewidth=2.5)
        plt.title(f"Held-Out Generalization Learning Curve\nFinal Seen: {metrics.get('seen_accuracy',0)*100:.1f}%, Unseen: {metrics.get('unseen_accuracy',0)*100:.1f}%", fontsize=14, pad=12)
    elif mode == "commutativity":
        if fwd:
            plt.plot(steps, fwd, label="Trained Fwd A+B (20 pairs)", color="#1f77b4", marker="o", linewidth=2)
        if rev:
            plt.plot(steps, rev, label="Held-Out Rev B+A (20 pairs)", color="#9467bd", marker="^", linewidth=2.5)
        plt.title(f"Commutative Generalization Learning Curve\nFinal Fwd: {metrics.get('commutative_source_accuracy',0)*100:.1f}%, Rev: {metrics.get('commutative_generalization_accuracy',0)*100:.1f}%", fontsize=14, pad=12)

    plt.axhline(100.0 / 19.0, color="gray", linestyle=":", label="Random Guess (5.26%)")
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    curve_png = RESULTS_DIR / "generalization_learning_curve.png"
    plt.savefig(curve_png, dpi=150)
    plt.close()
    print(f"Saved generalization learning curve to {curve_png}")

# 2. Plot Comparison Bar Chart
plt.figure(figsize=(7, 5))
if mode == "held-out":
    cats = ["Random Guess", "Seen Pairs (80)", "Unseen Pairs (20)"]
    vals = [5.26, metrics.get("seen_accuracy", 0) * 100, metrics.get("unseen_accuracy", 0) * 100]
    colors = ["#7f7f7f", "#1f77b4", "#d62728"]
    title = "Generalization to Unseen Operand Pairs"
elif mode == "commutativity":
    cats = ["Random Guess", "Trained Fwd (A+B)", "Held-Out Rev (B+A)"]
    vals = [5.26, metrics.get("commutative_source_accuracy", 0) * 100, metrics.get("commutative_generalization_accuracy", 0) * 100]
    colors = ["#7f7f7f", "#1f77b4", "#9467bd"]
    title = "Commutative Generalization Test (A+B vs B+A)"
else:
    cats = ["Random Guess", "Overall"]
    vals = [5.26, metrics.get("final_test_accuracy", 0) * 100]
    colors = ["#7f7f7f", "#1f77b4"]
    title = "Accuracy Comparison"

bars = plt.bar(cats, vals, color=colors, alpha=0.85, edgecolor="black", width=0.55)
plt.title(title, fontsize=13, pad=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, max(max(vals) * 1.25, 25))
plt.grid(axis="y", linestyle=":", alpha=0.6)

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
bar_png = RESULTS_DIR / "comparison_bar.png"
plt.savefig(bar_png, dpi=150)
plt.close()
print(f"Saved comparison bar chart to {bar_png}")

# 3. Plot Pair Accuracy Heatmap
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
    title=f"Operand Pair Accuracy ({mode})",
    ylabel="Left Operand (A)",
    xlabel="Right Operand (B)",
)

unseen_set = set(metrics.get("unseen_pairs", []) if mode == "held-out" else metrics.get("held_out_reversed_pairs", []))

for l in range(10):
    for r in range(10):
        key = f"{l}+{r}"
        val = pair_grid[l, r]
        is_held_out = key in unseen_set
        prefix = "*" if is_held_out else ""
        ax.text(
            r, l, f"{prefix}{val:.0f}%",
            ha="center", va="center",
            color="red" if is_held_out and val > 0 else "white" if val > 60 else "black",
            fontsize=8,
            fontweight="bold" if is_held_out else "normal",
        )

fig.tight_layout()
pair_png = RESULTS_DIR / "pair_accuracy_heatmap.png"
plt.savefig(pair_png, dpi=150)
plt.close()
print(f"Saved pair accuracy heatmap to {pair_png}")
