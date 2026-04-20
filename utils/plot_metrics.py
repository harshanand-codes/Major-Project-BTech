"""Generate training-curve plots from a metrics.jsonl log."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def line_plot(epochs, series, ylabel, out_path, *, logy=False, vline=None,
              title=None):
    plt.figure(figsize=(9, 4.5))
    for label, ys in series.items():
        plt.plot(epochs, ys, label=label, linewidth=1.5)
    if logy:
        plt.yscale("log")
    if vline is not None:
        plt.axvline(vline, ls="--", color="gray", alpha=0.6,
                    label=f"best img Dice (ep {vline})")
    if title:
        plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def summary_plot(epochs, rows, best_ep, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(epochs, [r["train"]["loss"] for r in rows], label="train")
    ax.plot(epochs, [r["img_val"]["loss"] for r in rows], label="img_val")
    ax.plot(epochs, [r["vid_val"]["loss"] for r in rows], label="vid_val")
    ax.set_yscale("log"); ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [r["img_val"]["dice"] for r in rows], label="img_val")
    ax.plot(epochs, [r["vid_val"]["dice"] for r in rows], label="vid_val")
    ax.axvline(best_ep, ls="--", color="gray", alpha=0.6)
    ax.set_title("Dice"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, [r["img_val"]["iou"] for r in rows], label="img_val")
    ax.plot(epochs, [r["vid_val"]["iou"] for r in rows], label="vid_val")
    ax.axvline(best_ep, ls="--", color="gray", alpha=0.6)
    ax.set_title("IoU"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [r["img_val"]["hausdorff"] for r in rows], label="img_val")
    ax.plot(epochs, [r["vid_val"]["hausdorff"] for r in rows], label="vid_val")
    ax.axvline(best_ep, ls="--", color="gray", alpha=0.6)
    ax.set_title("Hausdorff Distance"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for i in range(1, 5):
        ax.plot(epochs, [r["train"][f"loss_{i}"] for r in rows], label=f"L{i}")
    ax.set_yscale("log")
    ax.set_title("Train sub-losses"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, [r["lr"] for r in rows])
    ax.set_yscale("log")
    ax.set_title("Learning rate"); ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
    fig.suptitle(f"Training summary (best img-val Dice at epoch {best_ep})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="checkpoints/metrics.jsonl")
    parser.add_argument("--out-dir", default="predictions/plots1")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load(metrics_path)
    epochs = [r["epoch"] for r in rows]
    best_ep = max(rows, key=lambda r: r["img_val"]["dice"])["epoch"]

    line_plot(epochs, {
        "train":   [r["train"]["loss"] for r in rows],
        "img_val": [r["img_val"]["loss"] for r in rows],
        "vid_val": [r["vid_val"]["loss"] for r in rows],
    }, "Loss", out_dir / "losses.png", logy=True, vline=best_ep,
        title="Loss curves (log scale)")

    for m in ["dice", "iou", "f1"]:
        line_plot(epochs, {
            f"img_val_{m}": [r["img_val"][m] for r in rows],
            f"vid_val_{m}": [r["vid_val"][m] for r in rows],
        }, m.upper(), out_dir / f"{m}.png", vline=best_ep,
            title=f"{m.upper()} over epochs")

    line_plot(epochs, {
        "img_val": [r["img_val"]["hausdorff"] for r in rows],
        "vid_val": [r["vid_val"]["hausdorff"] for r in rows],
    }, "Hausdorff Distance", out_dir / "hausdorff.png", vline=best_ep,
        title="Hausdorff Distance over epochs")

    line_plot(epochs, {
        f"L{i}": [r["train"][f"loss_{i}"] for r in rows] for i in range(1, 5)
    }, "Train sub-loss", out_dir / "sublosses.png", logy=True,
        title="Train sub-loss decomposition (log scale)")

    line_plot(epochs, {"lr": [r["lr"] for r in rows]},
              "Learning rate", out_dir / "lr.png", logy=True,
              title="Learning-rate schedule")

    summary_plot(epochs, rows, best_ep, out_dir / "summary.png")
    print(f"Wrote plots to {out_dir}/ (best img-val Dice at epoch {best_ep})")


if __name__ == "__main__":
    main()
