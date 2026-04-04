#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+loss=(?P<loss>[-+]?\d*\.?\d+)\s+"
    r"lm=(?P<lm>[-+]?\d*\.?\d+)\s+valid=(?P<valid>[-+]?\d*\.?\d+)\s+"
    r"cmp=(?P<cmp>[-+]?\d*\.?\d+)\s+obj=(?P<obj>[-+]?\d*\.?\d+)"
)
EVAL_RE = re.compile(
    r"\[eval\]\s+step=(?P<step>\d+)\s+score=(?P<score>[-+]?\d*\.?\d+)\s+"
    r"val_loss=(?P<val_loss>[-+]?\d*\.?\d+)\s+val_lm=(?P<val_lm>[-+]?\d*\.?\d+)\s+"
    r"valid_acc=(?P<valid_acc>[-+]?\d*\.?\d+)\s+compact_acc=(?P<compact_acc>[-+]?\d*\.?\d+)"
)


def moving_average(values, window: int):
    if window <= 1 or len(values) < window:
        return None
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        smoothed.append(sum(chunk) / float(len(chunk)))
    return smoothed


def parse_log(log_path: Path):
    train = {
        "step": [],
        "loss": [],
        "lm": [],
        "valid": [],
        "cmp": [],
        "obj": [],
    }
    evals = {
        "step": [],
        "score": [],
        "val_loss": [],
        "val_lm": [],
        "valid_acc": [],
        "compact_acc": [],
    }

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            train["step"].append(int(m.group("step")))
            train["loss"].append(float(m.group("loss")))
            train["lm"].append(float(m.group("lm")))
            train["valid"].append(float(m.group("valid")))
            train["cmp"].append(float(m.group("cmp")))
            train["obj"].append(float(m.group("obj")))
            continue

        m = EVAL_RE.search(line)
        if m:
            evals["step"].append(int(m.group("step")))
            evals["score"].append(float(m.group("score")))
            evals["val_loss"].append(float(m.group("val_loss")))
            evals["val_lm"].append(float(m.group("val_lm")))
            evals["valid_acc"].append(float(m.group("valid_acc")))
            evals["compact_acc"].append(float(m.group("compact_acc")))
            continue

    return train, evals


def save_train_plot(train, out_file: Path, smooth_window: int):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    pairs = [
        ("loss", "Total Loss"),
        ("lm", "LM Loss"),
        ("valid", "Validity Loss"),
        ("cmp", "Compactness Loss"),
        ("obj", "Object Count Loss"),
    ]

    for idx, (key, title) in enumerate(pairs):
        r, c = divmod(idx, 2)
        ax = axes[r][c]
        ax.plot(
            train["step"],
            train[key],
            marker="o",
            markersize=2,
            linewidth=1,
            alpha=0.45,
            label="raw",
        )
        smooth = moving_average(train[key], smooth_window)
        if smooth is not None:
            ax.plot(train["step"], smooth, linewidth=2.0, color="tab:red", label=f"smoothed (w={smooth_window})")
            ax.legend(loc="best", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # Keep final cell as summary text.
    ax = axes[2][1]
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        (
            f"Train points: {len(train['step'])}\n"
            f"Last step: {train['step'][-1] if train['step'] else 'N/A'}\n"
            f"Smoothing window: {smooth_window}"
        ),
        va="top",
    )

    axes[2][0].set_xlabel("Step")
    fig.suptitle("Training Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def save_eval_plot(evals, out_file: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    pairs = [
        ("score", "Checkpoint Score"),
        ("val_loss", "Validation Loss"),
        ("valid_acc", "Validation Validity Accuracy"),
        ("compact_acc", "Validation Compactness Accuracy"),
    ]
    for idx, (key, title) in enumerate(pairs):
        r, c = divmod(idx, 2)
        ax = axes[r][c]
        ax.plot(evals["step"], evals[key], marker="o", linewidth=1.2)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Step")
    fig.suptitle("Evaluation Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Parse training log and generate metric charts.")
    parser.add_argument("--log", default="train.log", help="Path to train.log file")
    parser.add_argument("--out-dir", default="charts", help="Output directory for charts")
    parser.add_argument("--smooth-window", type=int, default=11, help="Moving-average window for training curves (odd recommended)")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, evals = parse_log(log_path)
    if not train["step"] and not evals["step"]:
        raise SystemExit("No train/eval metrics found in log.")

    if train["step"]:
        save_train_plot(train, out_dir / "train_metrics.png", max(1, args.smooth_window))
    if evals["step"]:
        save_eval_plot(evals, out_dir / "eval_metrics.png")

    print(f"Charts generated in: {out_dir.resolve()}")
    if train["step"]:
        print(f"Train points: {len(train['step'])}, last step: {train['step'][-1]}")
    if evals["step"]:
        print(f"Eval points: {len(evals['step'])}, last eval step: {evals['step'][-1]}")


if __name__ == "__main__":
    main()
