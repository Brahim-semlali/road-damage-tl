"""Exploratory Data Analysis: figures and tables for the paper (Section III.2).

Requires ``data/splits/all.csv`` (run ``make_splits`` first).

Outputs in ``results/eda/``:
    class_distribution.png   images per class (bar chart)
    split_distribution.png   images per class and per split
    image_sizes.png          width x height of the original images
    samples.png              a few random images of every class
    dataset_table.md         Markdown table ready to paste into the paper

Usage
-----
    python -m src.data.eda --raw data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works on Kaggle / servers)
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def class_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = df.label.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution (after cleaning)")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=200)
    plt.close(fig)


def split_distribution(df: pd.DataFrame, out: Path) -> None:
    table = pd.crosstab(df.label, df.split)[["train", "val", "test"]]
    ax = table.plot(kind="bar", figsize=(8, 4), rot=20)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_title("Stratified train / validation / test split")
    plt.tight_layout()
    plt.savefig(out / "split_distribution.png", dpi=200)
    plt.close()


def image_sizes(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, g in df.groupby("label"):
        ax.scatter(g.width, g.height, s=8, alpha=0.5, label=label)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Original image resolutions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "image_sizes.png", dpi=200)
    plt.close(fig)


def sample_grid(df: pd.DataFrame, raw: Path, out: Path, per_class: int = 5, seed: int = 42) -> None:
    labels = sorted(df.label.unique())
    fig, axes = plt.subplots(len(labels), per_class,
                             figsize=(2.2 * per_class, 2.2 * len(labels)), squeeze=False)
    for i, label in enumerate(labels):
        g = df[df.label == label]
        picks = g.sample(min(per_class, len(g)), random_state=seed)
        for j in range(per_class):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(picks):
                ax.imshow(Image.open(raw / picks.iloc[j].path).convert("RGB"))
        axes[i][0].set_title(label, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=150)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, out: Path) -> str:
    t = pd.crosstab(df.label, df.split, margins=True, margins_name="Total")
    t = t[["train", "val", "test", "Total"]]
    lines = ["| Class | Train | Val | Test | Total |", "|---|---|---|---|---|"]
    for label, r in t.iterrows():
        lines.append(f"| {label} | {r.train} | {r.val} | {r.test} | {r.Total} |")
    md = "\n".join(lines)
    ratio = df.label.value_counts()
    md += (f"\n\nImbalance ratio (largest / smallest class): "
           f"{ratio.max() / ratio.min():.2f}\n"
           f"Median original resolution: {int(df.width.median())} x {int(df.height.median())} px\n")
    (out / "dataset_table.md").write_text(md, encoding="utf-8")
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--csv", type=Path, default=Path("data/splits/all.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/eda"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)

    class_distribution(df, args.out)
    split_distribution(df, args.out)
    image_sizes(df, args.out)
    sample_grid(df, args.raw, args.out)
    print(markdown_table(df, args.out))
    print(f"Figures saved to {args.out}/")


if __name__ == "__main__":
    main()
