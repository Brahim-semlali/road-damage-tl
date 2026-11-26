"""Scan the raw dataset, clean it and create a reproducible stratified split.

Steps
-----
1. Find every image under ``--raw``; its class is the name of its parent folder.
2. Cleaning: drop unreadable/corrupted files and exact duplicates (MD5 hash).
   Removing duplicates *before* splitting prevents train/test leakage.
3. Stratified split train/val/test (default 70/15/15) with a fixed seed.
4. Save ``data/splits/{all,train,val,test}.csv`` — these CSVs are versioned in Git,
   so every team member trains and tests on exactly the same images.

Usage
-----
    python -m src.data.make_splits --raw data/raw
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_images(raw_dir: Path) -> pd.DataFrame:
    """Return one row per image: path, label, md5, width, height, readable."""
    rows = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        row = {"path": p.as_posix(), "label": p.parent.name,
               "md5": hashlib.md5(p.read_bytes()).hexdigest(),
               "width": None, "height": None, "readable": True}
        try:
            with Image.open(p) as im:
                im.verify()  # detects truncated / corrupted files
            with Image.open(p) as im:
                row["width"], row["height"] = im.size
        except Exception:
            row["readable"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    bad = (~df.readable).sum()
    df = df[df.readable]

    # Same image under two different labels = annotation conflict -> drop all copies.
    conflicts = df.groupby("md5").label.nunique()
    conflict_md5 = conflicts[conflicts > 1].index
    n_conflict_files = df.md5.isin(conflict_md5).sum()
    df = df[~df.md5.isin(conflict_md5)]

    dups = df.md5.duplicated().sum()
    df = df.drop_duplicates("md5").reset_index(drop=True)

    print(f"Images found            : {n0}")
    print(f"Corrupted removed       : {bad}")
    print(f"Label conflicts removed : {n_conflict_files} files "
          f"({len(conflict_md5)} images found under several classes)")
    print(f"Exact duplicates removed: {dups}")
    print(f"Clean images            : {len(df)}")
    return df


def split(df: pd.DataFrame, val: float, test: float, seed: int):
    train_df, tmp = train_test_split(df, test_size=val + test,
                                     stratify=df.label, random_state=seed)
    val_df, test_df = train_test_split(tmp, test_size=test / (val + test),
                                       stratify=tmp.label, random_state=seed)
    return train_df, val_df, test_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = scan_images(args.raw)
    if df.empty:
        raise SystemExit(f"No images found under {args.raw}. Download the dataset first.")

    df = clean(df)
    print("\nImages per class:\n" + df.label.value_counts().to_string())

    train_df, val_df, test_df = split(df, args.val, args.test, args.seed)
    df["split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    # Paths are stored relative to the raw folder so the CSVs work on any machine
    # (local PC, Kaggle, Colab): the training code joins them with its own data root.
    df["path"] = df.path.map(lambda p: Path(p).relative_to(args.raw).as_posix())

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "md5", "width", "height", "split"]
    df[cols].to_csv(args.out / "all.csv", index=False)
    for name in ("train", "val", "test"):
        df.loc[df.split == name, ["path", "label"]].to_csv(args.out / f"{name}.csv", index=False)

    print("\nSplit sizes (rows = class, cols = split):")
    print(pd.crosstab(df.label, df.split, margins=True).to_string())
    print(f"\nSaved to {args.out}/ (seed={args.seed})")


if __name__ == "__main__":
    main()
