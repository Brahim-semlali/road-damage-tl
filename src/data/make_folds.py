"""Stratified K-fold cross-validation splits (default 5 folds, seed 42).

Protocol
--------
* The **fixed test set** (``data/splits/test.csv``, 310 images) is kept out of the folds: it stays
  an untouched, common reference for every model and every fold.
* The **CV pool** = train + val images of the main split (1,756 images), divided into K stratified folds.
* For fold k:
    - ``test.csv``  = fold k (held-out fold, ~351 images): the cross-validation score is computed here;
    - ``val.csv``   = 15 % of the other folds (stratified): early stopping / checkpoint selection only,
                      so the held-out fold is never used to choose the model;
    - ``train.csv`` = the rest of the other folds.
  Every image of the pool is therefore evaluated exactly once over the K folds.

Duplicates (MD5) were already removed before the main split, so no image appears in two folds.

Outputs (versioned in Git, like the main split)::

    data/splits/cv/folds.csv                    path, label, fold
    data/splits/cv/fold{k}/{train,val,test}.csv

Usage
-----
    python -m src.data.make_folds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("data/splits"), help="folder with train.csv / val.csv")
    ap.add_argument("--out", type=Path, default=Path("data/splits/cv"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-val", type=float, default=0.15, help="share of the training folds used for early stopping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = pd.concat([pd.read_csv(args.splits / "train.csv"), pd.read_csv(args.splits / "val.csv")],
                     ignore_index=True)[["path", "label"]]
    test = pd.read_csv(args.splits / "test.csv")
    assert not set(pool.path) & set(test.path), "test images found in the CV pool"
    assert pool.path.is_unique, "duplicate paths in the CV pool"

    pool["fold"] = -1
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (_, held) in enumerate(skf.split(pool.path, pool.label)):
        pool.loc[held, "fold"] = k
    args.out.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.out / "folds.csv", index=False)

    for k in range(args.folds):
        held = pool[pool.fold == k]
        rest = pool[pool.fold != k]
        tr, va = train_test_split(rest, test_size=args.inner_val, stratify=rest.label, random_state=args.seed)
        d = args.out / f"fold{k}"
        d.mkdir(exist_ok=True)
        for name, part in [("train", tr), ("val", va), ("test", held)]:
            part[["path", "label"]].sort_values("path").to_csv(d / f"{name}.csv", index=False)

    table = pool.groupby(["label", "fold"]).size().unstack(fill_value=0)
    table["total"] = table.sum(axis=1)
    print(f"CV pool: {len(pool)} images, {args.folds} folds (fixed test set of {len(test)} images kept apart)")
    print(table.to_string())
    print(f"Inner split per fold: ~{len(rest) - len(va)} train / {len(va)} val, held-out fold ~{len(held)} images")
    print(f"Written to {args.out}/")


if __name__ == "__main__":
    main()
