"""Summarise the K-fold cross-validation runs (see ``src/data/make_folds.py`` for the protocol).

Reads ``<runs>/fold<k>/<model>__<strategy>__<timestamp>/metrics.json`` (GPU runs only; the latest
run is kept for each model/strategy/fold) and writes into ``--out``:

    cv_results.csv / .md     mean ± std over the folds, on the held-out folds (CV estimate) and on the
                             fixed test set (same 310 images for every fold model)
    cv_per_fold.csv          one row per model/strategy/fold
    cv_pairwise.md           paired comparison of the configurations on the same folds
                             (mean F1 difference, paired t-test and Wilcoxon p-values)
    cv_f1_per_fold.png       macro F1 of every fold, per configuration

Usage
-----
    python -m src.cv_summary --runs /content/drive/MyDrive/road-damage-tl/runs_cv --out results/cv
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.compare import MODEL_NAMES, MODEL_ORDER, STRATEGY_NAMES, STRATEGY_ORDER

METRICS = {"test_accuracy": "Accuracy", "test_precision_macro": "Precision", "test_recall_macro": "Recall",
           "test_f1_macro": "F1-score", "test_auc_macro_ovr": "AUC"}


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for mfile in runs_dir.glob("fold*/*/metrics.json"):
        m = json.load(open(mfile))
        if m.get("device", "cpu") == "cpu":
            print(f"Skipped (trained on CPU): {mfile.parent}")
            continue
        model, strategy, stamp = mfile.parent.name.split("__")
        rows.append({"model": model, "strategy": strategy, "fold": int(mfile.parent.parent.name[4:]),
                     "stamp": stamp, **m})
    if not rows:
        raise SystemExit(f"No finished cross-validation run found in {runs_dir}")
    df = pd.DataFrame(rows).sort_values("stamp").groupby(["model", "strategy", "fold"]).tail(1)
    df["config"] = (df.model.map(MODEL_NAMES).fillna(df.model) + " / "
                    + df.strategy.map(STRATEGY_NAMES).fillna(df.strategy))
    df["m_order"] = df.model.map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df["s_order"] = df.strategy.map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(99)
    return df.sort_values(["m_order", "s_order", "fold"]).reset_index(drop=True)


def summary_table(df: pd.DataFrame, n_folds: int, out: Path) -> pd.DataFrame:
    rows = []
    for (cfg, model, strategy), g in df.groupby(["config", "model", "strategy"], sort=False):
        r = {"Configuration": cfg, "Folds": f"{len(g)}/{n_folds}"}
        for key, name in METRICS.items():
            r[f"CV {name}"] = g[key].mean()
            r[f"CV {name} std"] = g[key].std(ddof=1) if len(g) > 1 else np.nan
        if "extra_test_f1_macro" in g:
            r["Fixed-test F1"] = g["extra_test_f1_macro"].mean()
            r["Fixed-test F1 std"] = g["extra_test_f1_macro"].std(ddof=1) if len(g) > 1 else np.nan
            r["Fixed-test AUC"] = g["extra_test_auc_macro_ovr"].mean()
        r["Training time (min)"] = g.train_time_s.mean() / 60
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "cv_results.csv", index=False)

    def pm(m, s):
        return f"{m:.4f}" if np.isnan(s) else f"{m:.4f} ± {s:.4f}"
    md = pd.DataFrame({"Configuration": t.Configuration, "Folds": t.Folds})
    for name in METRICS.values():
        md[f"CV {name}"] = [pm(m, s) for m, s in zip(t[f"CV {name}"], t[f"CV {name} std"])]
    if "Fixed-test F1" in t:
        md["Fixed-test F1"] = [pm(m, s) for m, s in zip(t["Fixed-test F1"], t["Fixed-test F1 std"])]
        md["Fixed-test AUC"] = t["Fixed-test AUC"].map("{:.4f}".format)
    md["Training time (min)"] = t["Training time (min)"].map("{:.1f}".format)
    lines = ["| " + " | ".join(md.columns) + " |", "|" + "---|" * len(md.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in md.values]
    lines += ["", f"Stratified {n_folds}-fold cross-validation on the train+val pool (1,756 images); "
              "CV = metrics on the held-out folds (mean ± std over folds). "
              "Fixed test = the same 310 test images, scored by each fold model."]
    (out / "cv_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return t


def pairwise(df: pd.DataFrame, out: Path) -> None:
    """Paired comparison on the folds that both configurations completed."""
    piv = df.pivot_table(index="fold", columns="config", values="test_f1_macro")
    lines = ["| Configuration A | Configuration B | Folds | Mean F1 (A - B) | Paired t-test p | Wilcoxon p |",
             "|---|---|---|---|---|---|"]
    for a, b in itertools.combinations(piv.columns, 2):
        pair = piv[[a, b]].dropna()
        if len(pair) < 2:
            continue
        diff = pair[a] - pair[b]
        t_p = stats.ttest_rel(pair[a], pair[b]).pvalue if diff.std() > 0 else np.nan
        try:
            w_p = stats.wilcoxon(pair[a], pair[b]).pvalue
        except ValueError:
            w_p = np.nan
        lines.append(f"| {a} | {b} | {len(pair)} | {diff.mean():+.4f} | {t_p:.3f} | {w_p:.3f} |")
    lines += ["", "With 5 folds the tests have little power (the smallest possible two-sided Wilcoxon "
              "p-value is 0.0625): a large p-value means 'no detectable difference', not 'equal'."]
    (out / "cv_pairwise.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_folds(df: pd.DataFrame, out: Path) -> None:
    configs = list(dict.fromkeys(df.config))
    fig, ax = plt.subplots(figsize=(max(5, 2.2 * len(configs)), 4.5))
    for i, c in enumerate(configs):
        g = df[df.config == c]
        ax.boxplot(g.test_f1_macro, positions=[i], widths=0.5, showmeans=True)
        ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)), g.test_f1_macro, s=25, zorder=3)
    ax.set_xticks(range(len(configs)), [c.replace(" / ", "\n") for c in configs], fontsize=9)
    ax.set(ylabel="Macro F1 on the held-out fold", title="Cross-validation: macro F1 per fold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "cv_f1_per_fold.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs_cv"))
    ap.add_argument("--out", type=Path, default=Path("results/cv"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = collect(args.runs)
    keep = ["model", "strategy", "fold", *METRICS, "train_time_s", "best_epoch"]
    keep += [c for c in df.columns if c.startswith("extra_test_")]
    df[keep].to_csv(args.out / "cv_per_fold.csv", index=False)
    table = summary_table(df, args.folds, args.out)
    pairwise(df, args.out)
    plot_folds(df, args.out)

    print(open(args.out / "cv_results.md", encoding="utf-8").read())
    print(open(args.out / "cv_pairwise.md", encoding="utf-8").read())
    incomplete = table[table.Folds != f"{args.folds}/{args.folds}"]
    if len(incomplete):
        print("NOT FINISHED:", ", ".join(f"{c} ({f})" for c, f in zip(incomplete.Configuration, incomplete.Folds)))
    print(f"Written to {args.out}/")


if __name__ == "__main__":
    main()
