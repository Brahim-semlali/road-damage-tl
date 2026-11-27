"""Compare all training runs: global results table + figures for the paper (Section IV).

Reads every ``<runs>/<model>__<strategy>__<timestamp>/`` folder written by ``src.train``
(GPU runs only; the latest run is kept for each model/strategy pair) and writes into ``--out``:

    global_results.csv / .md     table IV.6 (accuracy, precision, recall, F1, AUC, params, time…)
    roc/<model>__<strategy>.png  one-vs-rest ROC curve of every class (+ macro average)
    roc_best_per_model.png       macro-average ROC of the best strategy of each model
    confusion_best_per_model.png normalised confusion matrices of the best strategy of each model
    val_curves.png               validation accuracy/loss per epoch, one panel per strategy
    accuracy_vs_cost.png         F1 vs number of parameters and vs training time

Usage
-----
    python -m src.compare --runs /content/drive/MyDrive/road-damage-tl/runs --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

MODEL_ORDER = ["scratch_cnn", "alexnet", "convnextv2", "nfnet_f1", "vit_b16"]
MODEL_NAMES = {"scratch_cnn": "CNN (scratch)", "alexnet": "AlexNet", "convnextv2": "ConvNeXtV2-T",
               "nfnet_f1": "NFNet-F1", "vit_b16": "ViT-B/16"}
STRATEGY_ORDER = ["from_scratch", "feature_extraction", "partial_finetune", "full_finetune"]
STRATEGY_NAMES = {"from_scratch": "From scratch", "feature_extraction": "Feature extraction",
                  "partial_finetune": "Partial fine-tuning", "full_finetune": "Full fine-tuning"}
MARKERS = {"from_scratch": "*", "feature_extraction": "o", "partial_finetune": "s", "full_finetune": "^"}


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for mfile in runs_dir.glob("*/metrics.json"):
        run = mfile.parent
        model, strategy, stamp = run.name.split("__")
        metrics = json.load(open(mfile))
        if metrics.get("device", "cpu") == "cpu":
            print(f"Skipped (trained on CPU, not comparable): {run.name}")
            continue
        rows.append({"model": model, "strategy": strategy, "stamp": stamp, "dir": run, **metrics})
    if not rows:
        raise SystemExit(f"No finished run (metrics.json) found in {runs_dir}")
    df = pd.DataFrame(rows).sort_values("stamp").groupby(["model", "strategy"]).tail(1)
    df["m_order"] = df.model.map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df["s_order"] = df.strategy.map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(99)
    return df.sort_values(["m_order", "s_order"]).reset_index(drop=True)


def results_table(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    t = pd.DataFrame({
        "Model": df.model.map(MODEL_NAMES).fillna(df.model),
        "Strategy": df.strategy.map(STRATEGY_NAMES).fillna(df.strategy),
        "Accuracy": df.test_accuracy, "Precision": df.test_precision_macro,
        "Recall": df.test_recall_macro, "F1-score": df.test_f1_macro, "AUC": df.test_auc_macro_ovr,
        "Params (M)": df.params_total / 1e6, "Trainable (M)": df.params_trainable / 1e6,
        "Training time (min)": df.train_time_s / 60, "Peak GPU mem (MB)": df.peak_gpu_mem_mb,
        "Latency (ms)": df.latency_ms, "Best epoch": df.best_epoch,
    })
    t.to_csv(out / "global_results.csv", index=False)
    fmt = t.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]:
        fmt[c] = fmt[c].map("{:.4f}".format)
    for c in ["Params (M)", "Trainable (M)", "Training time (min)", "Latency (ms)"]:
        fmt[c] = fmt[c].map("{:.2f}".format)
    fmt["Peak GPU mem (MB)"] = fmt["Peak GPU mem (MB)"].map("{:.0f}".format)
    best = t["F1-score"].idxmax()
    fmt.loc[best, "F1-score"] = f"**{fmt.loc[best, 'F1-score']}**"
    lines = ["| " + " | ".join(fmt.columns) + " |", "|" + "---|" * len(fmt.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in fmt.values]
    (out / "global_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return t


def load_preds(run_dir: Path):
    p = np.load(run_dir / "predictions.npz", allow_pickle=True)
    return p["y_true"], p["probs"], [str(c) for c in p["classes"]]


def macro_roc(y, probs, n_classes):
    """Per-class one-vs-rest ROC curves and their macro average (mean TPR on a common FPR grid)."""
    grid = np.linspace(0, 1, 501)
    curves, tprs = [], []
    for k in range(n_classes):
        fpr, tpr, _ = roc_curve(y == k, probs[:, k])
        curves.append((fpr, tpr, auc(fpr, tpr)))
        tprs.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    return curves, (grid, mean_tpr, auc(grid, mean_tpr))


def plot_roc_per_run(df: pd.DataFrame, out: Path) -> None:
    (out / "roc").mkdir(exist_ok=True)
    for _, r in df.iterrows():
        y, probs, classes = load_preds(r.dir)
        curves, (g, m, a) = macro_roc(y, probs, len(classes))
        fig, ax = plt.subplots(figsize=(5, 5))
        for (fpr, tpr, ak), c in zip(curves, classes):
            ax.plot(fpr, tpr, lw=1.5, label=f"{c} (AUC={ak:.3f})")
        ax.plot(g, m, "k--", lw=2, label=f"macro avg (AUC={a:.3f})")
        ax.plot([0, 1], [0, 1], ":", color="grey")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"{MODEL_NAMES.get(r.model, r.model)} — {STRATEGY_NAMES.get(r.strategy, r.strategy)}")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "roc" / f"{r.model}__{r.strategy}.png", dpi=200)
        plt.close(fig)


def best_per_model(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby("model").test_f1_macro.idxmax()].sort_values("m_order")


def plot_best_roc(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for _, r in best_per_model(df).iterrows():
        y, probs, classes = load_preds(r.dir)
        _, (g, m, a) = macro_roc(y, probs, len(classes))
        ax.plot(g, m, lw=2, label=f"{MODEL_NAMES.get(r.model, r.model)} "
                                  f"({STRATEGY_NAMES.get(r.strategy, r.strategy)}, AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Macro-average ROC — best strategy of each model")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_best_per_model.png", dpi=200)
    plt.close(fig)


def plot_best_confusions(df: pd.DataFrame, out: Path) -> None:
    best = best_per_model(df)
    fig, axes = plt.subplots(1, len(best), figsize=(4.2 * len(best), 4.2), squeeze=False)
    for ax, (_, r) in zip(axes[0], best.iterrows()):
        y, probs, classes = load_preds(r.dir)
        cm = confusion_matrix(y, probs.argmax(1), labels=range(len(classes)), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set(xlabel="Predicted", ylabel="True",
               title=f"{MODEL_NAMES.get(r.model, r.model)}\n{STRATEGY_NAMES.get(r.strategy, r.strategy)}")
    fig.tight_layout()
    fig.savefig(out / "confusion_best_per_model.png", dpi=200)
    plt.close(fig)


def plot_val_curves(df: pd.DataFrame, out: Path) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in set(df.strategy)]
    fig, axes = plt.subplots(2, len(strategies), figsize=(4.5 * len(strategies), 7), squeeze=False)
    for j, s in enumerate(strategies):
        for _, r in df[df.strategy == s].iterrows():
            h = pd.read_csv(r.dir / "history.csv")
            label = MODEL_NAMES.get(r.model, r.model)
            axes[0][j].plot(h.epoch, h.val_acc, label=label)
            axes[1][j].plot(h.epoch, h.val_loss, label=label)
        axes[0][j].set(title=STRATEGY_NAMES.get(s, s), ylabel="Validation accuracy")
        axes[1][j].set(xlabel="Epoch", ylabel="Validation loss")
        for a in (axes[0][j], axes[1][j]):
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "val_curves.png", dpi=200)
    plt.close(fig)


def plot_cost(df: pd.DataFrame, out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {m: f"C{i}" for i, m in enumerate(MODEL_ORDER)}
    for _, r in df.iterrows():
        kw = dict(color=colors.get(r.model, "k"), marker=MARKERS.get(r.strategy, "x"), s=80)
        a1.scatter(r.params_trainable / 1e6 + 1e-3, r.test_f1_macro, **kw)
        a2.scatter(r.train_time_s / 60, r.test_f1_macro, **kw)
    a1.set(xscale="log", xlabel="Trainable parameters (M, log scale)", ylabel="Test macro F1")
    a2.set(xlabel="Training time (min)", ylabel="Test macro F1")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=colors[m], marker="o", ls="", label=MODEL_NAMES[m])
               for m in MODEL_ORDER if m in set(df.model)]
    handles += [plt.Line2D([], [], color="k", marker=MARKERS[s], ls="", label=STRATEGY_NAMES[s])
                for s in STRATEGY_ORDER if s in set(df.strategy)]
    a2.legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("Performance vs computational cost")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_cost.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = collect_runs(args.runs)
    table = results_table(df, args.out)
    plot_roc_per_run(df, args.out)
    plot_best_roc(df, args.out)
    plot_best_confusions(df, args.out)
    plot_val_curves(df, args.out)
    plot_cost(df, args.out)

    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    b = table.loc[table["F1-score"].idxmax()]
    print(f"\nBest macro-F1: {b.Model} / {b.Strategy} ({b['F1-score']:.4f})")
    print(f"Figures and tables written to {args.out}/")


if __name__ == "__main__":
    main()
