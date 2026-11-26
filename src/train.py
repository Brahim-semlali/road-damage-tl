"""Single training script for ALL models and strategies (same conditions for every run).

Examples
--------
    # Baseline CNN from scratch
    python -m src.train --model scratch_cnn --data-root /content/data/raw --out runs

    # A pretrained model with one of the three transfer-learning strategies
    python -m src.train --model convnextv2 --strategy feature_extraction --data-root ... --out ...

Every run writes into ``<out>/<model>__<strategy>__<timestamp>/``:
    config.json         exact configuration used
    history.csv         train/val loss, accuracy and macro-F1 per epoch
    curves.png          learning curves (accuracy and loss, train vs validation)
    best.pt             weights of the epoch with the lowest validation loss
    metrics.json        test metrics + parameters, training time, GPU memory, latency
    predictions.npz     test labels and predicted probabilities (for ROC curves, error analysis)
    confusion_matrix.png
and logs the same information to MLflow (if installed).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader

from src.data.dataset import RoadDamageDataset
from src.data.transforms import eval_transforms, train_transforms
from src.models.factory import apply_strategy, count_parameters, create_model, param_groups
from src.utils.seed import set_seed


# --------------------------------------------------------------------------- configuration
def load_config() -> dict:
    ap = argparse.ArgumentParser(description="Train one model with one strategy.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model")
    ap.add_argument("--strategy")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation (effective batch = batch-size x accum-steps)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lr-backbone", type=float)
    ap.add_argument("--timm-name", help="override the timm model id (e.g. convnextv2_base)")
    ap.add_argument("--data-root", default="data/raw", help="folder the split CSV paths are relative to")
    ap.add_argument("--out", default="runs", help="output folder (use Google Drive on Colab)")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (debug only)")
    ap.add_argument("--max-batches", type=int, default=0, help="debug: limit batches per epoch")
    ap.add_argument("--mlflow-uri", default=None,
                    help="tracking DB, e.g. sqlite:////content/drive/MyDrive/road-damage-tl/mlflow.db")
    ap.add_argument("--mlflow-artifacts", default=None,
                    help="folder for MLflow artifacts (used when the experiment is created)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    t, d = cfg["train"], cfg["data"]
    for key, val in [("model", a.model), ("strategy", a.strategy)]:
        if val is not None:
            cfg[key] = val
    for key, val in [("epochs", a.epochs), ("batch_size", a.batch_size), ("lr", a.lr),
                     ("lr_backbone", a.lr_backbone)]:
        if val is not None:
            t[key] = val
    if a.num_workers is not None:
        d["num_workers"] = a.num_workers
    if cfg["model"] == "scratch_cnn":
        cfg["strategy"] = "from_scratch"
    t["accum_steps"] = a.accum_steps
    cfg.update(timm_name=a.timm_name, data_root=a.data_root, out=a.out,
               pretrained=not a.no_pretrained, max_batches=a.max_batches, mlflow_uri=a.mlflow_uri,
               mlflow_artifacts=a.mlflow_artifacts)
    return cfg


# --------------------------------------------------------------------------- MLflow
def setup_mlflow(cfg: dict, run_name: str, params: dict):
    """Start an MLflow run. MLflow is optional: any problem only disables it, never the training
    (all results are written to files anyway)."""
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed: logging to files only.")
        return None
    try:
        if cfg["mlflow_uri"]:
            mlflow.set_tracking_uri(cfg["mlflow_uri"])
        name = cfg["tracking"]["mlflow_experiment"]
        if cfg["mlflow_artifacts"] and mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=cfg["mlflow_artifacts"])
        mlflow.set_experiment(name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        return mlflow
    except Exception as e:
        print(f"WARNING: MLflow disabled ({type(e).__name__}: {str(e)[:200]}). "
              "Results are still saved to files.")
        return None


def mlflow_call(mlflow, method: str, *args, **kwargs) -> None:
    if mlflow is None:
        return
    try:
        getattr(mlflow, method)(*args, **kwargs)
    except Exception as e:
        print(f"WARNING: mlflow.{method} failed ({type(e).__name__}: {str(e)[:200]})")


# --------------------------------------------------------------------------- helpers
def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict(net, loader, criterion, device, amp: bool, max_batches: int = 0):
    net.eval()
    losses, probs, targets = [], [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = net(x)
        losses.append(criterion(logits.float(), y).item() * len(y))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        targets.append(y.cpu())
    probs, targets = torch.cat(probs).numpy(), torch.cat(targets).numpy()
    return sum(losses) / len(targets), targets, probs


@torch.no_grad()
def measure_latency(net, device, image_size: int, runs: int = 50) -> float:
    """Mean inference time (ms) for one image (batch size 1)."""
    net.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(10):  # warm-up
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def plot_curves(hist: pd.DataFrame, path: Path, title: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist.epoch, hist.train_acc, label="train")
    a1.plot(hist.epoch, hist.val_acc, label="validation")
    a1.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    a2.plot(hist.epoch, hist.train_loss, label="train")
    a2.plot(hist.epoch, hist.val_loss, label="validation")
    a2.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    for a in (a1, a2):
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def test_metrics(y, probs, classes) -> dict:
    pred = probs.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    m = {"test_accuracy": accuracy_score(y, pred), "test_precision_macro": p,
         "test_recall_macro": r, "test_f1_macro": f1}
    try:
        m["test_auc_macro_ovr"] = roc_auc_score(y, probs, multi_class="ovr", average="macro",
                                                labels=list(range(len(classes))))
    except ValueError:  # a class missing from y (only possible in debug runs)
        m["test_auc_macro_ovr"] = float("nan")
    return {k: float(v) for k, v in m.items()}


# --------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    t, d = cfg["train"], cfg["data"]
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t["mixed_precision"]) and device.type == "cuda"

    run_name = f"{cfg['model']}__{cfg['strategy']}"
    out = Path(cfg["out"]) / f"{run_name}__{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"Run: {run_name} | device: {device} | AMP: {amp} | output: {out}")

    # ---- data
    splits = Path(d["splits_dir"])
    train_ds = RoadDamageDataset(splits / "train.csv", cfg["data_root"], train_transforms(d["image_size"]))
    val_ds = RoadDamageDataset(splits / "val.csv", cfg["data_root"], eval_transforms(d["image_size"]))
    test_ds = RoadDamageDataset(splits / "test.csv", cfg["data_root"], eval_transforms(d["image_size"]))
    g = torch.Generator().manual_seed(cfg["seed"])
    kw = dict(batch_size=t["batch_size"], num_workers=d["num_workers"], pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)
    classes = train_ds.classes

    # ---- model + strategy
    net = create_model(cfg["model"], len(classes), cfg["pretrained"], cfg["timm_name"])
    apply_strategy(net, cfg["model"], cfg["strategy"])
    net.to(device)
    info = count_parameters(net)
    print(f"Parameters: {info['params_total']:,} total, {info['params_trainable']:,} trainable")

    # Weighted loss against class imbalance (weights computed on the TRAIN split only)
    criterion = nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device),
                                    label_smoothing=t["label_smoothing"])
    optimizer = torch.optim.AdamW(param_groups(net, cfg["model"], t["lr"], t["lr_backbone"],
                                               t["weight_decay"]))
    steps_per_epoch = len(train_dl) // t["accum_steps"]
    if cfg["max_batches"]:
        steps_per_epoch = max(1, min(len(train_dl), cfg["max_batches"]) // t["accum_steps"])
    scheduler = cosine_with_warmup(optimizer, t["warmup_epochs"] * steps_per_epoch,
                                   t["epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    mlflow = setup_mlflow(cfg, run_name, {
        "model": cfg["model"], "strategy": cfg["strategy"], "timm_name": cfg["timm_name"],
        "image_size": d["image_size"], **t, **info})

    # ---- training loop
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_loss, best_epoch, patience = [], float("inf"), 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        net.train()
        run_loss, correct, seen = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl):
            if cfg["max_batches"] and i >= cfg["max_batches"]:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = net(x)
                loss = criterion(logits.float(), y)
            scaler.scale(loss / t["accum_steps"]).backward()
            if (i + 1) % t["accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        val_loss, vy, vprobs = predict(net, val_dl, criterion, device, amp, cfg["max_batches"])
        row = {"epoch": epoch, "train_loss": run_loss / seen, "train_acc": correct / seen,
               "val_loss": val_loss, "val_acc": accuracy_score(vy, vprobs.argmax(1)),
               "val_f1_macro": f1_score(vy, vprobs.argmax(1), average="macro", zero_division=0),
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"Epoch {epoch:3d}/{t['epochs']} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f}"
              f" | val loss {val_loss:.4f} acc {row['val_acc']:.3f} f1 {row['val_f1_macro']:.3f}")
        mlflow_call(mlflow, "log_metrics", {k: v for k, v in row.items() if k != "epoch"}, step=epoch)

        if val_loss < best_loss:  # checkpoint on the best validation loss
            best_loss, best_epoch, patience = val_loss, epoch, 0
            torch.save(net.state_dict(), out / "best.pt")
        else:
            patience += 1
            if patience >= t["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break
    train_time = time.perf_counter() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0

    hist = pd.DataFrame(history)
    hist.to_csv(out / "history.csv", index=False)
    plot_curves(hist, out / "curves.png", run_name)

    # ---- test evaluation with the best checkpoint (the test set is used ONCE, here)
    net.load_state_dict(torch.load(out / "best.pt", map_location=device))
    _, ty, tprobs = predict(net, test_dl, criterion, device, amp, cfg["max_batches"])
    metrics = test_metrics(ty, tprobs, classes)
    metrics.update(info, best_epoch=best_epoch, epochs_run=len(history),
                   train_time_s=round(train_time, 1), peak_gpu_mem_mb=round(peak_mem, 1),
                   latency_ms=round(measure_latency(net, device, d["image_size"]), 2),
                   device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    np.savez(out / "predictions.npz", y_true=ty, probs=tprobs, classes=np.array(classes),
             paths=test_ds.df.path.values[:len(ty)])

    labels = list(range(len(classes)))
    cm = confusion_matrix(ty, tprobs.argmax(1), labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", colorbar=False)
    plt.title(run_name)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close()
    print(classification_report(ty, tprobs.argmax(1), labels=labels, target_names=classes,
                                digits=3, zero_division=0))
    print(json.dumps(metrics, indent=2))

    mlflow_call(mlflow, "log_metrics",
                {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not math.isnan(v)})
    for f in ("history.csv", "curves.png", "metrics.json", "confusion_matrix.png", "config.json"):
        mlflow_call(mlflow, "log_artifact", str(out / f))
    mlflow_call(mlflow, "end_run")


if __name__ == "__main__":
    main()
