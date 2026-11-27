"""Grad-CAM explanations for a trained run (all 5 architectures, CNNs and ViT).

Grad-CAM (Selvaraju et al., 2017) weights the feature maps of a layer by the gradient of the
predicted class score and shows which image regions drove the decision.
For ViT-B/16 the same method is applied to the token features of the last transformer block,
reshaped to the 14x14 patch grid (the class token is dropped).

Besides the figures, this script computes a simple quantitative check of the dataset bias found
in the EDA: the **road-focus ratio** = share of the Grad-CAM energy that falls in the lower half
of the image, where the road surface is in most of our (forward-facing) photos. A ratio close to
0.5 or lower means the model also relies on the upper part (sky, buildings, vegetation).
It is only a proxy (the road does not always fill the lower half) and is reported as such.

Outputs in ``<out>/<model>__<strategy>/``:
    correct.png        examples correctly classified (per class, highest confidence)
    errors.png         misclassified test images
    road_focus.json    road-focus ratio over the whole test set (overall and per class)

Usage
-----
    python -m src.explain.gradcam --run <runs>/<model>__<strategy>__<stamp> --data-root data/raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.transforms import eval_transforms
from src.models.factory import create_model


def target_layers(net, model: str):
    """Last convolutional feature map (CNNs) or last transformer block (ViT)."""
    if model == "scratch_cnn":
        return [net.features[-1]]
    if model == "alexnet":
        return [net.features[10]]          # last conv layer (conv5)
    if model == "convnextv2":
        return [net.stages[-1]]
    if model == "nfnet_f1":
        return [net.final_conv]
    if model == "vit_b16":
        return [net.blocks[-1].norm1]
    raise ValueError(model)


def vit_reshape(net):
    """(B, 1+196, C) tokens -> (B, C, 14, 14) maps, dropping the class/prefix tokens."""
    n_prefix = getattr(net, "num_prefix_tokens", 1)

    def reshape(t):
        t = t[:, n_prefix:, :]
        side = int(t.shape[1] ** 0.5)
        return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)
    return reshape


def load_run(run: Path, device):
    cfg = json.load(open(run / "config.json"))
    p = np.load(run / "predictions.npz", allow_pickle=True)
    classes = [str(c) for c in p["classes"]]
    net = create_model(cfg["model"], len(classes), pretrained=False, timm_name=cfg.get("timm_name"))
    net.load_state_dict(torch.load(run / "best.pt", map_location=device))
    return cfg, net.to(device).eval(), p, classes


def load_image(path: Path, size: int):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    tensor = eval_transforms(size)(image=rgb)["image"]
    shown = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    return tensor, shown


def road_focus(cam: np.ndarray) -> float:
    h = cam.shape[0] // 2
    return float(cam[h:].sum() / max(cam.sum(), 1e-8))


def grid(items, classes, title, path: Path, cols: int = 4) -> None:
    if not items:
        return
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 2.1, rows * 2.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (img, overlay, y, pred, conf) in enumerate(items):
        r, c = divmod(k, cols)
        axes[r][2 * c].imshow(img)
        axes[r][2 * c].set_title(f"true: {classes[y]}", fontsize=8)
        axes[r][2 * c + 1].imshow(overlay)
        axes[r][2 * c + 1].set_title(f"pred: {classes[pred]} ({conf:.2f})", fontsize=8,
                                     color="green" if y == pred else "red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run folder written by src.train")
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("results/gradcam"))
    ap.add_argument("--per-class", type=int, default=2, help="correct examples shown per class")
    ap.add_argument("--max-errors", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, net, preds, classes = load_run(args.run, device)
    model, size = cfg["model"], cfg["data"]["image_size"]
    out = args.out / f"{model}__{cfg['strategy']}"
    out.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model=net, target_layers=target_layers(net, model),
                  reshape_transform=vit_reshape(net) if model == "vit_b16" else None)

    y_true, probs, paths = preds["y_true"], preds["probs"], preds["paths"]
    y_pred = probs.argmax(1)
    conf = probs.max(1)

    # Images shown in the figures: most confident correct predictions per class, then errors
    correct_idx = []
    for k in range(len(classes)):
        idx = np.where((y_true == k) & (y_pred == k))[0]
        correct_idx += list(idx[np.argsort(-conf[idx])][:args.per_class])
    err = np.where(y_true != y_pred)[0]
    error_idx = list(err[np.argsort(-conf[err])][:args.max_errors])
    keep = set(correct_idx) | set(error_idx)

    # 1) road-focus ratio on every test image (Grad-CAM w.r.t. the predicted class)
    ratios, per_class = [], {c: [] for c in classes}
    examples = {}
    for i, rel in enumerate(paths):
        tensor, shown = load_image(args.data_root / str(rel), size)
        heat = cam(input_tensor=tensor[None].to(device),
                   targets=[ClassifierOutputTarget(int(y_pred[i]))])[0]
        ratio = road_focus(heat)
        ratios.append(ratio)
        per_class[classes[y_true[i]]].append(ratio)
        if i in keep:
            examples[i] = (shown, show_cam_on_image(shown, heat, use_rgb=True))

    stats = {"model": model, "strategy": cfg["strategy"],
             "road_focus_mean": float(np.mean(ratios)),
             "road_focus_per_class": {c: float(np.mean(v)) for c, v in per_class.items() if v},
             "road_focus_correct": float(np.mean([r for r, t, p in zip(ratios, y_true, y_pred) if t == p] or [np.nan])),
             "road_focus_errors": float(np.mean([r for r, t, p in zip(ratios, y_true, y_pred) if t != p] or [np.nan])),
             "n_test": int(len(ratios)), "n_errors": int((y_true != y_pred).sum())}
    json.dump(stats, open(out / "road_focus.json", "w"), indent=2)

    # 2) figures
    def item(i):
        return (*examples[i], int(y_true[i]), int(y_pred[i]), float(conf[i]))
    correct = [item(i) for i in correct_idx]
    wrong = [item(i) for i in error_idx]
    name = f"{model} — {cfg['strategy']}"
    grid(correct, classes, f"Grad-CAM, correct predictions ({name})", out / "correct.png")
    grid(wrong, classes, f"Grad-CAM, misclassified images ({name})", out / "errors.png")
    print(json.dumps(stats, indent=2))
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
