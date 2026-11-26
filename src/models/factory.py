"""Model factory + the three transfer-learning strategies.

Strategies (for pretrained models):
    feature_extraction : backbone frozen, only the new classification layer is trained.
    partial_finetune   : the last stage(s) of the backbone + the classifier are trained.
    full_finetune      : every parameter is trained (backbone with a smaller learning rate).

Which modules are "head" and "last stage" for each architecture is defined in ``SPECS``
so that the definition of each strategy is explicit and documented in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from src.models.scratch_cnn import ScratchCNN


@dataclass(frozen=True)
class ModelSpec:
    source: str                 # "timm" | "torchvision" | "scratch"
    name: str                   # timm / torchvision identifier
    head: tuple[str, ...]       # parameter-name prefixes of the new classifier
    partial: tuple[str, ...]    # prefixes additionally unfrozen in partial fine-tuning


SPECS: dict[str, ModelSpec] = {
    "scratch_cnn": ModelSpec("scratch", "scratch_cnn", head=("head",), partial=()),
    # AlexNet: 5 conv layers (features.0/3/6/8/10) + 3 FC layers (classifier.1/4/6).
    # FE trains the new last FC layer; partial adds the last 2 conv layers and FC6/FC7.
    "alexnet": ModelSpec("torchvision", "alexnet", head=("classifier.6",),
                         partial=("features.8", "features.10", "classifier")),
    # ConvNeXtV2: stem + 4 stages + head(norm, fc). Partial = last stage + head.
    "convnextv2": ModelSpec("timm", "convnextv2_tiny.fcmae_ft_in22k_in1k", head=("head.fc",),
                            partial=("stages.3", "head")),
    # NFNet-F1: stem + 4 stages + final_conv + head. Partial = last stage + final conv + head.
    "nfnet_f1": ModelSpec("timm", "dm_nfnet_f1.dm_in1k", head=("head.fc",),
                          partial=("stages.3", "final_conv", "head")),
    # ViT-B/16: patch embedding + 12 transformer blocks + norm + head. Partial = last 4 blocks.
    "vit_b16": ModelSpec("timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k", head=("head",),
                         partial=("blocks.8", "blocks.9", "blocks.10", "blocks.11",
                                  "norm", "fc_norm", "head")),
}

STRATEGIES = ("feature_extraction", "partial_finetune", "full_finetune")


def create_model(model: str, num_classes: int, pretrained: bool = True,
                 timm_name: str | None = None) -> nn.Module:
    spec = SPECS[model]
    if spec.source == "scratch":
        return ScratchCNN(num_classes)
    if spec.source == "torchvision":
        from torchvision.models import AlexNet_Weights, alexnet

        net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier[6] = nn.Linear(net.classifier[6].in_features, num_classes)
        return net
    import timm

    return timm.create_model(timm_name or spec.name, pretrained=pretrained, num_classes=num_classes)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def apply_strategy(net: nn.Module, model: str, strategy: str) -> None:
    """Set ``requires_grad`` according to the transfer-learning strategy."""
    spec = SPECS[model]
    if spec.source == "scratch" or strategy == "full_finetune":
        for p in net.parameters():
            p.requires_grad = True
        return
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}, choose from {STRATEGIES}")
    trainable = spec.head if strategy == "feature_extraction" else spec.head + spec.partial
    for name, p in net.named_parameters():
        p.requires_grad = _matches(name, trainable)
    if not any(p.requires_grad for p in net.parameters()):
        raise RuntimeError(f"No trainable parameter for {model}/{strategy}: check SPECS prefixes")


def param_groups(net: nn.Module, model: str, lr_head: float, lr_backbone: float,
                 weight_decay: float) -> list[dict]:
    """Head trained with ``lr_head``, unfrozen backbone layers with ``lr_backbone``
    (discriminative learning rates). The scratch CNN uses ``lr_head`` everywhere."""
    spec = SPECS[model]
    head, backbone = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        is_head = spec.source == "scratch" or _matches(name, spec.head)
        (head if is_head else backbone).append(p)
    groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay}]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay})
    return groups


def count_parameters(net: nn.Module) -> dict:
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in net.parameters()) / 1024**2
    return {"params_total": total, "params_trainable": trainable, "model_size_mb": round(size_mb, 2)}
