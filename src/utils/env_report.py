"""Describe the software/hardware environment (reproducibility).

``environment_info()`` is stored in the config.json of every training run.
Run as a script on Colab to write the locked versions of the packages used by this project:

    python -m src.utils.env_report --out env

writes ``env/environment.md`` (hardware + versions) and ``env/requirements-lock.txt``
(exact versions, installable with ``pip install -r env/requirements-lock.txt``).
"""
from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata
from pathlib import Path

# Distribution names of the packages imported by the project
PACKAGES = ["torch", "torchvision", "timm", "albumentations", "opencv-python", "opencv-python-headless",
            "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pillow", "grad-cam", "mlflow",
            "pyyaml", "tqdm", "huggingface-hub", "safetensors", "kagglehub"]


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), **package_versions()}
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_memory_gb"] = round(p.total_memory / 1024**3, 1)
    except Exception:  # torch missing or broken: report what we have
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("env"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    info = environment_info()
    versions = package_versions()
    lock = [f"{k}=={v}" for k, v in versions.items() if k not in ("opencv-python-headless", "kagglehub")]
    (args.out / "requirements-lock.txt").write_text(
        f"# Exact versions used for the experiments (Google Colab, Python {info['python']}).\n"
        + "\n".join(lock) + "\n", encoding="utf-8")
    lines = ["| Component | Version |", "|---|---|"] + [f"| {k} | {v} |" for k, v in info.items()]
    (args.out / "environment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {args.out}/ (python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()
