"""PyTorch Dataset reading the versioned split CSVs (data/splits/*.csv)."""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ["good", "satisfactory", "poor", "very_poor"]  # ordered from best to worst road state


class RoadDamageDataset(Dataset):
    """Images listed in a split CSV (columns: ``path``, ``label``).

    Args:
        csv_file: one of data/splits/{train,val,test}.csv
        root: folder the CSV paths are relative to (e.g. data/raw or /content/data/raw)
        transform: Albumentations pipeline (must end with ToTensorV2)
    """

    def __init__(self, csv_file: str | Path, root: str | Path, transform=None,
                 classes: list[str] = CLASSES):
        self.df = pd.read_csv(csv_file)
        self.root = Path(root)
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        unknown = set(self.df.label) - set(classes)
        if unknown:
            raise ValueError(f"Unknown labels in {csv_file}: {unknown}")
        self.targets = self.df.label.map(self.class_to_idx).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self.root / self.df.path.iloc[idx]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)  # OpenCV loads BGR
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(self.targets[idx], dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights (normalised to mean 1) for the weighted loss."""
        counts = torch.bincount(torch.tensor(self.targets), minlength=len(self.classes)).float()
        w = counts.sum() / (len(self.classes) * counts)
        return w / w.mean()
