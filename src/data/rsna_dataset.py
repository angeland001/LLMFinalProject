"""RSNA dataset loader utilities.

Expected CSV columns:
- image_path: relative path from image_root or absolute path
- label: 0 (normal) / 1 (pneumonia)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class RSNADataset(Dataset):
    def __init__(self, csv_path: str, image_root: str, image_col: str, label_col: str, processor):
        self.df = pd.read_csv(csv_path)
        if self.df.empty:
            raise ValueError(f"Dataset is empty: {csv_path}")

        self.image_root = Path(image_root)
        self.image_col = image_col
        self.label_col = label_col
        self.processor = processor

        missing_cols = [c for c in [image_col, label_col] if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row[self.image_col])
        if not img_path.is_absolute():
            img_path = self.image_root / img_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = int(row[self.label_col])

        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": label,
            "image_path": str(img_path),
        }


def make_dataloaders(config: dict, processor):
    data_cfg = config["data"]
    train_cfg = config["train"]

    train_ds = RSNADataset(
        csv_path=data_cfg["train_csv"],
        image_root=data_cfg["image_root"],
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        processor=processor,
    )
    val_ds = RSNADataset(
        csv_path=data_cfg["val_csv"],
        image_root=data_cfg["image_root"],
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        processor=processor,
    )
    test_ds = RSNADataset(
        csv_path=data_cfg["test_csv"],
        image_root=data_cfg["image_root"],
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        processor=processor,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )

    return train_loader, val_loader, test_loader
