from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pydicom
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class RSNADataset(Dataset):
    """Minimal RSNA-style dataset from CSV with image_path/label/split columns."""

    def __init__(self, df: pd.DataFrame, processor, root_dir: str | None = None):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.root_dir = Path(root_dir) if root_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if self.root_dir and not path.is_absolute():
            path = self.root_dir / path

        if path.suffix.lower() == ".dcm":
            dcm = pydicom.dcmread(str(path))
            arr = dcm.pixel_array
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = (arr * 255).astype("uint8")
            image = Image.fromarray(arr).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")
        return image

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = self._load_image(row["image_path"])
        enc = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": int(row["label"]),
            "image_path": str(row["image_path"]),
        }


def make_rsna_dataloaders(
    csv_path: str,
    processor,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    root_dir: str | None = None,
) -> Dict[str, DataLoader]:
    df = pd.read_csv(csv_path)

    required_cols = {"image_path", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    loaders: Dict[str, DataLoader] = {}
    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split].copy()
        if sdf.empty:
            continue

        ds = RSNADataset(sdf, processor=processor, root_dir=root_dir)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders
