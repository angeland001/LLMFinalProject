#!/usr/bin/env python
"""Download hf-vision/chest-xray-pneumonia and generate metadata.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil
import pandas as pd
from datasets import load_dataset

DEST_IMAGES = Path("data/rsna/images")
DEST_CSV = Path("data/rsna/metadata.csv")

LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}
SPLIT_MAP = {"train": "train", "test": "test", "validation": "val"}

def main():
    DEST_IMAGES.mkdir(parents=True, exist_ok=True)

    print("Downloading hf-vision/chest-xray-pneumonia …")
    ds = load_dataset("hf-vision/chest-xray-pneumonia")

    rows = []
    for hf_split, csv_split in SPLIT_MAP.items():
        if hf_split not in ds:
            continue
        split_ds = ds[hf_split]
        print(f"  {hf_split}: {len(split_ds)} images")
        for i, sample in enumerate(split_ds):
            img = sample["image"]
            label_idx = sample["label"]
            label_name = split_ds.features["label"].int2str(label_idx)
            fname = f"{csv_split}_{i:05d}.jpg"
            dest = DEST_IMAGES / fname
            if not dest.exists():
                img.save(dest)
            rows.append({
                "image_path": str(dest).replace("\\", "/"),
                "label": LABEL_MAP[label_name],
                "split": csv_split,
            })

    df = pd.DataFrame(rows)
    DEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {DEST_CSV}")
    print(df["split"].value_counts().to_string())
    print(df["label"].value_counts().to_string())

if __name__ == "__main__":
    main()
