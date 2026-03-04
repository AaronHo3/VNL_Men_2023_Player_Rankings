#!/usr/bin/env python3
"""Download VNL Men 2023 dataset from Kaggle and copy files into data/raw/."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "yeganehbavafa/vnl-men-2023"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded cache path: {downloaded_path}")

    copied = 0
    for item in downloaded_path.iterdir():
        destination = RAW_DIR / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
        copied += 1

    print(f"Copied {copied} item(s) into: {RAW_DIR}")


if __name__ == "__main__":
    main()
