"""Portable paths and checks; no random-number calls or data mutation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_data_root():
    return Path(os.environ.get("GNN_DATA_ROOT", str(ROOT / "data"))).expanduser().resolve()


def get_save_root():
    return Path(os.environ.get("GNN_SAVE_ROOT", str(ROOT / "save"))).expanduser().resolve()


def get_data_reference(region):
    """Input fingerprints, fixed splits and test sampling in one configuration."""
    records = json.loads((ROOT / "data_preparation" / "reference_data.json").read_text(encoding="utf-8"))
    return records[region]


def check_fixed_split(region, train, validation, test):
    """Reject another event order instead of silently changing the paper split."""
    for name, actual in [("train", train), ("validation", validation), ("test", test)]:
        expected = get_data_reference(region)["splits"][name]
        if list(map(int, actual)) != expected:
            raise ValueError(f"{region} {name} event order differs from the published split. "
                             "Use the original catalogue order and processed data; see data_preparation/README.md.")


def check_input_hashes(region, directory):
    for name, expected in get_data_reference(region)["input_hashes"].items():
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Input differs from the paper snapshot: {path}. "
                             "A fresh download is not guaranteed to reproduce the original catalogue.")
