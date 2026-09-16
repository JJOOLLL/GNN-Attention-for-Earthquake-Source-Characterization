"""Portable paths and checks; no random-number calls or data mutation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_data_root():
    return Path(os.environ.get("GNN_DATA_ROOT", str(ROOT / "data"))).expanduser().resolve()


def get_save_root():
    return Path(os.environ.get("GNN_SAVE_ROOT", str(ROOT / "save"))).expanduser().resolve()


def check_fixed_split(region, train, validation, test):
    """Reject another event order instead of silently changing the paper split."""
    for name, actual in [("train", train), ("validation", validation), ("test", test)]:
        path = ROOT / "results" / "splits" / region / f"{name}.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            expected = [int(row["event_id"]) for row in csv.DictReader(stream)]
        if list(map(int, actual)) != expected:
            raise ValueError(f"{region} {name} event order differs from the published split. "
                             "Use the original catalogue order and processed data; see data_preparation/README.md.")


def check_input_hashes(region, directory):
    records = json.loads((ROOT / "data_preparation" / "input_hashes.json").read_text(encoding="utf-8"))
    for name, expected in records[region]["input_hashes"].items():
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Input differs from the paper snapshot: {path}. "
                             "A fresh download is not guaranteed to reproduce the original catalogue.")
