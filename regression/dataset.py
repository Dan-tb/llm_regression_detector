"""Golden dataset loading. The file itself is versioned (dataset_version)."""
from __future__ import annotations

import json
from pathlib import Path

from .models import GoldenDataset


def load_dataset(path: str | Path) -> GoldenDataset:
    with open(path, encoding="utf-8") as f:
        ds = GoldenDataset(**json.load(f))
    ids = [c.id for c in ds.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate test case ids in golden dataset")
    return ds
