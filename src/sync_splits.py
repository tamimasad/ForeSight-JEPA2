"""Copy the current split column from data/index.csv into features/main_clips.csv and
features/lead_clips.csv.

Usage (from the repo root):  python -m src.sync_splits
"""

import sys

import pandas as pd

from src.config import FEATURES_DIR, INDEX_CSV

split_of = pd.read_csv(INDEX_CSV).set_index("clip_id").split
for name in ("main", "lead"):
    path = FEATURES_DIR / f"{name}_clips.csv"
    clips = pd.read_csv(path)
    new_split = clips.clip_id.map(split_of)
    if new_split.isna().any():
        sys.exit(
            f"{path.name}: {int(new_split.isna().sum())} clips are missing from {INDEX_CSV.name}"
        )
    changed = int((new_split != clips.split).sum())
    clips["split"] = new_split
    clips.to_csv(path, index=False)
    print(
        f"{path.name}: {changed} rows changed; now {clips.split.value_counts().to_dict()}"
    )
