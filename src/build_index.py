"""Build data/index.csv with one row per HMDB51 clip.

Columns: clip_id, path, label, label_id, split, source, n_frames, fps, width, height,
duration_s, harm. The validation set is VAL_FRACTION of each class's split-1 training
clips, grouped by source video so clips cut from the same video stay on one side.

"""

import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import av
import pandas as pd
from tqdm import tqdm

from src.config import (
    HARM_CLASSES,
    INDEX_CSV,
    ROOT,
    SEED,
    SPLIT,
    SPLITS_DIR,
    VAL_FRACTION,
    VIDEOS_DIR,
)

SUFFIX = f"_test_split{SPLIT}.txt"
TAGS = {1: "train", 2: "test", 0: "unused"}

MATCH_KEYS = [
    ("exact", lambda name: name),
    (
        "letters, digits and _ (case kept)",
        lambda name: re.sub(r"[^A-Za-z0-9_]", "", Path(name).stem),
    ),
    (
        "letters and digits (case ignored)",
        lambda name: re.sub(r"[^a-z0-9]", "", Path(name).stem.lower()),
    ),
]


def read_splits() -> list:
    """[(official file name, label, tag)] from the 51 split files."""
    files = sorted(SPLITS_DIR.glob(f"*{SUFFIX}"))
    if len(files) != 51:
        sys.exit(
            f"Expected 51 split files in {SPLITS_DIR}, found {len(files)}. Run src.get_splits first."
        )
    rows = []
    for f in files:
        label = f.name[: -len(SUFFIX)]
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                rows.append((parts[0], label, int(parts[1])))
    return rows


def match_videos(official_names: list) -> dict:
    """{official name: path on disk}, one-to-one.

    Each key in MATCH_KEYS is tried in turn on the names still unmatched. A pair is
    accepted only when exactly one official name and exactly one unclaimed file share
    the key, so look-alike names never get paired by guesswork.
    """
    remaining_files = {p.name: p for p in VIDEOS_DIR.rglob("*.avi")}
    remaining_names = list(official_names)
    matched = {}

    for desc, key in MATCH_KEYS:
        names_by_key, files_by_key = defaultdict(list), defaultdict(list)
        for n in remaining_names:
            names_by_key[key(n)].append(n)
        for fname in remaining_files:
            files_by_key[key(fname)].append(fname)
        count = 0
        for k, names in names_by_key.items():
            files = files_by_key.get(k, [])
            if len(names) == 1 and len(files) == 1:
                matched[names[0]] = remaining_files.pop(files[0])
                count += 1
        remaining_names = [n for n in remaining_names if n not in matched]
        print(f"  matched by {desc}: {count}")

    if remaining_names:
        print(f"\n{len(remaining_names)} clips could not be matched:")
        for n in remaining_names[:10]:
            print(f"    {n}")
        print(f"Unclaimed files on disk ({len(remaining_files)}):")
        for fname in list(remaining_files)[:10]:
            print(f"    {fname}")
        sys.exit(1)
    return matched


def source_of(stem: str, label: str) -> str:
    """HMDB51 names follow <source>_<label>_<5 tags>_<n>; return <source>."""
    parts = stem.split("_")
    head = "_".join(parts[:-6]) if len(parts) > 6 else stem
    return head[: -len(label) - 1] if head.endswith("_" + label) else head


def probe(path: Path) -> dict:
    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate or s.guessed_rate or 30)
            n = s.frames or sum(1 for _ in c.decode(s))
            return {
                "n_frames": int(n),
                "fps": round(fps, 3),
                "width": s.codec_context.width,
                "height": s.codec_context.height,
                "duration_s": round(n / fps, 3),
            }
    except Exception as exc:
        print(f"  unreadable: {path.name} ({exc})")
        return {"n_frames": 0, "fps": 0.0, "width": 0, "height": 0, "duration_s": 0.0}


def assign_val(df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    for _, grp in df[df.split == "train"].groupby("label"):
        by_source = defaultdict(list)
        for idx, src in zip(grp.index, grp.source):
            by_source[src].append(idx)
        sources = sorted(by_source)
        rng.shuffle(sources)
        target, taken = VAL_FRACTION * len(grp), []
        for src in sources:
            if len(taken) >= target:
                break
            if len(taken) + len(by_source[src]) >= len(grp):
                continue
            taken += by_source[src]
        df.loc[taken, "split"] = "val"
    return df


def main():
    splits = read_splits()
    print("Matching split lists to video files")
    paths = match_videos([name for name, _, _ in splits])
    labels = sorted({label for _, label, _ in splits})
    label_id = {c: i for i, c in enumerate(labels)}

    rows = []
    for name, label, tag in tqdm(splits, desc="Reading video headers"):
        stem = Path(name).stem
        path = paths[name]
        rows.append(
            {
                "clip_id": stem,
                "path": path.relative_to(ROOT).as_posix(),
                "label": label,
                "label_id": label_id[label],
                "split": TAGS[tag],
                "source": source_of(stem, label),
                **probe(path),
                "harm": label in HARM_CLASSES,
            }
        )

    df = assign_val(pd.DataFrame(rows))
    bad = df[df.n_frames == 0]
    if len(bad):
        print(f"Dropping {len(bad)} unreadable clips")
        df = df[df.n_frames > 0]
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)

    def pairs(split):
        part = df[df.split == split]
        return set(zip(part.label, part.source))

    counts = df.split.value_counts()
    print(f"\nWrote {INDEX_CSV} ({len(df)} clips, {len(labels)} classes)")
    print(
        f"  train {counts.get('train', 0)}, val {counts.get('val', 0)}, "
        f"test {counts.get('test', 0)}, unused {counts.get('unused', 0)}"
    )
    print(
        f"  train + val = {counts.get('train', 0) + counts.get('val', 0)} (expected 3570), "
        f"test = {counts.get('test', 0)} (expected 1530)"
    )
    print(
        f"  harm-class test clips: {int((df.harm & (df.split == 'test')).sum())} (expected 210)"
    )
    print(
        f"  (class, source) pairs in both train and val: {len(pairs('train') & pairs('val'))} (expected 0)"
    )
    print(
        f"  duration: median {df.duration_s.median():.2f} s, min {df.duration_s.min():.2f} s, "
        f"max {df.duration_s.max():.2f} s"
    )
    print(
        f"  frames: median {int(df.n_frames.median())}, min {int(df.n_frames.min())}; "
        f"fps values: {sorted(df.fps.round().unique().tolist())}"
    )
    print(f"  resolution: {df.groupby(['width', 'height']).size().to_dict()}")


if __name__ == "__main__":
    main()
