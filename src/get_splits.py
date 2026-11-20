"""Download the official HMDB51 split 1 lists from a Hugging Face mirror and verify them
against the videos in data/raw/hmdb51_org.
"""

import re
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
VIDEOS_DIR = ROOT / "data" / "raw" / "hmdb51_org"
MIRRORS = ["MichiganNLP/hmdb", "CVML-TueAI/HMDB51"]
SUFFIX = "_test_split1.txt"
TAGS = {1: "train", 2: "test", 0: "unused"}


def from_text_files(repo: str, files: list) -> bool:
    """Copy the original <class>_test_split1.txt files if the mirror has them."""
    wanted = [f for f in files if f.endswith(SUFFIX)]
    print(f"  {len(wanted)} split-1 text files found")
    if len(wanted) != 51:
        return False
    for f in wanted:
        local = hf_hub_download(repo, f, repo_type="dataset")
        (SPLITS_DIR / Path(f).name).write_bytes(Path(local).read_bytes())
    return True


def from_metadata(repo: str, files: list) -> bool:
    """Rebuild the 51 text files from a (video_path, label, subset) table."""
    import pandas as pd

    tables = [
        f
        for f in files
        if "split1" in f.lower() and f.endswith((".jsonl", ".json", ".parquet", ".csv"))
    ]
    for f in tables:
        local = hf_hub_download(repo, f, repo_type="dataset")
        if f.endswith(".parquet"):
            df = pd.read_parquet(local)
        elif f.endswith(".csv"):
            df = pd.read_csv(local)
        else:
            df = pd.read_json(local, lines=f.endswith(".jsonl"))
        if not {"video_path", "label", "subset"} <= set(df.columns):
            print(f"  {f}: unexpected columns {list(df.columns)}, skipped")
            continue
        for label, grp in df.groupby("label"):
            names = [Path(str(p).replace("\\", "/")).name for p in grp["video_path"]]
            lines = [f"{n} {int(s)}" for n, s in zip(names, grp["subset"])]
            (SPLITS_DIR / f"{label}{SUFFIX}").write_text("\n".join(lines) + "\n")
        print(f"  rebuilt split files from {f}")
        return True
    return False


def fetch() -> bool:
    for repo in MIRRORS:
        print(f"Trying {repo} ...")
        try:
            files = list_repo_files(repo, repo_type="dataset")
        except Exception as exc:
            print(f"  cannot list files: {type(exc).__name__}: {exc}")
            continue
        print(f"  {len(files)} files in the repo")
        if from_text_files(repo, files) or from_metadata(repo, files):
            return True
    return False


def norm(name: str) -> str:
    """Clip name reduced to lowercase letters and digits.

    Some Kaggle copies of HMDB51 change characters such as [ ] ! in file names,
    so names are compared in this form.
    """
    return re.sub(r"[^a-z0-9]", "", Path(name).stem.lower())


def video_lookup() -> dict:
    """{normalised clip name: path} for every .avi under data/raw/hmdb51_org."""
    lookup, clashes = {}, []
    for p in VIDEOS_DIR.rglob("*.avi"):
        key = norm(p.name)
        if key in lookup:
            clashes.append(p.name)
        lookup[key] = p
    if clashes:
        print(
            f"Warning: {len(clashes)} videos share a normalised name, e.g. {clashes[:2]}"
        )
    return lookup


def verify() -> bool:
    files = sorted(SPLITS_DIR.glob(f"*{SUFFIX}"))
    classes = {p.name for p in VIDEOS_DIR.iterdir() if p.is_dir()}
    exact_names = {p.name for p in VIDEOS_DIR.rglob("*.avi")}
    lookup = video_lookup()
    total, bad_counts, unknown = Counter(), [], []
    exact, renamed, missing = 0, [], []

    for f in files:
        label = f.name[: -len(SUFFIX)]
        if label not in classes:
            unknown.append(label)
        per_class = Counter()
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            name, tag = parts[0], TAGS.get(int(parts[1]), "?")
            per_class[tag] += 1
            if name in exact_names:
                exact += 1
            elif norm(name) in lookup:
                renamed.append((name, lookup[norm(name)].name))
            else:
                missing.append(name)
        total += per_class
        if per_class["train"] != 70 or per_class["test"] != 30:
            bad_counts.append(
                f"{label} ({per_class['train']} train / {per_class['test']} test)"
            )

    print(f"\nSplit files: {len(files)} (expected 51)")
    print(
        f"Clips: train {total['train']} (expected 3570), test {total['test']} (expected 1530), "
        f"unused {total['unused']}"
    )
    print(f"Videos on disk: {len(exact_names)}")
    print(f"Classes without a matching video folder: {unknown or 'none'}")
    print(f"Classes not at 70 train / 30 test: {bad_counts or 'none'}")
    print(f"Clip names matched exactly: {exact}")
    print(f"Clip names matched after ignoring punctuation: {len(renamed)}")
    for official, on_disk in renamed[:2]:
        print(f"    {official}\n      -> {on_disk}")
    print(
        f"Clips still missing: {len(missing)}"
        + (f", e.g. {missing[:3]}" if missing else "")
    )
    return len(files) == 51 and not unknown and not bad_counts and not missing


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    if len(list(SPLITS_DIR.glob(f"*{SUFFIX}"))) != 51 and not fetch():
        sys.exit("None of the mirrors provided usable split 1 lists.")
    ok = verify()
    print(
        "\nRESULT:",
        "split 1 ready" if ok else "split lists do not match the videos on disk",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
