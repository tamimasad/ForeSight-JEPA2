"""Encode clips with the frozen V-JEPA 2 model and store the features on disk.

Two sets of clips:
  main   train, val and test clips at RATIOS (0.5, 0.7, 1.0)
  lead   harm-class test clips at the extra lead-time ratios (0.6, 0.8, 0.9)

For every clip and ratio the script stores the 128 pooled encoder tokens and the
174 SSv2 logits. Files written to features/:
  <set>_tokens.npy   (clips, ratios, 128, 1024) float16
  <set>_ssv2.npy     (clips, ratios, 174) float16
  <set>_done.npy     (clips,) bool, rows that are finished
  <set>_clips.csv    clips in row order
  <set>_info.json    ratios and model settings
  <set>_failed.txt   clips that could be resumed

Usage :
  python -m src.extract_features --set main --limit 20   (quick test on 20 clips)
  python -m src.extract_features --set main
  python -m src.extract_features --set lead
  python -m src.extract_features --set main --check-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    FEATURES_DIR,
    INDEX_CSV,
    LEAD_TIME_RATIOS,
    NUM_FRAMES,
    POOL_GRID,
    RATIOS,
    ROOT,
    VJEPA_REPO,
)
from src.sample import decode_frames, sample_partial
from src.vjepa import VJEPA

SAVE_EVERY = 50  # clips between checkpoints
CLIP_COLUMNS = [
    "clip_id",
    "path",
    "label",
    "label_id",
    "split",
    "harm",
    "n_frames",
    "duration_s",
]


def paths(which: str) -> dict:
    names = [
        "tokens.npy",
        "ssv2.npy",
        "done.npy",
        "clips.csv",
        "info.json",
        "failed.txt",
    ]
    return {n.split(".")[0]: FEATURES_DIR / f"{which}_{n}" for n in names}


def select(index: pd.DataFrame, which: str):
    if which == "main":
        clips = index[index.split.isin(["train", "val", "test"])]
        ratios = tuple(RATIOS)
    else:
        clips = index[(index.split == "test") & index.harm]
        ratios = tuple(r for r in LEAD_TIME_RATIOS if r not in RATIOS)
    return clips[CLIP_COLUMNS].reset_index(drop=True), ratios


def save_done(done: np.ndarray, path: Path) -> None:
    """Write the done mask to a temporary file first, so a crash never leaves it half written."""
    tmp = path.with_suffix(".tmp.npy")
    with open(tmp, "wb") as f:
        np.save(f, done)
    os.replace(tmp, path)


def open_store(which: str, clips: pd.DataFrame, info: dict):
    """Create the output arrays, or reopen them after an interrupted run."""
    p = paths(which)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    n, r = len(clips), len(info["ratios"])

    if p["done"].exists():
        old_ids = pd.read_csv(p["clips"]).clip_id.tolist()
        old_info = json.loads(p["info"].read_text())
        if old_ids != clips.clip_id.tolist() or old_info != info:
            sys.exit(
                f"features/{which}_* were made from a different index or different settings. "
                f"Delete the features/{which}_* files and run again."
            )
        tokens = np.lib.format.open_memmap(p["tokens"], mode="r+")
        ssv2 = np.lib.format.open_memmap(p["ssv2"], mode="r+")
        done = np.load(p["done"])
        return tokens, ssv2, done

    clips.to_csv(p["clips"], index=False)
    p["info"].write_text(json.dumps(info, indent=2))
    tokens = np.lib.format.open_memmap(
        p["tokens"],
        mode="w+",
        dtype=np.float16,
        shape=(n, r, info["tokens"], info["hidden"]),
    )
    ssv2 = np.lib.format.open_memmap(
        p["ssv2"], mode="w+", dtype=np.float16, shape=(n, r, info["ssv2_classes"])
    )
    done = np.zeros(n, dtype=bool)
    save_done(done, p["done"])
    return tokens, ssv2, done


def extract(which: str, limit: int) -> None:
    index = pd.read_csv(INDEX_CSV)
    clips, ratios = select(index, which)

    print(f"Loading {VJEPA_REPO}")
    vj = VJEPA()
    info = {
        "set": which,
        "ratios": list(ratios),
        "model": VJEPA_REPO,
        "num_frames": NUM_FRAMES,
        "pool_grid": POOL_GRID,
        "tokens": (NUM_FRAMES // vj.tubelet) * POOL_GRID**2,
        "hidden": vj.model.config.hidden_size,
        "ssv2_classes": len(vj.id2label),
    }
    tokens, ssv2, done = open_store(which, clips, info)
    p = paths(which)

    todo = np.flatnonzero(~done)
    if limit:
        todo = todo[:limit]
    print(
        f"Set '{which}': {len(clips)} clips x {len(ratios)} ratios {list(ratios)}; "
        f"{int(done.sum())} already done, {len(todo)} to do in this run"
    )

    failed, processed = [], 0
    t0 = time.perf_counter()
    try:
        for k, i in enumerate(tqdm(todo, desc=f"Encoding {which}", unit="clip")):
            row = clips.iloc[i]
            try:
                frames = decode_frames(ROOT / row.path)
                pixels = vj.preprocess([sample_partial(frames, r) for r in ratios])
                tok, logits = vj.encode(pixels)
                tokens[i] = vj.pool_tokens(tok).half().cpu().numpy()
                ssv2[i] = logits.half().cpu().numpy()
                done[i] = True
            except Exception as exc:
                failed.append(f"{row.clip_id}\t{type(exc).__name__}: {exc}")
            processed += 1
            if (k + 1) % SAVE_EVERY == 0:
                tokens.flush()
                ssv2.flush()
                save_done(done, p["done"])
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; run the same command again to continue.")
    finally:
        tokens.flush()
        ssv2.flush()
        save_done(done, p["done"])
        if failed:
            with open(p["failed"], "a", encoding="utf-8") as f:
                f.write("\n".join(failed) + "\n")

    elapsed = time.perf_counter() - t0
    if processed:
        print(
            f"Processed {processed} clips in {elapsed / 60:.1f} min "
            f"({elapsed / processed:.2f} s per clip); {len(failed)} failed"
        )
    del tokens, ssv2


def check(which: str) -> bool:
    p = paths(which)
    if not p["done"].exists():
        print(f"No features for set '{which}' yet.")
        return False
    info = json.loads(p["info"].read_text())
    clips = pd.read_csv(p["clips"])
    done = np.load(p["done"])
    tokens = np.load(p["tokens"], mmap_mode="r")
    ssv2 = np.load(p["ssv2"], mmap_mode="r")
    rows = np.flatnonzero(done)

    non_finite, empty = 0, 0
    norm_sum = np.zeros(len(info["ratios"]))
    for chunk in np.array_split(rows, max(1, len(rows) // 100)) if len(rows) else []:
        t = tokens[chunk].astype(np.float32)
        non_finite += int((~np.isfinite(t)).sum()) + int(
            (~np.isfinite(ssv2[chunk].astype(np.float32))).sum()
        )
        empty += int((np.abs(t).reshape(len(chunk), -1).max(axis=1) == 0).sum())
        norm_sum += np.linalg.norm(t, axis=-1).mean(axis=2).sum(axis=0)

    n_failed = len(p["failed"].read_text().splitlines()) if p["failed"].exists() else 0
    size_gb = (p["tokens"].stat().st_size + p["ssv2"].stat().st_size) / 1024**3
    print(f"\nCheck of set '{which}'")
    print(
        f"  arrays: tokens {tokens.shape}, ssv2 {ssv2.shape}, {size_gb:.2f} GB on disk"
    )
    print(f"  done: {len(rows)} of {len(clips)} clips; failed: {n_failed}")
    print(f"  done per split: {clips[done].split.value_counts().to_dict()}")
    print(
        f"  non-finite values: {non_finite} (expected 0); all-zero rows: {empty} (expected 0)"
    )
    if len(rows):
        norms = ", ".join(
            f"{r}: {v:.2f}" for r, v in zip(info["ratios"], norm_sum / len(rows))
        )
        print(f"  mean token norm per ratio: {norms}")
    ok = len(rows) == len(clips) and non_finite == 0 and empty == 0
    print(
        "RESULT:",
        f"set '{which}' complete" if ok else f"set '{which}' not complete yet",
    )
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["main", "lead"], required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process at most this many clips in this run",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        if not torch.cuda.is_available():
            sys.exit("CUDA is not available. Run python -m src.check_env first.")
        extract(args.set, args.limit)
    check(args.set)


if __name__ == "__main__":
    main()
