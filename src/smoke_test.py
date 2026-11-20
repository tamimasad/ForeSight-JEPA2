"""Check frozen V-JEPA 2 pipeline correctness and GPU runtime."""

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
import transformers

from src.config import (
    INDEX_CSV,
    LEAD_TIME_RATIOS,
    RATIOS,
    RESULTS_DIR,
    ROOT,
    SEED,
    VJEPA_REPO,
)
from src.sample import decode_frames, sample_partial
from src.vjepa import VJEPA  

MASKING_CLIPS, MASKING_CELLS = 300, 9


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def part_a(vj: VJEPA, clips: pd.DataFrame) -> dict:
    print(f"\nPart A: correctness on {len(clips)} test clips at ratio 0.5")
    batch = [sample_partial(decode_frames(ROOT / p), 0.5) for p in clips.path]
    pixels = vj.preprocess(batch)
    tokens, logits = vj.encode(pixels)
    reference = vj.library_logits(pixels)
    pooled = vj.pool_tokens(tokens)

    diff = (logits.float() - reference.float()).abs().max().item()
    same_top1 = bool((logits.argmax(-1) == reference.argmax(-1)).all())
    print(
        f"  input {tuple(pixels.shape)}  tokens {tuple(tokens.shape)}  pooled {tuple(pooled.shape)}"
    )
    print(
        f"  max |our logits - library logits| = {diff:.2e}, same top-1 on every clip: {same_top1}"
    )

    tops = vj.top_labels(logits)
    for (_, row), top in zip(clips.iterrows(), tops):
        phrases = "; ".join(f"{label} ({p:.2f})" for label, p in top[:3])
        print(f"  [{row.label}] {row.clip_id[:50]}\n      SSv2: {phrases}")

    ok = (
        tuple(tokens.shape[1:]) == (2048, 1024)
        and tuple(pooled.shape[1:]) == (128, 1024)
        and diff < 5e-2
        and same_top1
    )
    print("  PASS" if ok else "  FAIL")
    return {
        "tokens_shape": list(tokens.shape),
        "pooled_shape": list(pooled.shape),
        "max_logit_diff": diff,
        "same_top1": same_top1,
        "pass": ok,
        "examples": [
            {"clip": r.clip_id, "label": r.label, "ssv2_top5": t}
            for (_, r), t in zip(clips.iterrows(), tops)
        ],
    }


def part_b(vj: VJEPA, clips: pd.DataFrame, index: pd.DataFrame) -> dict:
    print(f"\nPart B: timing on {len(clips)} test clips, {len(RATIOS)} ratios each")
    t_decode, t_prep, t_encode = [], [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for i, path in enumerate(clips.path):
        t0 = time.perf_counter()
        frames = decode_frames(ROOT / path)
        t1 = time.perf_counter()
        pixels = vj.preprocess([sample_partial(frames, r) for r in RATIOS])
        sync()
        t2 = time.perf_counter()
        tokens, _ = vj.encode(pixels)
        vj.pool_tokens(tokens).half().cpu()
        sync()
        t3 = time.perf_counter()
        if i >= 2:  # the first two clips are for warmup
            t_decode.append(t1 - t0)
            t_prep.append(t2 - t1)
            t_encode.append(t3 - t2)

    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    decode = float(np.mean(t_decode))
    prep = float(np.mean(t_prep)) / len(RATIOS)
    encode = float(np.mean(t_encode)) / len(RATIOS)
    print(
        f"  decode {decode * 1000:.0f} ms per clip, preprocess {prep * 1000:.0f} ms and "
        f"encode {encode * 1000:.0f} ms per pass, peak GPU memory {peak_gb:.2f} GB"
    )

    used = index[index.split.isin(["train", "val", "test"])]
    harm_test = index[(index.split == "test") & index.harm]
    extra = [r for r in LEAD_TIME_RATIOS if r not in RATIOS]
    passes = {
        "features": len(used) * len(RATIOS),
        "lead_time": len(harm_test) * len(extra),
        "masking": MASKING_CLIPS * MASKING_CELLS,
    }
    clips_decoded = {
        "features": len(used),
        "lead_time": len(harm_test),
        "masking": MASKING_CLIPS,
    }
    hours = {
        k: (clips_decoded[k] * decode + passes[k] * (prep + encode)) / 3600
        for k in passes
    }
    print(
        f"  feature extraction: {passes['features']:,} passes, about {hours['features']:.1f} h"
    )
    print(
        f"  lead-time ratios:   {passes['lead_time']:,} passes, about {hours['lead_time']:.1f} h"
    )
    print(
        f"  masking search:     {passes['masking']:,} passes, about {hours['masking']:.1f} h"
    )
    return {
        "decode_s_per_clip": decode,
        "preprocess_s_per_pass": prep,
        "encode_s_per_pass": encode,
        "peak_gpu_gb": peak_gb,
        "passes": passes,
        "hours": hours,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-check", type=int, default=5)
    parser.add_argument("--n-time", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run python -m src.check_env first.")

    index = pd.read_csv(INDEX_CSV)
    test = index[index.split == "test"]
    rng = np.random.default_rng(SEED)
    check = test.loc[rng.choice(test.index, args.n_check, replace=False)]
    timing = test.loc[rng.choice(test.index, args.n_time, replace=False)]

    print(f"Loading {VJEPA_REPO} (the first run downloads about 1.5 GB)")
    t0 = time.perf_counter()
    vj = VJEPA()
    print(
        f"  loaded in {time.perf_counter() - t0:.0f} s, {len(vj.id2label)} SSv2 classes"
    )

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "model": VJEPA_REPO,
        "part_a": part_a(vj, check),
        "part_b": part_b(vj, timing, index),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "smoke_test.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")
    print(
        "RESULT:", "pipeline verified" if report["part_a"]["pass"] else "Part A failed"
    )


if __name__ == "__main__":
    main()
