"""Decode a clip once and cut it at any observed fraction."""

import sys
from pathlib import Path

import av
import numpy as np

from src.config import NUM_FRAMES


def decode_frames(path) -> np.ndarray:
    """All frames of a video as a uint8 array of shape (N, H, W, 3)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames)


def partial_indices(
    n_total: int, ratio: float, num_frames: int = NUM_FRAMES
) -> np.ndarray:
    """Indices of num_frames frames spread evenly over the first `ratio` of a clip.

    ratio=0.5 on a 100-frame clip only looks at frames 0..49. Short clips repeat frames.
    """
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    n_observed = max(1, int(round(ratio * n_total)))
    return np.linspace(0, n_observed - 1, num_frames).round().astype(np.int64)


def sample_partial(
    frames: np.ndarray, ratio: float, num_frames: int = NUM_FRAMES
) -> np.ndarray:
    return frames[partial_indices(len(frames), ratio, num_frames)]


if __name__ == "__main__":
    clip = Path(sys.argv[1])
    frames = decode_frames(clip)
    print(
        f"{clip.name}: {frames.shape[0]} frames of {frames.shape[2]}x{frames.shape[1]}"
    )
    for r in (0.3, 0.5, 0.7, 1.0):
        print(f"  ratio {r}: frames {partial_indices(len(frames), r).tolist()}")
