"""Paths and settings shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "raw" / "hmdb51_org"
SPLITS_DIR = DATA_DIR / "splits"
INDEX_CSV = DATA_DIR / "index.csv"
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"

SPLIT = 1  # official HMDB51 split used throughout
VAL_FRACTION = 0.15  # share of each class's training clips held out for validation
SEED = 0

# Classes treated as harmful by the warning module
HARM_CLASSES = ("fall_floor", "hit", "kick", "punch", "push", "shoot_gun", "sword")

# V-JEPA 2 encoder
VJEPA_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
NUM_FRAMES = 16  # frames sampled from the observed part of a clip
POOL_GRID = 4  # 16 x 16 patch grid pooled to 4 x 4 per time step

# Observed fractions of each clip (1.0 is the full clip, used as an upper bound)
RATIOS = (0.5, 0.7, 1.0)
LEAD_TIME_RATIOS = (0.5, 0.6, 0.7, 0.8, 0.9)
