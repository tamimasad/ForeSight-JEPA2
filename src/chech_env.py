"""Check: packages, GPU, and that every cache folder points inside the project."""

import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "av",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "kaggle",
    "huggingface_hub",
]
CACHE_VARS = [
    "CONDA_PKGS_DIRS",
    "PIP_CACHE_DIR",
    "HF_HOME",
    "TORCH_HOME",
    "KAGGLE_CONFIG_DIR",
    "YOLO_CONFIG_DIR",
    "MPLCONFIGDIR",
    "GRADIO_TEMP_DIR",
    "TEMP",
    "TMP",
]


def inside_project(path: str) -> bool:
    return path.lower().startswith(str(ROOT).lower())


def main():
    ok = True

    print(f"python   {sys.executable} ({sys.version.split()[0]})")
    if not inside_project(sys.executable):
        ok = False
        print(r"  <-- not the project env, run: conda activate [env_name]")

    print("\npackages")
    for name in PACKAGES:
        try:
            v = version(name)
        except PackageNotFoundError:
            v, ok = "MISSING  <-- pip install it", False
        print(f"  {name:<16}{v}")

    print("\ncache folders")
    for var in CACHE_VARS:
        val = os.environ.get(var, "")
        good = inside_project(val)
        ok = ok and good
        flag = "" if good else f"  <-- should be inside {ROOT}"
        print(f"  {var:<18}{val or 'NOT SET'}{flag}")

    import torch

    print()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        x = torch.randn(2048, 1024, device="cuda", dtype=torch.float16)
        (x @ x.T).sum().item()  # fp16 computation check on the GPU
        print(
            f"GPU      {p.name}, {p.total_memory / 1024**3:.1f} GB, "
            f"torch {torch.__version__}, CUDA {torch.version.cuda}, fp16 matmul ok"
        )
    else:
        ok = False
        print(
            "GPU      NOT visible to PyTorch  <-- reinstall torch from the cu132 index"
        )

    print(
        f"disk     {shutil.disk_usage(ROOT).free / 1024**3:.0f} GB free on {ROOT.drive}"
    )
    print("\nRESULT:", "environment ready" if ok else "fix the lines marked <--")


if __name__ == "__main__":
    main()
