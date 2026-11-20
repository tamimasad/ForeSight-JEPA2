<h1 align="center">ForeSight-JEPA2</h1>

<p align="center"><em>Early warning of harmful human actions from partial video, with evidence-grounded explanations.</em></p>

> **Status: under continuous development.** The code, results and this README change as the project moves forward.

## Overview

ForeSight-JEPA2 watches the **first 30 to 70 percent** of a video of a person and answers three questions:

1. **Anticipate:** which action is about to happen?
2. **Explain:** what in the video supports that prediction?
3. **Judge:** could the action hurt someone (a fall, a punch, a kick), and how early can we warn?

The system is built on [V-JEPA 2](https://arxiv.org/abs/2506.09985) from Meta FAIR. The video encoder stays frozen and every clip is encoded once, so the whole pipeline trains and runs on a **single 12 GB consumer GPU**. The only trained component is a small attentive probe of about **4M parameters**.

## How it works

- A **frozen V-JEPA 2 ViT-L encoder** turns a partial clip (16 frames, 256 px) into spatio-temporal tokens. The same forward pass gives motion phrases from Meta's released Something-Something v2 head.
- An **attentive probe** on pooled tokens predicts the clip's final action, and temperature scaling calibrates its confidence.
- A **harm score** (the summed probability of 7 harmful classes) triggers an early warning, reported with its lead time in seconds.
- **Explanations come from checkable evidence:** a masking search for the region that moves the prediction most, pose cues from a pose model, and the SSv2 motion phrases. A small local language model may reword the evidence, and a guard rejects any rewording that adds unsupported facts.

## Models and data

| Role                               | Model / dataset                                                                                            |
| :--------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **Video encoder + motion phrases** | [facebook/vjepa2-vitl-fpc16-256-ssv2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) (frozen) |
| **Pose cues**                      | [Ultralytics YOLO11n-pose](https://huggingface.co/Ultralytics/YOLO11)                                      |
| **Optional rewording**             | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)                            |
| **Dataset**                        | [HMDB51](https://serre.lab.brown.edu/hmdb51.html), official split 1 (51 actions, 6,766 clips)              |

## Project structure

```text
ForeSight-JEPA2/
├── .vscode/
│   └── settings.json      # keeps env, caches and temp files inside the project folder
├── src/
│   ├── config.py          # paths, model name, observed fractions, harm classes
│   ├── check_env.py       # verifies packages, GPU and cache locations
│   ├── get_splits.py      # downloads and verifies the official split 1 lists
│   ├── build_index.py     # builds data/index.csv (train / val / test, durations)
│   ├── sample.py          # decodes a clip and samples frames from its observed part
│   ├── vjepa.py           # frozen V-JEPA 2: tokens and SSv2 logits in one pass
│   └── smoke_test.py      # correctness check and GPU timing
├── data/                  # HMDB51 videos, split lists, index (not tracked)
├── features/              # cached encoder features (not tracked)
├── results/               # metrics tables and figures
├── requirements.txt
└── README.md
```

More modules (data preparation, feature extraction, probe training, harm judgment, explanations and a Gradio demo) are added as development continues.

## Setup

Tested on **Windows 11** with an **RTX 3060 (12 GB)**, Python 3.12, PyTorch 2.15 and CUDA 13.2.

```bash
git clone https://github.com/tamimasad/ForeSight-JEPA2.git
cd ForeSight-JEPA2

conda create --prefix ./env python=3.12 -y
conda activate ./env

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt

python -m src.check_env
```

> `.vscode/settings.json` points conda, pip, Hugging Face, Kaggle and temp folders at `E:\ForeSight-JEPA2`. **If you clone the repo somewhere else, change those paths first.**

## Reference

Assran et al., _[V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)_, FAIR at Meta, 2025.
