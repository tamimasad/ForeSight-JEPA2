"""Frozen V-JEPA 2: encoder tokens and SSv2 logits from a single forward pass.

VJEPA2ForVideoClassification.forward runs vjepa2(skip_predictor=True) -> pooler -> classifier.
Calling those three parts directly gives the same logits and also returns the encoder tokens.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

from src.config import NUM_FRAMES, POOL_GRID, VJEPA_REPO


class VJEPA:
    def __init__(
        self, repo: str = VJEPA_REPO, device: str = "cuda", dtype=torch.float16
    ):
        self.device, self.dtype = device, dtype
        self.processor = AutoVideoProcessor.from_pretrained(repo)
        self.model = (
            AutoModelForVideoClassification.from_pretrained(
                repo, dtype=dtype, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        self.id2label = self.model.config.id2label
        self.tubelet = self.model.config.tubelet_size

    def preprocess(self, clips: list) -> torch.Tensor:
        """List of uint8 arrays (T, H, W, 3) -> tensor (B, T, 3, 256, 256) on the device."""
        videos = [
            torch.from_numpy(np.ascontiguousarray(c)).permute(0, 3, 1, 2) for c in clips
        ]
        pixels = self.processor(videos, return_tensors="pt")["pixel_values_videos"]
        return pixels.to(self.device, self.dtype)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor):
        """Encoder tokens (B, N, D) and SSv2 logits (B, 174)."""
        tokens = self.model.vjepa2(
            pixel_values_videos=pixels, skip_predictor=True
        ).last_hidden_state
        logits = self.model.classifier(self.model.pooler(tokens))
        return tokens, logits

    @torch.no_grad()
    def library_logits(self, pixels: torch.Tensor) -> torch.Tensor:
        """The library's own forward pass, used to check encode()."""
        return self.model(pixel_values_videos=pixels).logits

    def pool_tokens(
        self, tokens: torch.Tensor, num_frames: int = NUM_FRAMES, grid: int = POOL_GRID
    ):
        """(B, T*G*G, D) -> (B, T*grid*grid, D) by average-pooling each time step's patch grid.

        Tokens are ordered (time, row, column) because the patch embedding is a Conv3d
        followed by a flatten.
        """
        b, n, d = tokens.shape
        t = num_frames // self.tubelet
        g = int(round((n / t) ** 0.5))
        if t * g * g != n:
            raise ValueError(f"cannot reshape {n} tokens into {t} x {g} x {g}")
        x = tokens.view(b, t, g, g, d).permute(0, 1, 4, 2, 3).reshape(b * t, d, g, g)
        x = F.adaptive_avg_pool2d(x.float(), grid)
        return (
            x.view(b, t, d, grid * grid)
            .permute(0, 1, 3, 2)
            .reshape(b, t * grid * grid, d)
        )

    def top_labels(self, logits: torch.Tensor, k: int = 5) -> list:
        probs = logits.float().softmax(-1)
        p, i = probs.topk(k, dim=-1)
        return [
            [(self.id2label[int(j)], float(q)) for q, j in zip(pr, ix)]
            for pr, ix in zip(p, i)
        ]
