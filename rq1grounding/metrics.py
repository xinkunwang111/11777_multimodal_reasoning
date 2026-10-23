"""Grounding metrics on attention over the image-token grid."""

from __future__ import annotations

import numpy as np
import torch


def grounding_metrics(attn_img: torch.Tensor, cov_flat: np.ndarray) -> dict[str, torch.Tensor]:
    """attn_img: [..., n_img] raw attention probabilities of a query onto the image tokens.

    img_mass  - total attention the query puts on the image (visual reliance)
    in_box    - share of the image attention landing on the target box (coverage weighted)
    ratio     - in_box / box area share; 1 = uniform, >1 = attends to the target more than chance
    hit       - pointing game: is the most-attended image token inside the box
    entropy   - normalized entropy of the image attention (1 = uniform)
    """
    a = attn_img.float()
    cov = torch.as_tensor(cov_flat, dtype=torch.float32)
    mass = a.sum(-1)
    p = a / mass.clamp_min(1e-12).unsqueeze(-1)
    in_box = (p * cov).sum(-1)
    area = cov.mean().clamp_min(1e-6)
    hit = (cov > 0)[p.argmax(-1)].float()
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1) / np.log(p.shape[-1])
    return {"img_mass": mass, "in_box": in_box, "ratio": in_box / area, "hit": hit, "entropy": ent}


def topk_heads(score: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """score: [layers, heads] -> [(layer, head, score)] of the k largest."""
    flat = np.argsort(score.reshape(-1))[::-1][:k]
    return [(int(i // score.shape[1]), int(i % score.shape[1]), float(score.reshape(-1)[i])) for i in flat]


def binding_error_type(sample, pred: int) -> str:
    """For synthetic attribute questions: correct / binding (another object's color) / hallucination."""
    if pred == sample.answer_idx:
        return "correct"
    src = sample.meta.get("option_source", {}).get(str(pred))
    return {"in_image": "binding", "absent": "hallucination"}.get(src, "wrong")
