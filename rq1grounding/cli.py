"""Shared command-line plumbing for the probe scripts."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import load_dataset
from .model import DEFAULT_MODEL, load_model


def common_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default="vstar", help="vstar | synthetic | path/to/samples.jsonl")
    p.add_argument("--data_root", default=None, help="V* folder (downloaded if missing) or synthetic dir")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=1024 * 32 * 32,
                   help="pixel budget; each LLM visual token covers 32x32 px")
    p.add_argument("--min_pixels", type=int, default=256 * 32 * 32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", required=True)
    return p


def setup(args):
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
    samples = load_dataset(args.dataset, args.data_root, args.categories, args.limit)
    model, processor = load_model(args.model, dtype=args.dtype)
    print(f"{len(samples)} samples | {args.model} | layers={model.config.text_config.num_hidden_layers} "
          f"heads={model.config.text_config.num_attention_heads} "
          f"deepstack_vit_layers={model.config.vision_config.deepstack_visual_indexes}")
    return model, processor, samples


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
