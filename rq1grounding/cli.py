"""Shared command-line plumbing and the files that pass results between scripts."""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from .data import load_dataset, split
from .model import DEFAULT_MODEL, load_model


def common_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default="vstar", help="vstar | synthetic | path/to/samples.jsonl")
    p.add_argument("--data_root", default=None, help="V* folder (downloaded if missing) or synthetic folder")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=1024 * 32 * 32, help="pixel budget; 1 visual token = 32x32 px")
    p.add_argument("--min_pixels", type=int, default=64 * 32 * 32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p


def add_split_args(p: argparse.ArgumentParser):
    p.add_argument("--calib_frac", type=float, default=0.3, help="share of --dataset used for calibration (hash split)")
    p.add_argument("--calib_dataset", default=None, help="calibrate on another dataset instead (e.g. synthetic)")
    p.add_argument("--calib_root", default=None)
    p.add_argument("--calib_limit", type=int, default=None)


def load_samples(args):
    samples = [s for s in load_dataset(args.dataset, args.data_root, args.categories, args.limit) if s.has_gt]
    if not hasattr(args, "calib_frac"):
        return samples
    if args.calib_dataset:
        calib = [s for s in load_dataset(args.calib_dataset, args.calib_root, None, args.calib_limit) if s.has_gt]
        return calib, samples
    return split(samples, args.calib_frac)


def setup(args):
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
    model, processor = load_model(args.model, dtype=args.dtype)
    c = model.config
    print(f"{args.model}: {c.text_config.num_hidden_layers} layers x {c.text_config.num_attention_heads} heads, "
          f"DeepStack from ViT blocks {c.vision_config.deepstack_visual_indexes}")
    return model, processor


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def write_jsonl(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(to_jsonable(r)) + "\n")


def append_jsonl(path, row):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(to_jsonable(row)) + "\n")
        f.flush()


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(to_jsonable(rows))


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def nanmean(x) -> float:
    x = np.asarray([np.nan if v is None else v for v in x], dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ---------------------------------------------------------------------------- probe outputs used downstream

def load_p1_config(p1_dir: str | None) -> dict | None:
    """Grounding heads, sink threshold and map->box threshold chosen by probe 1 on its calibration split."""
    path = os.path.join(p1_dir or "", "config.json")
    return json.load(open(path)) if p1_dir and os.path.exists(path) else None


def reading_layers(p2_dir: str | None, qids=None, group: str = "all", top: int = 1):
    """Layer window(s) where knocking out the target hurts most beyond the control region, computed only
    from probe-2 samples in `qids` (pass the calibration qids so test samples never choose layers).
    Returns (layers, ranking) or (None, None) if probe 2 has not been run."""
    rows = read_jsonl(os.path.join(p2_dir or "", "per_sample.jsonl")) if p2_dir else []
    rows = [r for r in rows if qids is None or r["qid"] in qids]
    if not rows or group not in rows[0]["knockout"]:
        return None, None
    windows = rows[0]["windows"]
    eff = np.array([[w["target"] - w["control"] for w in r["knockout"][group]] for r in rows])  # [N, n_windows]
    score = np.nanmean(eff, 0)
    order = np.argsort(score)  # most negative = target-specific reading
    layers = sorted({l for i in order[:top] for l in windows[i]})
    return layers, [{"layers": windows[i], "target_minus_control": float(score[i])} for i in order]
