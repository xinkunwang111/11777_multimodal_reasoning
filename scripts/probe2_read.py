"""Probe 2 - In which layers does the answer read the object?  Layer-window attention knockout.

For each window of consecutive LLM layers we cut the attention from a group of text queries to
  target   the visual tokens of the question's GT object(s)
  control  same-shaped boxes at random places that avoid every annotated object (same token budget)
  image    all visual tokens (how much the answer depends on vision in this window at all)
and record the change of log p(correct option). Windows where cutting the target hurts much more than
cutting the control are where the answer *reads* the object ("reading layers"); the method steers
attention in exactly these layers.

Query groups: all    = every text token after the image (question, options, answer slot)
              last   = the answer position only
              object = the object's name tokens in the question
Knockout: --mode scale -> attention weight x 0 (classic knockout)
          --mode shift -> the target's attention is moved onto the other image tokens, so the total
                          attention on the image is unchanged (separates "where" from "how much")
Synthetic set: when the target is cut, does probability flow to the neighbour's colour (binding leak)?

Outputs: per_sample.jsonl (the method picks its layers from the calibration samples in here),
         summary.json, knockout_<group>.png
"""

import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.attention import AttentionScaler, AttnEdit  # noqa: E402
from rq1grounding.cli import common_parser, load_samples, nanmean, reading_layers, setup, to_jsonable, write_jsonl  # noqa: E402
from rq1grounding.model import answer_from_logits, forward, prepare  # noqa: E402
from rq1grounding.viz import layer_curves  # noqa: E402

REGIONS = ("target", "control", "image")


def layer_windows(n_layers: int, size: int, stride: int) -> list:
    starts = list(range(0, max(1, n_layers - size + 1), stride))
    if starts[-1] + size < n_layers:
        starts.append(n_layers - size)
    return [list(range(s, min(s + size, n_layers))) for s in starts]


def knockout(model, prep, layers, queries, keys, mode, pool):
    if mode == "shift" and pool is not None:
        edit = AttnEdit(layers, queries, keys, "shift", -1.0, pool)
    else:
        edit = AttnEdit(layers, queries, keys, "scale", 0.0)
    with AttentionScaler(model, [edit]):
        return answer_from_logits(forward(model, prep).logits, prep)


def main():
    p = common_parser(__doc__)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--groups", nargs="*", default=["all", "last"], choices=["all", "last", "object"])
    p.add_argument("--mode", default="scale", choices=["scale", "shift"])
    args = p.parse_args()
    samples = load_samples(args)
    model, processor = setup(args)
    windows = layer_windows(model.config.text_config.num_hidden_layers, args.window, args.stride)
    rng = np.random.default_rng(args.seed)

    rows = []
    for s in tqdm(samples):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        base = answer_from_logits(forward(model, prep).logits, prep)
        img = prep.image_pos.cpu().numpy()
        avoid = s.boxes + [tuple(b) for b in s.meta.get("other_boxes", [])]
        target = prep.tokens_of(s.boxes)
        control = np.setdiff1d(prep.tokens_of([G.random_box_like(b, avoid, rng) for b in s.boxes]), target)
        keys = {"target": target, "control": control, "image": img}
        queries = {"all": prep.text_pos, "last": [prep.answer_pos], "object": sorted({q for o in prep.object_pos for q in o})}
        nb = next((int(k) for k, v in s.meta.get("option_source", {}).items() if v == "neighbor"), None)
        row = {"qid": s.qid, "category": s.category, "base": base, "windows": windows,
               "n_tokens": {k: len(v) for k, v in keys.items()}, "knockout": {}}
        for g in args.groups:
            out = []
            for w in windows:
                cell = {}
                for region in REGIONS:
                    pool = np.setdiff1d(img, keys[region]) if region != "image" else None
                    r = knockout(model, prep, w, queries[g], keys[region], args.mode, pool)
                    cell[region] = r["logp_correct"] - base["logp_correct"]
                    cell[f"{region}_correct"] = r["correct"]
                    if region == "target" and nb is not None:
                        cell["neighbor_gain"] = r["probs"][nb] - base["probs"][nb]
                out.append(cell)
            row["knockout"][g] = out
        rows.append(row)
    write_jsonl(os.path.join(args.out, "per_sample.jsonl"), rows)

    centers = [float(np.mean(w)) for w in windows]
    summary = {"n": len(rows), "windows": windows, "mode": args.mode, "groups": {}}
    for g in args.groups:
        d = {r: np.array([[c[r] for c in row["knockout"][g]] for row in rows]) for r in REGIONS}
        acc = {r: np.array([[c[f"{r}_correct"] for c in row["knockout"][g]] for row in rows]).mean(0) for r in REGIONS}
        layers, ranking = reading_layers(args.out, None, g, top=1)
        summary["groups"][g] = {
            "delta_logp": {r: d[r].mean(0) for r in REGIONS},
            "accuracy_after_knockout": acc,
            "base_accuracy": nanmean([row["base"]["correct"] for row in rows]),
            "reading_window": layers, "ranking": ranking,
            "by_category": {c: {r: d[r][[row["category"] == c for row in rows]].mean(0) for r in ("target", "control")}
                            for c in sorted({row["category"] for row in rows})},
        }
        gains = [[c.get("neighbor_gain") for c in row["knockout"][g]] for row in rows if "neighbor_gain" in row["knockout"][g][0]]
        if gains:
            summary["groups"][g]["neighbor_gain_when_target_cut"] = np.nanmean(np.array(gains, dtype=float), 0)
        layer_curves({f"cut {r}": d[r] for r in REGIONS}, "change in log p(correct)",
                     os.path.join(args.out, f"knockout_{g}.png"),
                     f"queries = {g}; knockout of {args.window}-layer windows ({args.mode})", hline=0.0, x=centers,
                     xlabel="LLM layer (window centre)")
    json.dump(to_jsonable(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    for g, v in summary["groups"].items():
        print(g, "reading window:", v["reading_window"])


if __name__ == "__main__":
    main()
