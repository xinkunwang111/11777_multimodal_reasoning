"""Probe 2 - Is the attention on the object actually used?  Layer-window attention knockout.

Probe 1 asks whether the attention *finds* the object; this probe asks whether the answer *depends* on it.
For each window of consecutive LLM layers we cut the attention from the text tokens to
  target   the visual tokens of the question's GT object(s)
  control  same-shaped boxes at random places that avoid every annotated object (same token budget)
  image    all visual tokens (how much the answer depends on vision in this window at all)
and record the change of log p(correct option).
  * cutting the target hurts much more than cutting the control -> the attention on the object is used
    (so amplifying it can help), and the windows where this happens are the "reading layers";
  * the method amplifies attention in exactly these layers (chosen from the calibration questions only).

Query groups: all = every text token after the image (default); last = the answer position;
              object = the object's name tokens.
Knockout: --mode scale -> attention weight x 0 (classic knockout)
          --mode shift -> the target's attention is moved onto the other image tokens instead, keeping the
                          total attention on the image unchanged
Synthetic set: when the target is cut, does probability flow to the neighbour's colour (binding leak)?

Outputs: per_sample.jsonl (read by the method), summary.json, knockout_<group>.png
"""

import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.attention import AttentionScaler, AttnEdit  # noqa: E402
from rq1grounding.cli import (append_jsonl, common_parser, load_samples, nanmean,
                              read_jsonl, reading_layers, setup, to_jsonable, write_jsonl)  # noqa: E402
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
    p.add_argument("--groups", nargs="*", default=["all"], choices=["all", "last", "object"])
    p.add_argument("--mode", default="scale", choices=["scale", "shift"])
    args = p.parse_args()
    samples = load_samples(args)
    model, processor = setup(args)
    windows = layer_windows(model.config.text_config.num_hidden_layers, args.window, args.stride)
    rng = np.random.default_rng(args.seed)

    out_file = os.path.join(args.out, "per_sample.jsonl")
    done_rows = read_jsonl(out_file)
    done_map = {r["qid"]: r for r in done_rows if "knockout" in r and r.get("knockout")}
    rows = []
    for s in tqdm(samples):
        if s.qid in done_map:
            rows.append(done_map[s.qid])
            continue
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
        append_jsonl(out_file, row)
    write_jsonl(out_file, rows)

    centers = [float(np.mean(w)) for w in windows]
    summary = {"n": len(rows), "windows": windows, "mode": args.mode, "groups": {}}
    for g in args.groups:
        d = {r: np.array([[c[r] for c in row["knockout"][g]] for row in rows]) for r in REGIONS}
        acc = {r: np.array([[c[f"{r}_correct"] for c in row["knockout"][g]] for row in rows]).mean(0) for r in REGIONS}
        layers, ranking = reading_layers(args.out, None, g, top=1)
        bi = windows.index(layers) if layers in windows else int(np.argmin((d["target"] - d["control"]).mean(0)))
        summary["groups"][g] = {
            # the answer to "is the attention on the object used?", at the reading window
            "used": {"window": windows[bi], "delta_target": float(d["target"][:, bi].mean()),
                     "delta_control": float(d["control"][:, bi].mean()),
                     "share_target_hurts_more": float((d["target"][:, bi] < d["control"][:, bi]).mean())},
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
        print(g, "reading window:", v["reading_window"], {k: round(x, 3) if isinstance(x, float) else x for k, x in v["used"].items()})


if __name__ == "__main__":
    main()
