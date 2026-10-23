"""Appendix C - What do Qwen3-VL's own alignment designs (DeepStack, MRoPE) contribute?

Conditions
  baseline
  ds_zero_all / ds_zero_<i>   DeepStack off (all levels / level i: ViT block deepstack_visual_indexes[i],
                              injected after LLM layer i) -> the LLM sees only the last ViT layer there
  pos_collapse                all image tokens share one MRoPE position: no 2D layout in RoPE
  pos_hflip / pos_vflip       MRoPE positions mirrored, pixels unchanged
Measured under every condition
  answer     accuracy
  attention  pointing of the object-name attention (grounding heads from probe 1 if --p1, else all heads)
  mirror     for flips only, on objects whose mirrored box does not overlap the original:
             follow_rope_attn   - the attention peak moves to the *mirrored* object position
             follow_rope_answer - left/right (above/below) answers flip
If mirroring RoPE moves the attention / flips the answers, the model's sense of "where" comes from MRoPE;
if not, from the visual content (the ViT's own position information).

Outputs: per_sample.jsonl, summary.json, summary.csv
"""

import csv
import json
import os
import sys
from contextlib import nullcontext

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.cli import common_parser, load_p1_config, load_samples, nanmean, setup, to_jsonable, write_jsonl  # noqa: E402
from rq1grounding.data import mirror_option_idx  # noqa: E402
from rq1grounding.grounding import aggregate, map_metrics, peak_point, record_attention  # noqa: E402
from rq1grounding.hooks import DeepStackRecorder, VisualFeatureEditor, mrope_position_ids  # noqa: E402
from rq1grounding.model import answer_from_logits, prepare  # noqa: E402

FLIP = {"pos_hflip": ("h", G.hflip), "pos_vflip": ("v", G.vflip)}


def conditions(n_ds, which):
    allc = ["baseline", "ds_zero_all"] + [f"ds_zero_{i}" for i in range(n_ds)] + ["pos_collapse", "pos_hflip", "pos_vflip"]
    return [c for c in allc if not which or c in which]


def run_condition(model, prep, cond, args, heads):
    s = prep.sample
    editor = None
    if cond.startswith("ds_zero_"):
        lvl = cond.rsplit("_", 1)[1]
        editor = VisualFeatureEditor(model, zero_deepstack="all" if lvl == "all" else [int(lvl)])
    mode = cond[4:] if cond.startswith("pos_") else "default"
    pos = mrope_position_ids(model, prep.inputs, prep.image_pos, mode, args.seed) if mode != "default" else None
    flip = FLIP.get(cond)
    with editor or nullcontext(), DeepStackRecorder(model) if cond == "baseline" else nullcontext() as ds:
        am = record_attention(model, prep, args.sink_k, pos)
    res = {**answer_from_logits(am.logits, prep), "objects": []}
    for k, obj in enumerate(s.objects):
        o = {}
        A = am.maps.get(f"object:{k}")
        if A is not None:
            M = aggregate(A, heads=heads, sinks=am.sinks)
            o["pointing"] = map_metrics(M, prep.object_cov(k))["pointing"]
            mirrored = [flip[1](b) for b in obj.boxes] if flip else []
            if flip and all(G.iou(m, b) == 0 for m, b in zip(mirrored, obj.boxes)):  # only where mirroring is visible
                pk = peak_point(M, prep.grid_h, prep.grid_w)
                o["follow_rope_attn"] = any(G.contains(m, pk) for m in mirrored)
        res["objects"].append(o)
    if flip:
        mi = mirror_option_idx(s.options, s.answer_idx, flip[0])
        if mi is not None:
            res["follow_rope_answer"] = res["pred"] == mi
    if ds is not None:
        res["deepstack"] = ds.stats
    return res


def main():
    p = common_parser(__doc__)
    p.add_argument("--p1", default=None, help="probe-1 output dir (grounding heads, sink_k)")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--sink_k", type=float, default=5.0)
    args = p.parse_args()
    samples = load_samples(args)
    model, processor = setup(args)
    cfg = load_p1_config(args.p1)
    heads = cfg["heads"] if cfg else None
    if cfg:
        args.sink_k = cfg.get("sink_k", args.sink_k)
    conds = conditions(len(model.config.vision_config.deepstack_visual_indexes), args.conditions)

    rows = []
    for s in tqdm(samples):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        rows.append({"qid": s.qid, "category": s.category,
                     "conds": {c: run_condition(model, prep, c, args, heads) for c in conds}})
    write_jsonl(os.path.join(args.out, "per_sample.jsonl"), rows)

    cats = sorted({r["category"] for r in rows})
    table = []
    for c in conds:
        rs = [r["conds"][c] for r in rows]
        objs = [o for x in rs for o in x["objects"]]
        line = {"condition": c, "acc": nanmean([x["correct"] for x in rs]),
                **{f"acc_{k}": nanmean([r["conds"][c]["correct"] for r in rows if r["category"] == k]) for k in cats},
                "attention_pointing": nanmean([o.get("pointing") for o in objs])}
        if c in FLIP:
            line["follow_rope_attn"] = nanmean([o["follow_rope_attn"] for o in objs if "follow_rope_attn" in o])
            line["follow_rope_answer"] = nanmean([x["follow_rope_answer"] for x in rs if "follow_rope_answer" in x])
        table.append(line)
    summary = {"n": len(rows), "heads": "probe-1 grounding heads" if heads else "all layers x heads", "table": table}
    if "baseline" in conds:
        st = [r["conds"]["baseline"]["deepstack"] for r in rows if r["conds"]["baseline"].get("deepstack")]
        if st:
            summary["deepstack_injection"] = [{k: nanmean([x[i][k] for x in st]) for k in st[0][i]} for i in range(len(st[0]))]
    json.dump(to_jsonable(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for t in table for k in t)))
        w.writeheader()
        w.writerows(to_jsonable(table))
    for t in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})


if __name__ == "__main__":
    main()
