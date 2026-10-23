"""Probe 3 - What do Qwen3-VL's alignment designs contribute to finding vs reading?

Conditions
  baseline
  ds_zero_all / ds_zero_<i>   DeepStack off (all levels / level i: ViT block deepstack_visual_indexes[i],
                              injected after LLM layer i) -> the LLM sees only the last ViT layer there
  pos_collapse                all image tokens share one MRoPE position: no 2D layout in RoPE
  pos_hflip / pos_vflip       MRoPE positions mirrored, pixels unchanged
Measured under every condition
  read     answer accuracy
  find     explicit: the model's own box (centre-in-GT, IoU); implicit: attention pointing of the object's
           name tokens (grounding heads from probe 1 if --p1 is given, else all layers x heads)
  mirror   for flips only, on objects whose mirrored box does not overlap the original:
           follow_rope_box  - the model's box lands on the *mirrored* GT (spatial sense from RoPE)
           follow_rope_attn - the attention peak lands on the mirrored GT
           follow_rope_answer - left/right (above/below) answers flip
Reading of the outcome: if mirroring RoPE moves the boxes, the model's notion of "where" comes from
MRoPE; if the boxes stay on the object, it comes from the visual content (ViT position embeddings).

Outputs: per_sample.jsonl, summary.json, summary.csv
"""

import csv
import json
import os
import sys
from contextlib import ExitStack, nullcontext

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.cli import common_parser, load_p1_config, load_samples, nanmean, setup, to_jsonable, write_jsonl  # noqa: E402
from rq1grounding.data import mirror_option_idx  # noqa: E402
from rq1grounding.grounding import aggregate, ground_explicit, map_metrics, peak_point, record_attention  # noqa: E402
from rq1grounding.hooks import DeepStackRecorder, VisualFeatureEditor, mrope_position_ids  # noqa: E402
from rq1grounding.model import answer_from_logits, prepare  # noqa: E402

FLIP = {"pos_hflip": ("h", G.hflip), "pos_vflip": ("v", G.vflip)}


def conditions(n_ds, which):
    allc = ["baseline", "ds_zero_all"] + [f"ds_zero_{i}" for i in range(n_ds)] + ["pos_collapse", "pos_hflip", "pos_vflip"]
    return [c for c in allc if not which or c in which]


def run_condition(model, processor, prep, cond, args, heads):
    s = prep.sample
    editor = None
    if cond.startswith("ds_zero_"):
        lvl = cond.rsplit("_", 1)[1]
        editor = VisualFeatureEditor(model, zero_deepstack="all" if lvl == "all" else [int(lvl)])
    mode = cond[4:] if cond.startswith("pos_") else "default"
    pos_fn = (lambda inputs, image_pos: mrope_position_ids(model, inputs, image_pos, mode, args.seed)) if mode != "default" else None
    flip = FLIP.get(cond)
    with ExitStack() as st:
        if editor:
            st.enter_context(editor)
        with DeepStackRecorder(model) if cond == "baseline" else nullcontext() as ds:  # answer pass only
            am = record_attention(model, prep, args.sink_k, pos_fn(prep.inputs, prep.image_pos) if pos_fn else None)
        res = {**answer_from_logits(am.logits, prep), "objects": []}
        for k, obj in enumerate(s.objects):
            o = {}
            A = am.maps.get(f"object:{k}")
            M = aggregate(A, heads=heads, sinks=am.sinks) if A is not None else None
            if M is not None:
                o["pointing"] = map_metrics(M, prep.object_cov(k))["pointing"]
            if not args.no_explicit:
                ex = ground_explicit(model, processor, prep.image, obj.name, args.max_new_tokens, pos_fn)
                o.update({"box": ex["box"], **G.match(ex["box"], obj.boxes)})
            if flip:
                mirrored = [flip[1](b) for b in obj.boxes]
                if all(G.iou(m, b) == 0 for m, b in zip(mirrored, obj.boxes)):  # only where mirroring is visible
                    if o.get("box") is not None:
                        o["follow_rope_box"] = G.match(o["box"], mirrored)["iou"] > G.match(o["box"], obj.boxes)["iou"]
                    if M is not None:
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
    p.add_argument("--no_explicit", action="store_true", help="skip box generation (faster)")
    p.add_argument("--max_new_tokens", type=int, default=128)
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
                     "conds": {c: run_condition(model, processor, prep, c, args, heads) for c in conds}})
    write_jsonl(os.path.join(args.out, "per_sample.jsonl"), rows)

    cats = sorted({r["category"] for r in rows})
    table = []
    for c in conds:
        rs = [r["conds"][c] for r in rows]
        objs = [o for x in rs for o in x["objects"]]
        line = {"condition": c, "acc": nanmean([x["correct"] for x in rs]),
                **{f"acc_{k}": nanmean([r["conds"][c]["correct"] for r in rows if r["category"] == k]) for k in cats},
                "find_explicit_center_hit": nanmean([o.get("center_hit") for o in objs]),
                "find_explicit_iou": nanmean([o.get("iou") for o in objs]),
                "find_attention_pointing": nanmean([o.get("pointing") for o in objs])}
        if c in FLIP:
            line["follow_rope_box"] = nanmean([o["follow_rope_box"] for o in objs if "follow_rope_box" in o])
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
