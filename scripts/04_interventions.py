"""Probe 4 - Causal tests of Qwen3-VL's alignment design.

Conditions (each = one extra forward per sample):
  baseline
  ablate_target   mean-ablate the visual tokens inside the GT box (main + DeepStack embeds)
  ablate_random   same number of random tokens outside the box (control)
  ds_zero_all     drop all DeepStack injections (LLM sees only the final ViT layer)
  ds_zero_<i>     drop DeepStack level i (ViT block deepstack_visual_indexes[i], injected after LLM layer i)
  pos_collapse    all image tokens share one MRoPE position -> 2D layout removed from RoPE
  pos_shuffle     MRoPE (h, w) permuted across image tokens -> layout decoupled from content
  pos_hflip/vflip MRoPE says "mirrored", pixels are not. On left/right (above/below) questions,
                  `follow_rope` = how often the answer flips to the mirrored option.
With --attention, the answer-query in-box attention is also tracked per condition, e.g.
to see whether removing DeepStack makes the visual tokens harder to localize.

Outputs: per_sample.jsonl, summary.csv / summary.json, attention_by_condition.png
"""

import csv
import json
import os
import sys
from contextlib import ExitStack

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding.cli import common_parser, setup, to_jsonable  # noqa: E402
from rq1grounding.data import mirror_option_idx  # noqa: E402
from rq1grounding.hooks import AttentionRecorder, DeepStackRecorder, VisualFeatureEditor, mrope_position_ids  # noqa: E402
from rq1grounding.metrics import grounding_metrics  # noqa: E402
from rq1grounding.model import answer_from_logits, forward, prepare  # noqa: E402
from rq1grounding.viz import layer_curves  # noqa: E402


def conditions(n_ds: int, which: list[str] | None):
    allc = (["baseline", "ablate_target", "ablate_random", "ds_zero_all"] + [f"ds_zero_{i}" for i in range(n_ds)]
            + ["pos_collapse", "pos_shuffle", "pos_hflip", "pos_vflip"])
    return [c for c in allc if not which or c in which]


def run_condition(model, prep, cond, rng, attention: bool, seed: int):
    cov = prep.cov_flat
    editor, pos_ids = None, None
    if cond == "ablate_target":
        editor = VisualFeatureEditor(model, ablate_tokens=np.flatnonzero(cov > 0))
    elif cond == "ablate_random":
        n = int((cov > 0).sum())
        rest = np.flatnonzero(cov == 0)
        editor = VisualFeatureEditor(model, ablate_tokens=rng.choice(rest, min(n, len(rest)), replace=False))
    elif cond.startswith("ds_zero_"):
        lvl = cond.split("_")[-1]
        editor = VisualFeatureEditor(model, zero_deepstack="all" if lvl == "all" else [int(lvl)])
    elif cond.startswith("pos_"):
        pos_ids = mrope_position_ids(model, prep, cond[4:], seed=seed)
    with ExitStack() as st:
        if editor is not None:
            st.enter_context(editor)
        ds = st.enter_context(DeepStackRecorder(model)) if cond == "baseline" else None
        rec = st.enter_context(AttentionRecorder(model, [prep.answer_pos])) if attention else None
        out = forward(model, prep, position_ids=pos_ids)
    res = answer_from_logits(out.logits, prep)
    if rec is not None:
        a = rec.attn[:, :, 0][..., prep.image_pos.cpu()].float().mean(1)  # [Ly, n_img]
        g = grounding_metrics(a, cov)
        res["attn_in_box"], res["attn_img_mass"] = g["in_box"].tolist(), g["img_mass"].tolist()
    if ds is not None:
        res["deepstack_stats"] = ds.stats
    return res


def main():
    p = common_parser(__doc__)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--attention", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    model, processor, samples = setup(args)
    conds = conditions(len(model.config.vision_config.deepstack_visual_indexes), args.conditions)
    rng = np.random.default_rng(args.seed)

    rows = []
    for s in tqdm(samples):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        mirror = {"h": mirror_option_idx(s.options, s.answer_idx, "h"), "v": mirror_option_idx(s.options, s.answer_idx, "v")}
        r = {"qid": s.qid, "category": s.category, "pair_id": s.meta.get("pair_id"), "answer_idx": s.answer_idx,
             "mirror_h": mirror["h"], "mirror_v": mirror["v"], "box_tokens": int((prep.cov_flat > 0).sum()), "conds": {}}
        for c in conds:
            r["conds"][c] = run_condition(model, prep, c, rng, args.attention, args.seed)
        rows.append(r)
    with open(os.path.join(args.out, "per_sample.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # --- summary table: category x condition
    cats = sorted({r["category"] for r in rows}) + ["ALL"]
    table, summary = [], {}
    for cat in cats:
        sub = [r for r in rows if cat == "ALL" or r["category"] == cat]
        base_p = np.array([r["conds"]["baseline"]["p_correct"] for r in sub]) if "baseline" in conds else None
        for c in conds:
            acc = np.mean([r["conds"][c]["correct"] for r in sub])
            pc = np.array([r["conds"][c]["p_correct"] for r in sub])
            line = {"category": cat, "condition": c, "n": len(sub), "acc": round(float(acc), 4),
                    "mean_p_correct": round(float(pc.mean()), 4),
                    "delta_p_vs_baseline": round(float((pc - base_p).mean()), 4) if base_p is not None else None}
            axis = {"pos_hflip": "mirror_h", "pos_vflip": "mirror_v"}.get(c)
            if axis:
                m = [r for r in sub if r[axis] is not None]
                line["n_mirrorable"] = len(m)
                line["follow_rope"] = (round(float(np.mean([r["conds"][c]["pred"] == r[axis] for r in m])), 4)
                                       if m else None)
            table.append(line)
    # binding consistency on color-swapped twins (synthetic): both members answered correctly
    pairs = {}
    for r in rows:
        if r["pair_id"] and r["category"] == "synth_attr":
            pairs.setdefault(r["pair_id"], []).append(r)
    pairs = [v for v in pairs.values() if len(v) == 2]
    if pairs:
        summary["pair_consistency"] = {c: float(np.mean([all(x["conds"][c]["correct"] for x in v) for v in pairs]))
                                       for c in conds}
    if "baseline" in conds and "deepstack_stats" in rows[0]["conds"]["baseline"]:
        st = [r["conds"]["baseline"]["deepstack_stats"] for r in rows]
        summary["deepstack_injection"] = [{k: float(np.mean([x[i][k] for x in st])) for k in st[0][i]}
                                          for i in range(len(st[0]))]
    summary["table"] = table
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        keys = sorted({k for t in table for k in t}, key=lambda k: list(table[0]).index(k) if k in table[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    json.dump(to_jsonable(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    if args.attention:
        layer_curves({c: np.array([r["conds"][c]["attn_in_box"] for r in rows]) for c in conds},
                     "answer-query in-box share", os.path.join(args.out, "attention_by_condition.png"),
                     "grounding under interventions")
    for t in table:
        if t["category"] == "ALL":
            print(t)


if __name__ == "__main__":
    main()
