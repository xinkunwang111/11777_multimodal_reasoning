"""Probe 1 - Where does Qwen3-VL look?  Attention-based grounding per layer and head.

For every sample we recompute the attention rows of two kinds of queries onto the
image tokens, in every LLM layer and head:
  * answer query: the last prompt token (the one that emits the option letter)
  * object query: the tokens of the target object's name in the question ("... color of the [glove]")
and score them against the GT box (in-box share, in-box/area ratio, pointing game).

Outputs (in --out):
  per_sample.jsonl, arrays.npz, summary.json
  layer_*.png           layer curves (all / correct / incorrect)
  heads_*.png           [layer x head] grounding maps -> candidate "grounding heads"
  vis/<qid>.png         heatmap overlays for the first --n_vis samples
"""

import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding.data import LETTERS  # noqa: E402
from rq1grounding.cli import common_parser, setup, to_jsonable  # noqa: E402
from rq1grounding.hooks import AttentionRecorder  # noqa: E402
from rq1grounding.metrics import binding_error_type, grounding_metrics, topk_heads  # noqa: E402
from rq1grounding.model import answer_from_logits, forward, prepare  # noqa: E402
from rq1grounding.viz import boxes_to_norm, head_heatmap, layer_curves, overlay, plt  # noqa: E402

KEYS = ("img_mass", "in_box", "ratio", "hit", "entropy")


def main():
    p = common_parser(__doc__)
    p.add_argument("--n_vis", type=int, default=12)
    p.add_argument("--vis_layers", type=int, nargs="*", default=None)
    p.add_argument("--top_heads", type=int, default=16)
    args = p.parse_args()
    model, processor, samples = setup(args)
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
    n_layers = model.config.text_config.num_hidden_layers
    vis_layers = args.vis_layers or sorted({n_layers // 6, n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 1})

    recs, arrs, vis_cache = [], {}, []
    for si, s in enumerate(tqdm(samples)):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        obj_q = [pos[-1] for pos in prep.object_pos.values() if pos]
        qpos = [prep.answer_pos] + obj_q
        with AttentionRecorder(model, qpos) as rec:
            out = forward(model, prep)
        ans = answer_from_logits(out.logits, prep)
        A = rec.attn[..., prep.image_pos.cpu()].float()  # [Ly, H, nQ, n_img]

        row = {}
        for qname, a in (("ans", A[:, :, 0]), ("obj", A[:, :, 1:].mean(2) if obj_q else None)):
            if a is None:
                for k in KEYS:
                    row[f"{qname}_{k}_head"] = np.full(A.shape[:2], np.nan, np.float32)
                    row[f"{qname}_{k}_avg"] = np.full(A.shape[:1], np.nan, np.float32)
                continue
            gh_ = grounding_metrics(a, prep.cov_flat)
            ga = grounding_metrics(a.mean(1), prep.cov_flat)
            for k in KEYS:
                row[f"{qname}_{k}_head"] = gh_[k].numpy()
                row[f"{qname}_{k}_avg"] = ga[k].numpy()
        if prep.distractor_coverage is not None:  # does attention leak onto the other objects?
            row["ans_distractor_in_box_avg"] = grounding_metrics(A[:, :, 0].mean(1), prep.distractor_coverage.reshape(-1))["in_box"].numpy()
        else:
            row["ans_distractor_in_box_avg"] = np.full(A.shape[:1], np.nan, np.float32)
        for k, v in row.items():
            arrs.setdefault(k, []).append(v)

        err = binding_error_type(s, ans["pred"]) if s.category.startswith("synth_attr") else (
            "correct" if ans["correct"] else "wrong")
        recs.append({"qid": s.qid, "category": s.category, **ans, "error_type": err,
                     "grid": [prep.grid_h, prep.grid_w], "box_area_share": float(prep.cov_flat.mean()),
                     "object_tokens_found": {k: len(v) for k, v in prep.object_pos.items()}})
        maps = {"ans": A[:, :, 0].mean(1).reshape(-1, prep.grid_h, prep.grid_w).numpy().astype(np.float16)}
        if obj_q:
            maps["obj"] = A[:, :, 1:].mean((1, 2)).reshape(-1, prep.grid_h, prep.grid_w).numpy().astype(np.float16)
        np.savez_compressed(os.path.join(args.out, "vis", f"{si:04d}_maps.npz"), coverage=prep.coverage, **maps)
        if si < args.n_vis:
            vis_cache.append((si, s, prep.image, prep.grid_h, prep.grid_w, A[:, :, 0].numpy().astype(np.float16),
                              maps.get("obj"), ans))
        del A, out

    arrs = {k: np.stack(v) for k, v in arrs.items()}
    np.savez_compressed(os.path.join(args.out, "arrays.npz"), **arrs)
    with open(os.path.join(args.out, "per_sample.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    correct = np.array([r["correct"] for r in recs])
    cats = sorted({r["category"] for r in recs})

    # --- layer curves
    for qn in ("ans", "obj"):
        for k, yl, h in (("ratio", "in-box / area (1 = chance)", 1.0), ("hit", "pointing-game hit", None),
                         ("img_mass", "attention mass on image", None), ("in_box", "in-box share", None)):
            a = arrs[f"{qn}_{k}_avg"]
            if np.all(np.isnan(a)):
                continue
            curves = {"all": a, "correct": a[correct], "incorrect": a[~correct]}
            curves.update({c: a[[r["category"] == c for r in recs]] for c in cats} if len(cats) > 1 else {})
            layer_curves(curves, yl, os.path.join(args.out, f"layer_{qn}_{k}.png"),
                         title=f"{qn} query, head-averaged attention", hline=h)
    if not np.all(np.isnan(arrs["ans_distractor_in_box_avg"])):
        layer_curves({"target box": arrs["ans_in_box_avg"], "distractor boxes": arrs["ans_distractor_in_box_avg"]},
                     "in-box share", os.path.join(args.out, "layer_ans_target_vs_distractor.png"),
                     title="answer query: target vs other objects")

    # --- head maps & grounding heads
    hit_head = np.nanmean(arrs["ans_hit_head"], 0)
    logratio_head = np.nanmean(np.log(arrs["ans_ratio_head"] + 1e-3), 0)
    obj_hit_head = np.nanmean(arrs["obj_hit_head"], 0) if not np.all(np.isnan(arrs["obj_hit_head"])) else None
    head_heatmap(hit_head, os.path.join(args.out, "heads_ans_hit.png"), "answer query: pointing hit rate")
    head_heatmap(logratio_head, os.path.join(args.out, "heads_ans_logratio.png"), "answer query: mean log(in-box/area)")
    if obj_hit_head is not None:
        head_heatmap(obj_hit_head, os.path.join(args.out, "heads_obj_hit.png"), "object query: pointing hit rate")
    rank_score = hit_head + 1e-3 * logratio_head
    top = topk_heads(rank_score, args.top_heads)

    # --- visualizations
    for si, s, img, gh, gw, A_ans, obj_map, ans in vis_cache:
        cols = 2 + len(vis_layers) + (obj_map is not None)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.2))
        W, H = Image.open(s.image_path).size
        boxes = boxes_to_norm(s.bboxes, W, H)
        overlay(axes[0], img, None, boxes, f"{s.qid}\npred={LETTERS[ans['pred']]} gt={LETTERS[s.answer_idx]}")
        for j, l in enumerate(vis_layers):
            overlay(axes[1 + j], img, A_ans[l].astype(np.float32).mean(0).reshape(gh, gw), boxes, f"answer q, L{l}")
        ens = np.mean([A_ans[l, h].astype(np.float32) for l, h, _ in top], 0).reshape(gh, gw)
        overlay(axes[1 + len(vis_layers)], img, ens, boxes, f"top-{len(top)} grounding heads")
        if obj_map is not None:
            best_l = int(np.nanargmax(np.nanmean(arrs["obj_hit_avg"], 0)))
            overlay(axes[-1], img, obj_map[best_l].astype(np.float32), boxes, f"object-word q, L{best_l}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "vis", f"{si:04d}_{s.qid.replace('/', '_')}.png"), dpi=120)
        plt.close(fig)

    summary = {
        "n": len(recs), "accuracy": float(correct.mean()),
        "accuracy_by_category": {c: float(np.mean([r["correct"] for r in recs if r["category"] == c])) for c in cats},
        "error_types": {e: sum(r["error_type"] == e for r in recs) for e in {r["error_type"] for r in recs}},
        "ans_layer_ratio_mean": np.nanmean(arrs["ans_ratio_avg"], 0),
        "ans_layer_hit_mean": np.nanmean(arrs["ans_hit_avg"], 0),
        "obj_layer_hit_mean": np.nanmean(arrs["obj_hit_avg"], 0),
        "best_layer_ans_hit": int(np.nanargmax(np.nanmean(arrs["ans_hit_avg"], 0))),
        "top_grounding_heads": [{"layer": l, "head": h, "hit_rate": float(hit_head[l, h])} for l, h, _ in top],
        # does looking at the object predict answering correctly?
        "hit_rate_best_layer_correct_vs_incorrect": None,
    }
    bl = summary["best_layer_ans_hit"]
    if correct.any() and (~correct).any():
        summary["hit_rate_best_layer_correct_vs_incorrect"] = [
            float(arrs["ans_hit_avg"][correct, bl].mean()), float(arrs["ans_hit_avg"][~correct, bl].mean())]
    json.dump(to_jsonable(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(to_jsonable({k: summary[k] for k in ("n", "accuracy", "accuracy_by_category", "error_types",
                                                          "best_layer_ans_hit")}), indent=2))


if __name__ == "__main__":
    main()
