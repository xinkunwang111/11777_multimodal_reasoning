"""Collect the outputs of run_all.sh into one Markdown report: <out>/REPORT.md.

    python scripts/collect_results.py outputs/Qwen3-VL-8B-Instruct
"""

import csv
import json
import os
import sys

import numpy as np


def _json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-" if x in (None, "") else str(x)
    return "-" if np.isnan(x) else f"{x:.{nd}f}"


def table(rows, cols, headers=None):
    if not rows:
        return "_(no results)_\n"
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_f(r.get(c)) if c not in ("map", "variant", "condition", "source", "feature")
                                     else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def img(rel, path, alt):
    return f"![{alt}]({os.path.relpath(path, rel)})\n" if os.path.exists(path) else ""


def argbest(v):
    v = np.asarray(v, dtype=float)
    return (int(np.nanargmax(v)), float(np.nanmax(v))) if np.isfinite(v).any() else (None, float("nan"))


def section_attention(d, root):
    s = _json(os.path.join(d, "summary.json"))
    if not s:
        return ""
    cv = s.get("hit_rate_best_layer_correct_vs_incorrect") or [None, None]
    md = ["### 01 Attention heatmaps\n",
          f"- accuracy {_f(s['accuracy'])} | by category: "
          + ", ".join(f"{k} {_f(v)}" for k, v in s["accuracy_by_category"].items()),
          f"- error types: {s['error_types']}",
          f"- best layer (answer-query pointing): L{s['best_layer_ans_hit']}; pointing there: correct {_f(cv[0])} "
          f"vs incorrect {_f(cv[1])}",
          "- top grounding heads (layer, head, hit): "
          + ", ".join(f"({h['layer']},{h['head']},{_f(h['hit_rate'], 2)})" for h in s["top_grounding_heads"][:8]), ""]
    md.append(img(root, os.path.join(d, "layer_ans_ratio.png"), "answer-query lift per layer"))
    return "\n".join(md) + "\n"


def section_lens(d, root):
    s = _json(os.path.join(d, "summary.json"))
    if not s:
        return ""
    li, vi = argbest(s["bind_in_layer_mean"])
    lo, vo = argbest(s["bind_out_layer_mean"])
    ri = np.nanmin(s["obj_rank_in_layer_mean"])
    ro = np.nanmin(s["obj_rank_out_layer_mean"])
    md = ["### 02 Logit lens on image tokens\n",
          f"- option-word binding accuracy: in-box best {_f(vi)} (layer {li}) vs out-box best {_f(vo)} (layer {lo})",
          f"- object name, best mean log10 rank: in-box {_f(ri, 2)} vs out-box {_f(ro, 2)}", ""]
    md.append(img(root, os.path.join(d, "layer_binding_acc_all.png"), "logit-lens binding"))
    return "\n".join(md) + "\n"


def section_probe(d, root):
    r = _json(os.path.join(d, "probe_results.json"))
    if not r:
        return ""
    md = ["### 03 Linear probes\n"]
    for task in ("attribute", "relation"):
        if task not in r:
            continue
        t = r[task]
        rows = []
        for feat, accs in t["llm"].items():
            layer, best = argbest(accs)
            rows.append({"feature": feat, "best_acc": best, "layer": layer})
        for feat, acc in t["vit"].items():
            rows.append({"feature": feat, "best_acc": acc, "layer": None})
        md.append(f"**{task}** (n={t['n']}, majority={_f(t['majority_baseline'])})\n")
        md.append(table(rows, ["feature", "best_acc", "layer"]))
    if "position" in r:
        rc = np.array(r["position"]["llm_r2_row_col"], dtype=float)
        md.append(f"- position probe (LLM residual): best R2 row {_f(np.nanmax(rc[:, 0]))}, col {_f(np.nanmax(rc[:, 1]))}; "
                  + ", ".join(f"{k.replace('_pos_r2_row_col', '')} {_f(v[0], 2)}/{_f(v[1], 2)}"
                              for k, v in r["position"].items() if k.startswith("vit_")) + "\n")
    md.append(img(root, os.path.join(d, "probe_attribute.png"), "attribute probe"))
    return "\n".join(md) + "\n"


def section_interventions(d, root):
    rows = _csv(os.path.join(d, "summary.csv"))
    if not rows:
        return ""
    s = _json(os.path.join(d, "summary.json")) or {}
    allr = [r for r in rows if r["category"] == "ALL"]
    md = ["### 04 Interventions (all categories)\n",
          table(allr, ["condition", "acc", "mean_p_correct", "delta_p_vs_baseline", "follow_rope"])]
    cats = sorted({r["category"] for r in rows} - {"ALL"})
    if len(cats) > 1:
        piv = []
        for c in dict.fromkeys(r["condition"] for r in rows):
            piv.append({"condition": c, **{k: next((r["acc"] for r in rows if r["condition"] == c and r["category"] == k), None)
                                           for k in cats}})
        md.append("accuracy by category:\n\n" + table(piv, ["condition"] + cats))
    if s.get("pair_consistency"):
        md.append("- color-swap twin consistency: " + ", ".join(f"{k} {_f(v)}" for k, v in s["pair_consistency"].items()))
    if s.get("deepstack_injection"):
        md.append("- DeepStack injection ||added||/||residual||: "
                  + ", ".join(f"L{i} {_f(x['norm_ratio'], 2)}" for i, x in enumerate(s["deepstack_injection"])))
    md.append("\n" + img(root, os.path.join(d, "attention_by_condition.png"), "grounding under interventions"))
    return "\n".join(md) + "\n"


def section_bench(d, root, cfg_path, title="05 Grounding bench (test split, calibrated thresholds)"):
    rows = _csv(os.path.join(d, "bench_table.csv"))
    s = _json(os.path.join(d, "bench_summary.json"))
    if not rows or not s:
        return ""
    cfg = _json(cfg_path) or {}
    rule = cfg.get("default_rule", "rel")
    rows = sorted(rows, key=lambda r: -float(r["auroc"]) if r["auroc"] not in ("", "nan") else 0)
    md = [f"### {title}\n",
          f"- n={s['n']}, model accuracy {_f(s['accuracy'])}, sink tokens {_f(100 * s['mean_sink_frac'], 1)}%, "
          f"quantization ceiling (oracle IoU) {_f(s['mean_oracle_iou'])}",
          f"- thresholded metrics use rule `{rule}` with thresholds fixed on the calibration split\n",
          table(rows, ["map", "pointing", "auroc", "lift", "own_gt_other", f"{rule}@cal:center_hit",
                       f"{rule}@cal:recall", f"{rule}@cal:tok_f1", f"{rule}@cal:area"],
                ["map", "pointing", "AUROC", "lift", "own>other", "center hit", "recall", "tok F1", "area"])]
    br = s.get("breakdown", {})
    best = s.get("best_map_by_auroc")
    if best and "correct" in br and "incorrect" in br:
        md.append(f"- `{best}` correct vs incorrect: pointing {_f(br['correct'][best]['pointing'])} vs "
                  f"{_f(br['incorrect'][best]['pointing'])}, AUROC {_f(br['correct'][best]['auroc'])} vs "
                  f"{_f(br['incorrect'][best]['auroc'])}")
    if cfg.get("heads"):
        md.append("- grounding heads (calibration): " + "; ".join(f"{q}: {[tuple(h) for h in hs][:6]}"
                                                                 for q, hs in cfg["heads"].items()))
    md.append("\n" + img(root, os.path.join(d, "layer_object_auroc.png"), "object-query AUROC per layer")
              + img(root, os.path.join(d, "threshold_sweep.png"), "threshold sweep"))
    return "\n".join(md) + "\n"


def section_zoom(d, root, title="06 Self-grounded zoom"):
    rows = _csv(os.path.join(d, "summary.csv"))
    if not rows:
        return ""
    cats = [k for k in rows[0] if k.startswith("acc_")]
    md = [f"### {title}\n", table(rows, ["variant"] + cats + ["mean_tokens"])]
    q = _csv(os.path.join(d, "crop_quality.csv"))
    if q:
        md.append("crop contains the target:\n\n" + table(q, ["source", "center_in", "recall", "area"]))
    g = _csv(os.path.join(d, "gating.csv"))
    if g:
        best = {}
        for r in g:
            if r["variant"] not in best or float(r["acc"]) > float(best[r["variant"]]["acc"]):
                best[r["variant"]] = r
        md.append("best confidence gate per variant (chosen on this split - optimistic):\n\n"
                  + table(list(best.values()), ["variant", "tau", "zoom_rate", "acc"]))
    return "\n".join(md) + "\n"


def main(out):
    md = [f"# RQ1 results: `{out}`\n"]
    cfg = os.path.join(out, "logs", "run_config.txt")
    if os.path.exists(cfg):
        md.append("```\n" + open(cfg).read().strip() + "\n```\n")
    for ds in sorted(x for x in os.listdir(out) if os.path.isdir(os.path.join(out, x)) and x != "logs"):
        d = os.path.join(out, ds)
        md.append(f"## {ds}\n")
        md.append(section_attention(os.path.join(d, "01_attention"), out))
        md.append(section_lens(os.path.join(d, "02_logit_lens"), out))
        md.append(section_probe(os.path.join(d, "03_linear_probe"), out))
        md.append(section_interventions(os.path.join(d, "04_interventions"), out))
        cfg = os.path.join(d, "05_bench", "grounding_config.json")
        md.append(section_bench(os.path.join(d, "05_bench"), out, cfg))
        md.append(section_zoom(os.path.join(d, "06_zoom"), out))
        for sw in sorted(x for x in os.listdir(d) if x.startswith("sweep_")):
            md.append(section_bench(os.path.join(d, sw, "05_bench"), out, cfg, f"05 bench @ {sw[6:]}"))
            md.append(section_zoom(os.path.join(d, sw, "06_zoom"), out, f"06 zoom @ {sw[6:]}"))
    path = os.path.join(out, "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(x for x in md if x))
    print(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
