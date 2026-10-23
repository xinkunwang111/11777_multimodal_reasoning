"""Collect the outputs of run_all.sh into one Markdown report: <out>/REPORT.md.

    python scripts/collect_results.py outputs/Qwen3-VL-8B-Instruct
Layout read: <out>/<dataset>/{p1_attention, p2_knockout, method, vea, appendix_*}
"""

import csv
import json
import os
import sys

import numpy as np

TEXT_COLS = {"mode", "condition", "source", "feature", "group", "attention_found_object", "category", "cell", "method",
             "image", "prompt", "scores", "vea_scores_peak_on_object"}


def _json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def f(x, nd=3):
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool) or (isinstance(x, str) and x.isdigit()):
        return str(int(x))  # counts
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-" if x in (None, "") else str(x)
    return "-" if np.isnan(x) else f"{x:.{nd}f}"


def table(rows, cols, headers=None):
    if not rows:
        return "_(no results)_\n"
    out = ["| " + " | ".join(headers or cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) if c in TEXT_COLS else f(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def img(root, path, alt):
    return f"![{alt}]({os.path.relpath(path, root)})\n" if os.path.exists(path) else ""


def p1(d, root):
    s = _json(os.path.join(d, "summary.json"))
    if not s:
        return ""
    cfg = _json(os.path.join(d, "config.json")) or {}
    t = s["attention_found_x_correct"]
    md = ["### Probe 1 - does the attention find the object, even when the answer is wrong?\n",
          f"- answer accuracy {f(s['accuracy'])} (" + ", ".join(f"{k} {f(v)}" for k, v in s["accuracy_by_category"].items()) + ")",
          f"- object name located in the question for {f(s['name_tokens_found'])} of the objects"]
    for k, name in (("attention_heads", f"{len(cfg.get('heads', [])) or 'all'} grounding heads"), ("attention_all_heads", "all heads")):
        if k in s:
            md.append(f"- attention, {name}: " + ", ".join(f"{m} {f(v)}" for m, v in s[k].items()))
    md.append(f"- **attention on the object when the answer is correct: {f(t['found_rate_when_correct'])}; "
              f"when it is wrong: {f(t['found_rate_when_wrong'])}**")
    if "binding_errors_with_attention_on_target" in s:
        md.append(f"- binding errors (answered another object's colour) with the attention on the right object: "
                  f"{f(s['binding_errors_with_attention_on_target'])}")
    md.append(f"- error types: {s['error_types']}" + (f"; colour-swap twins both correct {f(s['twin_both_correct'])}"
                                                      if "twin_both_correct" in s else ""))
    rows = [{"cell": "attention found the object", "correct": t["found&correct"], "wrong": t["found&wrong"], "accuracy": t["accuracy_if_found"]},
            {"cell": "attention missed the object", "correct": t["not_found&correct"], "wrong": t["not_found&wrong"], "accuracy": t["accuracy_if_not_found"]}]
    md.append("\n" + table(rows, ["cell", "correct", "wrong", "accuracy"], ["", "answer correct", "answer wrong", "accuracy"]))
    md.append(img(root, os.path.join(d, "layers.png"), "pointing per layer") + img(root, os.path.join(d, "heads.png"), "heads"))
    vis = sorted(os.listdir(os.path.join(d, "vis")))[:2] if os.path.isdir(os.path.join(d, "vis")) else []
    md += [img(root, os.path.join(d, "vis", v), v) for v in vis]
    return "\n".join(md) + "\n"


def p2(d, root):
    s = _json(os.path.join(d, "summary.json"))
    if not s:
        return ""
    md = [f"### Probe 2 - is the attention on the object used? (knockout: {s['mode']})\n"]
    for g, v in s["groups"].items():
        u = v["used"]
        md.append(f"**queries = {g}** - base accuracy {f(v['base_accuracy'])}; reading window **{u['window']}**: cutting the "
                  f"object changes log p(correct) by {f(u['delta_target'])} vs {f(u['delta_control'])} for the control region; "
                  f"the object cut hurts more in {f(u['share_target_hurts_more'])} of the questions\n")
        rows = [{"cell": str(w), "target": v["delta_logp"]["target"][i], "control": v["delta_logp"]["control"][i],
                 "image": v["delta_logp"]["image"][i], "acc after target cut": v["accuracy_after_knockout"]["target"][i]}
                for i, w in enumerate(s["windows"])]
        if "neighbor_gain_when_target_cut" in v:
            for r, gn in zip(rows, v["neighbor_gain_when_target_cut"]):
                r["p(neighbour colour) gain"] = gn
        md.append(table(rows, list(rows[0]), ["layers"] + list(rows[0])[1:]))
        md.append(img(root, os.path.join(d, f"knockout_{g}.png"), f"knockout {g}"))
    return "\n".join(md) + "\n"


def method(d, root):
    rows = _csv(os.path.join(d, "summary.csv"))
    if not rows:
        return ""
    cal = _json(os.path.join(d, "calibration.json")) or {}
    srcs = [c for c in rows[0] if c not in ("mode", "mean_tokens")]
    md = ["### Method A - amplify the attention on the attention-found region (accuracy; columns = region source)\n"]
    if cal:
        md.append(f"- steer: layers `{cal.get('layer_set')}` {cal.get('layers')}, {cal.get('mode')} {cal.get('value')} "
                  f"(chosen on the calibration split)\n")
    md.append(table(rows, ["mode"] + srcs + ["mean_tokens"]))
    q = _csv(os.path.join(d, "region_quality.csv"))
    if q:
        md.append("region quality:\n\n" + table(q, ["source", "center_hit", "iou", "no_region"]))
    bf = _csv(os.path.join(d, "by_found.csv"))
    if bf:
        md.append("accuracy split by whether the attention found the object:\n\n" + table(bf, list(bf[0])))
    bc = _csv(os.path.join(d, "summary_by_category.csv"))
    if bc:
        md.append("by category:\n\n" + table(bc, ["category", "mode"] + srcs))
    return "\n".join(md) + "\n"


def vea(d, root):
    s = _json(os.path.join(d, "summary.json"))
    if not s:
        return ""
    pr, sn, hp = s["profile"], s["seeing_not_believing"], s["hyperparameters"]
    md = ["### Method B - VEA: dim the image outside where the attention looks (Seeing but Not Believing, ICLR 2026)\n",
          f"- layers {pr['layers']} ({pr['source']}" + (f"; evidence AUROC {f(pr['mean_auroc_selected'])} vs "
                                                        f"{f(pr['mean_auroc_all'])} over all layers, {pr['n_samples']} calibration "
                                                        "questions" if "mean_auroc_all" in pr else "") + ")",
          f"- alpha {hp['alpha']}, sigma {hp['sigma']}, lambda {hp['lam']}",
          f"- **seeing but not believing** (last-token attention, test split): evidence AUROC when the answer is correct "
          f"{f(sn['evidence_auroc_when_correct'])} / wrong {f(sn['evidence_auroc_when_wrong'])}; peak on the object "
          f"{f(sn['found_rate_when_correct'])} / {f(sn['found_rate_when_wrong'])}; evidence / non-evidence attention per "
          f"token in the deeper half {f(sn['evidence_over_nonevidence_deep_when_correct'], 2)}x / "
          f"{f(sn['evidence_over_nonevidence_deep_when_wrong'], 2)}x\n"]
    rows = _csv(os.path.join(d, "summary.csv"))
    if rows:
        md.append("accuracy (fixed / broken: questions turned wrong -> right / right -> wrong vs base):\n\n" + table(rows, list(rows[0])))
    a = _csv(os.path.join(d, "attribution.csv"))
    if a:
        md.append("evidence attribution of the attention scores:\n\n" + table(a, ["scores", "auroc", "pointing"]))
    bf = _csv(os.path.join(d, "by_found.csv"))
    if bf:
        md.append("accuracy split by whether the VEA scores peak on the object:\n\n" + table(bf, list(bf[0])))
    md.append(img(root, os.path.join(d, "layers_rapt.png"), "evidence vs non-evidence attention")
              + img(root, os.path.join(d, "layers_auroc.png"), "evidence AUROC per layer"))
    vis = sorted(os.listdir(os.path.join(d, "vis")))[:2] if os.path.isdir(os.path.join(d, "vis")) else []
    md += [img(root, os.path.join(d, "vis", v), v) for v in vis]
    return "\n".join(md) + "\n"


def design(d, root):
    rows = _csv(os.path.join(d, "summary.csv"))
    if not rows:
        return ""
    s = _json(os.path.join(d, "summary.json")) or {}
    md = [f"### Appendix - DeepStack and MRoPE ablations ({s.get('heads', '')})\n", table(rows, ["condition"] + [c for c in rows[0] if c != "condition"])]
    if s.get("deepstack_injection"):
        md.append("- DeepStack injection ||added|| / ||residual||: "
                  + ", ".join(f"level {i}: {f(x['norm_ratio'], 2)}" for i, x in enumerate(s["deepstack_injection"])))
    return "\n".join(md) + "\n"


def appendix(d_lens, d_probe, root):
    md = []
    s = _json(os.path.join(d_lens, "summary.json"))
    if s:
        bi, bo = np.nanmax(s["bind_in_layer_mean"]), np.nanmax(s["bind_out_layer_mean"])
        md.append(f"- logit lens: best option-word accuracy at in-box tokens {f(bi)} vs out-box {f(bo)}")
    r = _json(os.path.join(d_probe, "probe_results.json"))
    if r:
        for task in ("attribute", "relation"):
            if task in r:
                best = {k: np.nanmax(np.array(v, dtype=float)) for k, v in r[task]["llm"].items()}
                md.append(f"- linear probe ({task}, majority {f(r[task]['majority_baseline'])}): "
                          + ", ".join(f"{k} {f(v)}" for k, v in best.items()))
    return ("### Appendix - is the attribute present at the object's tokens?\n\n" + "\n".join(md) + "\n") if md else ""


def main(out):
    md = [f"# RQ1 results: `{out}`\n"]
    cfg = os.path.join(out, "logs", "run_config.txt")
    if os.path.exists(cfg):
        md.append("```\n" + open(cfg).read().strip() + "\n```\n")
    for ds in sorted(x for x in os.listdir(out) if os.path.isdir(os.path.join(out, x)) and x != "logs"):
        d = os.path.join(out, ds)
        md += [f"## {ds}\n", p1(os.path.join(d, "p1_attention"), out), p2(os.path.join(d, "p2_knockout"), out),
               method(os.path.join(d, "method"), out), vea(os.path.join(d, "vea"), out),
               appendix(os.path.join(d, "appendix_logit_lens"), os.path.join(d, "appendix_linear_probe"), out),
               design(os.path.join(d, "appendix_design"), out)]
    path = os.path.join(out, "REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(x for x in md if x))
    print(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
