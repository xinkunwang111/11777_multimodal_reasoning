"""End-to-end: the scripts on a mock dataset in V*'s exact on-disk format (per-image JSON with
target_object / bbox [x, y, w, h] in pixels / question / options, plus test_questions.jsonl).
Tiny random model on CPU, a couple of minutes.  pytest -q tests/test_pipeline.py
"""

import json
import os
import subprocess
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq1grounding.data import LETTERS, load_jsonl  # noqa: E402
from rq1grounding.synthetic import generate  # noqa: E402
from tiny_model import build  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def make_mock_vstar(src_samples, out):
    """Write synthetic questions in V* format; the first half also get an official test_questions.jsonl entry."""
    for cat in ("direct_attributes", "relative_position"):
        os.makedirs(os.path.join(out, cat), exist_ok=True)
    official = []
    for i, s in enumerate(src_samples):
        cat = "direct_attributes" if s.category == "synth_attr" else "relative_position"
        W, H = s.image_size
        stem = f"sa_{i}"
        Image.open(s.image_path).save(os.path.join(out, cat, stem + ".jpg"))
        opts = s.options if cat == "direct_attributes" else [
            f"The {s.objects[0].name} is {kw} {'side of ' if kw in ('left', 'right') else ''}the {s.objects[1].name}."
            .replace(f"is {kw} side", f"is on the {kw} side") for kw in s.options]
        correct = opts[s.answer_idx]
        ann = {"target_object": [o.name for o in s.objects],
               "bbox": [[b[0] * W, b[1] * H, (b[2] - b[0]) * W, (b[3] - b[1]) * H] for o in s.objects for b in o.boxes],
               "question": s.question, "options": [correct] + [o for o in opts if o != correct]}
        json.dump(ann, open(os.path.join(out, cat, stem + ".json"), "w"))
        if i % 2 == 0:
            text = s.question + "\n" + "\n".join(f"({LETTERS[k]}) {o}" for k, o in enumerate(opts))
            official.append({"image": f"{cat}/{stem}.jpg", "text": text + "\nAnswer with the option's letter from the given choices directly.",
                             "category": cat, "question_id": i, "label": LETTERS[s.answer_idx]})
    with open(os.path.join(out, "test_questions.jsonl"), "w") as f:
        for o in official:
            f.write(json.dumps(o) + "\n")


@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    build(str(root / "tiny"))
    generate(str(root / "synth"), n_images=4, size=(480, 352), n_objects=(4, 6), obj_size=(0.08, 0.14), clutter=10)
    src = [s for s in load_jsonl(str(root / "synth" / "samples.jsonl")) if not s.qid.endswith("_v1")]
    make_mock_vstar(src, str(root / "vstar"))
    return root


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *map(str, args)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r


def test_vstar_pipeline(mock):
    common = ["--model", mock / "tiny", "--dtype", "float32", "--dataset", "vstar", "--data_root", mock / "vstar",
              "--max_pixels", 120000]
    out = mock / "out"
    run("probe1_attention.py", *common, "--calib_frac", 0.4, "--top_heads", 2, "--n_vis", 2, "--out", out / "p1")
    run("probe2_knockout.py", *common, "--window", 2, "--stride", 2, "--out", out / "p2")
    run("method_steer.py", *common, "--p1", out / "p1", "--p2", out / "p2", "--calib_frac", 0.4, "--lambdas", 0.5,
        "--crop_pixels", 32768, "--out", out / "m")
    run("appendix/qwen3vl_design.py", *common, "--p1", out / "p1", "--conditions", "baseline", "pos_hflip", "--out", out / "c")
    rows = [json.loads(line) for line in open(out / "p1" / "per_sample.jsonl")]
    assert rows and all(r["category"] in ("direct_attributes", "relative_position") for r in rows)
    rel = [r for r in rows if r["category"] == "relative_position"]
    assert all(len(r["objects"]) == 2 for r in rel)  # both GT objects of a relation question are scored
    cfg = json.load(open(out / "p1" / "config.json"))
    assert len(cfg["heads"]) == 2 and 0 < cfg["box_threshold"] < 1
    assert not set(cfg["calib_qids"]) & {r["qid"] for r in rows}  # calibration and test never overlap
    m = list(open(out / "m" / "summary.csv"))
    assert m[0].startswith("mode,attn,gt") and len(m) == 5  # header + base, base_hi, steer, crop
    p2 = json.load(open(out / "p2" / "summary.json"))
    assert set(p2["groups"]["all"]["used"]) == {"window", "delta_target", "delta_control", "share_target_hurts_more"}
    run("collect_results.py", out)
