"""Model-free checks: geometry, V* GT loading (official format), attention-map metrics, attention-edit math,
VEA's image steps, synthetic scenes.  pytest -q tests/test_core.py"""

import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding import vea  # noqa: E402
from rq1grounding.attention import edit_probs  # noqa: E402
from rq1grounding.data import load_jsonl, load_vstar, mirror_option_idx, option_keywords, split  # noqa: E402
from rq1grounding.grounding import auroc_rows, map_metrics, map_to_box  # noqa: E402
from rq1grounding.synthetic import generate  # noqa: E402


# ---------------------------------------------------------------------------- geometry

def test_box_conversions():
    b = G.from_xywh_px([64, 48, 32, 24], 640, 480)
    assert b == pytest.approx((0.1, 0.1, 0.15, 0.15))
    assert G.from_qwen(G.to_qwen(b)) == pytest.approx(b, abs=1e-3)
    assert G.from_xywh_px([600, 10, 100, 10], 640, 480) == pytest.approx((600 / 640, 10 / 480, 1.0, 20 / 480))  # clipped
    assert G.from_xywh_px([10, 10, 0, 5], 640, 480) is None  # degenerate
    assert G.hflip((0.1, 0.2, 0.3, 0.4)) == pytest.approx((0.7, 0.2, 0.9, 0.4))


def test_match_and_iou():
    gt = [(0.1, 0.1, 0.2, 0.2), (0.6, 0.6, 0.8, 0.8)]
    m = G.match((0.62, 0.62, 0.8, 0.8), gt)  # best instance is the second one
    assert m["center_hit"] and m["iou"] == pytest.approx(0.81)
    assert G.match(None, gt) == {"iou": 0.0, "center_hit": False}


def test_coverage_tiny_object_owns_its_cell():
    cov = G.coverage([(0.51, 0.51, 0.52, 0.52)], 4, 4)  # far smaller than one 32 px cell
    assert (cov > 0).sum() == 1 and cov[2, 2] > 0
    assert G.region_tokens([(0.51, 0.51, 0.52, 0.52)], 4, 4).tolist() == [10]
    assert len(G.region_tokens([(0.51, 0.51, 0.52, 0.52)], 4, 4, pad=1)) == 9


def test_expand_keeps_box_inside_with_context():
    b = G.expand((0.97, 0.97, 0.99, 0.99), 2000, 1000, margin=0.5, min_side=0.15)
    assert 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1
    assert (b[2] - b[0]) * 2000 == pytest.approx(300) and (b[3] - b[1]) * 1000 == pytest.approx(300)


# ---------------------------------------------------------------------------- V* GT

@pytest.fixture()
def vstar_root(tmp_path):
    for cat in ("direct_attributes", "relative_position"):
        os.makedirs(tmp_path / cat)
    Image.new("RGB", (640, 480)).save(tmp_path / "direct_attributes" / "a.jpg")
    json.dump({"target_object": ["glove"], "bbox": [[64, 48, 32, 24]], "question": "What is the color of the glove?",
               "options": ["red", "blue", "green", "yellow"]}, open(tmp_path / "direct_attributes" / "a.json", "w"))
    Image.new("RGB", (640, 480)).save(tmp_path / "direct_attributes" / "c.jpg")  # two instances of one object
    json.dump({"target_object": ["cup", "cup"], "bbox": [[0, 0, 64, 48], [320, 240, 64, 48]],
               "question": "What is the color of the cup?", "options": ["white", "black"]},
              open(tmp_path / "direct_attributes" / "c.json", "w"))
    Image.new("RGB", (800, 400)).save(tmp_path / "relative_position" / "b.jpg")
    json.dump({"target_object": ["cup", "dog"], "bbox": [[80, 40, 80, 40], [600, 200, 100, 100]],
               "question": "Is the cup on the left or right side of the dog?",
               "options": ["The cup is on the left side of the dog.", "The cup is on the right side of the dog."]},
              open(tmp_path / "relative_position" / "b.json", "w"))
    with open(tmp_path / "test_questions.jsonl", "w") as f:  # official prompt for a.jpg only
        f.write(json.dumps({"image": "direct_attributes/a.jpg", "category": "direct_attributes", "question_id": 0,
                            "text": "What is the color of the glove?\n(A) blue\n(B) green\n(C) red\n(D) yellow\n"
                                    "Answer with the option's letter from the given choices directly.", "label": "C"}) + "\n")
    return str(tmp_path)


def test_vstar_gt_loading(vstar_root):
    S = {s.qid: s for s in load_vstar(vstar_root)}
    a = S["direct_attributes/a"]
    assert a.options == ["blue", "green", "red", "yellow"] and a.answer_idx == 2 and a.meta["option_order"] == "official"
    assert [o.name for o in a.objects] == ["glove"] and a.objects[0].boxes[0] == pytest.approx((0.1, 0.1, 0.15, 0.15))
    assert a.image_size == (640, 480)
    c = S["direct_attributes/c"]
    assert len(c.objects) == 1 and len(c.objects[0].boxes) == 2  # instances merged
    assert c.options[c.answer_idx] == "white" and c.meta["option_order"] == "shuffled"
    b = S["relative_position/b"]
    assert [o.name for o in b.objects] == ["cup", "dog"]  # [subject, reference]
    assert b.objects[1].boxes[0] == pytest.approx((0.75, 0.5, 0.875, 0.75))
    assert b.options[b.answer_idx] == "The cup is on the left side of the dog."
    assert mirror_option_idx(b.options, b.answer_idx, "h") == 1 - b.answer_idx
    assert option_keywords(b.options) == ["left", "right"]


# ---------------------------------------------------------------------------- attention maps

def test_map_metrics_and_box():
    gh, gw = 6, 8
    cov = G.coverage([(0.25, 0.5, 0.5, 0.83)], gh, gw).reshape(-1)
    M = (cov > 0).astype(float) / (cov > 0).sum()
    m = map_metrics(M, cov)
    assert m["pointing"] == 1 and m["auroc"] == 1 and m["lift"] > 1
    assert G.iou(map_to_box(M, gh, gw, 0.5), (0.25, 0.5, 0.5, 0.8333)) > 0.95
    rng = np.random.default_rng(0)
    pos = rng.random(100) < 0.2
    S = rng.random((3, 100)).round(1)
    from sklearn.metrics import roc_auc_score

    np.testing.assert_allclose(auroc_rows(S, pos), [roc_auc_score(pos, s) for s in S])


# ---------------------------------------------------------------------------- attention edits

def test_edit_probs():
    p = torch.tensor([[0.2, 0.1, 0.3, 0.4]])  # [text, target, pool, pool]
    T = torch.tensor([False, True, False, False])
    P = torch.tensor([False, False, True, True])
    k = edit_probs(p, T, "scale", 0.0)
    assert k[0, 1] == 0 and k.sum() == pytest.approx(1) and k[0, 0] == pytest.approx(0.2 / 0.9)
    a = edit_probs(p, T, "scale", 3.0)
    assert a[0, 1] / a[0, 0] == pytest.approx(3 * 0.1 / 0.2)
    s = edit_probs(p, T, "shift", 0.5, P)  # half of the pool's 0.7 moves to the target
    assert s[0, 0] == pytest.approx(0.2) and s[0, 1] == pytest.approx(0.45) and s.sum() == pytest.approx(1)
    assert (s[0, 2] / s[0, 3]) == pytest.approx(0.3 / 0.4)  # pool keeps its internal distribution
    r = edit_probs(p, T, "shift", -1.0, P)  # mass-preserving knockout
    assert r[0, 1] == 0 and r[0, 0] == pytest.approx(0.2) and r[0, 2:].sum() == pytest.approx(0.8)
    z = edit_probs(torch.tensor([[0.5, 0.0, 0.0, 0.5]]), torch.tensor([False, True, True, False]), "shift", 0.5,
                   torch.tensor([False, False, False, True]))
    assert z[0, 1] == pytest.approx(0.125) and z[0, 2] == pytest.approx(0.125)  # no prior mass -> uniform


# ---------------------------------------------------------------------------- VEA

def test_vea_denoise_and_smoothing_match_the_original():
    rng = np.random.default_rng(0)
    gh, gw, cell = 6, 8, 4
    g = (0.5 + 0.5 * rng.random((gh, gw))).astype(np.float32)  # no ratio above 2: no spikes
    g[2, 3] = 50.0  # an isolated spike
    d = vea.denoise(g, 10.0)
    nb = [g[2 + dy, 3 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    assert d[2, 3] == pytest.approx(np.mean(nb)) and (d != g).sum() == 1
    c = g.copy()
    c[1:3, 1:3] = 50.0  # a cluster is evidence, not a spike
    np.testing.assert_array_equal(vea.denoise(c, 10.0), c)
    b = g.copy()
    b[0, 4] = 50.0  # border tokens see copies of themselves (edge padding) and are never spikes
    assert vea.denoise(b, 10.0)[0, 4] == 50.0
    from scipy import ndimage

    # the original (vea/highlight.py): paint each token onto its pixels, gaussian_filter with sigma = 0.5 x the
    # shorter side, min-max normalise
    canvas = np.kron(d, np.ones((cell, cell), np.float32))
    ref = ndimage.gaussian_filter(canvas, 0.5 * min(canvas.shape))
    np.testing.assert_allclose(vea.evidence_map(g.ravel(), gh, gw, cell), (ref - ref.min()) / (ref.max() - ref.min()), atol=1e-5)
    ref = ndimage.gaussian_filter(np.kron(g, np.ones((cell, cell))), 3.0)  # sigma > 1: in pixels
    np.testing.assert_allclose(vea.evidence_map(g.ravel(), gh, gw, cell, sigma=3.0, apply_denoise=False),
                               (ref - ref.min()) / (ref.max() - ref.min()), atol=1e-5)
    raw = vea.evidence_map(g.ravel(), gh, gw, cell, apply_denoise=False, apply_smooth=False)
    np.testing.assert_allclose(raw, np.kron((g - g.min()) / (g.max() - g.min()), np.ones((cell, cell))), atol=1e-6)
    assert not vea.evidence_map(np.ones(gh * gw), gh, gw, cell).any()  # constant map -> all zeros


def test_vea_highlight_layers_and_stats():
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (16, 24, 3), dtype=np.uint8))
    emap = np.zeros((16, 24), np.float32)
    emap[:8, :8] = 1.0
    arr, out = np.asarray(img).astype(int), np.asarray(vea.highlight(img, emap, 0.5)).astype(int)
    assert np.array_equal(out[:8, :8], arr[:8, :8]) and np.array_equal(out[8:], arr[8:] // 2)
    assert np.array_equal(np.asarray(vea.highlight(img, emap, 1.0)), np.asarray(img))
    assert len(vea.select_layers(np.zeros(36))) == 4 and len(vea.select_layers(np.zeros(28))) == 3  # 10%, rounded up
    assert vea.select_layers([0.5, 0.9, np.nan, 0.7, 0.8], 0.4) == [1, 4]
    assert vea.layer_span(36, 0.5, 1.0) == list(range(18, 36)) and vea.layer_span(36, 0.0, 0.5) == list(range(18))
    attn = np.array([[0.1, 0.1, 0.1, 0.1], [0.4, 0.0, 0.0, 0.0]])  # [layers, cells]
    st = vea.layer_stats(attn, np.array([1, 0, 0, 0]), n_tokens=10)
    np.testing.assert_allclose(st["rapt_evd"], [1.0, 4.0])
    np.testing.assert_allclose(st["rapt_nonevd"], [1.0, 0.0])
    np.testing.assert_allclose(st["auroc"], [0.5, 1.0])


# ---------------------------------------------------------------------------- synthetic set

def test_synthetic_scenes(tmp_path):
    generate(str(tmp_path), n_images=6, size=(512, 384), seed=1)
    S = load_jsonl(str(tmp_path / "samples.jsonl"))
    attr = [s for s in S if s.category == "synth_attr"]
    rel = [s for s in S if s.category == "synth_rel"]
    assert attr and rel
    for s in attr:
        src = s.meta["option_source"]
        assert src[str(s.answer_idx)] == "target" and "neighbor" in src.values() and len(s.options) == 4
        nb, tb = s.meta["neighbor"]["box"], s.objects[0].boxes[0]
        gap = max(nb[0] - tb[2], tb[0] - nb[2], nb[1] - tb[3], tb[1] - nb[3])
        assert gap < 0.1  # the neighbour really is next to the target
    twins = {}
    for s in attr:
        twins.setdefault(s.meta["pair_id"], []).append(s)
    for a, b in (v for v in twins.values() if len(v) == 2):
        assert a.options == b.options and a.answer_idx != b.answer_idx
    for s in rel:
        (t,), (r,) = s.objects[0].boxes, s.objects[1].boxes
        kw = option_keywords(s.options)[s.answer_idx]
        dx, dy = G.center(t)[0] - G.center(r)[0], G.center(t)[1] - G.center(r)[1]
        assert {"left": dx < 0, "right": dx > 0, "above": dy < 0, "below": dy > 0}[kw]
    calib, test = split(S, 0.5)
    for v in twins.values():
        assert len({s.qid in {c.qid for c in calib} for s in v}) == 1  # twins never straddle the split
