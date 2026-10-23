"""Checks of the grounding metrics, query-set localisation and the zoom / aligned-MRoPE machinery.

    pytest -q tests/test_bench.py
"""

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq1grounding import grounding_eval as ge  # noqa: E402
from rq1grounding import zoom as Z  # noqa: E402
from rq1grounding.data import Sample, bbox_coverage_grid, load_jsonl  # noqa: E402
from rq1grounding.maps import collect, grad_input_map, head_map  # noqa: E402
from rq1grounding.model import forward, load_model, prepare  # noqa: E402
from rq1grounding.synthetic import generate  # noqa: E402
from tiny_model import build  # noqa: E402


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    pos = rng.random(200) < 0.1
    S = rng.random((5, 200)).round(2)  # with ties
    ref = [roc_auc_score(pos, s) for s in S]
    np.testing.assert_allclose(ge.auroc_rows(S, pos), ref, atol=1e-9)


def test_perfect_and_uniform_maps():
    gh, gw = 8, 10
    cov = bbox_coverage_grid([[64, 32, 64, 64]], 320, 256, gh, gw).reshape(-1)  # 2x2 cells
    perfect = (cov > 0).astype(float)
    perfect /= perfect.sum()
    f = ge.threshold_free(perfect, cov, gh, gw)
    assert f["pointing"] == 1 and f["auroc"] == 1 and f["energy"] == pytest.approx(1)
    box = (64 / 320, 32 / 256, 128 / 320, 96 / 256)
    r = ge.thresholded(perfect, cov, gh, gw, [box], "rel", 0.5)
    assert r["box_iou"] == pytest.approx(1) and r["center_hit"] == 1 and r["tok_f1"] == pytest.approx(1)
    uni = np.full(gh * gw, 1 / (gh * gw))
    assert ge.threshold_free(uni, cov, gh, gw)["lift"] == pytest.approx(1)
    assert ge.threshold_free(uni, cov, gh, gw)["auroc"] == pytest.approx(0.5)


def test_binarize_rules():
    M = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    assert ge.binarize(M, "mass", 0.75).tolist() == [True, True, False, False, False]
    assert ge.binarize(M, "rel", 0.5).tolist() == [True, True, False, False, False]
    assert ge.binarize(M, "uni", 1.0).tolist() == [True, True, False, False, False]  # >= 1/5
    assert ge.binarize(M, "uni", 50).sum() == 1  # always keeps the peak


def test_expand_box_inside_image():
    b = Z.expand_box((0.95, 0.95, 0.99, 0.99), 1000, 800, margin=0.5, min_frac=0.2)
    assert 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1
    assert (b[2] - b[0]) * 1000 == pytest.approx(200)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    build(str(root / "tiny"))
    generate(str(root / "synth"), n_images=2, size=(320, 256), size_range=(0.15, 0.3), clutter=10)
    samples = load_jsonl(str(root / "synth" / "samples.jsonl"))
    model, proc = load_model(str(root / "tiny"), dtype="float32", device_map="cpu")
    return model, proc, samples


def test_object_query_found_in_question_not_options(env):
    model, proc, samples = env
    s = samples[0]
    s2 = Sample(**{**s.to_json(), "options": [f"{o} {s.target_objects[0]}" for o in s.options]})
    prep = prepare(model, proc, s2, 320 * 256, 64 * 64)
    tok = proc.tokenizer
    ids = prep.inputs["input_ids"][0].tolist()
    q_text = tok.decode([ids[i] for i in prep.question_pos])
    obj = prep.object_spans[0]
    assert s.question.lower() in q_text.lower()
    assert min(obj) >= min(prep.question_pos) and max(obj) <= max(prep.question_pos)


def test_collect_and_maps(env):
    model, proc, samples = env
    rel = next(s for s in samples if s.category == "synth_rel")
    prep = prepare(model, proc, rel, 320 * 256, 64 * 64)
    b = collect(model, prep)
    assert {"answer", "question", "object", "object:0", "object:1"} <= set(b.attn)
    L, H = model.config.text_config.num_hidden_layers, model.config.text_config.num_attention_heads
    assert b.attn["question"].shape == (L, H, prep.n_img)
    M = head_map(b, "object:1", heads=[(1, 0), (2, 3)], sink_mask=b.sinks(5.0))
    assert M.shape == (prep.n_img,) and M.sum() == pytest.approx(1)
    g = grad_input_map(model, prep)
    assert g.shape == (prep.n_img,) and np.isfinite(g).all()
    assert len(prep.object_cov) == 2


def test_zoom_and_aligned_positions(env):
    model, proc, samples = env
    s = samples[0]
    orig = Image.open(s.image_path).convert("RGB")
    prep = prepare(model, proc, s, 320 * 256, 64 * 64, image=orig)
    box = (0.25, 0.5, 0.75, 1.0)
    zp = Z.zoom_inputs(model, proc, s, prep.image, orig, box, 160 * 128)
    assert len(zp.spans) == 2
    pos = Z.aligned_position_ids(model, zp, box, dt=1.0)
    g, c = zp.spans
    base = pos[:, 0, g.pos[0]]
    rel = pos[:, 0, c.pos] - base[:, None]
    assert torch.allclose(rel[0], torch.ones_like(rel[0]))  # next temporal slice
    # crop tokens span exactly the region in global-grid units (cell centres)
    assert rel[1].min() >= box[1] * g.grid_h - 0.5 and rel[1].max() <= box[3] * g.grid_h - 0.5
    assert rel[2].min() >= box[0] * g.grid_w - 0.5 and rel[2].max() <= box[2] * g.grid_w - 0.5
    # text after the images keeps its default positions
    default, _ = model.model.get_rope_index(zp.inputs["input_ids"], zp.inputs["image_grid_thw"], None,
                                            zp.inputs["attention_mask"])
    tail = slice(int(c.pos[-1]) + 1, None)
    assert torch.equal(pos[:, :, tail], default[:, :, tail].float())
    a = forward(model, zp, position_ids=pos).logits
    d = forward(model, zp).logits
    assert torch.isfinite(a).all() and not torch.allclose(a, d)
    # integer-valued float positions reproduce the default forward exactly
    torch.testing.assert_close(forward(model, zp, position_ids=default.float()).logits, d)
