"""Instrumentation checks on a tiny random Qwen3-VL (CPU, offline).  pytest -q tests/test_model.py

The attention tools are verified against the model's own eager attention: recorded rows must equal
`output_attentions`, and a knockout must equal running the model with the same entries masked out.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq1grounding import cues, vea  # noqa: E402
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.attention import AttentionRecorder, AttentionScaler, AttnEdit  # noqa: E402
from rq1grounding.cli import reading_layers, write_jsonl  # noqa: E402
from rq1grounding.data import load_jsonl  # noqa: E402
from rq1grounding.grounding import record_attention  # noqa: E402
from rq1grounding.hooks import DeepStackRecorder, VisualFeatureEditor, mrope_position_ids  # noqa: E402
from rq1grounding.model import forward, load_model, prepare, rope_positions  # noqa: E402
from rq1grounding.synthetic import generate  # noqa: E402
from tiny_model import build  # noqa: E402

BUDGET = 320 * 256


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("rq1")
    build(str(root / "tiny"))
    generate(str(root / "synth"), n_images=3, size=(320, 256), n_objects=(4, 5), obj_size=(0.12, 0.2), clutter=10)
    samples = load_jsonl(str(root / "synth" / "samples.jsonl"))
    eager, proc = load_model(str(root / "tiny"), dtype="float32", device_map="cpu", attn_implementation="eager")
    sdpa, _ = load_model(str(root / "tiny"), dtype="float32", device_map="cpu", attn_implementation="sdpa")
    rel = next(s for s in samples if s.category == "synth_rel")
    return {"eager": eager, "sdpa": sdpa, "proc": proc, "samples": samples, "rel": rel, "root": root,
            "prep": prepare(eager, proc, rel, BUDGET)}


def test_layout(env):
    prep = env["prep"]
    assert prep.n_img == (256 // 32) * (320 // 32)
    assert len(prep.object_pos) == 2 and all(prep.object_pos)
    assert min(min(o) for o in prep.object_pos) >= min(prep.question_pos)
    assert max(max(o) for o in prep.object_pos) <= max(prep.question_pos)
    assert prep.text_pos[0] > int(prep.image_pos[-1]) and prep.text_pos[-1] == prep.answer_pos
    assert (prep.object_cov(0) > 0).any() and prep.other_cov(0) is not None
    assert set(prep.tokens_of(prep.sample.objects[0].boxes)) <= set(prep.image_pos.tolist())


def test_recorder_matches_eager(env):
    m, prep = env["eager"], env["prep"]
    groups = {"last": [prep.answer_pos], "object:0": prep.object_pos[0]}
    with AttentionRecorder(m, groups, prep.image_pos) as rec:
        out = forward(m, prep, output_attentions=True)
    ref = torch.stack([a[0] for a in out.attentions])  # [Ly, H, L, L]
    img = prep.image_pos
    torch.testing.assert_close(rec.maps["last"].float(), ref[:, :, prep.answer_pos][..., img], atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(rec.maps["object:0"].float(), ref[:, :, prep.object_pos[0]][..., img].mean(2), atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_vea_attribution_matches_eager(env, impl):
    m, prep = env[impl], env["prep"]
    attn, logits = vea.last_token_attention(m, prep)
    out = forward(env["eager"], prep, output_attentions=True)
    ref = torch.stack([a[0, :, prep.answer_pos] for a in out.attentions]).mean(1)[:, prep.image_pos]  # [Ly, n_img]
    np.testing.assert_allclose(attn, ref.numpy(), atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(logits, out.logits, atol=1e-4, rtol=1e-4)
    emap = vea.evidence_map(attn.mean(0), prep.grid_h, prep.grid_w, 32)
    assert emap.shape == (prep.image.size[1], prep.image.size[0]) and emap.max() == 1


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_scaler_identity_and_knockout(env, impl):
    m, prep = env[impl], env["prep"]
    base = forward(m, prep).logits
    keys = prep.tokens_of(prep.sample.boxes).tolist()
    pool = np.setdiff1d(prep.image_pos.numpy(), keys).tolist()
    L = m.config.text_config.num_hidden_layers
    for e in (AttnEdit(range(L), prep.text_pos, keys, "scale", 1.0), AttnEdit(range(L), prep.text_pos, keys, "shift", 0.0, pool)):
        with AttentionScaler(m, [e]):
            torch.testing.assert_close(forward(m, prep).logits, base, atol=1e-4, rtol=1e-4)
    # knockout in every layer == the model run with those (query, key) entries masked out
    with AttentionScaler(m, [AttnEdit(list(range(L)), prep.text_pos, keys, "scale", 0.0)]):
        ko = forward(m, prep).logits
    n = prep.inputs["input_ids"].shape[1]
    mask = torch.full((n, n), torch.finfo(torch.float32).min).triu(1)
    mask[torch.tensor(prep.text_pos)[:, None], torch.tensor(keys)[None, :]] = torch.finfo(torch.float32).min
    inputs = {**prep.inputs, "attention_mask": mask[None, None]}
    ref = env["eager"](**inputs, position_ids=rope_positions(env["eager"], prep.inputs), use_cache=False).logits[:, -1:]
    torch.testing.assert_close(ko, ref, atol=1e-4, rtol=1e-4)
    with AttentionScaler(m, [AttnEdit([1, 2], prep.text_pos, keys, "shift", 0.8, pool)]):
        assert not torch.allclose(forward(m, prep).logits, base)


def test_mrope_edits(env):
    m, prep = env["eager"], env["prep"]
    pos = rope_positions(m, prep.inputs)
    idx = prep.image_pos
    for mode in ("collapse", "shuffle", "hflip", "vflip"):
        e = mrope_position_ids(m, prep.inputs, idx, mode)
        assert torch.equal(e[:, :, : idx[0]], pos[:, :, : idx[0]]) and torch.equal(e[:, :, idx[-1] + 1:], pos[:, :, idx[-1] + 1:])
    h = mrope_position_ids(m, prep.inputs, idx, "hflip")[2, 0, idx].view(prep.grid_h, prep.grid_w)
    assert torch.equal(h, pos[2, 0, idx].view(prep.grid_h, prep.grid_w).flip(1))
    am = record_attention(m, prep, position_ids=mrope_position_ids(m, prep.inputs, idx, "hflip"))
    assert am.maps["object:0"].shape[-1] == prep.n_img and am.sinks.shape == (prep.n_img,)


def test_deepstack_tools(env):
    m, prep = env["eager"], env["prep"]
    base = forward(m, prep).logits
    with VisualFeatureEditor(m, zero_deepstack="all"):
        assert not torch.allclose(forward(m, prep).logits, base)
    with DeepStackRecorder(m) as ds:
        forward(m, prep)
    assert len(ds.stats) == len(m.config.vision_config.deepstack_visual_indexes)


def test_cues(env):
    m, proc, prep = env["eager"], env["proc"], env["prep"]
    s = prep.sample
    boxes = [list(o.boxes) for o in s.objects]
    t = cues.text_cue(s, boxes)
    assert str(G.to_qwen(s.objects[0].boxes[0])) in t and "0-1000" in t
    img, hint = cues.draw_cue(prep.image, s, boxes)
    assert img.size == prep.image.size and "box" in hint
    attr = next(x for x in env["samples"] if x.category == "synth_attr")
    _, hint = cues.draw_cue(prep.image, attr, [list(o.boxes) for o in attr.objects])
    used = hint.split(" marked with ")[1].split(" box")[0]
    assert used not in " ".join(attr.options)  # never draw in a colour that is an answer option
    from PIL import Image

    crop, hint, box = cues.crop_cue(m, Image.open(s.image_path).convert("RGB"), s, boxes, 64 * 32 * 32)
    assert crop.size[0] * crop.size[1] <= 64 * 32 * 32 and all(G.contains(box, G.center(b)) for bs in boxes for b in bs)
    p2 = prepare(m, proc, s, BUDGET, image=prep.image, prefix=hint, extra_images=[crop])
    assert len(p2.spans) == 2 and p2.question_pos
    e = cues.steer_edit(prep, boxes, [1, 2], "shift", 0.5)
    assert set(e.keys) == set(prep.tokens_of(s.boxes).tolist()) and not set(e.keys) & set(e.pool)


def test_reading_layers(tmp_path):
    windows = [[0, 1], [2, 3]]
    rows = [{"qid": q, "windows": windows,
             "knockout": {"all": [{"target": -0.1, "control": 0.0}, {"target": -1.0, "control": -0.1}]}} for q in "ab"]
    rows.append({"qid": "c", "windows": windows, "knockout": {"all": [{"target": -5.0, "control": 0.0}, {"target": 0, "control": 0}]}})
    write_jsonl(tmp_path / "per_sample.jsonl", rows)
    assert reading_layers(str(tmp_path), {"a", "b"})[0] == [2, 3]
    assert reading_layers(str(tmp_path), {"c"})[0] == [0, 1]  # only the given (calibration) qids count
    assert reading_layers(str(tmp_path / "missing"), None) == (None, None)
    json.dumps(reading_layers(str(tmp_path), None)[1])
