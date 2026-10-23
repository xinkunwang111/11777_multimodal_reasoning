"""Offline correctness checks of the instrumentation on a tiny random Qwen3-VL (CPU).

    pytest -q tests/test_hooks.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq1grounding.data import load_jsonl  # noqa: E402
from rq1grounding.hooks import (AttentionRecorder, DeepStackRecorder, HiddenStateRecorder,  # noqa: E402
                                VisualFeatureEditor, logit_lens, mrope_position_ids)
from rq1grounding.model import forward, load_model, prepare  # noqa: E402
from rq1grounding.synthetic import generate  # noqa: E402
from tiny_model import build  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("rq1")
    build(str(root / "tiny"))
    generate(str(root / "synth"), n_images=3, size=(320, 256), size_range=(0.15, 0.3), clutter=10)
    samples = load_jsonl(str(root / "synth" / "samples.jsonl"))
    model, proc = load_model(str(root / "tiny"), dtype="float32", device_map="cpu", attn_implementation="eager")
    prep = prepare(model, proc, samples[0], max_pixels=320 * 256, min_pixels=64 * 64)
    return model, proc, samples, prep


def test_layout(setup):
    _, _, samples, prep = setup
    assert prep.n_img == (256 // 32) * (320 // 32)
    assert prep.coverage.max() > 0 and prep.coverage.mean() < 0.5
    assert all(len(v) > 0 for v in prep.object_pos.values())
    assert len(prep.letter_ids) == len(samples[0].options)


def test_attention_recorder_matches_eager(setup):
    model, _, _, prep = setup
    q = [prep.answer_pos, int(prep.image_pos[-1]) + 3]
    with AttentionRecorder(model, q) as rec:
        out = forward(model, prep, output_attentions=True)
    ref = torch.stack([a[0, :, q] for a in out.attentions])  # [Ly, H, nQ, L]
    assert rec.attn.shape == ref.shape
    torch.testing.assert_close(rec.attn.float(), ref.float(), atol=2e-3, rtol=1e-2)


def test_hidden_recorder_logit_lens(setup):
    model, _, _, prep = setup
    with HiddenStateRecorder(model, [prep.answer_pos]) as rec:
        out = forward(model, prep)
    H = rec.stacked
    assert H.shape[0] == model.config.text_config.num_hidden_layers + 1
    torch.testing.assert_close(logit_lens(model, H[-1])[0], out.logits[0, -1].float(), atol=1e-4, rtol=1e-4)


def test_visual_editor(setup):
    model, _, _, prep = setup
    base = forward(model, prep).logits
    with VisualFeatureEditor(model, record=True) as v:
        same = forward(model, prep).logits
    torch.testing.assert_close(base, same)
    assert v.captured["main"].shape[0] == prep.n_img and len(v.captured["deepstack"]) == 2
    with VisualFeatureEditor(model, zero_deepstack="all"):
        nods = forward(model, prep).logits
    with VisualFeatureEditor(model, ablate_tokens=(prep.cov_flat > 0).nonzero()[0]):
        abl = forward(model, prep).logits
    assert not torch.allclose(base, nods) and not torch.allclose(base, abl)
    with DeepStackRecorder(model) as ds:
        forward(model, prep)
    assert len(ds.stats) == 2 and ds.stats[0]["norm_ratio"] > 0


def test_mrope_edits(setup):
    model, _, _, prep = setup
    base = forward(model, prep).logits
    torch.testing.assert_close(forward(model, prep, mrope_position_ids(model, prep, "default")).logits, base)
    pos = mrope_position_ids(model, prep, "default")
    idx = prep.image_pos
    for mode in ("collapse", "shuffle", "hflip", "vflip"):
        e = mrope_position_ids(model, prep, mode)
        assert torch.equal(e[:, :, : idx[0]], pos[:, :, : idx[0]])  # text before image untouched
        assert not torch.allclose(forward(model, prep, e).logits, base), mode
    h = mrope_position_ids(model, prep, "hflip")
    w = h[2, 0, idx].view(prep.grid_h, prep.grid_w)
    assert torch.equal(w, pos[2, 0, idx].view(prep.grid_h, prep.grid_w).flip(1))
