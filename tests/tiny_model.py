"""Build a tiny random Qwen3-VL (+ a toy tokenizer/processor) for offline CPU smoke tests.

    python tests/tiny_model.py /tmp/tiny-qwen3vl
    python scripts/probe1_attention.py --model /tmp/tiny-qwen3vl --dtype float32 --dataset synthetic ...
"""

import sys

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLConfig,
                          Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLVideoProcessor)

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
           "<|image_pad|>", "<|video_pad|>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% for c in m['content'] %}{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% else %}{{ c['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build(path: str):
    vocab = {t: i for i, t in enumerate(SPECIAL)}
    for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):  # byte-level: any text tokenizes
        vocab.setdefault(ch, len(vocab))
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tok = Qwen2TokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=SPECIAL[1:])
    tok.chat_template = CHAT_TEMPLATE
    ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL}

    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                         rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True}),
        vision_config=dict(depth=4, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=256,
                           deepstack_visual_indexes=[1, 2]),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
    )
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    # random deepstack mergers are tiny; scale them up so ablating DeepStack visibly changes outputs
    with torch.no_grad():
        for m in model.model.visual.deepstack_merger_list:
            m.linear_fc2.weight.mul_(10)
    model.save_pretrained(path)
    ip = Qwen2VLImageProcessorFast(patch_size=16, merge_size=2, temporal_patch_size=2,
                                   size={"shortest_edge": 64 * 64, "longest_edge": 4096 * 32 * 32})
    vp = Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
    Qwen3VLProcessor(image_processor=ip, tokenizer=tok, video_processor=vp, chat_template=CHAT_TEMPLATE).save_pretrained(path)
    return path


if __name__ == "__main__":
    build(sys.argv[1])
