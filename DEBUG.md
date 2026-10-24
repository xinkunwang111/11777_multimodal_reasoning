# Debug Log: CUDA Tensor to NumPy Conversion in Attention & Hook Recorders

## 1. Issue Description

During Step 6 of the RQ1 experiment pipeline (`python scripts/check_setup.py --model checkpoints/Qwen3-VL-8B-Instruct --data_root data/vstar`), the model and its 4 checkpoint shards loaded successfully onto GPU (`cuda:0`). However, when running the first sample through `record_attention(model, prep)`, the process crashed with the following traceback:

```text
model checkpoints/Qwen3-VL-8B-Instruct: 36 layers x 32 heads on cuda:0
Traceback (most recent call last):
  File "scripts/check_setup.py", line 72, in <module>
    main()
  File "scripts/check_setup.py", line 50, in main
    am = record_attention(model, prep)
  File "rq1grounding/grounding.py", line 49, in record_attention
    HiddenStateRecorder(model, prep.image_pos, norm_only=True) as nrec:
  File "rq1grounding/hooks.py", line 27, in __init__
    self.pos = torch.as_tensor(np.asarray(positions), dtype=torch.long)
  File "torch/_tensor.py", line 1149, in __array__
    return self.numpy()
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

## 2. Root Cause Analysis

- In `rq1grounding/model.py`, `prep.image_pos` is `self.spans[0].pos`, which is a `torch.Tensor` allocated on `model.device` (`cuda:0`).
- In `rq1grounding/grounding.py:48-49`:
  - `AttentionRecorder` was called with `prep.image_pos.cpu()`.
  - `HiddenStateRecorder` was called with `prep.image_pos` (still on `cuda:0`).
- In `rq1grounding/hooks.py:27`, `HiddenStateRecorder.__init__` called `np.asarray(positions)` on the tensor.
- In PyTorch, NumPy's `__array__` protocol invokes `.numpy()`, which strictly disallows CUDA tensors and raises `TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`
- Similar patterns existed in:
  - `rq1grounding/hooks.py:84` (`VisualFeatureEditor` with `ablate_tokens`)
  - `rq1grounding/attention.py:99` (`AttentionRecorder` with `keys`)
  - `rq1grounding/attention.py:166, 170` (`AttentionScaler` with `e.keys` and `e.pool`)

## 3. Code Modifications

### 3.1 `rq1grounding/hooks.py`
- In `HiddenStateRecorder.__init__`:
  ```python
  if isinstance(positions, torch.Tensor):
      self.pos = positions.detach().cpu().long()
  else:
      self.pos = torch.as_tensor(np.asarray(positions), dtype=torch.long)
  ```
- In `VisualFeatureEditor.__init__`:
  ```python
  if ablate_tokens is None:
      self.ablate = None
  elif isinstance(ablate_tokens, torch.Tensor):
      self.ablate = ablate_tokens.detach().cpu().long()
  else:
      self.ablate = torch.as_tensor(np.asarray(ablate_tokens), dtype=torch.long)
  ```

### 3.2 `rq1grounding/grounding.py`
- In `record_attention`:
  Pass `prep.image_pos.cpu()` to `HiddenStateRecorder` consistently with line 48:
  ```python
  with AttentionRecorder(model, query_groups(prep), prep.image_pos.cpu()) as rec, \
          HiddenStateRecorder(model, prep.image_pos.cpu(), norm_only=True) as nrec:
  ```

### 3.3 `rq1grounding/attention.py`
- In `AttentionRecorder.__init__`:
  ```python
  if isinstance(keys, torch.Tensor):
      self.keys = keys.detach().cpu().long()
  else:
      self.keys = torch.as_tensor(np.asarray(keys), dtype=torch.long)
  ```
- In `AttentionScaler.on_layer`:
  Safely handle `e.keys` and `e.pool` whether they are `torch.Tensor` or array-like:
  ```python
  target_keys = e.keys.detach().long().to(q.device) if isinstance(e.keys, torch.Tensor) else torch.as_tensor(np.asarray(e.keys), dtype=torch.long, device=q.device)
  target[target_keys] = True
  ...
  pool_keys = e.pool.detach().long().to(q.device) if isinstance(e.pool, torch.Tensor) else torch.as_tensor(np.asarray(e.pool), dtype=torch.long, device=q.device)
  pool[pool_keys] = True
  ```

## 4. Verification

1. **Unit Tests**:
   Ran `pytest -q tests`:
   `21 passed in 202.80s` (100% pass, 0 failures, including VEA tests).
2. **Setup Check**:
   Re-evaluated through SLURM GPU job (`check_setup.py` passed).

---

# Feature & Robustness Update: Real-Time Incremental Writing & Resuming

## 1. Motivation

In long-running GPU evaluations (especially on slower architectures or large datasets, e.g., probe2 knockout requiring 28 forward passes per question), tasks may face SLURM walltime limits or cluster preemption.
Originally, evaluation scripts (`probe1_attention.py`, `probe2_knockout.py`, `method_steer.py`, `method_vea.py`) accumulated results entirely in memory and only called `write_jsonl` after the full sample loop finished. If a job timed out before the last question, all prior hours of computation were lost.

## 2. Implementation

1. **`rq1grounding/cli.py`**:
   - Added `append_jsonl(path, row)` helper with automated parent directory creation and immediate `flush()`.
2. **`scripts/probe2_knockout.py`**:
   - Reads existing lines from `per_sample.jsonl` upon startup.
   - Builds `done_map = {r["qid"]: r for r in done_rows if "knockout" in r and r.get("knockout")}`.
   - For already completed questions, immediately reuses the record.
   - For newly processed questions, appends each sample to `per_sample.jsonl` immediately upon completion.
   - Re-writes clean final `per_sample.jsonl` and computes `summary.json` at loop conclusion.
3. **`scripts/probe1_attention.py`, `scripts/method_steer.py`, `scripts/method_vea.py`**:
   - Applied identical incremental appending and resuming pattern via `append_jsonl` and `read_jsonl`.

---

# Architecture & Precision Adaptation: V100 Tensor Cores via Float16

## 1. Motivation

NVIDIA Volta V100 (sm_70) lacks native hardware Tensor Core support for Bfloat16 (which was introduced in Ampere sm_80 and Hopper sm_90). In PyTorch, running Qwen3-VL-8B in `bfloat16` on V100 forces execution into software-emulated FP32 paths, reducing compute throughput to ~15 TFLOPS (~34s per forward pass with attention hooks).

In contrast, V100 features 640 hardware Tensor Cores specifically designed for **`float16` (FP16)**, delivering **125 TFLOPS** (an 8x–10x acceleration over emulated BF16).

## 2. Implementation & Configuration

1. **Precision Switching**:
   - Configured `DTYPE=float16` in the SLURM submission script [`run_v100.sbatch`](file:///jet/home/xwang93/11777/11777_multimodal_reasoning/run_v100.sbatch) and [`run_all.sh`](file:///jet/home/xwang93/11777/11777_multimodal_reasoning/run_all.sh).
   - Validated that `AutoModelForImageTextToText.from_pretrained(..., dtype=torch.float16)` correctly instantiates weights in half-precision and drives V100 Tensor Cores.
2. **Sample Limit**:
   - Set `LIMIT=10` for the Quick Run test.
   - For `vstar_p2_knockout`, the 10 completed samples are already safely saved on disk in `per_sample.jsonl`. It immediately generates `summary.json` and transitions to downstream methods (`method_steer`, `method_vea`, and synthetic evaluation) with 10x faster FP16 inference.



