#!/usr/bin/env bash
# One-click runner: the two probes and the two methods on V* and the synthetic binding set, then REPORT.md.
#
#   bash run_all.sh                          # everything
#   QUICK=1 bash run_all.sh                  # dry run: 20 questions per dataset, small synthetic set
#   STEPS="p1 method report" bash run_all.sh # only some steps
#   MODEL=Qwen/Qwen3-VL-4B-Instruct bash run_all.sh
#
# Steps, per dataset (V* and synthetic), in dependency order:
#   data      generate the synthetic set; download V* if missing
#   p1        probe 1 - does the attention find the object, also when the answer is wrong?
#                         -> grounding heads + region threshold
#   p2        probe 2 - is that attention used? (attention knockout: target vs control) -> reading layers
#   method    amplify the attention on the attention-found region in the reading layers;
#             compared with crop (ViCrop-style zoom) and with GT regions            (uses p1, p2)
#   vea       VEA (Seeing but Not Believing, ICLR 2026), reproduced: dim the image outside where
#             the attention looks and answer again; the paper's ablations and analyses (own profiling)
#   appendix  logit lens, linear probes, DeepStack / MRoPE ablations (off by default; uses p1)
#   report    $OUT/REPORT.md
# Calibration (grounding heads, thresholds, steer layers / strength, VEA layers) always uses a held-out split:
# CALIB_FRAC of each dataset by default, or the synthetic set for V* with CALIB_FROM_SYNTH=1.
#
# A finished step leaves <dir>/.done and is skipped next time (FORCE=1 re-runs), so a crashed or
# interrupted run is resumed by running the same command again. Logs: $OUT/logs/<step>.log.
# Several GPUs: run p1 first, then e.g.
#   CUDA_VISIBLE_DEVICES=0 STEPS="p2 method" bash run_all.sh &  CUDA_VISIBLE_DEVICES=1 STEPS="vea appendix" bash run_all.sh
# If huggingface.co is unreachable: export HF_ENDPOINT=https://hf-mirror.com

set -uo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
DTYPE=${DTYPE:-bfloat16}
VSTAR_ROOT=${VSTAR_ROOT:-data/vstar}
SYNTH_ROOT=${SYNTH_ROOT:-data/synthetic}
SYNTH_N=${SYNTH_N:-300}               # synthetic images (each -> 2 colour-twin attribute questions + 1 relation)
DATASETS=${DATASETS:-"vstar synthetic"}
OUT=${OUT:-outputs/$(basename "$MODEL")}
MAX_PIXELS=${MAX_PIXELS:-1048576}     # 1024 visual tokens (32x32 px each)
CROP_PIXELS=${CROP_PIXELS:-262144}    # 256 extra tokens for the zoomed crop baseline
CALIB_FRAC=${CALIB_FRAC:-0.3}
CALIB_FROM_SYNTH=${CALIB_FROM_SYNTH:-0}
P2_GROUPS=${P2_GROUPS:-"all"}
LIMIT=${LIMIT:-}
STEPS=${STEPS:-"data p1 p2 method vea report"}
QUICK=${QUICK:-0}
FORCE=${FORCE:-0}
KEEP_GOING=${KEEP_GOING:-1}
VERBOSE=${VERBOSE:-0}
PY=${PYTHON:-python}

if [[ $QUICK == 1 ]]; then
  LIMIT=${LIMIT:-20}; SYNTH_N=20
  [[ $OUT == outputs/* ]] && OUT=${OUT}_quick
fi
want() { [[ " $STEPS " == *" $1 "* ]]; }

mkdir -p "$OUT/logs"
cat > "$OUT/logs/run_config.txt" <<EOF
started $(date '+%F %T') on $(hostname)
MODEL=$MODEL DTYPE=$DTYPE DATASETS=$DATASETS VSTAR_ROOT=$VSTAR_ROOT SYNTH_ROOT=$SYNTH_ROOT SYNTH_N=$SYNTH_N
MAX_PIXELS=$MAX_PIXELS CROP_PIXELS=$CROP_PIXELS LIMIT=${LIMIT:-all} CALIB_FRAC=$CALIB_FRAC CALIB_FROM_SYNTH=$CALIB_FROM_SYNTH
P2_GROUPS=$P2_GROUPS STEPS=$STEPS
EOF

FAILED=()
run() {  # run <name> <out dir> <command...>
  local name=$1 dir=$2; shift 2
  if [[ -f $dir/.done && $FORCE != 1 ]]; then echo "[skip] $name (done; FORCE=1 to re-run)"; return 0; fi
  mkdir -p "$dir"
  local log="$OUT/logs/$name.log" t0=$SECONDS rc
  echo "[run ] $name -> $dir"
  if [[ $VERBOSE == 1 ]]; then "$@" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}; else "$@" > "$log" 2>&1; rc=$?; fi
  if [[ $rc == 0 ]]; then
    touch "$dir/.done"; echo "[ ok ] $name ($((SECONDS - t0))s)"
  else
    echo "[FAIL] $name (exit $rc) - last lines of $log:"; tail -n 8 "$log" | sed 's/^/       /'
    FAILED+=("$name"); [[ $KEEP_GOING == 1 ]] || { echo "stopping (KEEP_GOING=0)"; exit 1; }
  fi
}
needs() {  # needs <file> <step name>: dependency check
  [[ -f $1 ]] && return 0
  echo "[skip] $2: missing $1 (run its prerequisite step first)"; FAILED+=("$2"); return 1
}

data_args() {  # model + data arguments for dataset $1
  local a=(--model "$MODEL" --dtype "$DTYPE" --max_pixels "$MAX_PIXELS")
  if [[ $1 == synthetic ]]; then a+=(--dataset synthetic --data_root "$SYNTH_ROOT"); else a+=(--dataset "$1" --data_root "$VSTAR_ROOT"); fi
  [[ -n $LIMIT ]] && a+=(--limit "$LIMIT")
  printf '%s\n' "${a[@]}"
}
split_args() {  # calibration split for dataset $1
  if [[ $1 != synthetic && $CALIB_FROM_SYNTH == 1 ]]; then
    printf '%s\n' --calib_dataset synthetic --calib_root "$SYNTH_ROOT" --calib_limit "${LIMIT:-150}"
  else
    printf '%s\n' --calib_frac "$CALIB_FRAC"
  fi
}

echo "model=$MODEL datasets=[$DATASETS] out=$OUT limit=${LIMIT:-all} steps: $STEPS"

if want data; then
  if [[ ! -f $SYNTH_ROOT/samples.jsonl || $FORCE == 1 ]]; then
    run make_synthetic "$SYNTH_ROOT" $PY scripts/make_synthetic.py --out "$SYNTH_ROOT" --n_images "$SYNTH_N"
  else echo "[skip] make_synthetic ($SYNTH_ROOT exists)"; fi
  if [[ " $DATASETS " == *" vstar "* && ! -d $VSTAR_ROOT/direct_attributes ]]; then
    run download_vstar "$VSTAR_ROOT" $PY -c "from rq1grounding.data import load_vstar; print(len(load_vstar('$VSTAR_ROOT')), 'V* questions')"
  fi
fi

for dsarg in $DATASETS; do  # vstar | synthetic | path/to/samples.jsonl
  ds=$(basename "$dsarg" .jsonl)
  D=$OUT/$ds
  mapfile -t A < <(data_args "$dsarg")
  mapfile -t SPL < <(split_args "$dsarg")
  want p1 && run "${ds}_p1_attention" "$D/p1_attention" $PY scripts/probe1_attention.py "${A[@]}" "${SPL[@]}" --out "$D/p1_attention"
  # shellcheck disable=SC2086
  want p2 && run "${ds}_p2_knockout" "$D/p2_knockout" $PY scripts/probe2_knockout.py "${A[@]}" --groups $P2_GROUPS --out "$D/p2_knockout"
  if want method && needs "$D/p1_attention/config.json" "${ds}_method"; then
    [[ -f $D/p2_knockout/per_sample.jsonl ]] || echo "[warn] ${ds}_method: no probe-2 results, steering uses the middle layers"
    run "${ds}_method" "$D/method" $PY scripts/method_steer.py "${A[@]}" "${SPL[@]}" --p1 "$D/p1_attention" \
      --p2 "$D/p2_knockout" --crop_pixels "$CROP_PIXELS" --out "$D/method"
  fi
  want vea && run "${ds}_vea" "$D/vea" $PY scripts/method_vea.py "${A[@]}" "${SPL[@]}" --out "$D/vea"
  if want appendix; then
    run "${ds}_appendix_logit_lens" "$D/appendix_logit_lens" $PY scripts/appendix/logit_lens.py "${A[@]}" --out "$D/appendix_logit_lens"
    run "${ds}_appendix_linear_probe" "$D/appendix_linear_probe" $PY scripts/appendix/linear_probe.py "${A[@]}" --out "$D/appendix_linear_probe"
    if needs "$D/p1_attention/config.json" "${ds}_appendix_design"; then
      run "${ds}_appendix_design" "$D/appendix_design" $PY scripts/appendix/qwen3vl_design.py "${A[@]}" --p1 "$D/p1_attention" \
        --out "$D/appendix_design"
    fi
  fi
done

if want report; then
  $PY scripts/collect_results.py "$OUT" && echo "[ ok ] report -> $OUT/REPORT.md"
fi
if (( ${#FAILED[@]} )); then echo "finished with failures: ${FAILED[*]}"; exit 1; fi
echo "all done -> $OUT"
