#!/usr/bin/env bash
# One-click runner for every RQ1 experiment, followed by an aggregated report.
#
#   bash run_all.sh                         # everything, default settings
#   QUICK=1 bash run_all.sh                 # dry run: 20 samples per dataset, small synthetic set
#   STEPS="bench zoom report" bash run_all.sh
#   MODEL=Qwen/Qwen3-VL-4B-Instruct OUT=outputs/4b bash run_all.sh
#   SWEEP_TOKENS="256 4096" bash run_all.sh # also re-run bench + zoom at other token budgets
#
# Steps (in order; `probes` = attention logit_lens linear_probe interventions):
#   data          generate the synthetic set, download V* if missing
#   attention     01 attention heatmaps                 (V* and synthetic)
#   logit_lens    02 logit lens on image tokens         (V* and synthetic)
#   linear_probe  03 linear probes                      (V* and synthetic)
#   interventions 04 region / DeepStack / MRoPE ablations (V* and synthetic)
#   bench         05 grounding bench: calibrate on synthetic, test on V*; synthetic with a 30/70 split
#   zoom          06 self-grounded zoom (needs bench)
#   sweep         05 eval + 06 at each budget in SWEEP_TOKENS
#   report        collect everything into $OUT/REPORT.md
#
# A finished step leaves $dir/.done and is skipped next time (FORCE=1 re-runs it), so an
# interrupted run can simply be restarted. Steps can also be spread over GPUs, e.g.
#   CUDA_VISIBLE_DEVICES=0 STEPS="data probes" bash run_all.sh &
#   CUDA_VISIBLE_DEVICES=1 STEPS="bench zoom" bash run_all.sh &
# (run `data` first, or in the same job as the steps that need it).
# If huggingface.co is unreachable: export HF_ENDPOINT=https://hf-mirror.com

set -uo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
DTYPE=${DTYPE:-bfloat16}
DATASET=${DATASET:-vstar}            # main benchmark: vstar | synthetic | path/to/samples.jsonl
DATA_ROOT=${DATA_ROOT:-data/vstar}
SYNTH_ROOT=${SYNTH_ROOT:-data/synthetic}
SYNTH_N=${SYNTH_N:-300}              # synthetic images (x3 questions: attr, swapped twin, relation)
OUT=${OUT:-outputs/$(basename "$MODEL")}
MAX_PIXELS=${MAX_PIXELS:-1048576}    # 1024 visual tokens (32x32 px each)
MIN_PIXELS=${MIN_PIXELS:-262144}
CROP_PIXELS=${CROP_PIXELS:-1048576}
CALIB_LIMIT=${CALIB_LIMIT:-150}      # synthetic samples used to calibrate heads / thresholds for V*
LIMIT=${LIMIT:-}
GRAD=${GRAD:-1}                      # also evaluate gradient x input maps (one backward pass per sample)
SWEEP_TOKENS=${SWEEP_TOKENS:-}
STEPS=${STEPS:-"data probes bench zoom sweep report"}
QUICK=${QUICK:-0}
FORCE=${FORCE:-0}
KEEP_GOING=${KEEP_GOING:-1}          # 0: stop at the first failing step
VERBOSE=${VERBOSE:-0}                # 1: also stream each step's output to the terminal
PY=${PYTHON:-python}

if [[ $QUICK == 1 ]]; then
  LIMIT=${LIMIT:-20}; SYNTH_N=30; CALIB_LIMIT=30
  [[ $OUT == outputs/* ]] && OUT=${OUT}_quick
fi

STEPS=${STEPS//probes/attention logit_lens linear_probe interventions}
want() { [[ " $STEPS " == *" $1 "* ]]; }

MAIN=$(basename "$DATASET" .jsonl)
DATASETS=("$MAIN")
[[ $DATASET != synthetic ]] && DATASETS+=("synthetic")

ds_args() {  # common data/model arguments for dataset name $1
  local a=(--model "$MODEL" --dtype "$DTYPE" --min_pixels "$MIN_PIXELS")
  if [[ $1 == synthetic ]]; then a+=(--dataset synthetic --data_root "$SYNTH_ROOT")
  else a+=(--dataset "$DATASET" --data_root "$DATA_ROOT"); fi
  [[ -n $LIMIT ]] && a+=(--limit "$LIMIT")
  printf '%s\n' "${a[@]}"
}

mkdir -p "$OUT/logs"
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

echo "model=$MODEL dataset=$DATASET out=$OUT limit=${LIMIT:-all} budget=$MAX_PIXELS px steps: $STEPS"
cat > "$OUT/logs/run_config.txt" <<EOF
started $(date '+%F %T') on $(hostname)
MODEL=$MODEL DTYPE=$DTYPE DATASET=$DATASET DATA_ROOT=$DATA_ROOT SYNTH_ROOT=$SYNTH_ROOT SYNTH_N=$SYNTH_N
MAX_PIXELS=$MAX_PIXELS MIN_PIXELS=$MIN_PIXELS CROP_PIXELS=$CROP_PIXELS LIMIT=${LIMIT:-all} CALIB_LIMIT=$CALIB_LIMIT
GRAD=$GRAD SWEEP_TOKENS=${SWEEP_TOKENS:-none} STEPS=$STEPS
EOF

# ---------------------------------------------------------------- data
if want data; then
  if [[ ! -f $SYNTH_ROOT/samples.jsonl || $FORCE == 1 ]]; then
    run make_synthetic "$SYNTH_ROOT" $PY scripts/00_make_synthetic.py --out "$SYNTH_ROOT" --n_images "$SYNTH_N"
  else echo "[skip] make_synthetic ($SYNTH_ROOT exists)"; fi
  if [[ $DATASET == vstar && ! -d $DATA_ROOT/direct_attributes ]]; then
    run download_vstar "$DATA_ROOT" $PY -c "from rq1grounding.data import load_vstar; print(len(load_vstar('$DATA_ROOT')), 'V* samples')"
  fi
fi

# ---------------------------------------------------------------- probes 01-04
for ds in "${DATASETS[@]}"; do
  mapfile -t A < <(ds_args "$ds")
  D=$OUT/$ds
  want attention && run "${ds}_01_attention" "$D/01_attention" \
    $PY scripts/01_attention_grounding.py "${A[@]}" --max_pixels "$MAX_PIXELS" --out "$D/01_attention"
  want logit_lens && run "${ds}_02_logit_lens" "$D/02_logit_lens" \
    $PY scripts/02_logit_lens.py "${A[@]}" --max_pixels "$MAX_PIXELS" --out "$D/02_logit_lens"
  want linear_probe && run "${ds}_03_linear_probe" "$D/03_linear_probe" \
    $PY scripts/03_linear_probe.py "${A[@]}" --max_pixels "$MAX_PIXELS" --out "$D/03_linear_probe"
  want interventions && run "${ds}_04_interventions" "$D/04_interventions" \
    $PY scripts/04_interventions.py "${A[@]}" --max_pixels "$MAX_PIXELS" --attention --out "$D/04_interventions"
done

# ---------------------------------------------------------------- bench 05
G=(); [[ $GRAD == 1 ]] && G=(--grad)
bench_args() {  # how dataset $1 is calibrated
  if [[ $1 == synthetic ]]; then echo "--calib_frac 0.3"
  else echo "--calib_dataset synthetic --calib_root $SYNTH_ROOT --calib_limit $CALIB_LIMIT"; fi
}
if want bench; then
  for ds in "${DATASETS[@]}"; do
    mapfile -t A < <(ds_args "$ds")
    # shellcheck disable=SC2046
    run "${ds}_05_bench" "$OUT/$ds/05_bench" \
      $PY scripts/05_grounding_bench.py "${A[@]}" --max_pixels "$MAX_PIXELS" --mode both $(bench_args "$ds") "${G[@]}" \
      --out "$OUT/$ds/05_bench"
  done
fi

# ---------------------------------------------------------------- zoom 06
zoom() {  # zoom <dataset> <config> <out> <max_pixels>
  local ds=$1 cfg=$2 out=$3 px=$4
  if [[ ! -f $cfg ]]; then echo "[skip] ${ds} zoom: $cfg missing (run the bench step first)"; FAILED+=("${ds}_06_zoom"); return; fi
  mapfile -t A < <(ds_args "$ds")
  run "${ds}_06_zoom$5" "$out" $PY scripts/06_self_grounded_zoom.py "${A[@]}" --max_pixels "$px" \
    --crop_pixels "$CROP_PIXELS" --config "$cfg" "${G[@]}" --out "$out"
}
if want zoom; then
  for ds in "${DATASETS[@]}"; do
    zoom "$ds" "$OUT/$ds/05_bench/grounding_config.json" "$OUT/$ds/06_zoom" "$MAX_PIXELS" ""
  done
fi

# ---------------------------------------------------------------- resolution sweep
if want sweep && [[ -n $SWEEP_TOKENS ]]; then
  for T in $SWEEP_TOKENS; do
    px=$((T * 1024)); S=$OUT/$MAIN/sweep_${T}tok; cfg=$OUT/$MAIN/05_bench/grounding_config.json
    mapfile -t A < <(ds_args "$MAIN")
    if [[ -f $cfg ]]; then
      run "${MAIN}_sweep_${T}_bench" "$S/05_bench" $PY scripts/05_grounding_bench.py "${A[@]}" --max_pixels "$px" \
        --mode eval --config "$cfg" "${G[@]}" --out "$S/05_bench"
    fi
    zoom "$MAIN" "$cfg" "$S/06_zoom" "$px" "_sweep_${T}"
  done
fi

# ---------------------------------------------------------------- report
if want report; then
  $PY scripts/collect_results.py "$OUT" && echo "[ ok ] report -> $OUT/REPORT.md"
fi

if (( ${#FAILED[@]} )); then echo "finished with failures: ${FAILED[*]}"; exit 1; fi
echo "all done -> $OUT"
