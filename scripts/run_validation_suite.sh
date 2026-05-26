#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${PHENOBENCH_DIR:-/home/hhadhri/Bureau/data/PhenoBench}"
DEVKIT_ROOT="${DEVKIT_ROOT:-/home/hhadhri/Bureau/code/phenobench}"
PYTHON_BIN="${PYTHON_BIN:-$DEVKIT_ROOT/.venv/bin/python}"
EVAL_BIN="${EVAL_BIN:-$DEVKIT_ROOT/.venv/bin/phenobench-eval}"
HAPT_ROOT="${HAPT_ROOT:-/home/hhadhri/Bureau/code/HAPT}"
DEVICE="${DEVICE:-cuda}"
HAPT_INPUT_SIZE="${HAPT_INPUT_SIZE:-native}"
VIZ_LIMIT="${VIZ_LIMIT:-100000}"
SKIP_AUTHOR_HAPT="${SKIP_AUTHOR_HAPT:-0}"
RUN_ID="${RUN_ID:-$(date +%F_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/validation_suite_$RUN_ID}"
OUTPUT_ROOT="$RUN_ROOT/model_outputs"
LOG_DIR="$RUN_ROOT/logs"
METRICS_DIR="$RUN_ROOT/metrics"
TIMING_DIR="$RUN_ROOT/timing"
SUMMARY="$RUN_ROOT/metrics_summary.md"
TIMINGS_TSV="$RUN_ROOT/timings.tsv"
EXPECTED_IMAGES="$(find "$DATA/val/images" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"

DEFAULT_MODELS=(
  semantic_erfnet
  semantic_deeplab
  panoptic_maskrcnn
  panoptic_mask2former
  panoptic_panopticdeeplab
  leaf_instances_maskrcnn
  leaf_instances_mask2former
  hierarchical_weyler
  hierarchical_hapt
)

if [[ -n "${MODELS:-}" ]]; then
  read -r -a SELECTED_MODELS <<< "$MODELS"
else
  SELECTED_MODELS=("${DEFAULT_MODELS[@]}")
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$METRICS_DIR" "$TIMING_DIR"

task_for_model() {
  case "$1" in
    semantic_*) echo "semantics" ;;
    panoptic_*) echo "panoptic" ;;
    leaf_instances_*) echo "leaf_instances" ;;
    hierarchical_*) echo "hierarchical" ;;
    *) echo "unknown" ;;
  esac
}

gpu_only_model() {
  case "$1" in
    panoptic_mask2former|leaf_instances_mask2former|panoptic_panopticdeeplab) return 0 ;;
    *) return 1 ;;
  esac
}

should_skip_inference() {
  local model="$1"
  if [[ "$DEVICE" == "cpu" ]] && gpu_only_model "$model" && [[ "${ALLOW_GPU_DOCKER_ON_CPU:-0}" != "1" ]]; then
    echo "skipped_cpu_run_authors_makefile_requires_gpu_docker"
    return 0
  fi
  return 1
}

append_summary_header() {
  {
    echo "# PhenoBench Validation Suite"
    echo
    echo "- Run ID: \`$RUN_ID\`"
    echo "- Date: \`$(date -Iseconds)\`"
    echo "- Data: \`$DATA\`"
    echo "- Device: \`$DEVICE\`"
    echo "- Output root: \`$RUN_ROOT\`"
    echo "- Visualization limit: \`$VIZ_LIMIT\`"
    echo "- HAPT input size: \`$HAPT_INPUT_SIZE\`"
    echo "- Expected val images: \`$EXPECTED_IMAGES\`"
    echo "- Selected models: \`${SELECTED_MODELS[*]}\`"
    echo "- Timing note: inference time/FPS is measured inside the model wrapper before our normalization, visualization, and PhenoBench evaluation."
    echo "- Pipeline time includes model inference, prediction normalization, and visualization rendering."
    echo
  } > "$SUMMARY"
  echo -e "model\ttask\timages\tinference_seconds\tinference_seconds_per_image\tfps\tpipeline_seconds\tpipeline_seconds_per_image\tinference_status\tevaluation_status" > "$TIMINGS_TSV"
}

append_model_summary() {
  local model="$1"
  local task="$2"
  local infer_status="$3"
  local eval_status="$4"
  local pred_dir="$5"
  local infer_log="$6"
  local eval_log="$7"
  local inference_seconds="${8:-n/a}"
  local inference_seconds_per_image="${9:-n/a}"
  local fps="${10:-n/a}"
  local pipeline_seconds="${11:-n/a}"
  local pipeline_seconds_per_image="${12:-n/a}"
  local images="${13:-n/a}"
  local val_dir="${pred_dir%/predictions}"
  local viz_dir="$val_dir/visualizations"
  local viz_count=0

  if [[ -d "$viz_dir" ]]; then
    viz_count="$(find "$viz_dir" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"
  fi

  {
    echo "## $model"
    echo
    echo "- Task: \`$task\`"
    echo "- Inference status: \`$infer_status\`"
    echo "- Evaluation status: \`$eval_status\`"
    echo "- Predictions: \`$pred_dir\`"
    echo "- Visualizations: \`$viz_dir\` ($viz_count PNGs)"
    echo "- Inference log: \`$infer_log\`"
    echo "- Metrics log: \`$eval_log\`"
    echo "- Images: \`$images\`"
    echo "- Inference time: \`$inference_seconds\` seconds"
    echo "- Mean inference time: \`$inference_seconds_per_image\` seconds/image"
    echo "- Inference FPS: \`$fps\`"
    echo "- Pipeline wall time: \`$pipeline_seconds\` seconds"
    echo "- Mean pipeline time: \`$pipeline_seconds_per_image\` seconds/image"
    echo
    echo '```text'
    if [[ -f "$eval_log" ]]; then
      cat "$eval_log"
    else
      echo "No metrics log."
    fi
    echo '```'
    echo
  } >> "$SUMMARY"
  echo -e "$model\t$task\t$images\t$inference_seconds\t$inference_seconds_per_image\t$fps\t$pipeline_seconds\t$pipeline_seconds_per_image\t$infer_status\t$eval_status" >> "$TIMINGS_TSV"
}

notify_visualizations() {
  local model="$1"
  local output_dir="$2"
  local viz_dir="$output_dir/visualizations"
  local viz_count=0

  if [[ -d "$viz_dir" ]]; then
    viz_count="$(find "$viz_dir" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"
  fi

  if [[ "$viz_count" -gt 0 ]]; then
    echo
    echo ">>> Visualizations ready for $model"
    echo ">>> Open: $viz_dir"
    echo ">>> PNG files: $viz_count"
    echo
  else
    echo ">>> No visualization PNGs found yet for $model at $viz_dir"
  fi
}

count_predictions() {
  local model="$1"
  local pred_dir="$2"
  local task="$3"
  case "$task" in
    semantics) find "$pred_dir/semantics" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l ;;
    panoptic) find "$pred_dir/plant_instances" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l ;;
    leaf_instances) find "$pred_dir/leaf_instances" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l ;;
    hierarchical)
      local a b c
      a=$(find "$pred_dir/semantics" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
      b=$(find "$pred_dir/plant_instances" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
      c=$(find "$pred_dir/leaf_instances" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
      printf "%s\n" "$a $b $c" | awk '{m=$1; if ($2<m) m=$2; if ($3<m) m=$3; print m}'
      ;;
    *) echo 0 ;;
  esac
}

seconds_per_image() {
  local seconds="$1"
  local images="$2"
  awk -v s="$seconds" -v n="$images" 'BEGIN { if (s ~ /^[0-9.]+$/ && n > 0) printf "%.6f", s / n; else print "n/a" }'
}

read_timing_json() {
  local timing_json="$1"
  if [[ ! -f "$timing_json" ]]; then
    printf "n/a\tn/a\tn/a\n"
    return
  fi
  "$PYTHON_BIN" -c 'import json, sys
data = json.load(open(sys.argv[1]))
def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)
print("\t".join(fmt(data.get(key)) for key in ("inference_seconds", "seconds_per_image", "fps")))
' "$timing_json"
}

run_inference() {
  local model="$1"
  local infer_log="$LOG_DIR/${model}_infer.log"
  local timing_json="$TIMING_DIR/${model}_timing.json"
  local cmd=(
    "$PYTHON_BIN" "$ROOT/scripts/infer_and_visualize.py"
    --model "$model"
    --phenobench-dir "$DATA"
    --split val
    --output-root "$OUTPUT_ROOT"
    --python "$PYTHON_BIN"
    --devkit-root "$DEVKIT_ROOT"
    --device "$DEVICE"
    --viz-limit "$VIZ_LIMIT"
    --no-daily-output
    --timing-json "$timing_json"
  )

  if [[ "$model" == "hierarchical_hapt" ]]; then
    cmd+=(--hapt-root "$HAPT_ROOT" --hapt-input-size "$HAPT_INPUT_SIZE")
  fi

  if [[ "${SKIP_INFER:-0}" == "1" ]]; then
    cmd+=(--skip-infer)
  fi

  echo "==> Running $model"
  rm -f "$timing_json"
  printf '+ %q ' "${cmd[@]}" | tee "$infer_log"
  echo | tee -a "$infer_log"
  local start end status
  start="$(date +%s)"
  "${cmd[@]}" 2>&1 | tee -a "$infer_log"
  status="${PIPESTATUS[0]}"
  end="$(date +%s)"
  LAST_PIPELINE_SECONDS="$((end - start))"
  IFS=$'\t' read -r LAST_INFERENCE_SECONDS LAST_INFERENCE_SECONDS_PER_IMAGE LAST_INFERENCE_FPS < <(read_timing_json "$timing_json")
  echo "Inference time: ${LAST_INFERENCE_SECONDS}s" | tee -a "$infer_log"
  echo "Inference FPS: ${LAST_INFERENCE_FPS}" | tee -a "$infer_log"
  echo "Pipeline wall time: ${LAST_PIPELINE_SECONDS}s" | tee -a "$infer_log"
  return "$status"
}

run_eval() {
  local model="$1"
  local task="$2"
  local pred_dir="$3"
  local eval_log="$METRICS_DIR/${model}_${task}.txt"

  if [[ "$task" == "unknown" ]]; then
    echo "Unknown task for $model" | tee "$eval_log"
    return 2
  fi

  echo "==> Evaluating $model ($task)"
  "$EVAL_BIN" \
    --task "$task" \
    --phenobench_dir "$DATA" \
    --prediction_dir "$pred_dir" \
    --split val 2>&1 | tee "$eval_log"
  return "${PIPESTATUS[0]}"
}

write_skipped_eval() {
  local model="$1"
  local task="$2"
  local pred_dir="$3"
  local reason="$4"
  local eval_log="$METRICS_DIR/${model}_${task}.txt"

  {
    echo "Skipped evaluation for $model."
    echo "Reason: $reason"
    echo "Expected val images: $EXPECTED_IMAGES"
    echo "Prediction directory: $pred_dir"
  } | tee "$eval_log"
}

eval_author_hapt() {
  local zip_path="$ROOT/outputs/authors_predictions/hapt-val.zip"
  local author_dir="$RUN_ROOT/authors/hapt_val"
  local task="hierarchical"
  local eval_log="$METRICS_DIR/authors_hapt_${task}.txt"
  local images

  if [[ "$SKIP_AUTHOR_HAPT" == "1" ]]; then
    echo "==> Skipping authors_hapt reference evaluation (SKIP_AUTHOR_HAPT=1)"
    return 0
  fi

  if [[ ! -f "$zip_path" ]]; then
    return 0
  fi

  mkdir -p "$author_dir"
  unzip -q -o "$zip_path" -d "$author_dir"
  echo "==> Evaluating authors_hapt ($task)"
  "$EVAL_BIN" \
    --task "$task" \
    --phenobench_dir "$DATA" \
    --prediction_dir "$author_dir" \
    --split val 2>&1 | tee "$eval_log"
  local status="${PIPESTATUS[0]}"
  images="$(count_predictions "authors_hapt" "$author_dir" "$task")"
  append_model_summary "authors_hapt" "$task" "released_predictions" "$status" "$author_dir" "n/a" "$eval_log" "n/a" "n/a" "n/a" "n/a" "n/a" "$images"
}

append_summary_header
echo "==> Validation suite run root: $RUN_ROOT"
echo "==> Selected models: ${SELECTED_MODELS[*]}"
echo "==> Device: $DEVICE"
echo "==> Expected val images: $EXPECTED_IMAGES"
eval_author_hapt

for model in "${SELECTED_MODELS[@]}"; do
  task="$(task_for_model "$model")"
  output_dir="$OUTPUT_ROOT/$model/val"
  pred_dir="$OUTPUT_ROOT/$model/val/predictions"
  infer_log="$LOG_DIR/${model}_infer.log"
  eval_log="$METRICS_DIR/${model}_${task}.txt"

  echo
  echo "================================================================"
  echo "==> Model: $model"
  echo "==> Task: $task"
  echo "==> Output: $output_dir"
  echo "================================================================"

  LAST_INFERENCE_SECONDS="n/a"
  LAST_INFERENCE_SECONDS_PER_IMAGE="n/a"
  LAST_INFERENCE_FPS="n/a"
  LAST_PIPELINE_SECONDS="n/a"
  skip_reason=""
  if skip_reason="$(should_skip_inference "$model")"; then
    infer_status="$skip_reason"
    echo "==> Skipping $model: $skip_reason" | tee "$infer_log"
  else
    run_inference "$model"
    infer_status="$?"
    notify_visualizations "$model" "$output_dir"
  fi

  image_count="$(count_predictions "$model" "$pred_dir" "$task")"
  pipeline_seconds_per_image="$(seconds_per_image "$LAST_PIPELINE_SECONDS" "$image_count")"

  if [[ "$image_count" == "$EXPECTED_IMAGES" ]]; then
    run_eval "$model" "$task" "$pred_dir"
    eval_status="$?"
    echo ">>> Metrics ready for $model: $eval_log"
  else
    reason="prediction_count_${image_count}_does_not_match_expected_${EXPECTED_IMAGES}"
    write_skipped_eval "$model" "$task" "$pred_dir" "$reason"
    eval_status="skipped_missing_predictions"
  fi

  append_model_summary \
    "$model" \
    "$task" \
    "$infer_status" \
    "$eval_status" \
    "$pred_dir" \
    "$infer_log" \
    "$eval_log" \
    "$LAST_INFERENCE_SECONDS" \
    "$LAST_INFERENCE_SECONDS_PER_IMAGE" \
    "$LAST_INFERENCE_FPS" \
    "$LAST_PIPELINE_SECONDS" \
    "$pipeline_seconds_per_image" \
    "$image_count"
  echo ">>> Summary updated: $SUMMARY"
done

echo "Validation suite complete."
echo "Summary: $SUMMARY"
