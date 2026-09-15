#!/bin/bash
# Batch runner: reconstruct many windows unattended, N workers in parallel, each on its own GPU set.
# Every job runs the UNCHANGED pipeline (recover_clip.py) and is then evaluated (eval_recon.py -> eval.json).
#
# Job file: one job per line   TAG|VIDEO|CALIB_JSON|LO|HI|EXTRA_ARGS       (# comments and blank lines ignored)
#   e.g.   19-V1_tr|/data/19-V1.MP4|calib/19-V1_intrinsics.json|1040|1320|
#          46-V2_hi|/data/46-V2.MP4|calib/46-V2_intrinsics.json|540|780|--clahe 5.0
# Usage:   pipeline/run_queue.sh <jobs.txt> <out_root> <slot_name> <gpus>
#   start one instance per GPU pair, e.g.  run_queue.sh jobs.txt runs/queue A 0,1 &  run_queue.sh jobs.txt runs/queue B 2,3 &
# Jobs are claimed atomically with flock (an earlier grep-based pop let two workers take the same last line).
set -u
QUEUE="$1"; OUTROOT="$2"; SLOT="$3"; GPUS="$4"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${BRONCHO_PY:-python}"
mkdir -p "$OUTROOT"; LOG="$OUTROOT/runner.log"; LOCK="$QUEUE.lock"

pop_job() {                                     # atomically remove and print the first real job line
  exec 9>"$LOCK"; flock 9
  local job; job=$(grep -v '^#' "$QUEUE" | grep -v '^[[:space:]]*$' | head -1)
  if [ -n "$job" ]; then
    awk -v j="$job" 'BEGIN{done=0} { if (!done && $0==j) {done=1; next} print }' "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
  fi
  flock -u 9; exec 9>&-
  printf '%s' "$job"
}

while true; do
  job=$(pop_job); [ -z "$job" ] && { echo "[$SLOT] queue empty" | tee -a "$LOG"; break; }
  IFS='|' read -r tag vid calib lo hi extra <<<"$job"
  out="$OUTROOT/rr_$tag"; mkdir -p "$out"
  echo "[$SLOT $(date +%H:%M)] START $tag f$lo-$hi gpus $GPUS $extra" | tee -a "$LOG"
  timeout 7200 "$PY" -u "$HERE/recover_clip.py" --video "$vid" --calib "$calib" --session "$tag" --lo "$lo" --hi "$hi" --out "$out" --gpus "$GPUS" $extra > "$out/run.log" 2>&1
  model=$(grep -m1 '\[model\]' "$out/run.log")
  if [ -f "$out/dense0/fused.ply" ]; then
    "$PY" "$HERE/eval_recon.py" "$out" --label "$tag" > "$out/eval.log" 2>&1
    res=$("$PY" -c "import json;d=json.load(open('$out/eval.json'));print(f\"{d['n_gated']}/{d['n_stations']} gated cov {d['median_cov']:.2f} {d['n_clean']:,} pts -> {d['tier']}\")" 2>/dev/null)
    echo "[$SLOT $(date +%H:%M)] DONE  $tag  $model | $res" | tee -a "$LOG"
  else
    echo "[$SLOT $(date +%H:%M)] FAIL  $tag  $model | $(grep -m1 -E 'NO MODEL|no cloud|NO CALIB|Error' "$out/run.log" | head -c 100)" | tee -a "$LOG"
    rm -rf "$out/dense0"
  fi
done
