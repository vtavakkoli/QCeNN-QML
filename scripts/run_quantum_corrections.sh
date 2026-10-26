#!/usr/bin/env bash
# QCeNN-QML — Vahid Tavakkoli (2026). See LICENSE and CITATION.cff.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
qcenn_latest_file=${QCENN_LATEST_FILE:-results/quantum-corrections-latest.txt}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_quantum_corrections.sh [--smoke] [--mnist-only] [OUTPUT_DIRECTORY]
       bash scripts/run_quantum_corrections.sh --status [OUTPUT_DIRECTORY]

Full: original MNIST seeds 42/43/44, matched text diffusion, attention,
coherent text memory and Smol audit. Independent jobs run concurrently.
Smoke: explicit digits proxy and tiny corpora; not scientific benchmark data.
--mnist-only: rerun only the three MNIST seeds and their combined summary.
Each experiment gets a log, status, command and separate ZIP with checksum.

Environment: QCENN_PYTHON (default .venv-qcenn/bin/python), QCENN_JOBS (workers),
QCENN_CPUS (total CPU budget). Defaults respect CPU affinity/cgroup quota.
Exit: 0 DONE; 2 DONE_WITH_BLOCKED_TESTS (MNIST fit gate); 1 FAILED;
130/143 INTERRUPTED. Finished execution does not imply scientific success.
EOF
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if [[ ${1:-} == --status ]]; then
  if [[ -n ${2:-} ]]; then qcenn_status_dir=$2
  elif [[ -f $qcenn_latest_file ]]; then
    qcenn_status_dir=$(cat -- "$qcenn_latest_file")
  else printf 'No run found. Supply the output directory.\n' >&2; exit 1; fi
  cat -- "$qcenn_status_dir/STATUS.txt"
  if [[ $(head -n 1 "$qcenn_status_dir/STATUS.txt") == 'State: RUNNING' ]] &&
     ! kill -0 "$(cat "$qcenn_status_dir/runner.pid")" 2>/dev/null; then
    printf '\nSTALE: runner is no longer alive; completion was not recorded.\n'
  fi
  exit 0
fi
qcenn_mode=full; qcenn_mnist_only=0
while [[ ${1:-} == --smoke || ${1:-} == --mnist-only ]]; do
  if [[ $1 == --smoke ]]; then qcenn_mode=smoke; else qcenn_mnist_only=1; fi
  shift
done
if (( $# > 1 )) || [[ ${1:-} == --* ]]; then usage >&2; exit 1; fi
qcenn_python=${QCENN_PYTHON:-.venv-qcenn/bin/python}
command -v "$qcenn_python" >/dev/null || { printf 'Python missing: %s\n' "$qcenn_python" >&2; exit 1; }
command -v zip >/dev/null || { printf 'Install the native zip command first.\n' >&2; exit 1; }
qcenn_detected=$(nproc)
# nproc can see more cores than a container is allowed to consume.
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r qcenn_quota qcenn_period < /sys/fs/cgroup/cpu.max
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
  qcenn_quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
  qcenn_period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
else qcenn_quota=max; qcenn_period=1; fi
if [[ $qcenn_quota != max ]] && (( qcenn_quota > 0 )); then
  qcenn_quota_cores=$(( (qcenn_quota+qcenn_period-1)/qcenn_period ))
  if (( qcenn_quota_cores < qcenn_detected )); then qcenn_detected=$qcenn_quota_cores; fi
fi
qcenn_cpus=${QCENN_CPUS:-$qcenn_detected}
[[ $qcenn_cpus =~ ^[1-9][0-9]*$ ]] || { printf 'QCENN_CPUS must be positive.\n' >&2; exit 1; }
qcenn_independent=$((qcenn_mnist_only ? 3 : 6))
qcenn_workers=${QCENN_JOBS:-$((qcenn_cpus < qcenn_independent ? qcenn_cpus : qcenn_independent))}
[[ $qcenn_workers =~ ^[1-9][0-9]*$ ]] || { printf 'QCENN_JOBS must be positive.\n' >&2; exit 1; }
if (( qcenn_workers > qcenn_cpus || qcenn_workers > qcenn_independent )); then
  printf 'QCENN_JOBS must not exceed QCENN_CPUS or the selected independent job count (%s).\n' "$qcenn_independent" >&2; exit 1
fi
qcenn_output=${1:-results/quantum-corrections-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p -- "$(dirname -- "$qcenn_output")"
mkdir -- "$qcenn_output" # Atomic refusal to overwrite previous evidence.
qcenn_output=$(cd -- "$qcenn_output" && pwd -P)
mkdir -p "$qcenn_output"/{experiments,logs,status,commands,archives} results
printf '%s\n' "$$" > "$qcenn_output/runner.pid"
qcenn_started=$(date +%s)
qcenn_finalized=0; qcenn_state=RUNNING; qcenn_last_heartbeat=0; qcenn_last_snapshot=0
qcenn_audit_started=$qcenn_mnist_only
qcenn_names=(mnist-42 mnist-43 mnist-44 diffusion attention text smol-audit summary)
if ((qcenn_mnist_only)); then qcenn_names=(mnist-42 mnist-43 mnist-44 summary); fi
declare -A qcenn_pids=()

event() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$qcenn_output/events.log"; }
job_state() {
  local name=$1
  shift
  printf '%s\t%s\t%s\t%s\t%s\n' "$@" > "$qcenn_output/status/$name.$BASHPID.tmp"
  mv -- "$qcenn_output/status/$name.$BASHPID.tmp" "$qcenn_output/status/$name.tsv"
}
snapshot() {
  local name state code began ended threads now
  now=$(date +%s)
  {
    printf 'State: %s\nOutput: %s\nMode: %s\nElapsed: %ss\nUpdated: %s\n' "$qcenn_state" "$qcenn_output" "$qcenn_mode" "$((now-qcenn_started))" "$(date -Iseconds)"
    printf 'CPU budget: %s; workers: %s; backend: Aer CPU MPS\n' "$qcenn_cpus" "$qcenn_workers"
    printf '\n%-14s %-17s %-6s %-8s %s\n' EXPERIMENT STATE EXIT THREADS 'LATEST LOG LINE'
    for name in "${qcenn_names[@]}"; do
      read -r state code began ended threads < "$qcenn_output/status/$name.tsv"
      printf '%-14s %-17s %-6s %-8s ' "$name" "$state" "$code" "$threads"
      tail -n 1 "$qcenn_output/logs/$name.log" | cut -c 1-160
      [[ -s $qcenn_output/logs/$name.log ]] || printf '\n'
    done
    printf '\nArchives: %s/archives/*.zip\n' "$qcenn_output"
    printf 'DONE requires all jobs and archives to succeed; blocked/failed jobs stay labeled.\n'
  } > "$qcenn_output/STATUS.$BASHPID.tmp"
  mv -- "$qcenn_output/STATUS.$BASHPID.tmp" "$qcenn_output/STATUS.txt"
  qcenn_last_snapshot=$now
}
archive_job() {
  local name=$1
  (
    cd -- "$qcenn_output"
    zip -qr "archives/$name.partial.zip" "experiments/$name" "logs/$name.log" \
      "status/$name.tsv" "commands/$name.txt" runtime.txt &&
    mv -- "archives/$name.partial.zip" "archives/$name.zip" &&
    cd archives && sha256sum "$name.zip" > "$name.zip.sha256"
  )
}
run_job() {
  trap - EXIT
  local name=$1 threads=$2 began child=0 code state expected
  shift 2
  export QCENN_THREADS=$threads OMP_NUM_THREADS=$threads
  export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  export QISKIT_PARALLEL=FALSE QISKIT_NUM_PROCS=1
  began=$(date +%s)
  mkdir -p -- "$qcenn_output/experiments/$name"
  printf '%q ' "$@" > "$qcenn_output/commands/$name.txt"
  printf '\n' >> "$qcenn_output/commands/$name.txt"
  job_state "$name" RUNNING - "$began" 0 "$threads"
  event "START $name threads=$threads log=$qcenn_output/logs/$name.log"
  trap 'if ((child)); then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
    job_state "$name" INTERRUPTED 143 "$began" "$(date +%s)" "$threads"
    printf "INTERRUPTED; partial output only.\n" > "$qcenn_output/experiments/$name/RESULT_STATUS.txt"
    archive_job "$name" || true; exit 143' TERM INT
  "$@" > "$qcenn_output/logs/$name.log" 2>&1 &
  child=$!
  if wait "$child"; then code=0; else code=$?; fi
  child=0; state=DONE
  expected="$qcenn_output/experiments/$name/report.json"
  if [[ $name == summary ]]; then expected="$qcenn_output/experiments/$name/combined.json"; fi
  if ((code == 0)) && [[ ! -s $expected ]]; then
    printf 'FAILED: command exited 0 but expected report is missing: %s\n' "$expected" >> "$qcenn_output/logs/$name.log"
    code=1
  fi
  if ((code)); then
    state=FAILED
    if ((code == 2)) && [[ $name == mnist-* && -f $qcenn_output/experiments/$name/report.json ]]; then state=GATE_FAILED; fi
  fi
  job_state "$name" "$state" "$code" "$began" "$(date +%s)" "$threads"
  printf 'State: %s\nExit: %s\n' "$state" "$code" > "$qcenn_output/experiments/$name/RESULT_STATUS.txt"
  if ! archive_job "$name"; then
    code=1; state=ARCHIVE_FAILED
    job_state "$name" "$state" "$code" "$began" "$(date +%s)" "$threads"
  fi
  event "END $name state=$state exit=$code zip=$qcenn_output/archives/$name.zip"
  return "$code"
}
reap_workers() {
  local slot pid now
  for slot in "${!qcenn_pids[@]}"; do
    pid=${qcenn_pids[$slot]}
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then :; else :; fi # Per-job status retains the exit code.
      unset 'qcenn_pids[$slot]'
    fi
  done
  # The audit needs only attention, not the long MNIST/text simulations.
  if ((qcenn_audit_started == 0)); then
    local attention_state ignored
    read -r attention_state ignored < "$qcenn_output/status/attention.tsv"
    if [[ $attention_state == DONE ]] && ((${#qcenn_pids[@]} < qcenn_workers)); then
      qcenn_audit_started=1
      launch smol-audit "$qcenn_python" -u -m qcenn_qml.smol_feasibility \
        --attention-report "$qcenn_output/experiments/attention/report.json" --output-dir "$qcenn_output/experiments/smol-audit"
    elif [[ $attention_state == FAILED || $attention_state == ARCHIVE_FAILED || $attention_state == INTERRUPTED ]]; then
      qcenn_audit_started=1
      mkdir -p "$qcenn_output/experiments/smol-audit"
      printf 'SKIPPED: attention did not succeed.\n' > "$qcenn_output/logs/smol-audit.log"
      job_state smol-audit SKIPPED 1 0 "$(date +%s)" 0
      archive_job smol-audit
    fi
  fi
  now=$(date +%s)
  if ((now-qcenn_last_snapshot >= 5)); then snapshot; fi
  if ((now-qcenn_last_heartbeat >= 30)); then
    event "RUNNING elapsed=$((now-qcenn_started))s active=${#qcenn_pids[@]}; status=$qcenn_output/STATUS.txt"
    qcenn_last_heartbeat=$now
  fi
}
wait_workers() { while ((${#qcenn_pids[@]})); do reap_workers; ((${#qcenn_pids[@]} == 0)) || sleep 1; done; }
launch() {
  local name=$1 slot threads
  shift
  while ((${#qcenn_pids[@]} >= qcenn_workers)); do reap_workers; sleep 1; done
  for ((slot=0;slot<qcenn_workers;slot++)); do [[ -n ${qcenn_pids[$slot]:-} ]] || break; done
  threads=$((qcenn_cpus/qcenn_workers + (slot < qcenn_cpus%qcenn_workers)))
  run_job "$name" "$threads" "$@" &
  qcenn_pids[$slot]=$!
}
on_exit() {
  local code=$1 slot name state rest
  if ((qcenn_finalized)); then return; fi
  trap - EXIT INT TERM
  for slot in "${!qcenn_pids[@]}"; do kill -TERM "${qcenn_pids[$slot]}" 2>/dev/null || true; done
  for slot in "${!qcenn_pids[@]}"; do wait "${qcenn_pids[$slot]}" 2>/dev/null || true; done
  qcenn_state=FAILED
  if ((code == 130 || code == 143)); then qcenn_state=INTERRUPTED; fi
  for name in "${qcenn_names[@]}"; do
    read -r state rest < "$qcenn_output/status/$name.tsv"
    if [[ $state == PENDING ]]; then job_state "$name" SKIPPED "$code" 0 "$(date +%s)" 0; fi
  done
  snapshot
  printf 'State: %s\nExit: %s\n' "$qcenn_state" "$code" > "$qcenn_output/$qcenn_state"
  event "FINISHED state=$qcenn_state exit=$code; see $qcenn_output/STATUS.txt"
}
for qcenn_name in "${qcenn_names[@]}"; do
  job_state "$qcenn_name" PENDING - 0 0 0
  : > "$qcenn_output/logs/$qcenn_name.log"
  : > "$qcenn_output/commands/$qcenn_name.txt"
done
trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'mode=%s\ncpu_budget=%s\nworkers=%s\npython=%s\nbackend=CPU matrix_product_state\n' \
  "$qcenn_mode" "$qcenn_cpus" "$qcenn_workers" "$qcenn_python" > "$qcenn_output/runtime.txt"
printf 'mnist_only=%s\n' "$qcenn_mnist_only" >> "$qcenn_output/runtime.txt"
snapshot
mkdir -p -- "$(dirname -- "$qcenn_latest_file")"
printf '%s\n' "$qcenn_output" > "$qcenn_latest_file.$BASHPID.tmp"
mv -- "$qcenn_latest_file.$BASHPID.tmp" "$qcenn_latest_file"
event "RUN STARTED mode=$qcenn_mode cpus=$qcenn_cpus workers=$qcenn_workers output=$qcenn_output"
qcenn_dataset=mnist; qcenn_side=8; qcenn_shots=1024
qcenn_diffusion_shots=1024; qcenn_examples=8; qcenn_corpus=text
qcenn_attention_shots=8192; qcenn_tokens=3; qcenn_text_extra=()
qcenn_data_root=${QCENN_DATA_ROOT:-data}
if [[ $qcenn_mode == smoke ]]; then
  qcenn_dataset=digits-proxy; qcenn_side=4; qcenn_shots=128
  qcenn_diffusion_shots=128; qcenn_examples=2; qcenn_corpus=toy
  qcenn_attention_shots=256; qcenn_tokens=2
  qcenn_text_extra=(--corpus-file tests/fixtures/quantum_sequence.txt --prompt cat)
  event 'SMOKE ONLY: digits proxy and tiny corpora, not original-MNIST evidence.'
else
  event "PREPARING MNIST cache once before parallel readers; log=$qcenn_output/logs/data-prepare.log"
  QCENN_DATA_ROOT="$qcenn_data_root" "$qcenn_python" -u -c 'import os; from qcenn_qml.mnist_data import load_dataset; load_dataset("mnist",os.environ["QCENN_DATA_ROOT"]); print("MNIST cache ready", flush=True)' \
    > "$qcenn_output/logs/data-prepare.log" 2>&1 &
  qcenn_pids[0]=$!
  if wait "${qcenn_pids[0]}"; then unset 'qcenn_pids[0]'
  else qcenn_cache_exit=$?; unset 'qcenn_pids[0]'; exit "$qcenn_cache_exit"; fi
fi
for qcenn_seed in 42 43 44; do
  launch "mnist-$qcenn_seed" "$qcenn_python" -u -m qcenn_qml.recurrent_mnist \
    --dataset "$qcenn_dataset" --seeds "$qcenn_seed" --side "$qcenn_side" \
    --train-per-class 1 --validation-per-class 1 --test-per-class 1 --rounds 4 \
    --shots "$qcenn_shots" --include-reset-control --no-download --fail-on-fit-gate \
    --data-root "$qcenn_data_root" \
    --output-dir "$qcenn_output/experiments/mnist-$qcenn_seed"
done
if ((qcenn_mnist_only == 0)); then
launch diffusion "$qcenn_python" -u -m qcenn_qml.diffusion.benchmark \
  --corpus-kind "$qcenn_corpus" --steps 0 1 3 --examples "$qcenn_examples" \
  --shots "$qcenn_diffusion_shots" --readout-error 0.05 --output-dir "$qcenn_output/experiments/diffusion"
launch attention "$qcenn_python" -u -m qcenn_qml.quantum_attention \
  --shots "$qcenn_attention_shots" --output-dir "$qcenn_output/experiments/attention"
launch text "$qcenn_python" -u -m qcenn_qml.smol135 --prompt 'the cat' \
  --new-tokens "$qcenn_tokens" --shots "$qcenn_shots" "${qcenn_text_extra[@]}" --output-dir "$qcenn_output/experiments/text"
fi
wait_workers
# The combined report requires all experiment outputs.
launch summary "$qcenn_python" -u -m qcenn_qml.report \
  --mnist "$qcenn_output/experiments/legacy-mnist-not-run.json" \
  --recurrent "$qcenn_output/experiments/mnist-42/report.json" \
    "$qcenn_output/experiments/mnist-43/report.json" "$qcenn_output/experiments/mnist-44/report.json" \
  --diffusion "$qcenn_output/experiments/diffusion/report.json" --diffusion-noisy "$qcenn_output/experiments/diffusion/report.json" \
  --smol "$qcenn_output/experiments/text/report.json" --attention "$qcenn_output/experiments/attention/report.json" \
  --smol-feasibility "$qcenn_output/experiments/smol-audit/report.json" --output-dir "$qcenn_output/experiments/summary"
wait_workers
qcenn_exit=0; qcenn_state=DONE
for qcenn_name in "${qcenn_names[@]}"; do
  read -r qcenn_job_state _ < "$qcenn_output/status/$qcenn_name.tsv"
  if [[ $qcenn_job_state == GATE_FAILED ]]; then
    if ((qcenn_exit == 0)); then qcenn_exit=2; qcenn_state=DONE_WITH_BLOCKED_TESTS; fi
  elif [[ $qcenn_job_state != DONE ]]; then qcenn_exit=1; qcenn_state=FAILED; fi
done
snapshot
printf 'State: %s\nExit: %s\n' "$qcenn_state" "$qcenn_exit" > "$qcenn_output/$qcenn_state"
qcenn_finalized=1
event "FINISHED state=$qcenn_state exit=$qcenn_exit; status=$qcenn_output/STATUS.txt; ZIPs=$qcenn_output/archives/"
exit "$qcenn_exit"
