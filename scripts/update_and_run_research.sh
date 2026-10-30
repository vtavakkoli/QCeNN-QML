#!/usr/bin/env bash
# QCeNN-QML — Vahid Tavakkoli (2026). See LICENSE and CITATION.cff.
# Fetches immutable source, creates separate CPU/GPU environments, then runs.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash scripts/update_and_run_research.sh [--repository PATH] [--smoke]
       [--cpu-only] [--text-source bundled] [--study-only mnist|transformer] [--cpus N] [--jobs N]
       [--ref GIT_REF]
       [--replace-run OLD_OUTPUT] [--output-dir NEW_OUTPUT]
       [--resume OUTPUT | --status [OUTPUT] | --stop OUTPUT]

Full: larger measured MNIST memories/frozen test, complete small quantum
transformer replacement, attention, coherent text diagnostic and Smol audit.
Diffusion is retired; no diffusion jobs or WikiText downloads are launched.
Downloads public dataset snapshots and creates separate Python environments.
No system Python, GPU driver, running checkout or existing result is upgraded
in place. --replace-run stops ONLY the verified old experiment tree.
CPU and GPU workers share bounded resources; each experiment gets a native ZIP.
Exit 0 DONE; 2 execution finished with blocked scientific gates; 1 FAILED;
130 INTERRUPTED. Full Smol pretrained weights/training remain unimplemented.
EOF
}
qcenn_repo=${QCENN_REPO:-$PWD}
qcenn_smoke=0; qcenn_gpu=auto; qcenn_text=auto
qcenn_output=; qcenn_replace=; qcenn_action=run; qcenn_action_output=
qcenn_cpus=; qcenn_jobs=
qcenn_ref=origin/main
qcenn_study_only=all
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) qcenn_smoke=1; shift ;;
    --cpu-only) qcenn_gpu=off; shift ;;
    --repository|--ref|--text-source|--study-only|--cpus|--jobs|--replace-run|--output-dir|--resume|--stop)
      (($# >= 2)) || { usage >&2; exit 1; }
      case "$1" in
        --repository) qcenn_repo=$2 ;;
        --ref) qcenn_ref=$2 ;;
        --text-source) qcenn_text=$2 ;;
        --study-only) qcenn_study_only=$2 ;;
        --cpus) qcenn_cpus=$2 ;;
        --jobs) qcenn_jobs=$2 ;;
        --replace-run) qcenn_replace=$2 ;;
        --output-dir) qcenn_output=$2 ;;
        --resume) qcenn_action=resume; qcenn_action_output=$2 ;;
        --stop) qcenn_action=stop; qcenn_action_output=$2 ;;
      esac
      shift 2 ;;
    --status)
      qcenn_action=status; shift
      if (($#)) && [[ $1 != --* ]]; then qcenn_action_output=$1; shift; fi ;;
    *) usage >&2; exit 1 ;;
  esac
done
qcenn_repo=$(git -C "$qcenn_repo" rev-parse --show-toplevel)
cd -- "$qcenn_repo"
qcenn_base_python=${QCENN_BASE_PYTHON:-${QCENN_PYTHON:-$qcenn_repo/.venv-qcenn/bin/python}}
if [[ ! -x $qcenn_base_python ]] && ! command -v "$qcenn_base_python" >/dev/null 2>&1; then
  qcenn_base_python=$(command -v python3)
fi
if [[ $qcenn_action == status && -z $qcenn_action_output && -f results/research-setup-latest.txt ]]; then
  qcenn_setup_latest=$(cat results/research-setup-latest.txt)
  if [[ -f $qcenn_setup_latest/STATUS.txt ]] && {
       ! flock -n "$qcenn_setup_latest/setup.lock" true ||
       [[ $(head -n 1 "$qcenn_setup_latest/STATUS.txt") == 'State: SETUP_FAILED' ]]; }; then
    cat -- "$qcenn_setup_latest/STATUS.txt"
    exit 0
  fi
fi
if [[ -n $qcenn_action_output ]]; then
  qcenn_action_output=$("$qcenn_base_python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$qcenn_action_output")
elif [[ $qcenn_action == status || $qcenn_action == resume || $qcenn_action == stop ]]; then
  if [[ ! -s results/research-studies-latest.txt ]]; then
    printf 'State: NO_RUN_REGISTERED\nNo experiment output has been registered. Setup may still be in preflight or may have exited; inspect the launcher log.\n'
    if [[ $qcenn_action == status ]]; then exit 0; fi
    printf 'Resume/stop requires an explicit existing output directory.\n' >&2
    exit 1
  fi
  qcenn_action_output=$(cat results/research-studies-latest.txt)
fi
if [[ $qcenn_action == status ]]; then
  if [[ -f $qcenn_action_output/runtime.json ]]; then
    mapfile -t qcenn_status_env < <("$qcenn_base_python" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["source_root"]); print(d["python"])' "$qcenn_action_output/runtime.json")
    cd -- "${qcenn_status_env[0]}"
    export PYTHONPATH="${qcenn_status_env[0]}/src"
    exec "${qcenn_status_env[1]}" -m qcenn_qml.research_runner --status "$qcenn_action_output"
  fi
  cat -- "$qcenn_action_output/STATUS.txt"
  printf 'Legacy status snapshot; the v0.5 runner records live identity and verified terminal state.\n'
  exit 0
fi
if [[ $qcenn_action == resume || $qcenn_action == stop ]]; then
  mapfile -t qcenn_saved < <("$qcenn_base_python" - "$qcenn_action_output/runtime.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for value in (d['source_root'],d['python'],d.get('gpu_profile_path') or '',d['execution_policy']['mps_readout']):
 print(value)
PY
  )
  ((${#qcenn_saved[@]} == 4)) || { printf 'This output predates v0.5; use a fresh run with --replace-run instead.\n' >&2; exit 1; }
  export QCENN_GPU_PROFILE=${qcenn_saved[2]} QCENN_MPS_READOUT=${qcenn_saved[3]}
  cd -- "${qcenn_saved[0]}"
  export PYTHONPATH="${qcenn_saved[0]}/src"
  qcenn_resume_args=(--output-dir "$qcenn_action_output")
  [[ -z $qcenn_cpus ]] || qcenn_resume_args+=(--cpus "$qcenn_cpus")
  [[ -z $qcenn_jobs ]] || qcenn_resume_args+=(--jobs "$qcenn_jobs")
  if [[ $qcenn_action == stop ]]; then
    exec "${qcenn_saved[1]}" -u -m qcenn_qml.research_runner --stop "$qcenn_action_output"
  fi
  exec "${qcenn_saved[1]}" -u -m qcenn_qml.research_runner --resume "${qcenn_resume_args[@]}"
fi
[[ $qcenn_text == auto || $qcenn_text == bundled ]] || { printf 'Diffusion/WikiText is retired; use --text-source bundled.\n' >&2; exit 1; }
[[ $qcenn_study_only == all || $qcenn_study_only == mnist || $qcenn_study_only == transformer ]] || { usage >&2; exit 1; }
for qcenn_tool in git flock timeout zip unzip sha256sum; do
  command -v "$qcenn_tool" >/dev/null || { printf 'Required command missing: %s\n' "$qcenn_tool" >&2; exit 1; }
done
"$qcenn_base_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else "The accelerated environment requires Python >=3.11")'
git fetch origin main
qcenn_revision=$(git rev-parse "$qcenn_ref^{commit}")
qcenn_worktree="$(dirname -- "$qcenn_repo")/QCeNN-QML-accelerated-${qcenn_revision:0:12}"
if [[ ! -e $qcenn_worktree && ! -L $qcenn_worktree ]]; then
  # Recover only this absent checkout if Git retains its registration. One
  # --force still refuses a locked worktree; never prune other registrations.
  git worktree add --force --detach "$qcenn_worktree" "$qcenn_revision"
fi
[[ $(git -C "$qcenn_worktree" rev-parse HEAD) == "$qcenn_revision" ]] &&
  git -C "$qcenn_worktree" diff --quiet HEAD -- || { printf 'Worktree changed; refusing to update an active source snapshot.\n' >&2; exit 1; }
qcenn_config="$qcenn_worktree/.venv-qcenn-run-config"
mkdir -p -- "$qcenn_config" "$qcenn_repo/results"
exec 9> "$qcenn_config/setup.lock"
flock -n 9 || { printf 'SETUP_BUSY: another setup process owns this source environment.\n' >&2; exit 1; }
printf '%s\n' "$qcenn_config" > "$qcenn_repo/results/research-setup-latest.txt"
qcenn_setup_started=$(date +%s)
qcenn_setup_state() {
  local stage=$1 message=$2 state=${3:-SETUP}
  {
    printf 'State: %s\nStage: %s\nUpdated: %s\nElapsed: %ss\nSource: %s\nMessage: %s\n' \
      "$state" "$stage" "$(date -Iseconds)" "$(( $(date +%s)-qcenn_setup_started ))" "$qcenn_worktree" "$message"
  } > "$qcenn_config/STATUS.tmp"
  mv -- "$qcenn_config/STATUS.tmp" "$qcenn_config/STATUS.txt"
  printf '%s: %s\n' "$stage" "$message"
}
qcenn_setup_state ENVIRONMENT 'Preparing isolated environments; no new scientific run has started yet.'
(
  exec 9>&-
  while true; do
    sleep 30
    printf '[setup heartbeat %s] %s\n' "$(date -Iseconds)" "$(sed -n '2p' "$qcenn_config/STATUS.txt")"
  done
) &
qcenn_setup_heartbeat=$!
qcenn_setup_cleanup() {
  local code=$?
  kill "$qcenn_setup_heartbeat" 2>/dev/null || true
  wait "$qcenn_setup_heartbeat" 2>/dev/null || true
  if ((code)); then qcenn_setup_state FAILED "Setup exited $code; inspect logs in $qcenn_config" SETUP_FAILED; fi
}
trap qcenn_setup_cleanup EXIT
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export QISKIT_PARALLEL=FALSE QISKIT_NUM_PROCS=1
qcenn_python="$qcenn_worktree/.venv-qcenn/bin/python"
if [[ ! -f $qcenn_config/cpu-ready.txt ]]; then
  qcenn_setup_state CPU_INSTALL "Creating isolated CPU environment at $qcenn_worktree/.venv-qcenn"
  "$qcenn_base_python" -m venv "$qcenn_worktree/.venv-qcenn"
  timeout 1800 "$qcenn_python" -m pip install -e "$qcenn_worktree[mnist,test]" \
    'qiskit==2.6.1' 'qiskit-aer==0.18.2' > "$qcenn_config/cpu-install.log" 2>&1 || {
      cat "$qcenn_config/cpu-install.log" >&2; exit 1; }
  "$qcenn_python" -m pip check
  "$qcenn_python" -m pip freeze > "$qcenn_config/cpu-requirements.txt"
  printf '%s\n' "$qcenn_revision" > "$qcenn_config/cpu-ready.txt"
fi
export QCENN_PYTHON="$qcenn_python" QCENN_DATA_ROOT=${QCENN_DATA_ROOT:-$qcenn_repo/data}
export QCENN_LATEST_FILE="$qcenn_repo/results/research-studies-latest.txt"
export QCENN_MPS_READOUT=full_marginal
export QCENN_REQUEST_CPUS=${qcenn_cpus:-${QCENN_CPUS:-}}
export QCENN_TASK_TIMEOUT_SECONDS=${QCENN_TASK_TIMEOUT_SECONDS:-3600}
export QCENN_IDLE_TIMEOUT_SECONDS=${QCENN_IDLE_TIMEOUT_SECONDS:-1800}
unset QCENN_GPU_PROFILE QCENN_GPU_PYTHON
cd -- "$qcenn_worktree"
export PYTHONPATH="$qcenn_worktree/src"
qcenn_setup_state TESTS "Running the project tests; log=$qcenn_config/tests.log"
timeout 1800 "$qcenn_python" -m pytest -q > "$qcenn_config/tests.log" 2>&1 || {
  tail -n 80 "$qcenn_config/tests.log" >&2; exit 1; }
tail -n 2 "$qcenn_config/tests.log"
if [[ $qcenn_gpu != off ]] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  qcenn_gpu_python="$qcenn_worktree/.venv-qcenn-gpu/bin/python"
  if [[ ! -f $qcenn_config/gpu-ready.txt ]]; then
    qcenn_setup_state GPU_INSTALL "Trying the separate CUDA 12 Aer environment; log=$qcenn_config/gpu-install.log"
    # -I excludes the CPU project's PYTHONPATH/editable metadata. The GPU
    # environment intentionally contains the GPU Aer wheel, not qcenn-qml.
    if "$qcenn_base_python" -m venv "$qcenn_worktree/.venv-qcenn-gpu" &&
        timeout 1800 "$qcenn_gpu_python" -I -m pip install 'qiskit==2.6.1' 'qiskit-aer-gpu==0.18.2' > "$qcenn_config/gpu-install.log" 2>&1 &&
        "$qcenn_gpu_python" -I -m pip check > "$qcenn_config/gpu-check.log" 2>&1 &&
        "$qcenn_gpu_python" -I -m pip freeze > "$qcenn_config/gpu-requirements.txt"; then
      printf '%s\n' "$qcenn_revision" > "$qcenn_config/gpu-ready.txt"
    else
      printf 'GPU SETUP UNAVAILABLE; continuing with the accelerated CPU sampler. Logs: %s and %s\n' "$qcenn_config/gpu-install.log" "$qcenn_config/gpu-check.log"
    fi
  fi
  if [[ -f $qcenn_config/gpu-ready.txt ]]; then
    qcenn_setup_state GPU_QUALIFICATION "Checking device methods and measured native circuits; log=$qcenn_config/gpu-validation.log"
    qcenn_gpu_libs=$("$qcenn_gpu_python" -I -c 'import os,sysconfig; from pathlib import Path; p=Path(sysconfig.get_paths()["purelib"]); print(os.pathsep.join(str(x) for x in [*p.glob("nvidia/*/lib"),*p.glob("cuquantum/lib")] if x.is_dir()))')
    mapfile -t qcenn_gpu_launch_info < <("$qcenn_python" - "$qcenn_gpu_python" "$qcenn_gpu_libs:${LD_LIBRARY_PATH:-}" "$qcenn_config" <<'PY'
import hashlib,shlex,sys
from pathlib import Path
body='#!/usr/bin/env bash\nexport LD_LIBRARY_PATH='+shlex.quote(sys.argv[2])+'\nexec '+shlex.quote(sys.argv[1])+' -I "$@"\n'
path=Path(sys.argv[3])/('gpu-python-'+hashlib.sha256(body.encode()).hexdigest()[:16])
if path.exists() and path.read_text()!=body:
 raise ValueError('GPU launcher changed')
path.write_text(body); path.chmod(0o700)
print(path)
PY
    )
    qcenn_gpu_launcher=${qcenn_gpu_launch_info[0]}
    if timeout 180 "$qcenn_gpu_launcher" src/qcenn_qml/gpu_worker.py --probe "$qcenn_config/gpu-probe.json" > "$qcenn_config/gpu-probe.log" 2>&1; then
      "$qcenn_python" - "$qcenn_config/gpu-probe.json" "$qcenn_gpu_launcher" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); d['python_binary']=d['python']; d['python']=sys.argv[2]
p.write_text(json.dumps(d,indent=2))
PY
      qcenn_gpu_hash=$(sha256sum "$qcenn_config/gpu-probe.json"); qcenn_gpu_hash=${qcenn_gpu_hash%% *}
      qcenn_gpu_profile="$qcenn_config/gpu-$qcenn_gpu_hash.json"
      [[ -f $qcenn_gpu_profile ]] || cp -- "$qcenn_config/gpu-probe.json" "$qcenn_gpu_profile"
      export QCENN_GPU_PROFILE="$qcenn_gpu_profile"
      export QCENN_GPU_LOCK_DIR="/tmp/qcenn-gpu-slots-$(id -u)"
      "$qcenn_python" -c 'import json,os; d=json.load(open(os.environ["QCENN_GPU_PROFILE"])); print("GPU VERIFIED:",d["verified_gpu_ids"],"methods:",d["verified_methods"])'
      if "$qcenn_python" -c 'import json,os,sys; sys.exit(0 if json.load(open(os.environ["QCENN_GPU_PROFILE"])).get("enabled") else 1)'; then
        if timeout 600 "$qcenn_python" -m pytest -q tests/test_gpu_dispatch.py -k hardware > "$qcenn_config/gpu-validation.log" 2>&1; then
          export QCENN_GPU_FRACTION=${QCENN_GPU_FRACTION:-$("$qcenn_python" -c 'import json,os; from qcenn_qml.resources import detect_resources; d=json.load(open(os.environ["QCENN_GPU_PROFILE"])); print(min(.5,len(d["verified_gpu_ids"])/int(os.environ.get("QCENN_REQUEST_CPUS") or detect_resources()["cpu_budget"])))')}
        else
          printf 'GPU NATIVE-CIRCUIT VALIDATION FAILED; using CPU. Log: %s\n' "$qcenn_config/gpu-validation.log"
          unset QCENN_GPU_PROFILE
        fi
      fi
    else
      printf 'GPU PROBE FAILED; CPU execution remains available. Log: %s\n' "$qcenn_config/gpu-probe.log"
    fi
  fi
fi
qcenn_arguments=(--study-only "$qcenn_study_only")
if ((qcenn_smoke)); then qcenn_arguments+=(--smoke); fi
[[ -z $qcenn_cpus ]] || qcenn_arguments+=(--cpus "$qcenn_cpus")
[[ -z $qcenn_jobs ]] || qcenn_arguments+=(--jobs "$qcenn_jobs")
if [[ -n $qcenn_replace ]]; then
  qcenn_replace=$("$qcenn_python" -c 'from pathlib import Path; import sys; print((Path(sys.argv[1])/sys.argv[2]).resolve())' "$qcenn_repo" "$qcenn_replace")
  qcenn_arguments+=(--replace-run "$qcenn_replace")
fi
qcenn_output=${qcenn_output:-$qcenn_repo/results/accelerated-$(date +%Y%m%d-%H%M%S)-$$}
qcenn_output=$("$qcenn_python" -c 'from pathlib import Path; import sys; print((Path(sys.argv[1])/sys.argv[2]).resolve())' "$qcenn_repo" "$qcenn_output")
export QCENN_SETUP_DIR="$qcenn_config"
qcenn_setup_state HANDOFF "Launching new output $qcenn_output" RUN_LAUNCHED
kill "$qcenn_setup_heartbeat" 2>/dev/null || true
wait "$qcenn_setup_heartbeat" 2>/dev/null || true
flock -u 9; exec 9>&-
printf 'STARTING: %s\n' "$qcenn_output"
printf 'STATUS: bash scripts/run_research_studies.sh --status %q\n' "$qcenn_output"
exec "$qcenn_python" -u -m qcenn_qml.research_runner --output-dir "$qcenn_output" "${qcenn_arguments[@]}"
