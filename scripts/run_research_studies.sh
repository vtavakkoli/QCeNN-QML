#!/usr/bin/env bash
# QCeNN-QML — Vahid Tavakkoli (2026). See LICENSE and CITATION.cff.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export QCENN_LATEST_FILE=${QCENN_LATEST_FILE:-results/research-studies-latest.txt}
if [[ ${QCENN_LEGACY_RUNNER:-0} == 1 ]]; then
  if [[ ${1:-} == --status || ${1:-} == --help || ${1:-} == -h ]]; then
    exec bash scripts/run_quantum_corrections.sh "$@"
  fi
  exec bash scripts/run_quantum_corrections.sh --studies "$@"
fi
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 QISKIT_PARALLEL=FALSE QISKIT_NUM_PROCS=1
exec "${QCENN_PYTHON:-.venv-qcenn/bin/python}" -u -m qcenn_qml.research_runner "$@"
