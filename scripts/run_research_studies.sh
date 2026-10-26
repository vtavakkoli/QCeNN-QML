#!/usr/bin/env bash
# QCeNN-QML — Vahid Tavakkoli (2026). See LICENSE and CITATION.cff.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export QCENN_LATEST_FILE=${QCENN_LATEST_FILE:-results/research-studies-latest.txt}
if [[ ${1:-} == --status || ${1:-} == --help || ${1:-} == -h ]]; then
  exec bash scripts/run_quantum_corrections.sh "$@"
fi
exec bash scripts/run_quantum_corrections.sh --studies "$@"
