#!/usr/bin/env bash
set -euo pipefail

# Launch from the repository root with nohup bash scripts/run_quantum_corrections.sh.
qcenn_python="${QCENN_PYTHON:-.venv-qcenn/bin/python}"
qcenn_output="${1:-results/quantum-corrections-$(date +%Y%m%d-%H%M%S)}"
command -v zip >/dev/null
test ! -e "$qcenn_output"

"$qcenn_python" -u -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 43 44 \
  --train-per-class 1 --validation-per-class 1 --test-per-class 1 \
  --rounds 4 --shots 1024 --include-reset-control --output-dir "$qcenn_output/mnist"
"$qcenn_python" -u -m qcenn_qml.diffusion.benchmark --corpus-kind text \
  --steps 0 1 3 --examples 8 --shots 1024 --readout-error 0.05 \
  --output-dir "$qcenn_output/diffusion"
"$qcenn_python" -u -m qcenn_qml.quantum_attention --shots 8192 \
  --output-dir "$qcenn_output/attention"
"$qcenn_python" -u -m qcenn_qml.smol_feasibility \
  --attention-report "$qcenn_output/attention/report.json" --output-dir "$qcenn_output/smol-audit"
"$qcenn_python" -u -m qcenn_qml.smol135 --prompt 'the cat' --new-tokens 3 \
  --shots 1024 --output-dir "$qcenn_output/text"
"$qcenn_python" -u -m qcenn_qml.report --recurrent "$qcenn_output/mnist/report.json" \
  --diffusion "$qcenn_output/diffusion/report.json" --smol "$qcenn_output/text/report.json" \
  --attention "$qcenn_output/attention/report.json" --smol-feasibility "$qcenn_output/smol-audit/report.json" \
  --output-dir "$qcenn_output/summary"
zip -rq "${qcenn_output}.zip" "$qcenn_output"
printf 'Results archived: %s.zip\n' "$qcenn_output"
