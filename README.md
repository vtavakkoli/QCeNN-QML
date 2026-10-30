# QCeNN-QML

**Measured quantum-memory MNIST and complete small quantum transformer replacement tests.**

Version 0.6 adds balanced larger memory banks, measured nearest-record readout,
fixed image centering/deskew, and one/two complete coherent transformer blocks
with quantum embeddings and a quantum vocabulary head. Diffusion is retired
from launchers, automatic data downloads, and CI benchmark runs. Its historical
implementation and negative evidence remain available for reproducibility.

MNIST remains a nonparametric raw-record quantum memory, not neural-weight
training. Predictions use final measured counts. Loading, fixed input encoding,
compilation, simulation and decoding are classical. The transformer forward is
coherent; its separate autoregressive integration test explicitly uses a hybrid
host between generated tokens. The entire host pipeline is not purely quantum.
**100% held-out MNIST accuracy is a target, not an achieved or guaranteed result.**
Full pretrained Smol weights, quantum neural training, QPU runs and advantage
remain unimplemented or undemonstrated.

## License, attribution, and commercial use

**QCeNN-QML uses a custom source-available license, not an OSI-approved open-source license.** The current code is free for **non-commercial personal, educational, and academic research use**, provided the required attribution is included. **All commercial use requires a separately negotiated and paid written license before use**, including commercial research and development, internal business use, products, APIs, and consulting.

**Mandatory project credit:**

> QCeNN-QML — Vahid Tavakkoli (2026)  
> https://github.com/vtavakkoli/QCeNN-QML

If you publish research, benchmarks, presentations, or reports materially using this repository, you must also cite QCeNN-QML using [CITATION.cff](CITATION.cff). To request a commercial license and pricing, contact the copyright holder through [the GitHub repository](https://github.com/vtavakkoli/QCeNN-QML). No fixed price or automatic payment authorization is offered here.

See [LICENSE](LICENSE) for legally controlling terms, [NOTICE.md](NOTICE.md) for attribution and upstream provenance, and [CITATION.cff](CITATION.cff) for the recommended software citation. Versions previously distributed under MIT retain their original MIT permissions; this change does not retroactively remove them.

## Install and verify

```bash
python -m pip install -e '.[accelerated,mnist,test]'
python -m pytest -q
```

Original MNIST uses checksummed IDX archives and does not require PyTorch.
The tested simulator is Qiskit 2.6.1 / Aer 0.18.2; see requirements-tested.txt.

## Run the improved experiments

From the project directory, first run the small diagnostic:

```bash
bash scripts/update_and_run_research.sh --smoke --cpu-only
```

Then start a fresh full run:

```bash
mkdir -p results
nohup bash scripts/update_and_run_research.sh \
  > results/research-launch.log 2>&1 < /dev/null &
bash scripts/update_and_run_research.sh --status
```

The updater creates immutable source and isolated environments. Results and
separate per-experiment ZIPs are saved inside the original project's `results/`.
CPU/GPU workers share bounded resources. GPU selection uses the supported Aer
runtime option; actual CUDA qualification is still required on the execution
machine. Unsupported wide circuits remain on CPU MPS.

Run just one track with `--study-only mnist` or `--study-only transformer`.
Direct commands using an already prepared environment:

```bash
.venv-qcenn/bin/python -m qcenn_qml.mnist_study --accuracy-profile \
  --image-batch-size 1 --shot-budgets 4096 16384 65536 \
  --gate-shot-budgets 4096 16384 65536 --validation-per-class 5 \
  --jobs 6 --cpu-budget 6 --output-dir results/mnist-accuracy
.venv-qcenn/bin/python -m qcenn_qml.transformer_replacement \
  --layers 1 2 --measurement-seeds 101 202 303 --shots 1024 \
  --jobs 6 --cpu-budget 6 --output-dir results/transformer-replacement
```

Use fresh output directories and CPU budgets within your allocation. A full run
can take substantially longer than a smoke test. `DONE` means requested work and
archives completed; scientific metrics and target achievement are separate.
Exit 2 means MNIST fit/stability gates blocked testing. Source/configuration
changes require a fresh run; existing evidence is never silently relabeled.

## MNIST accuracy protocol

- Establish convergence at 10 records, then evaluate 20 and 30 records.
- Compare fixed centering/deskew and class-sum/nearest-record measured readout.
- Grow to 40, 80 and 160 records, in balanced 20-record quantum banks. Each bank
  receives equal shots; rejected shots and every bank's CNOT cost are charged.
- Require 100% measured stored-image retrieval before each candidate's validation,
  and at least 95% measurement agreement. Numeric roundoff does not lower that gate.
- Select on validation only; freeze before evaluating 3,000 reserved fresh test
  images across seeds 42/43/44. Keep k=0 and both equal-shot/equal-CNOT comparisons.

Damping is off by default. Request it explicitly with `--damping-values .05`
for a separate bounded investigation. The accuracy profile uses independently
resumable measurement tasks and a locked, hashed native-QPY compilation cache.
`--growth-memories 4 8` bounds a smaller campaign; values through 64 are supported.
No test-driven tuning or analytic prediction fallback is used.

## Complete small transformer replacement

The new test executes quantum token embeddings, all transformer block operators
(RMSNorm, QKV, RoPE, causal softmax, weighted values, output projection, residuals
and SwiGLU), and quantum vocabulary projection/argmax. One and two complete
blocks are checked against an independent classical reference. All 16 input-token
pairs are sampled in small coherent groups that can run in parallel; separate interference tests
check phase coherence and clean workspace. An explicit hybrid host then checks
two-token autoregressive generation using the same measured forward operation.

This is a four-token, two-context-token, width-two, three-bit, fixed-weight model.
It tests replacement correctness for that declared model. It does not demonstrate
pretrained Smol replacement, useful language quality, or neural training.

See [the detailed v0.6 protocol](docs/ACCURACY_AND_REPLACEMENT.md),
[bounded execution and recovery](docs/ACCELERATED_RUNS.md),
[research rules](AGENTS.md), and [historical instructions](docs/README_V05_ARCHIVE.md).

## Attribution

QCeNN-QML — Vahid Tavakkoli (2026), https://github.com/vtavakkoli/QCeNN-QML.
Cite the software in all public research and results materially using it.
QCeNN-MI research inspiration — Vahid Tavakkoli and Kabeh Mohsenzadegan (2026),
https://github.com/vtavakkoli/QCeNN-MI. Its separate license/citation obligations
and all third-party rights remain. No upstream source or pretrained weights
have been copied or relicensed by these new modules.
