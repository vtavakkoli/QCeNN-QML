# QCeNN-QML

**Quantum-memory MNIST, coherent text memory, matched denoising, and signed quantum softmax attention.**

Version 0.3 defaults have no classical optimizer, fitted classical prediction
weights, or intermediate CPU state/token updates. Raw records are loaded into
quantum memory; model outputs come from gates and final measurements. These
are **nonparametric memories, not quantum neural-weight training**. Input
encoding, compilation, Aer simulation and final counting remain classical,
so the entire host pipeline is not purely quantum.

The full pretrained SmolLM2-135M is **not converted or executed** here. A small
attention component and an explicit full-model coverage/resource audit are
implemented. Historical hybrid experiments remain explicit legacy references.
See [the v0.3 protocol](docs/QUANTUM_CORE.md) and [research rules](AGENTS.md).

## License, attribution, and commercial use

**QCeNN-QML uses a custom source-available license, not an OSI-approved open-source license.** The current code is free for **non-commercial personal, educational, and academic research use**, provided the required attribution is included. **All commercial use requires a separately negotiated and paid written license before use**, including commercial research and development, internal business use, products, APIs, and consulting.

**Mandatory project credit:**

> QCeNN-QML — Vahid Tavakkoli (2026)  
> https://github.com/vtavakkoli/QCeNN-QML

If you publish research, benchmarks, presentations, or reports materially using this repository, you must also cite QCeNN-QML using [CITATION.cff](CITATION.cff). To request a commercial license and pricing, contact the copyright holder through [the GitHub repository](https://github.com/vtavakkoli/QCeNN-QML). No fixed price or automatic payment authorization is offered here.

See [LICENSE](LICENSE) for legally controlling terms, [NOTICE.md](NOTICE.md) for attribution and upstream provenance, and [CITATION.cff](CITATION.cff) for the recommended software citation. Versions previously distributed under MIT retain their original MIT permissions; this change does not retroactively remove them.

## Installation

Python 3.10+; tested environment versions are in `requirements-tested.txt`.
Original MNIST uses checksummed IDX archives and does not require PyTorch.

```bash
git clone https://github.com/vtavakkoli/QCeNN-QML.git
cd QCeNN-QML
python -m pip install -e ".[mnist,test]"
```

## Quantum-memory MNIST

```bash
python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 \
  --train-per-class 1 --validation-per-class 1 --test-per-class 1 \
  --rounds 4 --shots 1024 --include-reset-control --output-dir results/quantum-mnist
```

A resident quantum example index survives repeated overlap filters. With
fidelity p, r rounds give monotone weights p**r; labels are read from quantum
registers. Tiny training-memory retrieval must reach 100% by default before
validation/test.
This checks memorization, not fitted neural weights or generalization.
The measured one-update **k=0 quantum reference** is always included; k=1 is
not a default. Optional reset control uses a fresh quantum index per update.
All postselection, raw counts, source/split hashes, gates and QPYs are retained.

Validate alternative encodings with a matched small image resolution:

```bash
python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 --side 4 \
  --encoding angle --rounds 2 --damping 0 --shots 1024 --validation-only \
  --output-dir results/mnist-angle
```

Use `--encoding amplitude` with the same side for a comparison. Other controls
are `--rounds 1`, `2`, `4` and `--damping 0`, `0.05`. Choose configurations
using training/validation, never test scores. `--validation-only` never
evaluates test, even after a passed retrieval gate. Use a fresh output directory;
the default raw memory limit is 32 records. Details: [QUANTUM_CORE.md](docs/QUANTUM_CORE.md).

The original per-exemplar reference remains available, explicitly using k=0:

```bash
python -m qcenn_qml.mnist --dataset mnist --classes 01 --train-per-class 2 --test-per-class 5 --shots 1024 --iterations 0 --output-dir results/mnist-binary
python -m qcenn_qml.mnist --dataset mnist --classes 0123456789 --train-per-class 2 --test-per-class 5 --shots 1024 --iterations 0 --output-dir results/mnist-ten
```

## Matched quantum denoising

```bash
python -m qcenn_qml.diffusion.benchmark --corpus-kind text --shots 1024 \
  --examples 8 --beta 1 --eta 0.65 --steps 0 1 3 --readout-error 0.05 \
  --output-dir results/diffusion-matched
```

Raw corpus records enter QROM; no fitted CPU conditional table compiles the
quantum probabilities. A direct classical conditional-table baseline is a
separate reference. Noiseless/noisy runs use the identical native circuits
and cases. The target stays quantum across replacements. Invalid readout
codes keep their error mass, and unseen contexts abstain.

The default small English corpus splits whole sentences before windowing;
`--corpus-file` accepts local text within the declared 256-record/64-token
budget. `--corpus-kind toy` retains the earlier three-token sanity check.
This is a nonparametric memory experiment, not a trained diffusion LLM.

## Coherent text memory

```bash
python -m qcenn_qml.smol135 --prompt 'the cat' --new-tokens 3 --shots 1024 --output-dir results/quantum-text
```

Later queries depend on earlier quantum token registers. All generated tokens
are measured at the end; there is no teacher, CPU transition table or
intermediate classical token update. Complete-sequence postselection can
reweight earlier outputs, and is reported. This is not SmolLM2-135M.

## Quantum attention and full-Smol audit

```bash
python -m qcenn_qml.quantum_attention --shots 8192 --output-dir results/attention
python -m qcenn_qml.smol_feasibility --attention-report results/attention/report.json --output-dir results/smol-audit
```

Reversible gates compute signed ternary dot products and postselected softmax
sampling. Values are loaded quantum mechanically and measured to estimate
weighted means. No CPU attention score/softmax drives the prediction. The
component is not a full-precision coherent transformer block.

The audit uses the [actual Smol configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json),
lists missing operators and input/readout costs, and reports
**NOT_IMPLEMENTED_NOT_DEMONSTRATED** for full pretrained model conversion.
See [scope and accounting](docs/QUANTUM_CORE.md).

## Run everything in the background and zip results

```bash
nohup bash scripts/run_quantum_corrections.sh > quantum-corrections.log 2>&1 < /dev/null &
```

The script uses `.venv-qcenn/bin/python`, creates a fresh timestamped directory,
and runs Linux `zip` after successful completion. Override `QCENN_PYTHON` to
choose another interpreter. Monitor `tail -f quantum-corrections.log`.

## Validation and historical references

```bash
python -m pytest -q
```

Tests independently check quantum memories, reversible signed arithmetic,
softmax distributions, retained recurrence versus refreshed memory, truth-free
fully masked gates, coherent generation, paired noise, data splits and fit-gate
failure handling. CI uses clearly identified digits-proxy and small text cases.
All execution here is **Qiskit Aer simulation**, with no physical QPU or quantum
advantage claim. Larger scientific comparisons remain necessary.

The older experiments remain reproducible only through explicit flags:
`recurrent_mnist --legacy-hybrid`, `smol135 --legacy-classical-memory`, and
`diffusion.benchmark --legacy-count-table`. Classical teacher text generation
requires the text legacy flag and the optional `teacher` dependencies.
See the [v0.2 recurrent protocol](docs/RECURRENT_MNIST.md),
[legacy diffusion protocol](docs/QUANTUM_DIFFUSION.md), and
[archived original-MNIST results](evidence/recurrent-mnist-smoke/README.md).
Negative historical results are preserved.

## Attribution

QCeNN-QML — Vahid Tavakkoli (2026), https://github.com/vtavakkoli/QCeNN-QML.
Cite the software in all public research and results materially using it.

QCeNN-MI research inspiration — Vahid Tavakkoli and Kabeh Mohsenzadegan (2026),
https://github.com/vtavakkoli/QCeNN-MI. Its separate license/citation obligations
and all third-party rights remain. No upstream source or pretrained weights
have been copied or relicensed by these new modules.
