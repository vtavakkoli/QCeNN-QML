# Quantum model cores and honest limits

Version 0.3 changes the default MNIST, text, and denoising paths to quantum
memory inference with no classical optimizer, fitted classical prediction
weights, or intermediate CPU reconstruction of quantum state. Numerical model
operations are explicit gates; final outputs are measured counts.

Input encoding, circuit compilation, classical quantum-circuit simulation, and
final count decoding remain classical. Consequently the **entire host program
is not purely quantum**. These are nonparametric raw-record memories, not a
quantum-only optimizer for the old neural model. Quantum neural-weight training
and a fully converted SmolLM2-135M are **not implemented**. Reports expose each
of these facts independently; no hybrid neural model is silently relabeled.

## MNIST: retrieve a tiny training memory before testing

```bash
python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 \
  --train-per-class 1 --validation-per-class 1 --test-per-class 1 \
  --side 8 --rounds 4 --shots 1024 --include-reset-control \
  --output-dir results/quantum-mnist
```

Uniformly prepare the raw training-example index and load each example's label
into a quantum register. A query and each raw example enter a multiplexed
overlap circuit. Each recurrence keeps the same quantum index and appends a
fresh quantum environment. Final overlap flags herald the accepted label
distribution. For fidelity p_i, r rounds give a class weight proportional to
sum_{i in class} p_i^r. This filter is monotone, unlike ordinary Grover overlap
scoring. The discarded branches remain in the environment until final readout;
postselection probability, actual shots and native gates are charged.

The tiny training-memory retrieval accuracy must reach `--fit-threshold`
(default 0.9) before validation/test are evaluated. This is a memorization
sanity check, **not** fitted neural weights or evidence of generalization.
Failed gates preserve a report and skip later splits. Split IDs and hashes,
raw memory vectors, raw counts, Wilson intervals and QPY examples are saved.

The one-update quantum k=0 reference is always included when test is evaluated.
`--include-reset-control` adds an actual independent-index control: a fresh
quantum index is prepared for each update, with old memory retained as
environment. Conditional output matches one update, but acceptance is a^r
instead of mean(p_i^r), and the extra qubits/gates are counted. No CPU state
update or classical prediction fallback is used.

For validation experiments, add `--validation-only` and vary one fixed
configuration at a time. This mode never evaluates test, even after a passed
retrieval gate:

- `--rounds 1`, `2`, `4`: recurrence/sharpening and postselection cost.
- `--damping 0`, `0.05`: optional physical amplitude-damping channel on the
  index. Fresh retained environment qubits implement the channel; their
  width is charged, and no mid-circuit reset or classical feedback is used.
  Default zero eliminates the previous learned forgetting mechanism.
- `--encoding amplitude` or `angle`, both with `--side 4` for a matched
  16-feature ablation. Angle loading uses product RY states; it is bounded to
  16 features. Amplitude loading supports sides 4, 8 and 16.

Angle success checks use a clean Toffoli ladder, with its additional work
qubits included in the reported width; this avoids exponential synthesis of
a many-control gate without workspace.

Configuration choices must use training/validation results. Do not select
rounds, encodings or damping with held-out test accuracy. No automatic CPU
hyperparameter fitting is disguised as quantum training. The default raw
record limit is 32; large quantum memories require explicit resource budgets.

## Denoising: direct baseline and matched noise

```bash
python -m qcenn_qml.diffusion.benchmark --corpus-kind text \
  --shots 1024 --examples 8 --steps 0 1 3 --readout-error 0.05 \
  --output-dir results/diffusion-matched
```

The default is a small authored English corpus with more than three ordinary
tokens. Whole sentences are split before constructing windows; vocabulary is
encoded from training text only. `--corpus-file` accepts a larger local text
file within explicit limits (256 raw training records, 64 vocabulary entries).
`--corpus-kind toy` reproduces the tiny earlier task for controlled checks.

Raw context/target records are loaded into a superposition; a quantum context
oracle and fixed context-amplification circuit retrieve candidate values.
No CPU conditional-count table compiles the quantum probabilities. The same
target is repeatedly replaced using quantum-controlled swaps and a retained
environment. For beta=1 the complete reverse circuit is independent of the
held-out target label. CNOT copies correlate basis values; they do not clone
arbitrary unknown quantum states.

A direct classical empirical conditional table and its matched replacement
channel are separate references, never inputs to quantum inference. Every
configuration runs noiselessly and with the requested readout-bit flips on
**the identical compiled circuits** and cases. Noise affects output bits, not
the query-success flag. Out-of-vocabulary readout errors remain in the
probability denominator. Unseen contexts abstain; there is no CPU fallback.
Reports retain coverage, acceptance, raw counts, source records and costs.

## Coherent text generation

```bash
python -m qcenn_qml.smol135 --prompt 'the cat' --new-tokens 3 \
  --shots 1024 --output-dir results/quantum-text
```

Raw corpus records supply a one- or two-token memory. Later queries are
controlled by previous **quantum** token registers. Every generated token and
success flag is measured only at the end, with no CPU token-state update.
Final conditioning on all query flags can bias the earlier-token distribution;
this is explicitly reported. It is a small corpus-memory model, not SmolLM2.

## Quantum attention and full-Smol feasibility

```bash
python -m qcenn_qml.quantum_attention --shots 8192 --output-dir results/attention
python -m qcenn_qml.smol_feasibility --attention-report results/attention/report.json \
  --output-dir results/smol-audit
```

The attention component supports signed ternary queries, keys and values.
Toffoli-controlled reversible arithmetic computes signed dot products into a
score register. A data-independent exponential-function rotation implements
success amplitude exp((score-bound)/(2 sqrt(d))). Postselection normalizes
index probabilities to softmax(q dot k / sqrt(d)). Score and key workspaces
are uncomputed; values are loaded quantum mechanically. Final measurements
estimate weighted value means. No CPU score/softmax is a model prediction;
NumPy references are calculated only after measured execution for validation.

The universal function-rotation table is fixed by the score domain, independent
of the query/key data, like fixed arithmetic gates. This is a small-domain
implementation, with its table synthesis cost charged. It does not provide an
efficient arbitrary-precision exp oracle or a coherent hidden-vector output.

The audit uses the actual [SmolLM2-135M configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json):
30 layers, width 576, 9 query heads, 3 KV heads, MLP width 1536 and vocabulary
49,152. Shape counting gives 134,515,008 parameters with tied embeddings.
An explicit resident 16-bit basis encoding would require **2,152,240,128
logical weight qubits**, before workspace or error correction. This is one
encoding estimate, not a universal lower bound or an impossibility theorem.
QROM can trade registers for loading circuits; loading is not free.

Full-precision projections, RoPE, RMSNorm, SiLU MLPs, a coherent weighted-sum
hidden vector, the full vocabulary head, cache and all 30 composed layers are
unimplemented. No actual pretrained weights or physical QPU jobs execute in
the audit. It reports **NOT_IMPLEMENTED_NOT_DEMONSTRATED** for a full quantum
Smol model, even when the small attention component passes. Classical MAC
counts in the ledger are not measured quantum-gate counts.

## Historical experiments

The previous measured results remain intact. Reproduce them explicitly with
`recurrent_mnist --legacy-hybrid`, `smol135 --legacy-classical-memory`, or
`diffusion.benchmark --legacy-count-table`. CPU SPSA training or count-fitted
gates remain honestly labeled; they are not the new defaults. k=1 overlap
scoring is available only through explicit experimental arguments.

QCeNN-QML — Vahid Tavakkoli (2026), https://github.com/vtavakkoli/QCeNN-QML.
QCeNN-MI inspiration — Vahid Tavakkoli and Kabeh Mohsenzadegan (2026),
https://github.com/vtavakkoli/QCeNN-MI. The active custom non-commercial / paid
commercial LICENSE and independent upstream citation obligations remain.
