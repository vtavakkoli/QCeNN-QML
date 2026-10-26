# Staged quantum studies (v0.4)

QCeNN-QML — Vahid Tavakkoli (2026). See LICENSE and CITATION.cff.
QCeNN-MI inspiration and separate attribution to Vahid Tavakkoli and Kabeh
Mohsenzadegan remain in NOTICE.md. These modules are independently written.

Execution is Qiskit Aer gate simulation. There is no fitted CPU prediction
head, classical optimizer or intermediate measured hidden-state update in the
default models. Input preparation, compilation, numerical simulation, validation
selection of experimental settings and final count decoding remain classical.
The entire host program is not purely quantum. MNIST/diffusion are raw-record
memories; the block has fixed demonstration weights. Neural-weight training,
pretrained Smol execution, physical QPU execution and advantage are not claimed.

## Parallel run, status and archives

```bash
QCENN_CPUS=10 QCENN_JOBS=3 \
nohup bash -c 'exec bash scripts/run_research_studies.sh' \
  > research-studies.log 2>&1 < /dev/null &
bash scripts/run_research_studies.sh --status
tail -f research-studies.log
```

Omit CPU variables for affinity/cgroup-aware defaults. Three independent tracks
share the budget without nested BLAS pools; studies parallelize independent
configurations/data seeds inside their share. Fit-before-validation and
freeze-before-test remain sequential. Each track gets a log, command, status,
native ZIP and checksum. The Smol audit starts when the block finishes; the
summary waits for all selected jobs. States distinguish DONE,
DONE_WITH_BLOCKED_TESTS (exit 2), FAILED and INTERRUPTED. Execution success is
not scientific success. Study logs print progress every 30 seconds; per-task
worker logs and per-image JSON give finer progress.

Use `--study-only mnist`, `diffusion` or `transformer`. `--smoke` uses digits-proxy
and small structured-text budgets. A smoke can correctly stop at a failed gate.
Full defaults can take substantially longer than ten-image demonstrations.

Keep a running study's checkout unchanged. With an older run still active:

```bash
git fetch origin
git worktree add --detach ../QCeNN-QML-studies origin/main
QCENN_PYTHON="$PWD/.venv-qcenn/bin/python" \
PYTHONPATH="$PWD/../QCeNN-QML-studies/src" \
QCENN_DATA_ROOT="$PWD/data" QCENN_CPUS=10 QCENN_JOBS=3 \
nohup bash -c 'exec bash ../QCeNN-QML-studies/scripts/run_research_studies.sh' \
  > results/research-studies.log 2>&1 < /dev/null &
bash ../QCeNN-QML-studies/scripts/run_research_studies.sh --status
```

## MNIST protocol

Defaults: original checksummed MNIST; data seeds 42/43/44; independent
measurement seeds 101/202/303; shot budgets 1024/4096/16384; two validation
images per digit; 100 fresh test images per digit per data seed. Validation
images/labels are fixed across candidates. Memory records are nested at 1/2/3
per digit, with a seeded address permutation so damping is not evaluated on
class-sorted addresses. The logical width limit is 128.

1. Convergence: amplitude encoding, side 8, damping 0, rounds 1/2/4.
2. Memory: repeat those rounds for 20 records, then 30 records.
3. Ablation: best eligible memory/rounds, amplitude side 8/4, angle side 4,
   damping 0/0.05. Match amplitude side 4 when interpreting angle encoding.
4. Freeze validation selection, then test that configuration and k=0 reference
   on reserved fresh images. k=1 is disabled throughout.

Every configuration must first achieve 100% finite-shot retrieval of every
stored training image. Retry budgets 1024/4096/16384 are declared beforehand.
Failure preserves counts and blocks that candidate's validation/test. This is
memorization, not neural training. Expansion/ablation requires an eligible
convergence candidate; failed expanded memories remain visible in reports.

Equal-shot comparisons give each image the declared shots. Equal-CNOT uses the
same hard per-image budget for **all** memory/encoding/round candidates: the
native CX count of the ten-record, amplitude, one-round k=0 anchor multiplied
by the base shot budget. Shots are `floor(budget/native_CX)`; the remainder is
unspent. Rejected shots and postselection are charged. Native compilation and
budget-anchor compilation are separately recorded, amortized across repeats.

Agreement is the mean per-image modal prediction fraction across independent
measurement seeds. Eligibility requires >=0.95 agreement at that budget and
the largest budget, >=0.95 modal-prediction agreement with the largest budget,
and an accepted shot for every image/repetition. This empirical criterion is
not a convergence proof. Curves, accuracies and counts are retained. Selection
maximizes eligible equal-CNOT validation accuracy, then minimizes actual CX,
then uses deterministic configuration order. Data-seed cluster intervals are
descriptive; small validation sets and three seeds leave selection uncertainty.

One exemplar per class cannot change the exact winning class under monotone
`p**r`. Multiple exemplars use `sum(p_i**r)` and can change class rankings;
a circuit/statevector regression verifies this distinction.

Test IDs are reserved before selection and disjoint across data seeds. The
30 already inspected real-MNIST test IDs in the uploaded seed 42/43/44 archives
are excluded by `assets/evaluated-mnist-indices.json`. Add any other inspected
IDs using `--exclusions-file previous.json`, with
`{"dataset":"mnist","indices":[...]}`. Freshness is relative to declared
exclusions. Test images/labels never enter validation workers.

```bash
.venv-qcenn/bin/python -m qcenn_qml.mnist_study --phase validation \
  --jobs 6 --cpu-budget 6 --output-dir results/mnist-validation
.venv-qcenn/bin/python -m qcenn_qml.mnist_study --phase test \
  --freeze-file results/mnist-validation/frozen-configuration.json \
  --jobs 6 --cpu-budget 6 --output-dir results/mnist-fresh-test
```

Testing verifies the freeze digest, prepared-data hashes, source and package
versions. A blocked study writes no freeze/test and exits 2. Testing compares
the selected model with one-round/no-damping k=0 at identical memory/encoding,
with equal shots and equal CX, and reports paired correct/incorrect cases.
Test outcomes cannot modify selection or trigger another configuration search.

## Broader held-out diffusion

```bash
.venv-qcenn/bin/python -m qcenn_qml.diffusion.study \
  --corpus-files corpus-a.txt corpus-b.txt corpus-c.txt \
  --data-seeds 42 43 44 --measurement-seeds 101 202 \
  --shots 1024 --examples 0 --readout-errors .01 .05 \
  --max-train-records 128 --max-vocabulary 32 \
  --jobs 4 --cpu-budget 4 --output-dir results/diffusion-broader
```

Files must have unique filenames. Defaults are three authored, structured
40-sentence corpora for reproducible diagnostics. They do not establish
language understanding; substantive language evaluation needs independent real
text. Tokenization is English `[a-z]+`, with three-token windows.

Identical normalized sentences stay in one partition. Split sentence groups
75/25 before windowing. Vocabulary selection uses training sentences only;
seeded training-only sampling caps loaded quantum records at 256. Corpus
snapshots/hashes, sentence IDs, sampled training-window IDs and all held-out
windows are retained. `--examples 0` evaluates every held-out window in batches.

A separate classical table receives the same raw training records and never
compiles the quantum probabilities. Every readout-error level uses identical
native gate/angle/bit-order hashes and cases, and has a matched classical channel
expectation. Readout flips are distinct from forward masking and device gate noise.
Unknown targets receive zero credit even if output is `<unk>`. Unknown contexts
use unused query addresses and abstain, preventing accidental matches to an
aggregate `<unk>` context. Invalid noisy codes retain their denominator mass.
Sentence-cluster bootstrap intervals avoid treating shots as independent text.
Repeated-measurement summaries cluster by data split; overlapping splits are
not independent corpora. Single-cluster intervals are null.

## Complete small coherent block

```bash
.venv-qcenn/bin/python -m qcenn_qml.quantum_transformer \
  --shots 256 --output-dir results/coherent-block
.venv-qcenn/bin/python -m qcenn_qml.smol_feasibility \
  --transformer-report results/coherent-block/report.json --output-dir results/smol-audit
```

Two tokens, hidden/MLP width two, one causal head, three-bit signed basis words
with one fractional bit. Gates implement pre-RMSNorm, QKV, RoPE, causal scaled
dot-product softmax, a **coherent weighted vector**, output projection,
attention residual, second RMSNorm, SwiGLU and MLP residual. Token 0 attends
only to itself; token 1 attends to both. Intermediate vectors remain quantum.
Compute/copy/uncompute clears workspace before final readout.

Signed addition/multiplication, floor arithmetic shifts and two-complement
modular overflow are explicit. Norm epsilon is 0.25; inverse-RMS and softmax
use two fractional bits. Input-independent scalar tables implement inverse
square root, sigmoid/softmax and SiLU; no full input/output block table is
compiled. Their exponential precision/table cost is charged. These functions
operate reversibly on **basis words** with retained inputs, not nonlinearly on
arbitrary quantum amplitudes. RoPE angles 0/pi/2 and the reported fixed toy
weights differ substantially from Smol precision and position frequencies.

The integer reference runs after measured inference. Reports/tests verify
basis/superposition outputs, functional phase kickback, clean workspace, extreme
signed inputs, causal independence and final-only measurements. The block uses 241 logical qubits
and approximately 175,000 native CX gates per execution in the local diagnostic;
these costs do not establish near-term hardware feasibility.

The audit records tiny-block coverage separately. Full Smol precision, GQA,
135M pretrained weights, vocabulary head, KV cache, 30-layer composition,
physical hardware and quantum-only neural training remain absent. Full-model
status stays `NOT_IMPLEMENTED_NOT_DEMONSTRATED`.

## Checkpoints and reproducibility

Studies archive recursive source hashes, including diffusion modules and
bundled assets; package versions; settings; and seed hierarchies. Runtime
versions are in `requirements-versions.txt`. Workers refuse changed source.
Hashed freezes link tests to validation and prepared data. Completed task
receipts hash results; atomic count files preserve partial work.

Resume a Python study with its **same original arguments** plus `--resume`.
Settings/source/package changes require a fresh directory. MNIST reuses raw
readouts, never analytic reconstructions. Diffusion recompiles and checks native
identities before reusing counts. Bash creates fresh batch roots; resume a
recorded Python command, then archive its directory with native `zip`.
Termination is forwarded to inner worker process groups. CI uses small proxy
studies; it is not original-MNIST or natural-language benchmark evidence.
