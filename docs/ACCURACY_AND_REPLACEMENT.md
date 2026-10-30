# MNIST accuracy and complete small transformer replacement — v0.6

QCeNN-QML — Vahid Tavakkoli (2026). See LICENSE, NOTICE.md and CITATION.cff.

## Scope and accuracy target

100% fresh-test MNIST accuracy is a research target, not a promised result.
The implementation never selects easy test images, substitutes memorization
accuracy, or tunes after seeing test outcomes. Measured errors, abstentions,
Wilson intervals and resource costs remain visible. Loading raw examples is
nonparametric memory preparation, not quantum neural-weight training.

## MNIST protocol

The active runner enables `mnist_study --accuracy-profile`. Convergence and
10/20/30-record memories precede fixed preprocessing and readout selection.
Centering and moment deskew are classical input encoding; neither fits weights
nor uses labels. Original encoding remains a candidate. Nearest-record readout
measures memory addresses, labels and overlap flags. A class score is its largest
measured record frequency, instead of a sum over class records. No CPU distance
or analytic probability is used to predict. Ties use the lowest class ID.

Memory then grows to 40/80/160 raw examples in balanced 20-record quantum banks,
each containing two records per digit. Equal shots in equal-size banks preserve
the uniform record prior. Raw bank counts and every bank's cost are retained.
For B banks and native CX counts C_j, equal-shot execution assigns floor(S/B)
shots per bank. Equal-CNOT assigns floor(C_anchor*S / sum(C_j)) per bank. All
rejected shots are charged; integer remainders remain unspent. Every candidate
retains the identical ten-record, one-round amplitude budget anchor.

Every stored image must be retrieved correctly before that candidate evaluates
validation. Fit remains 100%; stability remains at least 95%. A 1e-12 absolute
tolerance handles numeric representation at the boundary, not lower agreement.
Selection is validation-only. A hashed freeze precedes tests of the selected
model and k=0 under equal-shot and equal-CNOT budgets. k=1 remains disabled.

Full-run defaults use 4096/16384/65536 shots, five validation images per digit
per seed, and 100 fresh test images per digit per seed: 3,000 test images across
seeds 42/43/44. Prior declared test IDs are excluded. Maximum planned memory is
reserved before selection so growth cannot consume validation images. Override
`--growth-memories 4 8` for a smaller declared campaign. Damping is off by default.

Accuracy-profile validation schedules independent image/mode/budget/seed tasks.
A locked, hashed QPY cache shares compilation across measurements. Failed tasks
update the top-level scientific report to FAILED instead of leaving VALIDATING.
Existing immutable source/configuration/count receipts cannot be relabeled.

## Complete small transformer replacement

The new model includes quantum token embeddings, one/two complete coherent
blocks, and a quantum four-class vocabulary projection/argmax. Fixed logits are
x+y, -x+y, x-y and -x-y. Their lowest-ID argmax equals the two sign bits, copied
by gates. The transformer hidden words use the original three-bit modular
arithmetic; the head's declared mathematical logits do not wrap.

The circuit computes embeddings and hidden layers, copies vocabulary outputs,
then uncomputes every intermediate register. Clean workspace is shared across
layers. Width is 249 qubits for one block and 261 for two; neither is sent to a
dense statevector backend. All block operators execute: RMSNorm, QKV, RoPE,
causal softmax, weighted values, output projection, residuals and SwiGLU.

Tests require all 16 input pairs across independently scheduled two-branch
coherent groups, with correct outputs and clean workspace, across independent
measurement seeds. Coverage must contain every input exactly once per layer/seed.
Small groups bound simulation entanglement and expose more parallel work. Their
additional circuit executions and shots are fully charged. The one-block smoke
retains the already qualified 16-branch diagnostic; use --coherent-batch-size 16
explicitly for that expensive full diagnostic. Direct runs also cap workers by
available RAM using the same 4 GiB estimate as the shared runner.
Separate functional phase-kickback/inversion tests verify relative coherence.
Native QPY, measured counts, instrument audits and resources are retained.

An explicitly hybrid host separately generates two tokens. Each full forward
uses quantum gates; the host decodes its final token and updates context before
the next call. It never substitutes CPU attention or MLP predictions. These
classical between-token updates are labeled hybrid in reports.

This is a four-token vocabulary, two-token context, width-two, three-bit model
with fixed toy weights. Its independent classical comparator is never an
inference fallback. Full pretrained Smol, neural training, useful language
quality, gate-noise robustness, physical QPU execution and advantage remain
unimplemented or undemonstrated.

## Retired diffusion and GPU recovery

Diffusion is absent from active launchers, automatic WikiText downloads,
system-check suggested commands and CI benchmark runs. Historical source,
regression tests and negative evidence remain for reproducibility. Active
`--study-only diffusion` is rejected.

Aer 0.18.2 GPU selection is passed to `AerSimulator.run(target_gpus=[id])`, not
the constructor that rejected it in the uploaded runs. A real CPU API regression
checks this transport without claiming CUDA qualification. Actual GPU/native/
noise qualification still runs on the execution machine before GPU use.

Use the commands in [README.md](../README.md) or the integrated updater. Results
remain inside the original project's `results/`, while source and environments
use an immutable sibling worktree. Each track has a separate verified ZIP.
