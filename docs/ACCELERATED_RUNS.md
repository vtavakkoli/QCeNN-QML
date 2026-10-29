# Accelerated, bounded research runs (v0.5)

QCeNN-QML — Vahid Tavakkoli (2026). See LICENSE, NOTICE.md and CITATION.cff.

The uploaded LRZ snapshot from 28 October 2026 exposed 160 physical cores / 320
logical CPUs, about 8 TB available memory and eight 24 GB NVIDIA A30 cards. Three
diffusion workers were each using about one CPU core despite having 322 threads.
This was evidence of active computation and poor resource use, not a deadlock.
The installed Aer exposed CPU only; installed project metadata was also stale.
These are observations from that snapshot, not a current reservation or timing.

## Update and run without changing an active checkout

From the original repository:

```bash
git fetch origin
qcenn_updater=$(mktemp)
git show origin/main:scripts/update_and_run_research.sh > "$qcenn_updater"
nohup bash -c 'exec bash "$1" --replace-run "$2"' qcenn-run \
  "$qcenn_updater" results/OLD_RUN_DIRECTORY \
  > accelerated-research.log 2>&1 < /dev/null &
```

Replace `OLD_RUN_DIRECTORY` with the exact old output directory, or omit
`--replace-run` and its argument to leave an old run alone. The script verifies
owner, process identity/start time and output/task paths before stopping that
experiment tree. It does not use name-based `pkill` and preserves old results.

The updater creates an immutable Git worktree and separate CPU/GPU environments.
It installs matching project metadata, Qiskit 2.6.1 and Aer 0.18.2, runs the full
test suite, tries GPU qualification, prepares text snapshots, then starts a new
output directory. It does not upgrade system Python, drivers or the live venv.
GPU setup/qualification failures are logged and leave CPU execution available.
The CPU and GPU Aer wheels are never overlaid in one environment.
GPU package commands and the transport interpreter use Python isolated mode
(`-I`) so CPU-project metadata on `PYTHONPATH` cannot contaminate GPU checks.
Creation, installation, dependency-check and freeze failures all retain CPU fallback;
dependency-check output is saved in `gpu-check.log`.

```bash
bash "$qcenn_updater" --status
bash "$qcenn_updater" --stop results/NEW_OUTPUT_DIRECTORY
bash "$qcenn_updater" --resume results/NEW_OUTPUT_DIRECTORY
```

Resume uses the run's saved source, environment and backend policy. Scientific
settings, source hashes, prepared inputs and runtime versions must match.
CPU/worker limits can be reduced with `--cpus N --jobs N`. v0.4 outputs are not
silently migrated or relabeled as v0.5 measurements; start a fresh v0.5 run.

For an initial diagnostic, use `--smoke --cpu-only`. Smoke uses the explicitly
labeled digits proxy and small text data. `--text-source bundled` selects the
previous authored diffusion corpora; the full updater defaults to WikiText.
Direct `run_research_studies.sh` uses bundled diffusion unless supplied corpus
files. `QCENN_LEGACY_RUNNER=1` selects the old static runner explicitly.

## Resource use and sampling

All tracks share one kernel-locked worker pool. `QCENN_CPUS`/`--cpus` bounds CPU
use; `QCENN_JOBS`/`--jobs` now bounds the **total** one-thread workers, not the
number of outer tracks. Defaults use affinity, ancestor cgroup/scheduler limits
and an 80% memory allowance with a 4 GiB worker estimate. Each worker is bound to
its leased CPU before NumPy/Qiskit imports. This also gives newly allocated data
NUMA first-touch locality. Rayon, BLAS and nested OpenMP pools are capped.

MNIST gates, validation and frozen-test images are independent image batches;
diffusion splits corpus/seed/step/window work into independent batches. Complete
splits/raw text are prepared once per corpus/data seed. A finished track releases
its slots to the remaining tracks. Fit-before-validation and freeze-before-test
dependencies remain sequential.

The tested Aer 0.18.2 MPS kernel has a fast full-register shot sampler. The runner
adds final Z measurements of otherwise unobserved qubits and discards those
auxiliary bits after sampling. Their marginal is the original readout; all native
model gates still execute and all shots are sampled. No analytic model score is
substituted for a quantum count. Extra physical measurements and execution depth
are recorded separately from the unchanged native-CX gate budget. Full instrument
counts/QPY audits are saved for the first image of each configuration/seed or text
step, with full-count hashes and original model counts for other images. Tests
cover noisy readout, unused classical bits, tensor components and 241 qubits.

Fast mode is enabled only for the inspected/tested Aer 0.18.2 implementation;
earlier versions keep the compatibility sampler. The accelerated runner fails
early on stale package metadata or unsupported pinned versions. Native source:
[MPS sampler](https://github.com/Qiskit/qiskit-aer/blob/0.18.2/src/simulators/matrix_product_state/matrix_product_state.hpp).

GPU statevector or tensor-network execution is used only when the separate GPU
environment verifies a device/method and passes actual native-memory/noise tests.
Dense statevectors have a VRAM/qubit bound; the 241-qubit block is never sent to a
dense backend. Wide unsupported circuits retain CPU MPS. A hash of the native
circuit and measurement seed chooses GPU eligibility, independent of scheduling
and readout-noise level. The saved GPU fraction defaults to verified GPUs / CPU
budget, capped at 0.5. Chosen GPU calls wait for a kernel-released GPU slot with a
finite timeout. Actual devices, methods and GPU package versions are archived;
GPU availability alone is not a speedup claim.

## Completion, timeouts and recovery

| State | Meaning |
|---|---|
| `RUNNING` | Coordinators/tasks/archives are still active. |
| `DONE` | Requested reports and measurements completed; every ZIP passed CRC and has SHA256. |
| `DONE_WITH_BLOCKED_TESTS` | Execution finished, but MNIST fit/stability gates prevented test evaluation. |
| `FAILED` | A task, report, timeout or archive check failed; partial evidence remains. |
| `INTERRUPTED` | A stop/signal interrupted execution; partial evidence remains. |

`run-state.json`, `STATUS.txt` and the terminal marker agree. Archive creation is
a separate state; file existence or a zero child exit alone does not imply DONE.
Seven separate ZIPs cover MNIST, diffusion, the block, attention, the coherent
text diagnostic, the Smol audit and the combined summary in a full run.

The updater defaults to `QCENN_TASK_TIMEOUT_SECONDS=3600`,
`QCENN_IDLE_TIMEOUT_SECONDS=1800`, and one bounded retry
(`QCENN_TASK_RETRIES=1`). Direct APIs default to a six-hour task wall limit.
CPU activity, live helper CPU time and completed measurements are distinguished;
printing a heartbeat is not counted as scientific progress. GPU waits also have
a five-minute bound. Failed simulator/compilation attempts and wall time are
retained separately from logical CX budgets for completed reported shots.

Locks are OS-held leases, not permanent marker files: a leftover filename does
not lock a finished experiment. Concurrent writers are rejected. Status reports
a dead coordinator and identifiable orphan workers explicitly; stop that run
before resuming. Stop/escalation targets only the same verified process identities.

## Scientific protocol changes and limits

MNIST still requires 100% measured tiny-memory retrieval and >=0.95 validation
stability. Equal-shot stability/accuracy selects the configuration by default;
matched-CNOT eligibility and curves are reported separately. An unstable
cost-matched result no longer blocks otherwise stable sufficient-shot memory and
encoding research. Use `--selection-mode equal_cnot` to request the former policy.
All ten-/twenty-/thirty-record candidates still pass their own fit gate before
validation; the test remains frozen and excludes declared previously used IDs.

WikiText preparation freezes three disjoint 1,000-row slices of the upstream
training split. These are slices of **one dataset**, not three independent
language collections or canonical WikiText perplexity evaluations. Whole-sentence
splits precede windows and training-only vocabulary selection. Full updater runs
evaluate 256 held-out windows per slice/data/measurement seed, steps 0/1/3 and
paired noiseless/0.01/0.05 readout noise, retaining the direct classical-table
reference and all OOV/invalid-code mass. The 128-record/32-token caps remain.
Set `QCENN_DIFFUSION_EXAMPLES` before a fresh run to change that declared sample.

Each text snapshot includes row provenance, content hashes and WikiText
CC-BY-SA-3.0/GFDL attribution. Those data licenses remain separate from the code's
custom non-commercial/paid-commercial license. See the
[dataset card](https://huggingface.co/datasets/Salesforce/wikitext) and
[public rows API](https://huggingface.co/docs/dataset-viewer/en/rows).

The coherent text track is explicitly a bounded two-token diagnostic using the
packaged small corpus. `--text-corpus-file`/`--text-new-tokens` allow larger
declared experiments in the runner, but increasing coherent history can make
simulation expensive. The complete transformer remains the two-token,
width-two, three-bit fixed-weight block. Full pretrained Smol weights, quantum
neural-weight training, physical QPU runs and quantum advantage are not claimed.
