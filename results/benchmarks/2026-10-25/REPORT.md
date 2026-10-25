# QCeNN-QML — measured research report (25 October 2026)

**Evidence:** [passing GitHub Actions workflow 40178235764](https://github.com/vtavakkoli/QCeNN-QML/actions/runs/40178235764) at source commit f12610b4dbab9ec2b21ff885359850ecb8f608d7. All described quantum runs are **classically simulated quantum circuits** (Qiskit Aer), not physical quantum-computer executions. Full raw machine-readable results are in the workflow's **qcenn-qml-all-simulator-results** artifact. The archived original-MNIST report is independent historical evidence under evidence/recurrent-mnist-smoke/.

## 1. New quantum diffusion experiment

**Status:** implemented and validated as a small conditional masked-token denoising *quantum channel*; **not a trained diffusion LLM**.

**Setup:** Three-token toy sequences; vocabulary cat, dog, sat, [MASK]; 36 training and 12 held-out toy triples, four held-out cases sampled for CI. Six data/context qubits and three environment/flag qubits (nine total). The forward masking probability is beta=1.0, so the middle token is fully hidden; the reverse replacement probability is eta=0.65 per iteration. Each configuration measures 256 shots per example, 1024 shots in total, with data/simulator seed 42. The reverse quantum circuits preserve the target register across steps and use resettable ancillas; conditional output amplitudes are synthesized from classical counts of the toy triples (not quantum-trained weights). Device noise is not used, only engineered replacement-channel dissipation and a **separate optional simulated readout error**.

| Denoising steps | No readout error: measured correct-token fraction | Independent ideal-channel expectation | 5% readout bit-flip: measured correct-token fraction | No-error transpiled depth | Readout-noisy transpiled depth |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.00% | 3.22% | 3 | 9 |
| 1 | 62.89% | 61.57% | 57.42% | 1781 | 2675 |
| 3 | 88.28% | 90.66% | 80.76% | 5337 | 8007 |

**Interpretation:** Iterated quantum replacement moves the masked state toward a **precomputed conditional token distribution**, as independently predicted by the CPTP-channel formula. It is not evidence that uncontrolled quantum noise helps. The simulated **readout error degrades** correct-token recovery after denoising, by about 5.47 and 7.52 percentage points at steps 1 and 3. A nonzero correct-token readout at step 0 with readout noise is a bit-flip artifact, not model reconstruction.

With only four selected held-out cases and 1024 shots per configuration, the observed top-1 case accuracy after one and three steps (4/4 in both settings) is not a general language benchmark. The shot-only normal-approximation 95% intervals, which omit dataset uncertainty, are approximately 59.93–65.85% (1 step) and 86.31–90.25% (3 steps) with no readout error; 54.39–60.45% (1 step) and 78.35–83.18% (3 steps) with 5% readout noise. Compiled depth is backend/transpiler dependent; differing depth numbers under the noise model are not direct hardware costs.

**Causal caution:** Correct-token probability can rise even for an equivalent classical stochastic replacement process, since the toy conditional probability table is estimated classically. No quantum advantage or improved language-learning capacity is demonstrated.

## 2. Original ten-class MNIST recurrent results (archived independent run)

The archive evidence/recurrent-mnist-smoke/summary.json records real ten-class MNIST results for two seeds (42 and 43), 200 training, 50 validation, and 20 test images per seed, 150 classical optimizer steps, and 1024 shots per measured circuit. This original-MNIST run uses hybrid training (SPSA-Adam), but measured quantum readout at test time.

| Method | Mean test accuracy across two seeds |
|---|---:|
| Recurrent measured quantum classifier | 12.5% |
| Measured reset-channel control | 7.5% |
| Measured quantum exemplar overlap (k=0) | 57.5% |
| Measured Grover overlap (k=1) | 0.0% |
| Classical normalized centroid | 87.5% |

**Conclusion:** The recurrent classifier currently performs poorly compared with the classical centroid. Grover k=1 produces a nonmonotonic response and reverses useful rankings in these fixtures. This evidence does not validate recurrent quantum advantage.

The newer CI smoke run is separate: it tests two **sklearn digits-proxy** classes, one test image per class, and achieves 2/2 accuracy. That figure is a software smoke check and **must not be labeled MNIST performance**. The digits-proxy recurrent CI also ran with only one training step per seed; it is not a quality benchmark.

## 3. SmolLM2-135M-inspired quantum memory

The CI quantum autoregressive toy-memory experiment (six qubits, eight-token vocabulary, order-one context, 512 shots per generated token) generated **"sat on"** from prompt **"the cat"**. Its corpus is 352 toy tokens. No pretrained SmolLM2-135M weights ran as quantum gates and no real SmolLM2 classical teacher was used in this CI. This is an amplitude-encoded quantum n-gram associative-memory prototype, **not a 135-million-parameter quantum LLM** and not a diffusion LLM.

## 4. Verification, cost and limitations

- 29 unit/integration tests passed in the CI run, plus example MNIST-proxy, toy-language, ten-class recurrent-proxy, and clean/noisy diffusion executions.
- Tests and experiments use Qiskit Aer; **no QPU jobs** were submitted in this work.
- Quantum denoising is a contextual **state-replacement channel**, not a learned quantum neural denoiser; classical conditional-frequency estimation and quantum gate compilation are substantial.
- Dissipative reset operations are quantum-device channels, but this is not a fully coherent, measurement-free, unitary-only implementation.
- A full diffusion LLM requires multi-mask generation, learned conditional denoising, richer corpora, statistically meaningful held-out evaluation, matched classical/quantum baselines, noise sensitivity, and resource accounting.

## Reproduction

    python -m pip install -e '.[test]'
    python -m pytest -q
    python -m qcenn_qml.diffusion.benchmark --shots 256 --examples 4 --beta 1 --eta 0.65 --steps 0 1 3 --output-dir results/ideal
    python -m qcenn_qml.diffusion.benchmark --shots 256 --examples 4 --beta 1 --eta 0.65 --steps 0 1 3 --readout-error 0.05 --output-dir results/noisy
    python -m qcenn_qml.smol135 --new-tokens 2 --shots 512 --output-dir results/smol
    python -m qcenn_qml.report --mnist results/ci/mnist/report.json --smol results/ci/smol/report.json --diffusion results/ci/diffusion/report.json --diffusion-noisy results/ci/diffusion-noisy/report.json --output-dir results/ci/summary

Results and detailed JSON/Markdown are downloadable from the corresponding GitHub Actions artifact; a local reproduction should use fresh output directories. The original-MNIST historical evidence is documented separately.

**Research priority:** integrate the QCeNN-MI-style residual and polynomial recurrence more directly into a trainable conditional quantum diffusion architecture, and test learning beyond a classically prepared n-gram count table.
