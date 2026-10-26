# Gate-only model-core execution evidence (v0.3)

These are small **Aer simulator** checks, not a QPU run, accuracy advantage,
quantum neural-weight training, or a converted pretrained Smol model. The new
models are nonparametric raw-record memories with no fitted CPU prediction
weights or classical optimizer. Input loading, compilation and final decoding
remain classical; the entire host program is not purely quantum.

## Original-MNIST tiny retrieval gate

One seed (42), ten training images, ten validation images, ten test images;
4x4 images, two overlap-product updates, zero damping, 1,024 shots per circuit.

| Split/control | Measured accuracy | Accepted shots |
|---|---:|---:|
| Tiny training memory | 90% | 57.7% |
| Validation | 40% | 60.0% |
| Validation k=0 reference | 40% | 76.2% |
| Test | 20% | 59.2% |
| Test k=0 reference | 20% | 76.1% |
| Test refreshed-index control | 20% | 57.6% |

The configured 0.9 retrieval gate passes; this demonstrates memorization, not
neural fitting or useful generalization. The test result does not improve on
k=0. These tiny selections differ from the earlier three-seed neural benchmark;
their percentages must not be presented as a like-for-like improvement.

```bash
python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 --side 4 \
  --rounds 2 --shots 1024 --fit-threshold 0.9 --include-reset-control \
  --output-dir results/reproduce-quantum-mnist
```

## Training/validation diagnostics before further benchmarking

All diagnostics use the same seed, source IDs, 4x4 features and two rounds.
The 1,024-shot settings compare encoding/damping at matched shot budgets.
No test inference is performed by these `--validation-only` runs.

| Encoding | Damping | Shots | Tiny training retrieval | Validation |
|---|---:|---:|---:|---:|
| Amplitude | 0 | 256 | 80% | Gate failed; skipped |
| Amplitude | 0 | 1,024 | 90% | 40%; k=0 also 40% |
| Angle | 0 | 1,024 | 80% | Gate failed; skipped |
| Amplitude | 0.05 | 1,024 | 50% | Gate failed; skipped |
| Angle | 0.05 | 1,024 | 70% | Gate failed; skipped |

These small results support keeping damping disabled on this split and show
that the retrieval gate is sensitive to shot budget. They do not establish a
general ranking of encoders. Angle success checks use a clean Toffoli ladder;
damping uses fresh retained environments. Both charge their extra qubits.
No optimizer is used or claimed: quantum neural-weight optimization remains
unimplemented. No larger benchmark follows from these checks.

```bash
python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 --side 4 \
  --rounds 2 --encoding amplitude --damping 0 --shots 1024 --validation-only \
  --output-dir results/reproduce-mnist-validation
```

## Broader held-out text and matched noise

The authored small English corpus has 48 training windows, 16 held-out windows
from disjoint sentences, and 18 vocabulary entries. Four held-out windows are
selected using the recorded seed. Two have unseen contexts and abstain.
The direct classical conditional table scores 50% top-1 and 50% mean target
probability. No classical table supplies the quantum gate probabilities.

| Replacement steps | Noiseless correct-token frequency | Readout-noisy frequency |
|---|---:|---:|
| 0 | 0.0% | 0.0% |
| 1 | 31.8% | 26.7% |
| 3 | 47.5% | 38.5% |

Both modes use the same native circuit and cases, 256 shots per case, beta=1,
eta=0.65 and readout flip probability 0.05. Fractions average cases; unseen
contexts score zero. Invalid noisy token codes retain their probability mass.
This reproduces a conditional-memory channel, with no language-model advantage.

```bash
python -m qcenn_qml.diffusion.benchmark --corpus-kind text --shots 256 \
  --examples 4 --steps 0 1 3 --readout-error 0.05 \
  --output-dir results/reproduce-matched-text
```

## Attention and coherent sequence memory

Two signed ternary attention cases ran with 4,096 shots. Maximum absolute
attention-probability errors versus the independent softmax reference were
0.0188 and 0.0284. Arithmetic and exact statevector tests separately verify the
operator; finite-shot agreement alone is not the proof. The component samples
values, and does not produce a coherent full-precision Smol hidden vector.

Default raw-corpus coherent generation produced `ran on`, with 287/1,024 shots
accepted. Previous quantum token registers control later queries; tokens are
measured only at the end. This is a small corpus-memory demonstration, with
whole-sequence postselection bias, not a pretrained or trained neural LLM.

## Full Smol status and provenance

The full Smol audit reports **NOT_IMPLEMENTED_NOT_DEMONSTRATED**. The public
configuration implies 134,515,008 parameters. The explicit 16-bit resident
basis-encoding estimate is 2,152,240,128 logical weight qubits, before workspace;
it is not a lower bound on all quantum encodings. Pretrained weights, all
transformer operators and quantum neural-weight training were not executed.

`summary.json` records the detailed resources, limitations and configuration.
`raw-evidence.tar.gz` preserves all raw reports, memories, split IDs, QPY examples
and the combined audit. Verification recalculated 158 readouts, reloaded 20 QPY
examples and checked 160 source-file fingerprints. Runtime local commit
`bbbf0be` and public core commit `9dc7a43` have identical Git trees; source hashes
are checked against the archived public-core and current source snapshots.
The final validation runs include the clean success ladder and retained damping
environments. JSON reports preserve their original runtime provenance.

```bash
sha256sum -c evidence/quantum-core-v03/raw-evidence.tar.gz.sha256
mkdir -p results/extracted-evidence
tar -xzf evidence/quantum-core-v03/raw-evidence.tar.gz -C results/extracted-evidence
python scripts/verify_quantum_evidence.py results/extracted-evidence
```

57 software/operator tests pass. Classical input preparation and host control,
numerical MPS simulation (threshold 1e-20, direct measurement sampling), tiny
datasets and postselection costs are explicit limits. No QPU performance,
fully quantum host pipeline or quantum advantage is claimed.

QCeNN-QML — Vahid Tavakkoli (2026), https://github.com/vtavakkoli/QCeNN-QML.
See LICENSE and CITATION.cff for mandatory attribution/research citation and
the separate paid commercial-license requirement.
