# Recurrent ten-class MNIST

This is the **historical v0.2 hybrid neural experiment**. Use
`--legacy-hybrid` explicitly. The current default quantum-memory model and
its tiny retrieval gate are described in [QUANTUM_CORE.md](QUANTUM_CORE.md).

`python -m qcenn_qml.recurrent_mnist` trains a small quantum recurrent classifier
and evaluates it with explicit Qiskit Aer circuits and finite-shot measurements.
It has four persistent memory qubits, one reset/reused environment qubit, and
19 trained quantum gate parameters. The inference circuit contains no trained
classical prediction head. Training is **hybrid**: classical SPSA-Adam minimizes
cross-entropy using an exact NumPy reduced-density simulator.

## Run

Python 3.10+ is supported. The tested environment uses Python 3.12, NumPy 2.3.5,
Qiskit 2.6.1, Aer 0.18.0, Pillow 12.3.0, pytest 9.1.1 and scikit-learn 1.8.0.
For that environment, install `requirements-tested.txt` before the editable
package. Original MNIST is downloaded as checksummed IDX archives from the
torchvision mirror; PyTorch is not required. `--no-download` requires the
archives in `data/MNIST/raw/`. A missing or corrupt MNIST dataset fails loudly.

```bash
python -m pip install -e '.[mnist,test]'

# Small genuine-MNIST end-to-end smoke, including both measured overlap baselines.
python -m qcenn_qml.recurrent_mnist --legacy-hybrid --dataset mnist --seeds 42 \
  --train-per-class 2 --validation-per-class 1 --test-per-class 1 \
  --exemplars-per-class 1 --training-steps 5 --shots 256 \
  --output-dir results/recurrent-smoke

# Larger comparison, independently retrained across three seeded splits.
python -m qcenn_qml.recurrent_mnist --legacy-hybrid --dataset mnist --seeds 42 43 44 \
  --train-per-class 100 --validation-per-class 20 --test-per-class 20 \
  --exemplars-per-class 2 --training-steps 500 --shots 1024 \
  --include-reset-control --output-dir results/recurrent-comparison
```

Every run requires a fresh output directory. The second command is substantially
more expensive: each overlap method executes 4,000 circuits per seed, versus
200 for the recurrent classifier. Both use the specified shots **per circuit**;
these are not equal total-shot or total-gate comparisons. `--skip-quantum-overlap`
retains classical exact overlap references and measured recurrent inference for
faster development. `--side 4` is a cheaper resolution ablation, while the
default `--side 8` streams 16 patches. Any choice of resolution or hyperparameters
must be made with training/validation data, before examining test scores.

For offline CI only, explicitly choose `--dataset digits-proxy`. That dataset
is identified in every manifest and report and is never substituted for MNIST.

## Quantum cell and readout

Resize an image to 8x8 and divide intensities by 255. Stream its sixteen
nonoverlapping 2x2 patches in row-major order; each patch supplies four angles.
Initialize the memory once in `|0000>`. For each patch:

1. Apply input-driven RY and RZ rotations to each memory cell.
2. Apply RXX and RZZ interactions on the ring 0–1–2–3–0. The seven cell
   parameters are shared across all memory sites and all timesteps.
3. Couple each memory cell to the environment, then reset only the environment.
   The memory and its inter-cell correlations remain quantum.

After the last patch, a trained quantum rotation/CX readout layer is followed
by four computational-basis measurements. Outcomes 0..9 identify digit classes.
Outcomes 10..15 are rejected; the report includes the accepted-shot fraction.
Classical counting and argmax decode the measured histogram. An example with no
accepted shots predicts -1 and counts as an error. There is no ideal-probability
or classical-classifier fallback. The training objective uses **unconditional**
class probabilities, so putting population in rejected outcomes is penalized.

## Physical channel and connection to QCeNN-MI

QCeNN-MI — Vahid Tavakkoli and Kabeh Mohsenzadegan (2026),
https://github.com/vtavakkoli/QCeNN-MI . This is an independently written local
specialization inspired by its recurrent collision/reset construction. It is
distinct from its dense matrix-inversion residual solver and its polynomial
and Chebyshev filters. No upstream source has been copied or relicensed.

For one cell, let gamma = pi/(2(1+exp(-theta_d))) and define

    K0 = diag(1, cos(gamma)),
    K1 = sin(gamma) |0><1|.

K0†K0 + K1†K1 = I proves that the local map is completely positive and trace
preserving. The circuit realizes these Kraus operators by CRY(2 gamma) from
memory to an environment initially in zero, followed by CX from environment
to memory and environment reset. Tensoring the local Kraus operators with
identity preserves the other cells' correlations. Composing these maps with
the input-dependent ring unitary gives the full CPTP recurrent cell. No
measurement or classically corrected state drives a later memory update.

The map is linear in the density matrix; no nonlinear quantum-state evolution
is claimed. Input-conditioned gates prevent the architecture from imposing a
single input-independent attracting state. Tests demonstrate different stationary
output distributions for different constant inputs in a solvable subfamily,
and sensitivity to earlier patches with an identical final patch. These tests
do **not** prove that all ten digit classes remain separated at convergence.
The classifier reads out after a fixed finite image sweep.

## Comparisons and evidence

For each seed, stratified training and validation examples come from the
official 60,000-image training source; test examples come from the official
10,000-image test source. Training and validation IDs never overlap. Seeds
change data selection, parameter initialization, minibatches, transpilation
and shot sampling. Test data never enter optimization or checkpoint selection.

The runner compares:

- The trained recurrent classifier, with predictions from measured circuits.
- Optional independently trained last-patch/reset control, with the same
  optimizer budget. Resetting memory before every patch is equivalent to
  processing only the final patch, so this cheaper circuit is used explicitly.
- Classical normalized centroids using the full selected training set.
- Exact classical exemplar overlap and its Grover response.
- Measured quantum exemplar overlap for k=0 and k=1, using the same selected
  exemplars, test IDs and shots per circuit. Grover scoring is nonmonotonic.

The dense amplitude loader packs its probability-tree rotations into uniformly
controlled RY multiplexors, decomposed into RY/CX gates. This gives the same
prepared amplitudes while making the baseline comparison practical. Dense
loading remains exponential in data-register width and its costs are charged.

Classical references never provide a recurrent prediction. Exact recurrent
test probabilities are labeled diagnostics and are computed only after fitting.
Within-run accuracy uses Wilson 95% intervals; paired bootstrap intervals
resample test examples conditional on fitted models and recorded shots.
Across-seed mean and sample standard deviation are reported separately. These
are distinct uncertainty scopes; test IDs may overlap between seeds.

`report.json` contains raw counts, predictions, confusion matrices, intervals,
training traces, acceptance rates and per-circuit resources. Native compilation
uses RZ/SX/X/CX/reset/measurement; both mean/max depth and shot-weighted CX/reset
totals are reported. Loading, circuit construction, compilation, training and
sampling times are separated. The inference is a classical quantum-circuit
simulation, not a physical QPU job or an end-to-end speedup demonstration.

Each `seed-N/` also contains `splits.json` with source IDs and SHA256 hashes of
selected original bytes, `exemplars.json`, `recurrent/model.json`, and an actual
native QPY circuit. Models include all 19 parameters and source/package hashes.
Use the same versions, seed and batch size when reproducing shot counts;
cross-platform floating-point and timing identity are not guaranteed.

Reload a saved model to build a measured circuit:

```python
import json
from qcenn_qml.recurrent import recurrent_circuit
from qcenn_qml.recurrent_mnist import patches_from_images

with open("results/recurrent-smoke/seed-42/recurrent/model.json") as stream:
    model = json.load(stream)
# image is one grayscale uint8 array, e.g. an original MNIST image.
patches = patches_from_images([image], model["image_side"])[0]
circuit = recurrent_circuit(patches, model["parameters"],
                            retain_memory=model["retain_memory"])
```

Run `python -m pytest -q` for complete-circuit versus reduced-state comparisons,
Kraus/CPTP checks, memory and input-conditioned convergence tests, shot readout,
seeded split/optimizer reproducibility, QPY reload and a ten-class proxy smoke.
