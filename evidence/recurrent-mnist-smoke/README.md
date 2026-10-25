# Original-MNIST execution evidence

This is a small end-to-end execution check of the v0.2 implementation, run on
original MNIST with all ten digit classes. It is **not** a publication-quality
accuracy study. The recurrent model does not improve on the reference methods
in this run. No physical QPU was used.

Each seed selects 200 training images, 50 disjoint validation images, and 20
test images from the official test source. Classical SPSA-Adam runs for 150
steps with batch size 32. Final quantum inference uses 1,024 shots per circuit.
The memory-reset control is independently retrained with the same optimizer
budget. The overlap baselines use one training exemplar per class and therefore
ten times as many inference circuits/shots as the recurrent method.

| Method | Seed 42 accuracy | Seed 43 accuracy | Mean |
|---|---:|---:|---:|
| Measured recurrent classifier | 15% | 10% | 12.5% |
| Measured memory-reset control | 10% | 5% | 7.5% |
| Classical normalized centroid | 95% | 80% | 87.5% |
| Classical exact exemplar overlap, k=0 | 60% | 55% | 57.5% |
| Measured quantum exemplar overlap, k=0 | 60% | 55% | 57.5% |
| Classical exact Grover response, k=1 | 0% | 0% | 0% |
| Measured quantum Grover response, k=1 | 0% | 0% | 0% |

The k=1 outcome is retained: the nonmonotonic Grover response reverses useful
similarity rankings in these fixtures. Accuracy agreement between the measured
and classical overlap baselines is not a quantum advantage. The small difference
between recurrent and reset-control accuracy does not establish useful learned
memory or a statistically supported improvement. Hyperparameters were not
retuned after seeing these test results.

`summary.json` includes per-seed Wilson intervals, acceptance, resource totals,
configuration and source/package fingerprints. `raw-evidence.tar.gz` preserves
the full JSON reports with every prediction and measurement count, split IDs
and hashes, trained quantum parameters and actual native QPY examples. Source
hashes were verified against the tested working tree, and all four exported
native QPY examples were independently reloaded. The recorded Git parent and
dirty status describe that working tree before the implementation commit.

Reproduce in a fresh directory:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 43 \
  --train-per-class 20 --validation-per-class 5 --test-per-class 2 \
  --exemplars-per-class 1 --training-steps 150 --shots 1024 \
  --include-reset-control --output-dir results/recurrent-validation
```

Counts are seeded; exact cross-platform floating-point and timing identity are
not guaranteed. This evidence establishes executable measured inference and
reproducible comparisons. Architecture/training improvements require subsequent
validation-selected experiments.
