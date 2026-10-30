# AGENTS.md — QCeNN-QML research contract

## Absolute constraints

- v0.6 accuracy research adds fixed label-independent image centering/deskew, measured nearest-record readout and balanced raw-record banks beyond 32 total records. Equal bank sizes/shot counts preserve the prior. Record rankings must use measured counts, never CPU distances. 100% fresh-test accuracy is a target, never a guaranteed result or a reason to tune on test labels.
- The complete small replacement test includes quantum embeddings, one/two full three-bit blocks and a quantum four-token vocabulary argmax, with exhaustive 16-input-pair coverage, interference checks and workspace auditing. It is not pretrained Smol. A separately labeled hybrid autoregressive HOST may update token IDs between measured full-model calls, as explicitly requested; no classical attention or MLP may substitute for the quantum forward.

- Version 0.6 defaults follow the user's no-hybrid MODEL-CORE requirement: no CPU optimizer, fitted CPU inference weights, or intermediate classical hidden-state/token updates. Classical input encoding, gate compilation, simulator execution and final counting remain and MUST be disclosed. Never describe the entire host pipeline as purely quantum.
- The accelerated runner uses OS-released CPU/GPU leases, immutable source/runtime manifests, independent image/window shards and bounded retries/timeouts. DONE requires complete requested reports plus native ZIP CRC/SHA256 receipts. Never turn a blocked scientific gate or a missing report into successful benchmark completion. Stop only verified identities belonging to the requested run tree; no broad process-name kills.
- Fast full-register MPS readout is enabled only for the inspected/tested Aer 0.18.2 implementation and marginalized to the original model bits. Preserve all native gates, actual shots, extra measurement costs and first-query full-instrument audits. GPU wheels belong in a separate environment, are qualified with actual native-circuit/noise tests and have explicit width/VRAM limits. Never allocate a dense statevector for the 241-qubit block.
- MNIST v0.6 selects eligible equal-shot validation configurations by default; matched-CNOT stability/eligibility remains an independently reported resource comparison. Do not lower the 100% fit or 0.95 stability gates. Both budget curves, validation-only selection, frozen testing and declared prior-test exclusions remain mandatory.
- New default MNIST/text/diffusion models are nonparametric raw-record quantum memories. They do NOT implement quantum neural-weight training. Do not relabel exemplar retrieval or raw corpus loading as training the old neural model or transferring Smol weights.
- Retain old experiments only as explicitly named legacy references: recurrent `--legacy-hybrid`, text `--legacy-classical-memory`, diffusion `--legacy-count-table`. Never execute the old CPU optimizer implicitly. Default overlap comparisons use k=0; k=1 requires an explicit experimental request.
- Quantum attention currently verifies signed ternary dot products and postselected softmax/value sampling. It is not a full-precision coherent transformer block. The full Smol feasibility audit must keep missing operators, loading/compilation/readout costs and the absence of actual pretrained-weight execution explicit.
- The separate v0.4 coherent transformer block implements all operators for a two-token, width-two, three-bit basis-encoded toy with fixed weights and quarter-turn positions. Keep toy coverage separate from full Smol precision, pretrained weights and quantum neural training. MNIST studies must preserve fit/stability gates, identical native-CX budget anchors, validation-only selection, immutable freezes and declared fresh-test exclusions.
- Investigate QCeNN-MI's coherent recurrence and polynomial/Chebyshev channel research: https://github.com/vtavakkoli/QCeNN-MI. Inspect source, docs, LICENSE and CITATION.cff. Preserve original attribution when reusing any source.
- No classical trained neural inference head in MNIST or quantum-text paths. Measurements from quantum circuits generate similarity or token probabilities; classical image/token loading, parameter compilation, aggregation, output decoding and optional teacher-text generation are allowed but MUST be reported. These classical operations mean the entire pipeline is not purely quantum.
- The original `qcenn_qml.mnist` code is **quantum exemplar similarity** with coherent Grover overlap polynomial, not a trained quantum neural network nor QCeNN-MI's original recurrent collision channel. The separate `qcenn_qml.recurrent_mnist` path is a **trainable local recurrent collision classifier**, inspired by QCeNN-MI's reset-channel construction; it does not implement the matrix-inversion residual solver. Keep these methods distinct.
- The LEGACY recurrent training uses classical SPSA-Adam on an exact reduced-density simulator and MUST be called hybrid training. Default quantum-memory predictions MUST come only from explicit circuit measurement counts and must pass the tiny training retrieval gate before validation/test. Simulator probabilities are diagnostics, never a fallback for rejected shots. Never tune with the test split. Archive source hashes, split IDs, raw records, counts and resources.
- Smol path is **tiny quantum associative n-gram memory** (<=8 tokens, <=2-token context), NOT a converted or executed SmolLM2-135M transformer. Its optional teacher generates classical corpus text; quantum inference uses prepared memory and measured counts, not teacher neural logits at runtime. Never claim 135M weights ran on a quantum processor.
- Do not quietly substitute sklearn 8x8 digits for real MNIST. The default mnist dataset is actual torchvision MNIST. The optional digits-proxy is only for offline unit/CI smoke.
- No fabrication of performance, benchmark numbers, QPU executions or advantage. Show circuit width, transpiled depth, shot counts, accepted-shot fraction, classical loading cost, accuracy and uncertainty.

## Commands to maintain

    python -m pip install -e '.[mnist,test]'
    python -m pytest -q
    python -m qcenn_qml.mnist --dataset mnist --classes 01 --train-per-class 2 --test-per-class 5 --shots 1024 --iterations 0
    python -m qcenn_qml.mnist --dataset mnist --classes 0123456789 --train-per-class 2 --test-per-class 5 --shots 1024 --iterations 1
    python -m qcenn_qml.smol135 --prompt 'the cat' --new-tokens 5 --shots 4096
    python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 43 44

## Next milestones

1. Validate all Aer results against independently derived exact statevector and analytic overlap tests, including endian order; CI must pass.
2. Maintain the new ten-class recurrent comparison runner: stratified train/validation/test, uncertainty, multiple seeds, exact classical references and matched k=0/k=1 quantum exemplar comparisons. Report unequal total resources across architectures. Be alert: Grover overlap polynomial is nonmonotonic.
3. Validate the local recurrent classifier on larger real-MNIST studies before claiming improvement. Its unitary-plus-local-collision construction is CPTP; tests establish retained history and distinct input-conditioned stationary outputs in a solvable subfamily, not general class separation after convergence. A classifier built from the original matrix-inversion residual channel remains future work. Read upstream recurrent_quantum.py and polynomial_recurrent.py; do not pass a classical residual vector between quantum updates.
4. Build quantum recurrent sequence/cellular processing, rather than relabeling n-gram memory as a full quantum language model. If classical gradient descent tunes quantum angles, document training as hybrid. To claim wholly quantum training, provide physical circuits for parameter update and certify resources.
5. SmolLM2-135M feasibility study: validate and publish an explicit one-block operator map with honest encoding/query/readout complexity before contemplating 30 transformer layers / full vocabulary. Do not claim quantum execution of weights unless device jobs prove it.
6. Only then add optional hardware backend and noise models with real backend IDs, shots, qubit connectivity, synthesis and cost budgets.

## License and citation safeguards

- New original QCeNN-QML material is released under the custom non-commercial research / paid-commercial license in LICENSE. Preserve non-commercial attribution, mandatory public-research citations, and the separate paid written commercial-license requirement in docs and package metadata. Never describe the current release as MIT or OSI open source.
- Keep NOTICE.md and CITATION.cff synchronized when updating attribution or citations. Do not claim this change revokes the earlier MIT permissions for previously released copies. Preserve the independent QCeNN-MI and third-party license obligations.

## Engineering style

Use Python modules, not a notebook-only implementation. Keep CLI entry points stable, seeded and reproducible, with JSON reports and pinned/tested dependency ranges. Do not implement an unnecessary framework or make smoke tests silently appear to be scientific results. Preserve source provenance, dataset identity and license requirements.


## Retired quantum masked diffusion track

- Diffusion is retired from active launchers, dataset preparation and CI benchmark runs at the user's request. Preserve historical code/results for reproducibility only.
- Historical src/qcenn_qml/diffusion/ and docs/QUANTUM_DIFFUSION.md describe the toy
  quantum denoising baseline. It uses 3-token sequences and controlled
  replacement channels with ancilla resets.
- The LEGACY toy corpus is counted classically and quantum-controlled amplitude
  loaders encode the conditional distributions. This is NOT quantum-trained
  diffusion or natural-language understanding.
- The new default uses raw-corpus QROM and retained quantum memory, with a separate classical conditional-table reference and paired noiseless/readout-noisy runs of the identical native circuit. Broader text splits whole sentences before making windows. Invalid noisy tokens and unseen contexts must retain their error/abstention mass.
- Keep noisy forward masking beta distinct from readout bit-flip noise or
  uncontrolled quantum-device errors; an improvement over fully masked
  inputs alone is not evidence that physical noise improves modeling.
- Verify CPTP, target preservation between resets, conditional expectation,
  no test-label-conditioned reverse gates, and ideal-versus-noisy
  reproducibility. Benchmark non-recurrent, classical frequency, and
  independent quantum baselines at matched resources.
- Continue honoring the active custom non-commercial/paid-commercial license.
