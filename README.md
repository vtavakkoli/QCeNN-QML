# QCeNN-QML

**Recurrent quantum MNIST classification**, quantum exemplar baselines, and a tiny SmolLM2-135M-inspired autoregressive language experiment. See [AGENTS.md](AGENTS.md) for the research rules and [the roadmap](docs/RESEARCH_ROADMAP.md).

This independent work is inspired by [QCeNN-MI](https://github.com/vtavakkoli/QCeNN-MI). Version 0.2 adds a **trainable local recurrent collision classifier** with persistent quantum memory and a measured quantum readout. It specializes the collision/reset idea locally; it does not implement the upstream matrix-inversion residual solver. The original `qcenn_qml.mnist` remains an exemplar-overlap / Grover polynomial baseline. Experimental quantum predictions come from measured circuits; data preparation, orchestration and readout are classical. Recurrent gate training uses classical optimization and is explicitly **hybrid**.

## License, attribution, and commercial use

**QCeNN-QML uses a custom source-available license, not an OSI-approved open-source license.** The current code is free for **non-commercial personal, educational, and academic research use**, provided the required attribution is included. **All commercial use requires a separately negotiated and paid written license before use**, including commercial research and development, internal business use, products, APIs, and consulting.

**Mandatory project credit:**

> QCeNN-QML — Vahid Tavakkoli (2026)  
> https://github.com/vtavakkoli/QCeNN-QML

If you publish research, benchmarks, presentations, or reports materially using this repository, you must also cite QCeNN-QML using [CITATION.cff](CITATION.cff). To request a commercial license and pricing, contact the copyright holder through [the GitHub repository](https://github.com/vtavakkoli/QCeNN-QML). No fixed price or automatic payment authorization is offered here.

See [LICENSE](LICENSE) for legally controlling terms, [NOTICE.md](NOTICE.md) for attribution and upstream provenance, and [CITATION.cff](CITATION.cff) for the recommended software citation. Versions previously distributed under MIT retain their original MIT permissions; this change does not retroactively remove them.

## Installation

Requires Python 3.10+.

    git clone https://github.com/vtavakkoli/QCeNN-QML.git
    cd QCeNN-QML
    python -m pip install -e '.[mnist,test]'

## Recurrent ten-class MNIST (v0.2)

Small genuine-MNIST end-to-end check:

    python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 --train-per-class 2 --validation-per-class 1 --test-per-class 1 --exemplars-per-class 1 --training-steps 5 --shots 256 --output-dir results/recurrent-smoke

Larger comparison across three seeded train/validation/test selections:

    python -m qcenn_qml.recurrent_mnist --dataset mnist --seeds 42 43 44 --train-per-class 100 --validation-per-class 20 --test-per-class 20 --exemplars-per-class 2 --training-steps 500 --shots 1024 --include-reset-control --output-dir results/recurrent-comparison

The model streams 2x2 image patches through four memory qubits, local shared
RY/RZ/RXX/RZZ gates and one reset/reused environment qubit. Nineteen quantum
gate parameters are trained with classical SPSA-Adam on a reduced-density
simulator. Final test predictions use finite-shot circuit measurements only;
there is no trained classical prediction head. Basis outcomes 10..15 are
rejected and the accepted-shot fraction is reported.

Comparators include measured quantum exemplar fidelity and Grover scoring,
exact classical versions, normalized classical centroids, and an optional
independently trained memory-reset control. Reports preserve split IDs and
data hashes, source/package versions, model parameters, raw counts, Wilson and
paired-bootstrap intervals, training traces and full native resource totals.
Quantum overlap methods have more circuits/shots than the recurrent model;
costs are reported rather than claiming equal resources or quantum advantage.
Always use a fresh output directory. Original MNIST downloads use checksummed
IDX archives; PyTorch is not required. Full details and model reloading are in
[the recurrent experiment protocol](docs/RECURRENT_MNIST.md).
The [archived two-seed original-MNIST smoke](evidence/recurrent-mnist-smoke/README.md)
records the initial recurrent model's weak accuracy alongside every comparator;
it establishes execution and reproducibility, with no accuracy improvement.

## Two Python commands for genuine MNIST

Binary 0 versus 1, quantum-fidelity baseline:

    python -m qcenn_qml.mnist --dataset mnist --classes 01 --train-per-class 2 --test-per-class 5 --shots 1024 --iterations 0 --output-dir results/mnist-binary

Ten classes, coherent Grover polynomial response (may perform worse than baseline):

    python -m qcenn_qml.mnist --dataset mnist --classes 0123456789 --train-per-class 2 --test-per-class 5 --shots 1024 --iterations 1 --output-dir results/mnist-ten

The original MNIST dataset is downloaded automatically as checksummed 28x28 IDX images; it is resized to 8x8, normalized, and amplitude-loaded into six quantum data qubits. A per-exemplar quantum fidelity circuit uses U_exemplar inverse times U_image, and the result is estimated through finite-shot zero-state measurements in Qiskit Aer. The optional coherent Grover recurrence is a polynomial in overlap probability, but **nonmonotonic**; the same samples/shots should be evaluated with iterations 0 and 1 for any research comparison. This baseline has no neural weights; class prototypes are stored classically and class choice uses a classical argmax. Dense state preparation has real cost. Very small samples in these commands are smoke experiments, not publication-quality scores.

Offline CI alternative (NOT genuine MNIST; clearly identified in output):

    python -m qcenn_qml.mnist --dataset digits-proxy --classes 01 --train-per-class 1 --test-per-class 1 --shots 128

## SmolLM2-135M quantum research path

The actual [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) **is not transferred to or executed on quantum hardware**. It is a 135M-parameter transformer, and directly implementing its layers/weights has not been achieved here. This is a small quantum autoregressive **associative n-gram memory** with maximum eight-token vocabulary and one- or two-token context, not a trained quantum LLM. Memory amplitudes encode (context, next token) counts; a Qiskit circuit prepares the joint state, optionally amplifies a query context coherently, and measures the output token. It does not perform a prediction-time classical n-gram lookup. Dense classical amplitude construction is required; after each measured token, the next circuit is compiled classically.

Runnable quantum memory with built-in toy text:

    python -m qcenn_qml.smol135 --prompt 'the cat' --new-tokens 5 --shots 4096 --output-dir results/smol135-toy

Optional **classical teacher corpus only**, executed separately to create training text from real SmolLM2-135M; no teacher weights are converted into quantum gates:

    python -m pip install -e '.[teacher]'
    python -m qcenn_qml.smol135 --prompt 'The cat' --teacher-samples 4 --teacher-max-new-tokens 48 --new-tokens 5 --shots 4096 --output-dir results/smol135-teacher

Alternatively use --corpus-file to supply any text file. Report JSON truthfully distinguishes toy, corpus-file, and classical-teacher modes. Training a quantum model by classical parameter optimization would constitute hybrid training and is not claimed here.

## Tests and status

    python -m pytest -q

CI runs NumPy invariants, independent ideal statevector assertions, complete recurrent circuit/density cross-checks, memory/CPTP tests, seeded training/split reproducibility, and explicitly labeled **digits-proxy** and toy-language Aer smokes. All actual circuits run on **Qiskit Aer (classical simulation of quantum gates)**; no physical QPU test, runtime speedup, or SmolLM2 emulation is claimed. Reports are saved under the chosen output directory as report.json. Accuracy improvement must be established by the comparisons, not assumed from successful execution.

## Attribution and citation

**QCeNN-QML software citation:** V. Tavakkoli, "QCeNN-QML: Quantum-Circuit Inference Prototypes for MNIST and Quantum Language Experiments," software, 2026, https://github.com/vtavakkoli/QCeNN-QML. Cite the software in all public research and results materially using it.

**Upstream research inspiration:** QCeNN-MI — Vahid Tavakkoli and Kabeh Mohsenzadegan (2026), https://github.com/vtavakkoli/QCeNN-MI. Its separate custom license has its own mandatory attribution and citation conditions. This repository does **not** relicense upstream code or third-party dependencies. See [NOTICE.md](NOTICE.md).


## Quantum masked diffusion and engineered noise (research prototype)

Third independent track: a **three-token conditional masked-denoising circuit**
using 9 qubits (six data/context, two environment, one flag).
A mask-replacement forward channel and recurrent conditional replacement
channels are executed with quantum-controlled swaps and environment resets.
Quantum context-controlled token amplitudes come from a **classically counted
tiny toy corpus**, not from learned quantum neural-network weights. The
data register is not measured and classically reconstructed between updates.
This is **not a trained diffusion LLM**, not an implementation of the exact
QCeNN-MI inversion solver, and not a demonstration of benefit from
uncontrolled physical-device noise.

    python -m qcenn_qml.diffusion.benchmark --shots 1024 --examples 8 --beta 1 --eta 0.65 --steps 0 1 3 --output-dir results/diffusion/ideal
    python -m qcenn_qml.diffusion.benchmark --shots 1024 --examples 8 --beta 1 --eta 0.65 --steps 0 1 3 --readout-error 0.05 --output-dir results/diffusion/noisy

Each run saves report.json and report.md with measured conditional token
frequencies, independent channel-theory checks, shots, and compiled depths.
Separate measured experiments can be merged using:

    python -m qcenn_qml.report --output-dir results/ci/summary

See [the detailed quantum-noise research protocol](docs/QUANTUM_DIFFUSION.md).
CI uploads an artifact named qcenn-qml-all-simulator-results with all
three smoke experiment reports and the combined summary.
