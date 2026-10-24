# QCeNN-QML

**Quantum-circuit inference prototypes** for MNIST and a tiny SmolLM2-135M-inspired autoregressive language experiment. See [AGENTS.md](AGENTS.md) for the mandatory research rules and [the roadmap](docs/RESEARCH_ROADMAP.md).

This independent work is inspired by [QCeNN-MI](https://github.com/vtavakkoli/QCeNN-MI). It **does not yet implement the upstream recurrent collision channel**: the implemented quantum classifier is an exemplar-overlap / Grover polynomial baseline, not a trained quantum neural network. All predictions in the experimental inference paths come from measured quantum circuits, but data preparation, orchestration and readout are classical.

## Installation

Requires Python 3.10+.

    git clone https://github.com/vtavakkoli/QCeNN-QML.git
    cd QCeNN-QML
    python -m pip install -e '.[mnist,test]'

## Two Python commands for genuine MNIST

Binary 0 versus 1, quantum-fidelity baseline:

    python -m qcenn_qml.mnist --dataset mnist --classes 01 --train-per-class 2 --test-per-class 5 --shots 1024 --iterations 0 --output-dir results/mnist-binary

Ten classes, coherent Grover polynomial response (may perform worse than baseline):

    python -m qcenn_qml.mnist --dataset mnist --classes 0123456789 --train-per-class 2 --test-per-class 5 --shots 1024 --iterations 1 --output-dir results/mnist-ten

The torchvision MNIST dataset is downloaded automatically as original 28x28 images; it is resized to 8x8, normalized, and amplitude-loaded into six quantum data qubits. A per-exemplar quantum fidelity circuit uses U_exemplar inverse times U_image, and the result is estimated through finite-shot zero-state measurements in Qiskit Aer. The optional coherent Grover recurrence is a polynomial in overlap probability, but **nonmonotonic**; the same samples/shots should be evaluated with iterations 0 and 1 for any research comparison. No trainable classical prediction head or neural weights are present, but class prototypes are stored classically and class choice uses a classical argmax. Dense state preparation has real cost. Very small samples in these commands are smoke experiments, not publication-quality scores.

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

CI runs the NumPy invariant tests, independent ideal quantum statevector assertions, a small sklearn **digits-proxy** Aer smoke, and the quantum toy-language Aer smoke. All actual circuits run on **Qiskit Aer (classical simulation of quantum gates)**; no physical QPU test, runtime speedup, or SmolLM2 emulation is claimed. Reports are saved under the chosen output directory as report.json. The initial quantum experiments are intended to establish baselines before porting QCeNN-MI recurrent/polynomial channel methods.

## Attribution

QCeNN-MI — Vahid Tavakkoli and Kabeh Mohsenzadegan (2026), https://github.com/vtavakkoli/QCeNN-MI. Upstream has a custom attribution/citation license; follow it when using or adapting that work. Newly authored code in this repository uses the MIT license and **does not relicense upstream code**. See [NOTICE.md](NOTICE.md).
