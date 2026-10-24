"""MNIST exemplar classifier: quantum fidelity and coherent Grover polynomial.

This is a measured quantum kernel experiment, NOT a trained quantum neural
network and NOT a direct QCeNN-MI collision-channel implementation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .quantum import sample_counts, unit_vector, zero_reflection


def image_amplitudes(image, side=8):
    from PIL import Image
    if side < 2 or side & (side - 1):
        raise ValueError("side must be a power of two")
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("Grayscale image required")
    if arr.shape != (side, side):
        arr = np.asarray(Image.fromarray(arr).resize((side, side), Image.Resampling.BILINEAR))
    return unit_vector(arr.astype(float).reshape(-1) + 1e-4)


def load_samples(dataset, train, root):
    if dataset == "digits-proxy":
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        data = load_digits()
        train_i, test_i = train_test_split(np.arange(len(data.target)), test_size=.25,
                                          random_state=42, stratify=data.target)
        for i in (train_i if train else test_i):
            yield np.asarray(data.images[i] * (255.0 / 16), dtype=np.uint8), int(data.target[i])
        return
    if dataset != "mnist":
        raise ValueError("Expected mnist or digits-proxy")
    from torchvision.datasets import MNIST
    for image, label in MNIST(root=str(root), train=train, download=True):
        yield np.asarray(image, dtype=np.uint8), int(label)


def select_exemplars(items, classes, count, side):
    selected = {k: [] for k in classes}
    for image, label in items:
        if label in selected and len(selected[label]) < count:
            selected[label].append(image_amplitudes(image, side))
        if all(len(selected[k]) == count for k in classes):
            break
    if any(len(selected[k]) != count for k in classes):
        raise RuntimeError("Not enough examples")
    return selected


def fidelity_circuit(query, exemplar, iterations):
    """Measure |<exemplar|query>|^2, or its coherent Grover polynomial."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    query, exemplar = unit_vector(query), unit_vector(exemplar)
    if len(query) != len(exemplar) or iterations < 0:
        raise ValueError("Invalid vectors or iterations")
    n = int(np.log2(len(query)))
    transform = QuantumCircuit(n)
    transform.compose(StatePreparation(query).definition, qubits=list(range(n)), inplace=True)
    transform.compose(StatePreparation(exemplar).inverse().definition, qubits=list(range(n)), inplace=True)
    qc = transform.copy()
    reflection = zero_reflection(n)
    for _ in range(iterations):
        qc.compose(reflection, inplace=True)
        qc.compose(transform.inverse(), inplace=True)
        qc.compose(reflection, inplace=True)
        qc.compose(transform, inplace=True)
    qc.measure_all()
    return qc


def predict_from_counts(counts, keys, shots):
    scores = {}
    for result, (label, _) in zip(counts, keys, strict=True):
        clean = {k.replace(" ", ""): v for k, v in result.items()}
        bit_width = len(next(iter(clean)))
        scores.setdefault(label, []).append(clean.get("0" * bit_width, 0) / shots)
    averaged = {k: float(np.mean(v)) for k, v in scores.items()}
    return min(averaged, key=lambda k: (-averaged[k], k)), averaged


def evaluate(dataset="mnist", classes="01", train_per_class=2,
             test_per_class=4, shots=1024, iterations=0, side=8, seed=42,
             data_root="data", output_dir="results/mnist"):
    labels = sorted({int(ch) for ch in classes})
    if not labels or any(ch < 0 or ch > 9 for ch in labels):
        raise ValueError("Invalid class list")
    if min(train_per_class, test_per_class, shots) < 1 or iterations < 0:
        raise ValueError("Invalid experiment budget")
    prototypes = select_exemplars(load_samples(dataset, True, data_root),
                                  labels, train_per_class, side)
    test = select_exemplars(load_samples(dataset, False, data_root),
                            labels, test_per_class, side)
    targets, predictions, rows = [], [], []
    circuit_depth = None
    for true_label in labels:
        for image in test[true_label]:
            circuits, keys = [], []
            for candidate in labels:
                for idx, prototype in enumerate(prototypes[candidate]):
                    circuits.append(fidelity_circuit(image, prototype, iterations))
                    keys.append((candidate, idx))
            counts, compiled = sample_counts(circuits, shots, seed + len(targets))
            choice, scores = predict_from_counts(counts, keys, shots)
            targets.append(true_label)
            predictions.append(choice)
            rows.append({"target": true_label, "predicted": choice, "scores": scores})
            if circuit_depth is None:
                circuit_depth = int(compiled[0].depth())
    report = {
        "dataset": dataset, "is_real_mnist": dataset == "mnist",
        "backend": "qiskit_aer_shots", "hardware_tested": False,
        "method": "quantum_exemplar_fidelity_with_polynomial_amplification",
        "is_trained_quantum_neural_network": False,
        "is_qcenn_mi_collision_channel": False,
        "classes": labels, "train_per_class": train_per_class,
        "test_per_class": test_per_class, "shots_per_circuit": shots,
        "polynomial_iterations": iterations, "image_side": side,
        "data_qubits": int(np.log2(side * side)),
        "transpiled_example_depth": circuit_depth,
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "confusion_matrix": [[sum(t == a and p == b for t, p in zip(targets, predictions))
                              for b in labels] for a in labels],
        "predictions": rows,
        "limitations": ["Classical image resize and dense amplitude preparation",
                        "Classical support-example selection and final argmax",
                        "Tiny sample by default; Aer simulation is not real QPU",
                        "Grover response is nonmonotonic; no quantum advantage claim"]
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "predictions"}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "digits-proxy"], default="mnist")
    parser.add_argument("--classes", default="01")
    parser.add_argument("--train-per-class", type=int, default=2)
    parser.add_argument("--test-per-class", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/mnist")
    return evaluate(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
