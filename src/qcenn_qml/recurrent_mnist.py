"""Ten-class recurrent MNIST: hybrid training, measured quantum inference.

python -m qcenn_qml.recurrent_mnist --help
"""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from .mnist import fidelity_circuit, image_amplitudes, predict_from_counts
from .mnist_data import load_dataset, split_manifest, stratified_indices
from .recurrent import (PARAMETER_COUNT, decode_counts, initial_parameters,
                        probabilities, recurrent_circuit)


def patches_from_images(images, side=8):
    """Resize, scale to [0,1], and stream nonoverlapping 2x2 patches row-major."""
    from PIL import Image
    if side not in (4, 8, 16):
        raise ValueError("Image side must be 4, 8, or 16")
    pixels = np.asarray([np.asarray(Image.fromarray(np.asarray(im, dtype=np.uint8)).resize(
        (side, side), Image.Resampling.BILINEAR)) for im in images], dtype=float) / 255
    patches = pixels.reshape(-1, side//2, 2, side//2, 2).transpose(0, 1, 3, 2, 4)
    return patches.reshape(-1, side*side//4, 4)


def classification_metrics(targets, predictions):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    n = len(targets)
    if n == 0 or targets.shape != predictions.shape:
        raise ValueError("Nonempty matching targets and predictions required")
    accuracy = float(np.mean(targets == predictions))
    z = 1.959963984540054
    denominator = 1 + z*z/n
    middle = (accuracy + z*z/(2*n))/denominator
    radius = z*np.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n))/denominator
    confusion = np.zeros((10, 11), dtype=int)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[int(target), int(prediction) if prediction >= 0 else 10] += 1
    return {"n": n, "accuracy": accuracy,
            "accuracy_wilson_95": [max(0., float(middle-radius)), min(1., float(middle+radius))],
            "confusion_prediction_labels": list(range(10)) + [-1],
            "confusion_matrix": confusion.tolist(),
            "rejected_examples": int(np.sum(predictions == -1))}


def cross_entropy(patches, labels, theta, *, retain_memory=True):
    distribution = probabilities(patches, theta, retain_memory=retain_memory)
    # Unconditional Born probabilities penalize mass in rejected outcomes 10..15.
    return float(-np.log(np.maximum(distribution[np.arange(len(labels)), labels], 1e-12)).mean())


def train_model(train, labels, validation, validation_labels, *, seed, steps=150,
                batch_size=32, learning_rate=.04, validation_interval=25,
                retain_memory=True):
    """Seeded SPSA-Adam on the exact reduced density simulator: HYBRID training.

    Only training examples enter updates. Validation selects the checkpoint.
    No test examples or measurement-derived test scores enter model selection.
    """
    if steps < 0 or batch_size < 1 or validation_interval < 1 or learning_rate <= 0:
        raise ValueError("Invalid training budget")
    rng = np.random.default_rng(seed)
    start = perf_counter()
    theta = initial_parameters(seed)
    best = theta.copy()
    best_loss = cross_entropy(validation, validation_labels, theta, retain_memory=retain_memory)
    history = [{"step": 0, "validation_nll": best_loss}]
    best_step, first, second = 0, np.zeros_like(theta), np.zeros_like(theta)
    evaluated_examples = len(validation)
    for step in range(1, steps+1):
        ids = rng.choice(len(train), size=min(batch_size, len(train)), replace=False)
        direction = rng.choice([-1., 1.], size=PARAMETER_COUNT)
        perturbation = .12 / step**.101
        plus = cross_entropy(train[ids], labels[ids], theta + perturbation*direction,
                             retain_memory=retain_memory)
        minus = cross_entropy(train[ids], labels[ids], theta - perturbation*direction,
                              retain_memory=retain_memory)
        gradient = (plus-minus)/(2*perturbation)*direction
        gradient *= min(1., 5 / max(1e-12, float(np.linalg.norm(gradient))))
        first = .9*first + .1*gradient
        second = .999*second + .001*gradient**2
        theta -= learning_rate*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step)) + 1e-8)
        theta[[1, 3]] = np.clip(theta[[1, 3]], -2, 2)
        theta[6] = np.clip(theta[6], -6, 3)
        evaluated_examples += 2*len(ids)
        if step % validation_interval == 0 or step == steps:
            loss = cross_entropy(validation, validation_labels, theta, retain_memory=retain_memory)
            evaluated_examples += len(validation)
            history.append({"step": step, "validation_nll": loss,
                            "training_minibatch_perturbed_nll": (plus+minus)/2})
            if loss < best_loss:
                best_loss, best, best_step = loss, theta.copy(), step
            print(f"  seed={seed} memory={retain_memory} step={step}/{steps} val_nll={loss:.5f}", flush=True)
    return best, {"mode": "hybrid_classical_SPSA_Adam_exact_density_training",
                  "quantum_gate_parameters": PARAMETER_COUNT, "optimizer_seed": seed,
                  "steps": steps, "batch_size": batch_size, "learning_rate": learning_rate,
                  "validation_interval": validation_interval, "selected_step": best_step,
                  "selected_validation_nll": best_loss,
                  "simulated_example_evaluations": evaluated_examples,
                  "quantum_training_shots": 0, "seconds": perf_counter()-start,
                  "history": history}


def run_circuits(circuits, shots, seed):
    """Native-gate compilation and finite-shot execution, including device resets."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    if shots < 1 or not circuits:
        raise ValueError("Positive shots and nonempty circuit list required")
    backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
    start = perf_counter()
    compiled = transpile(circuits, basis_gates=["rz", "sx", "x", "cx", "reset"],
                         optimization_level=1, seed_transpiler=seed, num_processes=1)
    compilation_seconds = perf_counter()-start
    start = perf_counter()
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(f"Aer circuit execution failed: {result.status}")
    counts = [result.get_counts(i) for i in range(len(compiled))]
    if any(sum(row.values()) != shots for row in counts):
        raise RuntimeError("Measured shot count differs from the requested budget")
    return counts, compiled, {"compilation_seconds": compilation_seconds,
                              "sampling_seconds": perf_counter()-start}


def circuit_record(circuit, shots):
    operations = circuit.count_ops()
    return {"qubits": circuit.num_qubits, "depth": int(circuit.depth()),
            "cx": int(operations.get("cx", 0)), "resets": int(operations.get("reset", 0)),
            "one_qubit_gates": sum(int(operations.get(g, 0)) for g in ("rz", "sx", "x")),
            "shots": shots}


def resource_summary(records, timings):
    return {"compiled_circuits": len(records),
            "total_shots": sum(row["shots"] for row in records),
            "qubits_max": max(row["qubits"] for row in records),
            "depth_mean": float(np.mean([row["depth"] for row in records])),
            "depth_max": max(row["depth"] for row in records),
            "cx_per_circuit_mean": float(np.mean([row["cx"] for row in records])),
            "cx_shot_weighted_total": sum(row["cx"]*row["shots"] for row in records),
            "reset_shot_weighted_total": sum(row["resets"]*row["shots"] for row in records),
            "native_basis": ["rz", "sx", "x", "cx", "reset", "measure"],
            **timings, "per_circuit": records}


def measured_recurrent(patches, theta, targets, *, shots, seed, batch_size,
                       output_dir, retain_memory=True):
    from qiskit import qpy
    rows, resources = [], []
    timings = {"circuit_construction_seconds": 0., "compilation_seconds": 0., "sampling_seconds": 0.}
    for start in range(0, len(patches), batch_size):
        timer = perf_counter()
        circuits = [recurrent_circuit(p, theta, retain_memory=retain_memory)
                    for p in patches[start:start+batch_size]]
        timings["circuit_construction_seconds"] += perf_counter()-timer
        counts, compiled, elapsed = run_circuits(circuits, shots, seed+start)
        for key, value in elapsed.items():
            timings[key] += value
        if start == 0:
            with (Path(output_dir) / "example-native.qpy").open("wb") as stream:
                qpy.dump(compiled[0], stream)
        for measured, native in zip(counts, compiled, strict=True):
            rows.append(decode_counts(measured))
            resources.append(circuit_record(native, shots))
        print(f"  measured {'recurrent' if retain_memory else 'reset control'}: {len(rows)}/{len(patches)} images", flush=True)
    accepted = sum(row["accepted_shots"] for row in rows)
    return {"kind": "measured_quantum_inference", "backend": "qiskit_aer_density_matrix",
            "physical_qpu_tested": False, "retained_memory": retain_memory,
            "prediction_source": "final_four_bit_measurement_counts_only",
            "accepted_shots": accepted, "accepted_shot_fraction": accepted/(shots*len(rows)),
            "rejection_rule": "basis outcomes 10..15 discarded; all-rejected samples predict -1",
            **classification_metrics(targets, [row["prediction"] for row in rows]),
            "predictions": rows, "resources": resource_summary(resources, timings)}


def exemplar_scores(test, prototypes, prototype_labels, iterations):
    overlap = np.clip((test @ prototypes.T)**2, 0, 1)
    response = np.sin((2*iterations+1)*np.arcsin(np.sqrt(overlap)))**2
    return np.column_stack([response[:, prototype_labels == label].mean(axis=1) for label in range(10)])


def measured_overlap(test, prototypes, prototype_labels, targets, *, iterations,
                     shots, seed, batch_size):
    """Same test IDs/prototypes/shots for k=0 and k=1; costs are charged separately."""
    rows, records = [], []
    timings = {"circuit_construction_seconds": 0., "compilation_seconds": 0., "sampling_seconds": 0.}
    keys = [(int(label), i) for i, label in enumerate(prototype_labels)]
    # Batch size counts circuits, rather than loading the entire experiment at once.
    per_batch = max(1, batch_size // len(prototypes))
    for start in range(0, len(test), per_batch):
        batch = test[start:start+per_batch]
        timer = perf_counter()
        circuits = [fidelity_circuit(image, prototype, iterations) for image in batch for prototype in prototypes]
        timings["circuit_construction_seconds"] += perf_counter()-timer
        counts, compiled, elapsed = run_circuits(circuits, shots, seed+start)
        for key, value in elapsed.items():
            timings[key] += value
        records.extend(circuit_record(circuit, shots) for circuit in compiled)
        for i in range(len(batch)):
            selected = counts[i*len(prototypes):(i+1)*len(prototypes)]
            prediction, scores = predict_from_counts(selected, keys, shots)
            rows.append({"prediction": prediction, "scores": scores, "counts": selected})
        print(f"  measured overlap k={iterations}: {len(rows)}/{len(test)} images", flush=True)
    return {"kind": "measured_quantum_exemplar_baseline", "iterations": iterations,
            "backend": "qiskit_aer_density_matrix", "physical_qpu_tested": False,
            "accepted_shot_fraction": 1., "acceptance_definition": "all shots used to estimate zero-state frequency",
            **classification_metrics(targets, [row["prediction"] for row in rows]),
            "predictions": rows, "resources": resource_summary(records, timings)}


def paired_accuracy_interval(targets, first, second, seed):
    differences = (np.asarray(first) == targets).astype(float) - (np.asarray(second) == targets)
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(targets), size=(2000, len(targets)))].mean(axis=1)
    return {"accuracy_difference": float(differences.mean()),
            "paired_bootstrap_95": np.quantile(means, [.025, .975]).tolist(),
            "scope": "test-example resampling conditional on fitted models and recorded shots"}


def provenance():
    packages = {}
    for package in ("numpy", "qiskit", "qiskit-aer", "Pillow", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[2]
    hashes = {f"src/qcenn_qml/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    for name in ("pyproject.toml",):
        if (root / name).exists():
            hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "git_commit": commit, "git_dirty": dirty, "source_sha256": hashes}


def write_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def evaluate(dataset="mnist", seeds=(42,), train_per_class=20, validation_per_class=5,
             test_per_class=5, exemplars_per_class=2, side=8, training_steps=150,
             batch_size=32, learning_rate=.04, validation_interval=25, shots=1024,
             overlap_iterations=(0,), skip_quantum_overlap=False,
             include_reset_control=False, data_root="data", no_download=False,
             output_dir="results/recurrent-mnist"):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**31-10000 for seed in seeds):
        raise ValueError("Provide unique nonnegative 31-bit seeds")
    if min(train_per_class, validation_per_class, test_per_class, exemplars_per_class, shots, batch_size, validation_interval) < 1:
        raise ValueError("Split, exemplar, shot, batch and validation budgets must be positive")
    if exemplars_per_class > train_per_class or training_steps < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid exemplar or training budget")
    iterations = sorted(set(overlap_iterations))
    if not iterations or any(k not in (0, 1) for k in iterations):
        raise ValueError("Compare overlap iterations 0 and/or 1")
    if side not in (4, 8, 16):
        raise ValueError("Image side must be 4, 8, or 16")
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a fresh output directory to preserve experiment evidence")
    config = {key: value for key, value in locals().copy().items()
              if key not in ("directory", "iterations", "output_dir")}
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", config)
    start = perf_counter()
    train_images, train_labels, test_images, test_labels = load_dataset(dataset, data_root, download=not no_download)
    loading_seconds = perf_counter()-start
    source = provenance()
    runs = []
    for seed in seeds:
        print(f"Starting {dataset} ten-class seed={seed}", flush=True)
        run_dir = directory / f"seed-{seed}"
        run_dir.mkdir()
        train_ids, validation_ids, test_ids = stratified_indices(
            train_labels, test_labels, train_per_class, validation_per_class, test_per_class, seed)
        manifest = split_manifest(dataset, seed, train_ids, validation_ids, test_ids,
                                  train_images, train_labels, test_images, test_labels)
        write_json(run_dir / "splits.json", manifest)
        start = perf_counter()
        train = patches_from_images(train_images[train_ids], side)
        validation = patches_from_images(train_images[validation_ids], side)
        test = patches_from_images(test_images[test_ids], side)
        preprocessing_seconds = perf_counter()-start
        targets = test_labels[test_ids]
        methods = {}
        for retain_memory, name in ((True, "recurrent"), (False, "reset_control")):
            if not retain_memory and not include_reset_control:
                continue
            model_dir = run_dir / name
            model_dir.mkdir()
            theta, training = train_model(train, train_labels[train_ids], validation,
                train_labels[validation_ids], seed=seed+1000, steps=training_steps,
                batch_size=batch_size, learning_rate=learning_rate,
                validation_interval=validation_interval, retain_memory=retain_memory)
            write_json(model_dir / "model.json", {"format_version": 1, "parameters": theta.tolist(),
                "image_side": side, "retain_memory": retain_memory, "training": training,
                "readout_basis_labels": list(range(10)), "rejected_basis_outcomes": list(range(10, 16)),
                "source": source})
            methods[name] = measured_recurrent(test, theta, targets, shots=shots,
                seed=seed+2000, batch_size=batch_size, output_dir=model_dir, retain_memory=retain_memory)
            methods[name]["training"] = training
            exact = probabilities(test, theta, retain_memory=retain_memory)
            methods[name]["independent_exact_diagnostic"] = {
                **classification_metrics(targets, exact[:, :10].argmax(axis=1)),
                "accepted_probability_mean": float(exact[:, :10].sum(axis=1).mean()),
                "not_used_for_measured_predictions": True}
        start = perf_counter()
        amplitudes = np.asarray([image_amplitudes(image, side) for image in train_images[train_ids]])
        test_amplitudes = np.asarray([image_amplitudes(image, side) for image in test_images[test_ids]])
        prototype_ids = np.concatenate([np.flatnonzero(train_labels[train_ids] == label)[:exemplars_per_class]
                                        for label in range(10)])
        prototypes, prototype_labels = amplitudes[prototype_ids], train_labels[train_ids][prototype_ids]
        baseline_loading_seconds = perf_counter()-start
        write_json(run_dir / "exemplars.json", {"training_source_indices": train_ids[prototype_ids].tolist(),
            "labels": prototype_labels.tolist(), "classical_loading_seconds": baseline_loading_seconds})
        start = perf_counter()
        centroids = np.asarray([amplitudes[train_labels[train_ids] == label].mean(axis=0) for label in range(10)])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        centroid_training_seconds = perf_counter()-start
        start = perf_counter()
        centroid_predictions = (test_amplitudes @ centroids.T).argmax(axis=1).tolist()
        methods["classical_centroid"] = {"kind": "classical_reference_only",
            **classification_metrics(targets, centroid_predictions),
            "predictions": [{"prediction": p} for p in centroid_predictions],
            "inference_seconds": perf_counter()-start, "training_examples": len(train_ids),
            "training_seconds": centroid_training_seconds}
        for k in iterations:
            start = perf_counter()
            scores = exemplar_scores(test_amplitudes, prototypes, prototype_labels, k)
            predictions = scores.argmax(axis=1).tolist()
            methods[f"classical_overlap_k{k}"] = {"kind": "classical_exact_reference_only",
                **classification_metrics(targets, predictions), "inference_seconds": perf_counter()-start,
                "predictions": [{"prediction": p, "scores": s.tolist()} for p, s in zip(predictions, scores, strict=True)]}
            if not skip_quantum_overlap:
                methods[f"quantum_overlap_k{k}"] = measured_overlap(test_amplitudes, prototypes,
                    prototype_labels, targets, iterations=k, shots=shots, seed=seed+3000,
                    batch_size=batch_size)
        recurrent_predictions = [row["prediction"] for row in methods["recurrent"]["predictions"]]
        paired = {name: paired_accuracy_interval(targets, recurrent_predictions,
                    [row["prediction"] for row in result["predictions"]], seed+4000)
                  for name, result in methods.items() if name != "recurrent"}
        run = {"seed": seed, "targets": targets.tolist(), "methods": methods,
               "recurrent_minus_comparator": paired, "preprocessing_seconds": preprocessing_seconds,
               "exemplar_loading_seconds": baseline_loading_seconds,
               "split_manifest": f"seed-{seed}/splits.json"}
        write_json(run_dir / "report.json", run)
        runs.append(run)
        print(json.dumps({"seed": seed, "accuracy": {name: result["accuracy"] for name, result in methods.items()}}), flush=True)
    aggregate = {}
    for name in runs[0]["methods"]:
        values = [run["methods"][name]["accuracy"] for run in runs]
        aggregate[name] = {"seeds": len(seeds), "accuracy_mean": float(np.mean(values)),
            "accuracy_std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "accuracy_by_seed": values}
    report = {"experiment": "recurrent_ten_class_mnist", "dataset": dataset,
        "is_real_mnist": dataset == "mnist", "physical_qpu_tested": False,
        "backend": "qiskit_aer_density_matrix_finite_shots", "classes": list(range(10)),
        "training_is_hybrid": True, "classical_trained_prediction_head": False,
        "configuration": config, "dataset_loading_seconds": loading_seconds,
        "source": source, "aggregate": aggregate, "runs": runs,
        "comparison_protocol": {
            "same_test_examples_within_seed": True, "test_used_for_model_selection": False,
            "quantum_overlap_same_exemplars_and_shots_for_each_k": True,
            "equal_total_gate_or_shot_budget_across_architectures": False,
            "reset_control": "independently trained, identical optimizer budget, last-patch-only channel",
            "uncertainty": "Wilson within each test run; paired test bootstrap; seed dispersion separately",
            "reproducibility": "fixed source/package versions, IDs, seeds and batch size; timing is machine-dependent"},
        "limitations": ["Small-register simulator experiment; no QPU or quantum advantage claim",
            "Classical resizing, angle loading, SPSA training, compilation, counting and argmax",
            "Exact reduced density training is classically simulated; final predictions use measured circuits",
            "Outcomes 10..15 are rejected and acceptance costs are reported",
            "QCeNN-inspired local collision network, not the upstream matrix-inversion solver",
            "Tiny runs are smoke checks; no accuracy improvement is presumed"]}
    write_json(directory / "report.json", report)
    print(json.dumps({"report": str(directory / "report.json"), "aggregate": aggregate}, indent=2), flush=True)
    return report


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--legacy-hybrid" not in argv:
        from .quantum_mnist import main as quantum_main
        return quantum_main(argv)
    argv.remove("--legacy-hybrid")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "digits-proxy"], default="mnist")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--test-per-class", type=int, default=5)
    parser.add_argument("--exemplars-per-class", type=int, default=2)
    parser.add_argument("--side", type=int, choices=[4, 8, 16], default=8)
    parser.add_argument("--training-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.04)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--overlap-iterations", type=int, choices=[0, 1], nargs="+", default=[0])
    parser.add_argument("--skip-quantum-overlap", action="store_true")
    parser.add_argument("--include-reset-control", action="store_true")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-dir", default="results/recurrent-mnist")
    return evaluate(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
