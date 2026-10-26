"""Quantum-memory MNIST: tiny retrieval gate before validation/test evaluation.

This is nonparametric quantum inference, with no classical optimizer. The
historical SPSA-trained neural experiment remains explicitly legacy/hybrid.
"""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np

from .coherent import CORE_DISCLOSURE, decode_postselection, execute
from .quantum_memory import memory_circuit, reference_distribution
from .mnist_data import load_dataset, stratified_indices, split_manifest
from .recurrent_mnist import classification_metrics, patches_from_images, provenance, write_json


def measure_split(images, labels, keys, key_labels, *, rounds, encoding, damping,
                  shots, seed, batch_size, max_qubits, output_dir, reset_memory=False):
    from qiskit import qpy
    rows, resources = [], []
    construction_seconds = 0.
    for start in range(0, len(images), batch_size):
        timer = perf_counter()
        circuits = [memory_circuit(query, keys, key_labels, rounds=rounds,
            encoding=encoding, damping=damping, max_qubits=max_qubits,reset_memory=reset_memory)
            for query in images[start:start+batch_size]]
        construction_seconds += perf_counter()-timer
        counts, native, costs = execute(circuits, shots, seed+start)
        resources.append(costs)
        if start == 0:
            with (Path(output_dir)/"example-native.qpy").open("wb") as stream:
                qpy.dump(native[0], stream)
        rows.extend(decode_postselection(c, 4, rounds, 10) for c in counts)
        print(f"  measured {len(rows)}/{len(images)} images", flush=True)
    result = {**classification_metrics(labels, [r["prediction"] for r in rows]),
        "predictions": rows, "accepted_shot_fraction": sum(r["accepted_shots"] for r in rows)/(shots*len(rows)),
        "resources": resources, "circuit_construction_seconds": construction_seconds}
    if damping == 0:
        exact = [reference_distribution(q, keys, key_labels, rounds=1 if reset_memory else rounds, encoding=encoding)
                 for q in images]
        result["classical_reference_only"] = {
            **classification_metrics(labels, [p.argmax() if a else -1 for p,a in exact]),
            "acceptance_mean": float(np.mean([a**rounds if reset_memory else a for _,a in exact])),
            "used_for_quantum_predictions": False}
    return result


def evaluate(dataset="mnist", seeds=(42,), train_per_class=1, validation_per_class=1,
             test_per_class=1, side=8, rounds=4, encoding="amplitude", damping=0.,
             fit_threshold=.9, shots=1024, batch_size=2, max_qubits=96,
             max_records=32, include_reset_control=False, data_root="data",
             no_download=False, output_dir="results/quantum-mnist"):
    config = dict(locals())
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    if min(train_per_class, validation_per_class, test_per_class, shots, batch_size) < 1:
        raise ValueError("Positive split/shot/batch budgets required")
    if 10*train_per_class > max_records or not 0 < fit_threshold <= 1:
        raise ValueError("Tiny-memory budget exceeded or invalid fit threshold")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Provide unique seeds")
    directory.mkdir(parents=True, exist_ok=True)
    config["seeds"] = list(seeds)
    write_json(directory/"config.json", config)
    started = perf_counter()
    train_images, train_labels, test_images, test_labels = load_dataset(dataset, data_root, download=not no_download)
    loading_seconds = perf_counter()-started
    runs = []
    for seed in seeds:
        run_dir = directory/f"seed-{seed}"
        run_dir.mkdir()
        ti, vi, ei = stratified_indices(train_labels, test_labels, train_per_class,
            validation_per_class, test_per_class, seed)
        write_json(run_dir/"splits.json", split_manifest(dataset,seed,ti,vi,ei,
            train_images,train_labels,test_images,test_labels))
        timer = perf_counter()
        # Fixed image preparation only; no fitted encoder or trained prototype.
        transform = lambda x: patches_from_images(x,side).reshape(len(x),-1) + 1e-8
        keys = transform(train_images[ti])
        validation = transform(train_images[vi])
        test = transform(test_images[ei])
        if encoding == "angle":
            keys, validation, test = [np.clip(x,0,1) for x in (keys,validation,test)]
        preparation_seconds = perf_counter()-timer
        np.savez_compressed(run_dir/"memory.npz", raw_image_vectors=keys,
            labels=train_labels[ti], training_source_indices=ti)
        splits = {}
        for name, data, truth in (("tiny_train",keys,train_labels[ti]),
                                   ("validation",validation,train_labels[vi]),
                                   ("test",test,test_labels[ei])):
            if name != "tiny_train" and not splits["tiny_train"]["fit_gate_passed"]:
                break
            target = run_dir/name
            target.mkdir()
            print(f"seed={seed} {name}: quantum memory, no optimizer", flush=True)
            splits[name] = measure_split(data,truth,keys,train_labels[ti],rounds=rounds,
                encoding=encoding,damping=damping,shots=shots,seed=seed+1000,
                batch_size=batch_size,max_qubits=max_qubits,output_dir=target)
            if name == "tiny_train":
                splits[name]["fit_gate_passed"] = splits[name]["accuracy"] >= fit_threshold
            if name == 'validation':
                target=run_dir/'validation_k0_reference'; target.mkdir()
                splits['validation_k0_reference']=measure_split(validation,train_labels[vi],keys,train_labels[ti],
                    rounds=1,encoding=encoding,damping=damping,shots=shots,seed=seed+1000,
                    batch_size=batch_size,max_qubits=max_qubits,output_dir=target)
        if "test" in splits:
            target = run_dir/"quantum_k0_reference"
            target.mkdir()
            splits["quantum_k0_reference"] = measure_split(test,test_labels[ei],keys,train_labels[ti],
                rounds=1,encoding=encoding,damping=damping,shots=shots,seed=seed+1000,
                batch_size=batch_size,max_qubits=max_qubits,output_dir=target)
            if include_reset_control:
                target=run_dir/'reset_memory_control'; target.mkdir()
                splits['reset_memory_control']=measure_split(test,test_labels[ei],keys,train_labels[ti],
                    rounds=rounds,encoding=encoding,damping=damping,shots=shots,seed=seed+1000,
                    batch_size=batch_size,max_qubits=max_qubits,output_dir=target,reset_memory=True)
        run = {"seed":seed,"splits":splits,"test_evaluated":"test" in splits,
            "classical_preparation_seconds":preparation_seconds,
            "configuration_selection":"fixed_before_test; no validation-driven CPU fitting"}
        write_json(run_dir/"report.json",run)
        runs.append(run)
    report = {"experiment":"coherent_recurrent_quantum_exemplar_memory",
        **CORE_DISCLOSURE, "dataset":dataset,"is_real_mnist":dataset=="mnist",
        "configuration":config,"source":provenance(),"loading_seconds":loading_seconds,
        "tiny_fit_gate_required":True,"runs":runs,
        "test_accuracy_mean":float(np.mean([r['splits']['test']['accuracy'] for r in runs if r['test_evaluated']]))
            if any(r['test_evaluated'] for r in runs) else None,
        "inference":"resident quantum index, repeated quantum overlap filters, measured label register",
        "filter":"p**rounds, monotone on [0,1], with postselection costs",
        "limitations":["Nonparametric memory retrieval, not trained neural weights",
            "Classical raw input loading, compilation and final count decoding remain",
            "No quantum-only training of the historical 19-angle network is claimed",
            "All quantum execution here uses Aer simulation; no QPU or advantage claim"]}
    write_json(directory/"report.json",report)
    print(f"report: {directory/'report.json'}; test accuracy: {report['test_accuracy_mean']}",flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["mnist","digits-proxy"],default="mnist")
    parser.add_argument("--seeds",type=int,nargs="+",default=[42])
    for name,default in (("train-per-class",1),("validation-per-class",1),("test-per-class",1),
        ("side",8),("rounds",4),("shots",1024),("batch-size",2),("max-qubits",96),("max-records",32)):
        parser.add_argument("--"+name,type=int,default=default)
    parser.add_argument("--encoding",choices=["amplitude","angle"],default="amplitude")
    parser.add_argument("--damping",type=float,default=0.)
    parser.add_argument("--fit-threshold",type=float,default=.9)
    parser.add_argument("--include-reset-control",action="store_true")
    parser.add_argument("--data-root",default="data")
    parser.add_argument("--no-download",action="store_true")
    parser.add_argument("--output-dir",default="results/quantum-mnist")
    return evaluate(**vars(parser.parse_args(argv)))


if __name__=="__main__":
    main()
