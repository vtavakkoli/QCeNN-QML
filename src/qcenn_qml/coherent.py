"""Shared gate-only memory primitives; classical inputs/control are disclosed.

No optimizer, input-dependent classical attention scores, or fitted CPU weights.
Raw records are compiled into QROM gates. This is nonparametric quantum memory,
not a claim of quantum neural-weight training or a fully quantum host pipeline.
"""
import numpy as np

from .quantum import amplitude_preparation


def load_basis(circuit, qubits, value):
    if not 0 <= int(value) < 1 << len(qubits):
        raise ValueError("Value does not fit its basis register")
    for bit, qubit in enumerate(qubits):
        if int(value) >> bit & 1:
            circuit.x(qubit)


def mark_pattern(circuit, controls, pattern, target):
    controls = list(controls)
    flips = [q for bit, q in enumerate(controls) if not int(pattern) >> bit & 1]
    if flips:
        circuit.x(flips)
    if controls:
        circuit.mcx(controls, target)
    else:
        circuit.x(target)
    if flips:
        circuit.x(flips)


def mcx_clean(circuit, controls, target, scratch):
    """Toffoli ladder with clean work qubits, restored to zero."""
    controls=list(controls)
    if len(controls)==1:
        circuit.cx(controls[0],target)
    elif len(controls)==2:
        circuit.ccx(*controls,target)
    elif len(controls)>2:
        if len(scratch)<len(controls)-2:
            raise ValueError('Insufficient clean work qubits')
        circuit.ccx(controls[0],controls[1],scratch[0])
        for i in range(2,len(controls)-1):
            circuit.ccx(controls[i],scratch[i-2],scratch[i-1])
        circuit.ccx(controls[-1],scratch[len(controls)-3],target)
        for i in reversed(range(2,len(controls)-1)):
            circuit.ccx(controls[i],scratch[i-2],scratch[i-1])
        circuit.ccx(controls[0],controls[1],scratch[0])
    else:
        circuit.x(target)


def clean_pattern(circuit, controls, pattern, target, scratch, phase=False):
    controls=list(controls)
    flips=[q for bit,q in enumerate(controls) if not int(pattern)>>bit&1]
    if flips: circuit.x(flips)
    if phase:
        circuit.h(controls[-1])
        mcx_clean(circuit,controls[:-1],controls[-1],scratch)
        circuit.h(controls[-1])
    else:
        mcx_clean(circuit,controls,target,scratch)
    if flips: circuit.x(flips)


def qrom(circuit, address, target, records):
    """Load RAW basis records into a ZERO target; inverse unloads them.

    Uniform RY(0/pi) multiplexors implement |i,0> -> |i,record_i> without
    record-dependent phases on this input subspace. This is not XOR lookup
    on an arbitrary nonzero target. No conditional prediction table is fitted.
    """
    from qiskit.circuit.library import UCRYGate
    if len(records) > 1 << len(address):
        raise ValueError("Too many QROM records")
    for value in records:
        if not 0 <= int(value) < 1 << len(target):
            raise ValueError("QROM record does not fit its register")
    for bit, qubit in enumerate(target):
        angles = [np.pi if i < len(records) and int(records[i]) >> bit & 1 else 0.
                  for i in range(1 << len(address))]
        circuit.append(UCRYGate(angles), [qubit, *address])


def uniform_records(count):
    if count < 1:
        raise ValueError("Empty quantum memory")
    width = max(1, (count-1).bit_length())
    values = np.zeros(1 << width)
    values[:count] = 1
    return amplitude_preparation(values)


def controlled_amplitudes(records):
    """Multiplexed state loading from raw vectors, without pairwise scores."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UCRYGate
    records = np.asarray(records, dtype=float)
    if records.ndim != 2 or len(records) < 1 or np.any(records < 0) or not np.isfinite(records).all():
        raise ValueError("Expected nonnegative finite raw record vectors")
    size = records.shape[1]
    if size < 2 or size & (size-1) or np.any(np.linalg.norm(records, axis=1) == 0):
        raise ValueError("Nonzero power-of-two record vectors required")
    index_width = max(1, (len(records)-1).bit_length())
    data_width = int(np.log2(size))
    circuit = QuantumCircuit(index_width+data_width, name="raw_record_loader")
    for target in range(data_width-1, -1, -1):
        upper = list(range(target+1, data_width))
        span = 1 << target
        angles = []
        for record in range(1 << index_width):
            for prefix in range(1 << len(upper)):
                if record >= len(records):
                    angles.append(0.)
                    continue
                start = prefix*2*span
                left = np.linalg.norm(records[record, start:start+span])
                right = np.linalg.norm(records[record, start+span:start+2*span])
                angles.append(float(2*np.arctan2(right, left)))
        circuit.append(UCRYGate(angles), [index_width+target,
            *[index_width+q for q in upper], *range(index_width)])
    return circuit


def execute(circuits, shots, seed=42, readout_error=0., precompiled=False):
    """Measured native circuits. Noise affects declared output bits, not flags."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    from time import perf_counter
    import os
    if shots < 1 or not circuits or not 0 <= readout_error < .5:
        raise ValueError("Invalid circuit/shot/noise budget")
    threads=int(os.environ.get('QCENN_THREADS','1'))
    if threads<1:
        raise ValueError('QCENN_THREADS must be a positive integer')
    noise = None
    if readout_error:
        noise = NoiseModel()
        for qubit in circuits[0].metadata["output_qubits"]:
            noise.add_readout_error(ReadoutError([[1-readout_error, readout_error],
                [readout_error, 1-readout_error]]), [qubit])
    # One OpenMP layer per process. The Bash runner divides the total CPU
    # budget among independent workers; BLAS/transpilation do not nest pools.
    backend = AerSimulator(method="matrix_product_state", max_parallel_threads=threads,
        max_parallel_experiments=1, max_parallel_shots=1, mps_omp_threads=threads,
        matrix_product_state_truncation_threshold=1e-20, noise_model=noise,
        # Direct MPS sampling avoids constructing a large probability vector
        # for wide coherent memories, even when few classical bits are read.
        mps_sample_measure_algorithm="mps_apply_measure")
    start = perf_counter()
    native = list(circuits) if precompiled else transpile(circuits,
        basis_gates=["rz", "sx", "x", "cx", "reset"],
        optimization_level=1, seed_transpiler=seed, num_processes=1)
    compile_seconds = perf_counter()-start
    start = perf_counter()
    result = backend.run(native, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(str(result.status))
    counts = [result.get_counts(i) for i in range(len(native))]
    if any(sum(c.values()) != shots for c in counts):
        raise RuntimeError("Missing measured shots")
    return counts, native, {"compilation_seconds": compile_seconds,
        "cpu_thread_budget":threads,"mps_omp_threads":threads,
        "parallel_experiments":1,"parallel_shots":1,"device":"CPU",
        "mps_truncation_threshold":1e-20,
        "mps_sampling_algorithm":"mps_apply_measure",
        "sampling_seconds": perf_counter()-start, "total_shots": shots*len(native),
        "qubits_max": max(c.num_qubits for c in native),
        "depth_max": max(c.depth() for c in native),
        "cx_total_per_execution": sum(c.count_ops().get("cx", 0) for c in native),
        "cx_shot_weighted_total": shots*sum(c.count_ops().get("cx", 0) for c in native)}


def decode_postselection(counts, value_bits, success_bits, value_count):
    histogram = np.zeros(1 << value_bits, dtype=int)
    accepted = 0
    for key, count in counts.items():
        value = int(key.replace(" ", ""), 2)
        if value >> value_bits == (1 << success_bits)-1:
            token = value & ((1 << value_bits)-1)
            histogram[token] += count
            accepted += count
    total = sum(counts.values())
    winner = int(histogram.argmax()) if accepted else -1
    return {"prediction": winner if 0 <= winner < value_count else -1,
        "histogram": histogram[:value_count].tolist(),
        "invalid_value_shots": int(histogram[value_count:].sum()),
        "accepted_shots": int(accepted),
        "total_shots": int(total), "accepted_shot_fraction": accepted/total,
        "counts": counts}


CORE_DISCLOSURE = {
    "classical_optimizer": False, "fitted_classical_weights": False,
    "numerical_model_inference": "quantum_gates_and_final_measurements",
    "learning_mode": "nonparametric_raw_record_memory_loading",
    "quantum_neural_weight_training_implemented": False,
    "classical_input_loading_compilation_and_count_decoding": True,
    "entire_host_pipeline_pure_quantum": False,
    "physical_qpu_tested": False,
}
