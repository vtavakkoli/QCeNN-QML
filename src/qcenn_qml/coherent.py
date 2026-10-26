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


def independent_components(circuit):
    """Exact tensor-product simulation optimization, never a model prediction.

    Only explicitly opted-in circuits with disjoint quantum components and
    distinct final classical destinations qualify. All original clbits remain
    in each component so measured shot strings can be combined without scores.
    """
    if not (circuit.metadata or {}).get('factorize_independent_registers'):
        return [circuit]
    from qiskit.converters import circuit_to_dag,dag_to_circuit
    measured=set()
    for instruction in circuit.data:
        op=instruction.operation
        if getattr(op,'condition',None) is not None or hasattr(op,'blocks'):
            return [circuit]
        if instruction.clbits:
            if op.name!='measure' or instruction.clbits[0] in measured:
                return [circuit]
            measured.add(instruction.clbits[0])
    parts=[]
    for dag in circuit_to_dag(circuit).separable_circuits(remove_idle_qubits=True):
        part=dag_to_circuit(dag)
        if not any(e.operation.name=='measure' for e in part.data):
            continue # An independent, unobserved component has trace one.
        original=[circuit.find_bit(q).index for q in part.qubits]
        part.metadata={'output_qubits':[original.index(q) for q in
            (circuit.metadata or {}).get('output_qubits',[]) if q in original],
            'original_qubits':original}
        parts.append(part)
    return sorted(parts,key=lambda p:min(p.metadata['original_qubits'])) if parts else [circuit]


def _aer_result(native,shots,seed,readout_error,threads,*,memory=False):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    from time import perf_counter
    from concurrent.futures import TimeoutError as FutureTimeout
    noise = None
    if readout_error and (native[0].metadata or {}).get('output_qubits'):
        noise = NoiseModel()
        for qubit in native[0].metadata["output_qubits"]:
            noise.add_readout_error(ReadoutError([[1-readout_error, readout_error],
                [readout_error, 1-readout_error]]), [qubit])
    # Parallelize independent circuits when present; otherwise allow one MPS
    # thread pool. Never nest both pools or exceed the worker's CPU budget.
    parallel=min(threads,len(native))
    mps_threads=threads if parallel==1 else 1
    backend = AerSimulator(method="matrix_product_state", max_parallel_threads=threads,
        max_parallel_experiments=parallel, max_parallel_shots=1, mps_omp_threads=mps_threads,
        matrix_product_state_truncation_threshold=1e-20, noise_model=noise,
        # Direct MPS sampling avoids constructing a large probability vector
        # for wide coherent memories, even when few classical bits are read.
        mps_sample_measure_algorithm="mps_apply_measure")
    print(f'Aer sampling: circuits={len(native)} qubits={max(c.num_qubits for c in native)} '
        f'shots={shots} CPU-thread-budget={threads} parallel-circuits={parallel}',flush=True)
    started=perf_counter()
    job=backend.run(native,shots=shots,seed_simulator=seed,memory=memory)
    while True:
        try:
            result=job.result(timeout=30)
            break
        except FutureTimeout:
            print(f'Aer still running: elapsed={perf_counter()-started:.0f}s; '
                'no shot-progress estimate available',flush=True)
    if not result.success:
        raise RuntimeError(str(result.status))
    return result


def _compile_components(circuit,seed):
    """Prevent MCX synthesis borrowing dirty ancillas from unrelated blocks."""
    from qiskit import transpile
    options=dict(basis_gates=['rz','sx','x','cx','reset'],
        optimization_level=1,seed_transpiler=seed,num_processes=1)
    parts=independent_components(circuit)
    if len(parts)==1:
        return transpile(circuit,**options)
    native=circuit.copy_empty_like()
    measurements=[]
    for part in parts:
        mapping=part.metadata['original_qubits']
        compiled=transpile(part,**options)
        if compiled.num_qubits!=len(mapping) or compiled.layout is not None:
            raise RuntimeError('Unexpected component qubit remapping')
        native.global_phase+=compiled.global_phase
        for instruction in compiled.data:
            qubits=[native.qubits[mapping[compiled.find_bit(q).index]] for q in instruction.qubits]
            clbits=[native.clbits[compiled.find_bit(c).index] for c in instruction.clbits]
            if instruction.operation.name=='measure':
                measurements.append((instruction.operation,qubits,clbits))
            else:
                native.append(instruction.operation,qubits,clbits)
    # All component measurements commute; retain the model's final-only readout.
    for operation,qubits,clbits in measurements:
        native.append(operation,qubits,clbits)
    return native


def execute(circuits, shots, seed=42, readout_error=0., precompiled=False):
    """Measured native circuits. Noise affects declared output bits, not flags."""
    from time import perf_counter
    from collections import Counter
    import os
    if shots < 1 or not circuits or not 0 <= readout_error < .5:
        raise ValueError("Invalid circuit/shot/noise budget")
    threads=int(os.environ.get('QCENN_THREADS','1'))
    if threads<1:
        raise ValueError('QCENN_THREADS must be a positive integer')
    print(f'Transpiling {len(circuits)} circuits',flush=True)
    start = perf_counter()
    native = list(circuits) if precompiled else [_compile_components(c,seed) for c in circuits]
    compile_seconds = perf_counter()-start
    start = perf_counter()
    components=[independent_components(c) for c in native]
    if all(len(parts)==1 for parts in components):
        result=_aer_result(native,shots,seed,readout_error,threads)
        counts=[result.get_counts(i) for i in range(len(native))]
    else:
        counts=[]
        for index,(circuit,parts) in enumerate(zip(native,components,strict=True)):
            joint=[0]*shots
            print(f'Tensor-product sampling: {circuit.num_qubits} logical qubits; '
                f'independent components={[p.num_qubits for p in parts]}',flush=True)
            for component,part in enumerate(parts):
                component_seed=int(np.random.SeedSequence([seed,index,component]).generate_state(1)[0])
                result=_aer_result([part],shots,component_seed,readout_error,threads,memory=True)
                for shot,bits in enumerate(result.get_memory(0)):
                    joint[shot] |= int(bits.replace(' ',''),2)
            counts.append(dict(Counter(format(value,f'0{circuit.num_clbits}b') for value in joint)))
    if any(sum(c.values()) != shots for c in counts):
        raise RuntimeError("Missing measured shots")
    factorized=any(len(parts)>1 for parts in components)
    parallel=1 if factorized else min(threads,len(native))
    return counts, native, {"compilation_seconds": compile_seconds,
        "cpu_thread_budget":threads,"mps_omp_threads":threads if parallel==1 else 1,
        "parallel_experiments":parallel,"parallel_shots":1,"device":"CPU",
        "simulation_component_qubits":[[p.num_qubits for p in parts] for parts in components],
        "tensor_product_sampling":factorized,
        "simulated_component_shots":shots*sum(len(parts) for parts in components),
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
