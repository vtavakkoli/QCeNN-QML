"""Coherent resident-index recurrence: monotone p**rounds kernel attention."""
import numpy as np
from .coherent import controlled_amplitudes, load_basis, qrom, uniform_records
from .quantum import amplitude_preparation


def memory_circuit(query, keys, labels, *, rounds=4, encoding="amplitude",
                   damping=0., value_bits=4, max_qubits=96, reset_memory=False):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UCRYGate
    query, keys = np.asarray(query, float), np.asarray(keys, float)
    labels = np.asarray(labels, int)
    if keys.ndim != 2 or query.shape != keys.shape[1:] or len(keys) != len(labels):
        raise ValueError("Query/key/value shapes disagree")
    if not 1 <= rounds <= 8 or not 0 <= damping < 1 or encoding not in ("amplitude", "angle"):
        raise ValueError("Invalid recurrence, damping or encoding")
    if not np.isfinite(keys).all() or not np.isfinite(query).all() or np.any(keys < 0) or np.any(query < 0):
        raise ValueError("Finite nonnegative inputs required")
    index_width = max(1, (len(keys)-1).bit_length())
    if encoding == "amplitude":
        loader = controlled_amplitudes(keys)
        query_loader = amplitude_preparation(query)
        data_width = query_loader.num_qubits
    else:
        if np.any(keys > 1) or np.any(query > 1):
            raise ValueError("Angle encoding requires [0,1] intensities")
        data_width = len(query)
        if data_width>16:
            raise MemoryError("The angle-encoding ablation is bounded to 16 features; use --side 4")
        loader, query_loader = None, None
    width = index_width*(rounds if reset_memory else 1) + rounds*(data_width+1) + value_bits + int(damping > 0)
    if width > max_qubits:
        raise MemoryError(f"Circuit needs {width} qubits, budget {max_qubits}")
    circuit = QuantumCircuit(width, value_bits+rounds)
    index = list(range(index_width))
    if not reset_memory:
        circuit.compose(uniform_records(len(keys)), index, inplace=True)
    offset = 0 if reset_memory else index_width
    flags = []
    for _ in range(rounds):
        if reset_memory:
            # Fresh independent index; old memory is retained as environment.
            # This is the coherent dilation of reset/reprepare, with no CPU update.
            index=list(range(offset,offset+index_width))
            circuit.compose(uniform_records(len(keys)),index,inplace=True)
            offset+=index_width
        work = list(range(offset, offset+data_width)); flag = offset+data_width
        if encoding == "amplitude":
            circuit.compose(query_loader, work, inplace=True)
            circuit.compose(loader.inverse(), [*index, *work], inplace=True)
        else:
            for feature, qubit in enumerate(work):
                circuit.ry(np.pi*query[feature], qubit)
                angles = [-np.pi*keys[i, feature] if i < len(keys) else 0.
                          for i in range(1 << index_width)]
                circuit.append(UCRYGate(angles), [qubit, *index])
        circuit.x(work)
        circuit.mcx(work, flag)
        circuit.x(work)
        flags.append(flag)
        if damping:
            environment = width-1
            for qubit in index:
                circuit.cry(2*np.arcsin(np.sqrt(damping)), qubit, environment)
                circuit.cx(environment, qubit)
                circuit.reset(environment)
        offset += data_width+1
    values = list(range(offset, offset+value_bits))
    qrom(circuit, index, values, labels)
    circuit.measure(values, range(value_bits))
    circuit.measure(flags, range(value_bits, value_bits+rounds))
    circuit.metadata = {"output_qubits": values, "resident_index_qubits": index_width,
        "quantum_recurrence_rounds": rounds, "classical_state_feedback": False,
        "retained_index_across_rounds":not reset_memory,
        "classical_optimizer": False, "encoding": encoding, "damping": damping,
        "readout": "label_register_conditioned_on_all_overlap_flags"}
    return circuit


def reference_distribution(query, keys, labels, *, rounds=4, encoding="amplitude", values=10):
    """Independent CLASSICAL BENCHMARK ONLY. Never called by memory_circuit."""
    keys, query = np.asarray(keys, float), np.asarray(query, float)
    if encoding == "amplitude":
        q = query/np.linalg.norm(query)
        k = keys/np.linalg.norm(keys, axis=1, keepdims=True)
        scores = np.clip((k@q)**2, 0, 1)**rounds
    else:
        scores = np.prod(np.cos(np.pi*(keys-query)/2)**2, axis=1)**rounds
    probabilities = np.bincount(labels, weights=scores, minlength=values)/len(keys)
    acceptance = float(probabilities.sum())
    return probabilities/max(acceptance, 1e-300), acceptance
