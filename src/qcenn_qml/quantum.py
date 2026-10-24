"""Quantum state-preparation and reflection primitives.

Dense classical state loading is required; it is not a free oracle.
Measurement outcomes are the sole source of runtime prediction scores.
"""
import numpy as np


def unit_vector(values):
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or len(raw) < 2 or len(raw) & (len(raw) - 1):
        raise ValueError("Amplitudes require a power-of-two vector")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Nonfinite amplitudes")
    norm = float(np.linalg.norm(raw))
    if norm <= 0:
        raise ValueError("Zero amplitude vector")
    return raw / norm


def zero_reflection(num_qubits):
    from qiskit import QuantumCircuit
    if num_qubits < 1:
        raise ValueError("Must use at least one qubit")
    qc = QuantumCircuit(num_qubits)
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    return qc


def basis_reflection(num_qubits, qubits, target_bits):
    """Mark one little-endian basis pattern on a selected qubit subset."""
    from qiskit import QuantumCircuit
    qubits = list(qubits)
    if not qubits or len(set(qubits)) != len(qubits) or max(qubits) >= num_qubits:
        raise ValueError("Invalid qubit subset")
    if not 0 <= target_bits < (1 << len(qubits)):
        raise ValueError("Invalid target")
    qc = QuantumCircuit(num_qubits)
    flips = [q for i, q in enumerate(qubits) if not (target_bits >> i) & 1]
    qc.x(flips)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    qc.x(flips)
    return qc


def sample_counts(circuits, shots, seed):
    """Run actual quantum gates with final measurements in Aer."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    if shots < 1:
        raise ValueError("Shots must be positive")
    backend = AerSimulator(method="statevector", max_parallel_threads=1)
    compiled = transpile(circuits, backend, optimization_level=1,
                         seed_transpiler=seed)
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(str(result.status))
    return [result.get_counts(i) for i in range(len(circuits))], compiled


def grover_response(probability, iterations):
    """Independent analytic TEST ORACLE, never an inference shortcut."""
    if not 0 <= probability <= 1 or iterations < 0:
        raise ValueError("Bad probability or iterations")
    return float(np.sin((2 * iterations + 1) * np.arcsin(np.sqrt(probability))) ** 2)
