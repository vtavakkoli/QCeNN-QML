"""Small conditional quantum replacement channels for masked-token denoising.

The noisy forward channel masks a token. Recurrent collision-style reverse
channels draw replacement token amplitudes from a fixed classical toy-corpus
count table encoded in quantum controlled gates. There is no trainable QNN,
no SmolLM weights, and no reproduction of QCeNN-MI's inverse solver.

In a noiseless simulator, tracing out auxiliary registers implements
E_c(rho) = (1 - eta) rho + eta |psi_c><psi_c|, a valid CPTP map.
Each iteration keeps the same target register, using device-level resets
instead of measuring/reconstructing that target between iterations.
"""
from collections import Counter
from math import asin, sqrt
import numpy as np

VOCAB = ("cat", "dog", "sat", "[MASK]")
MASK = 3
# Small, deliberately repetitive toy corpus; not general-purpose language.
TOY_COUNTS = (
    ("cat", "sat", "cat", 12),
    ("dog", "sat", "dog", 12),
    ("cat", "dog", "dog", 8),
    ("dog", "cat", "cat", 8),
    ("sat", "cat", "dog", 4),
    ("sat", "dog", "cat", 4),
)

def dataset(seed=42):
    rng = np.random.default_rng(seed)
    data = [(VOCAB.index(a), VOCAB.index(b), VOCAB.index(c))
            for a, b, c, repeats in TOY_COUNTS for _ in range(repeats)]
    indices = rng.permutation(len(data))
    split = int(0.75 * len(data))
    return [data[i] for i in indices[:split]], [data[i] for i in indices[split:]]


def context_number(left, right):
    if not all(isinstance(x, (int, np.integer)) and 0 <= x < 4 for x in (left, right)):
        raise ValueError("Invalid context tokens")
    return int(left) + 4 * int(right)


def conditional_distribution(train, smoothing=0.2):
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be positive and finite")
    counts = np.zeros((16, 4), dtype=float)
    counts[:, :3] = smoothing
    for left, target, right in train:
        if MASK in (left, target, right):
            raise ValueError("Toy training sequences cannot contain masks")
        counts[context_number(left, right), target] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def expected_clean_probability(original, distribution, beta, eta, steps):
    """Independent analytic oracle, never used to make a model prediction."""
    if not 0 <= beta <= 1 or not 0 <= eta <= 1 or steps < 0:
        raise ValueError("Invalid channel parameters")
    p = float(distribution[original])
    return (1-eta)**steps * (1-beta) + (1-(1-eta)**steps) * p


def _x_state(qc, qubits, value):
    for i, qubit in enumerate(qubits):
        if value & (1 << i):
            qc.x(qubit)


def _flag_swap(qc, probability, target, environment, flag):
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Swap probability must lie in [0,1]")
    if probability == 0:
        return
    if probability == 1:
        for a, b in zip(target, environment, strict=True):
            qc.swap(a, b)
    else:
        qc.ry(2*asin(sqrt(probability)), flag)
        for a, b in zip(target, environment, strict=True):
            qc.cswap(flag, a, b)
    qc.reset(list(environment) + [flag])


def controlled_token_loader(qc, table, context=(0,1,2,3), ancillas=(6,7)):
    """Context-controlled quantum distribution, synthesized from RY gates.

    For each of 16 context basis states, apply controlled rotations that
    prepare sqrt(p0)|00>+sqrt(p1)|01>+sqrt(p2)|10>+sqrt(p3)|11>.
    The gate sequence is shared across all inputs; no CPU conditional lookup
    determines the output at prediction time.
    """
    from qiskit.circuit.library import RYGate
    if np.shape(table) != (16,4):
        raise ValueError("Expected sixteen conditional distributions")
    controls = list(context)
    low, high = ancillas
    for c in range(16):
        probabilities = np.asarray(table[c], dtype=float)
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(),1.0):
            raise ValueError("Invalid conditional probability row")
        flips = [q for j,q in enumerate(controls) if not (c >> j) & 1]
        if flips:
            qc.x(flips)
        left = float(np.sqrt(probabilities[0]+probabilities[1]))
        right = float(np.sqrt(probabilities[2]+probabilities[3]))
        theta_high = float(2*np.arctan2(right,left))
        if abs(theta_high) > 1e-14:
            qc.append(RYGate(theta_high).control(len(controls)), controls+[high])
        # Conditional probability of low bit when high bit = 0.
        theta_low0 = float(2*np.arctan2(np.sqrt(probabilities[1]),
                                       np.sqrt(probabilities[0])))
        if abs(theta_low0) > 1e-14:
            qc.x(high)
            qc.append(RYGate(theta_low0).control(len(controls)+1),
                      controls+[high,low])
            qc.x(high)
        # Conditional probability of low bit when high bit = 1.
        theta_low1 = float(2*np.arctan2(np.sqrt(probabilities[3]),
                                       np.sqrt(probabilities[2])))
        if abs(theta_low1) > 1e-14:
            qc.append(RYGate(theta_low1).control(len(controls)+1),
                      controls+[high,low])
        if flips:
            qc.x(flips)


def denoising_circuit(left, original, right, *, table, beta=1.0,
                      eta=0.65, steps=3):
    """Prepare 3-token input, corrupt middle token, then denoise coherently.

    Context qubits 0..3 stay fixed and quantum-controlled. Target qubits 4,5
    are never measured or classically reconstructed until final measurement.
    Environment qubits 6,7 and flag qubit 8 are reset between rounds.
    """
    from qiskit import QuantumCircuit
    if steps < 0 or not isinstance(steps, int) or steps > 8:
        raise ValueError("steps must be 0 through 8")
    if any(x not in range(4) for x in (left, original, right)):
        raise ValueError("Invalid vocabulary IDs")
    qc = QuantumCircuit(9,2)
    _x_state(qc, (0,1), left)
    _x_state(qc, (2,3), right)
    _x_state(qc, (4,5), original)
    # Forward stochastic masking is a genuine quantum swap/reset channel.
    _x_state(qc, (6,7), MASK)
    _flag_swap(qc, beta, (4,5), (6,7), 8)
    for _ in range(steps):
        controlled_token_loader(qc, table)
        _flag_swap(qc, eta, (4,5), (6,7), 8)
    qc.measure([4,5],[0,1])
    qc.metadata = {
        "architecture": "toy_context_conditional_recurrent_quantum_replacement",
        "no_intermediate_target_measurements": True,
        "quantum_data_qubits": 6, "total_qubits": 9,
        "classical_count_table_compilation": True, "trained_qnn": False,
        "qcenn_mi_collision_solver_imported": False,
    }
    return qc


def counts_to_distribution(counts, shots):
    out = np.zeros(4,dtype=float)
    for bits, n in counts.items():
        out[int(bits.replace(" ",""),2)] += n
    return out / shots
