"""Trainable local collision channels with four retained memory qubits.

Inspired by QCeNN-MI — Vahid Tavakkoli and Kabeh Mohsenzadegan (2026),
https://github.com/vtavakkoli/QCeNN-MI . Independently authored local-channel
specialization, not the upstream matrix-inversion residual solver.

The NumPy reduced-density evolution is for hybrid training/verification only.
Inference uses the explicit five-qubit Qiskit circuit and final measurements.
"""
import numpy as np


MEMORY_QUBITS = 4
PARAMETER_COUNT = 19
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_BASIS = np.arange(16)


def initial_parameters(seed):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, .2, PARAMETER_COUNT)
    theta[:7] = (0, .7, 0, .4, .15, .2, -2)
    return theta


def validate_parameters(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (PARAMETER_COUNT,) or not np.isfinite(theta).all():
        raise ValueError(f"Expected {PARAMETER_COUNT} finite quantum gate parameters")
    return theta


def collision_angle(value):
    # Bounded physical angle: p = sin(gamma)**2, with 0 < p < 1.
    return float(np.pi / (2 * (1 + np.exp(-np.clip(value, -30, 30)))))


def rotation(axis, angles):
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    c, s = np.cos(angles / 2), np.sin(angles / 2)
    gates = np.zeros((len(angles), 2, 2), dtype=complex)
    if axis == "y":
        gates[:, 0, 0] = gates[:, 1, 1] = c
        gates[:, 0, 1], gates[:, 1, 0] = -s, s
    elif axis == "z":
        gates[:, 0, 0], gates[:, 1, 1] = np.exp(-.5j * angles), np.exp(.5j * angles)
    else:
        raise ValueError("Expected y or z rotation")
    return gates


def apply_single(density, gates, qubit):
    """Batched U rho U†; basis bit zero is Qiskit's least-significant bit."""
    lo = _BASIS[(_BASIS & (1 << qubit)) == 0]
    hi = lo | (1 << qubit)
    gates = np.asarray(gates)
    if gates.ndim == 2:
        gates = gates[None]
    a, b, c, d = (gates[:, i, j, None, None] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    left = np.empty_like(density)
    left[:, lo, :] = a * density[:, lo, :] + b * density[:, hi, :]
    left[:, hi, :] = c * density[:, lo, :] + d * density[:, hi, :]
    out = np.empty_like(density)
    out[:, :, lo] = a.conj() * left[:, :, lo] + b.conj() * left[:, :, hi]
    out[:, :, hi] = c.conj() * left[:, :, lo] + d.conj() * left[:, :, hi]
    return out


def apply_damping(density, gamma, qubit):
    """Local C rho C† + Tr(F rho F†)|0><0| collision channel.

    On this cell C=diag(1,cos(gamma)), F=sin(gamma)|0><1|. The full-memory
    channel tensors the two local Kraus operators with identity, preserving
    correlations with all other memory qubits.
    """
    lo = _BASIS[(_BASIS & (1 << qubit)) == 0]
    hi = lo | (1 << qubit)
    scale = np.ones(16)
    scale[hi] = np.cos(gamma)
    out = density * scale[None, :, None] * scale[None, None, :]
    out[:, lo[:, None], lo] += np.sin(gamma)**2 * density[:, hi[:, None], hi]
    return out


def cell_density(density, features, theta):
    """One input-driven, shared-template update; no state reconstruction."""
    for qubit in range(4):
        density = apply_single(density, rotation("y", theta[0] + np.pi * theta[1] * features[:, qubit]), qubit)
        density = apply_single(density, rotation("z", theta[2] + np.pi * theta[3] * features[:, qubit]), qubit)
    for first, second in EDGES:
        permutation = _BASIS ^ ((1 << first) | (1 << second))
        left, right = density[:, permutation, :], density[:, :, permutation]
        both = left[:, :, permutation]
        c, s = np.cos(theta[4] / 2), np.sin(theta[4] / 2)
        density = c*c*density + s*s*both + 1j*c*s*(right-left)
        signs = 1 - 2 * (((_BASIS >> first) ^ (_BASIS >> second)) & 1)
        phase = np.exp(-.5j * theta[5] * signs)
        density = density * phase[None, :, None] * phase.conj()[None, None, :]
    for qubit in range(4):
        density = apply_damping(density, collision_angle(theta[6]), qubit)
    return density


def evolve(patches, theta, *, retain_memory=True):
    """Exact reduced-state simulator for training, never measured inference."""
    theta = validate_parameters(theta)
    patches = np.asarray(patches, dtype=float)
    if patches.ndim != 3 or patches.shape[2] != 4 or patches.shape[1] < 1:
        raise ValueError("Expected (examples, patches, 4) input")
    if not np.isfinite(patches).all() or np.any((patches < 0) | (patches > 1)):
        raise ValueError("Patch intensities must lie in [0,1]")
    density = np.zeros((len(patches), 16, 16), dtype=complex)
    density[:, 0, 0] = 1
    # Resetting before each patch is exactly equivalent to using only the last.
    sequence = patches if retain_memory else patches[:, -1:]
    for step in range(sequence.shape[1]):
        density = cell_density(density, sequence[:, step], theta)
    for qubit in range(4):
        density = apply_single(density, rotation("y", theta[7 + qubit]), qubit)
        density = apply_single(density, rotation("z", theta[11 + qubit]), qubit)
    for control, target in ((0, 1), (1, 2), (2, 3)):
        permutation = _BASIS ^ (((_BASIS >> control) & 1) << target)
        density = density[:, permutation, :][:, :, permutation]
    for qubit in range(4):
        density = apply_single(density, rotation("y", theta[15 + qubit]), qubit)
    return density


def probabilities(patches, theta, *, retain_memory=True):
    diagonal = np.diagonal(evolve(patches, theta, retain_memory=retain_memory), axis1=1, axis2=2).real
    return np.clip(diagonal, 0, 1)


def recurrent_circuit(patches, theta, *, retain_memory=True, measure=True):
    """All memory updates are quantum; one environment qubit is reset/reused."""
    from qiskit import QuantumCircuit
    theta = validate_parameters(theta)
    patches = np.asarray(patches, dtype=float)
    if patches.ndim != 2 or patches.shape[1] != 4 or len(patches) < 1:
        raise ValueError("Expected nonempty (patches, 4) input")
    if not np.isfinite(patches).all() or np.any((patches < 0) | (patches > 1)):
        raise ValueError("Patch intensities must lie in [0,1]")
    circuit = QuantumCircuit(5, 4 if measure else 0)
    sequence = patches if retain_memory else patches[-1:]
    for features in sequence:
        for qubit in range(4):
            circuit.ry(theta[0] + np.pi * theta[1] * features[qubit], qubit)
            circuit.rz(theta[2] + np.pi * theta[3] * features[qubit], qubit)
        for first, second in EDGES:
            circuit.rxx(theta[4], first, second)
            circuit.rzz(theta[5], first, second)
        for qubit in range(4):
            circuit.cry(2 * collision_angle(theta[6]), qubit, 4)
            circuit.cx(4, qubit)
            circuit.reset(4)
    for qubit in range(4):
        circuit.ry(theta[7 + qubit], qubit)
        circuit.rz(theta[11 + qubit], qubit)
    for control, target in ((0, 1), (1, 2), (2, 3)):
        circuit.cx(control, target)
    for qubit in range(4):
        circuit.ry(theta[15 + qubit], qubit)
    if measure:
        circuit.measure(range(4), range(4))
    circuit.metadata = {"model": "local_recurrent_collision_classifier",
                        "retained_memory_qubits": 4, "reused_environment_qubits": 1,
                        "classical_iterate_updates": 0, "intermediate_measurements": 0,
                        "patches_processed": len(sequence), "physical_qpu_tested": False}
    return circuit


def decode_counts(counts):
    """Digits are basis outcomes 0..9; 10..15 are explicitly rejected.

    All-rejected samples predict -1 and count as errors. No ideal probabilities,
    classical classifier, or true labels enter this readout.
    """
    histogram = np.zeros(16, dtype=int)
    for bits, count in counts.items():
        value = int(bits.replace(" ", ""), 2)
        if not 0 <= value < 16 or int(count) != count or count < 0:
            raise ValueError("Invalid four-bit measurement counts")
        histogram[value] += int(count)
    accepted = int(histogram[:10].sum())
    return {"prediction": int(np.argmax(histogram[:10])) if accepted else -1,
            "counts": histogram.tolist(), "accepted_shots": accepted,
            "total_shots": int(histogram.sum()),
            "accepted_shot_fraction": accepted / max(1, int(histogram.sum()))}
