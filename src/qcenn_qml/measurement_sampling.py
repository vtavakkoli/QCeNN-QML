"""Final-readout instrument and measured-bit marginalization, never a predictor.

Aer 0.18.2 MPS sample_measure_all uses a linear pass through the MPS per shot.
Measuring extra qubits at the END commutes with the original Z readout and
preserves its marginal. We still evolve the native gates and sample all shots.
"""
from collections import Counter
import gzip
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path


def readout_policy():
    requested = os.environ.get('QCENN_MPS_READOUT', 'auto')
    try:
        aer_version = version('qiskit-aer')
    except PackageNotFoundError:
        aer_version = version('qiskit-aer-gpu')
    if requested not in ('auto', 'full_marginal', 'partial'):
        raise ValueError('QCENN_MPS_READOUT must be auto, full_marginal or partial')
    # Earlier implementations may materialize an exponential probability vector.
    # Enable only the implementation whose source and wide-circuit tests we verify.
    verified = aer_version == '0.18.2'
    if requested == 'full_marginal' and not verified:
        raise RuntimeError('Fast full-register readout requires the tested qiskit-aer 0.18.2; use the isolated upgrade script')
    return verified and requested != 'partial'


def full_readout(circuit):
    """Return an equivalent final-readout instrument and old-bit -> qubit map.

    Restrict to unique, final measurements. Arbitrary classical conditions,
    overwritten bits and intermediate readouts retain the original instrument.
    """
    from qiskit import QuantumCircuit
    mapping, seen, measuring = {}, set(), False
    for instruction in circuit.data:
        operation = instruction.operation
        if getattr(operation, 'condition', None) is not None or hasattr(operation, 'blocks'):
            return circuit, None
        if operation.name == 'measure':
            measuring = True
            qubit = circuit.find_bit(instruction.qubits[0]).index
            bit = circuit.find_bit(instruction.clbits[0]).index
            if bit in mapping or qubit in seen:
                return circuit, None
            mapping[bit] = qubit; seen.add(qubit)
        elif instruction.clbits or (measuring and operation.name != 'barrier'):
            return circuit, None
    if not mapping:
        return circuit, None
    executed = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
    executed.global_phase = circuit.global_phase
    for instruction in circuit.data:
        if instruction.operation.name != 'measure':
            executed.append(instruction.operation,
                [executed.qubits[circuit.find_bit(q).index] for q in instruction.qubits], [])
    executed.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    executed.metadata = dict(circuit.metadata or {})
    return executed, mapping


def project(bits, mapping, width):
    value = int(bits.replace(' ', ''), 2)
    output = sum(((value >> qubit) & 1) << bit for bit, qubit in mapping.items())
    return format(output, f'0{width}b')


_AUDIT_WRITTEN = False


class MarginalResult:
    def __init__(self, result, original, executed, mappings, seed, shots, memory):
        global _AUDIT_WRITTEN
        self.result, self.original, self.mappings = result, original, mappings
        self.success, self.status = result.success, result.status
        full = [dict(result.get_counts(i)) for i in range(len(original))]
        self.counts = []
        for counts, circuit, mapping in zip(full, original, mappings, strict=True):
            marginal = Counter()
            for bits, count in counts.items():
                marginal[project(bits, mapping, circuit.num_clbits) if mapping is not None else bits] += count
            if sum(marginal.values()) != shots:
                raise RuntimeError('Full-register marginalization lost measured shots')
            self.counts.append(dict(marginal))
        identity = hashlib.sha256(json.dumps(full, sort_keys=True).encode()).hexdigest()
        audit = None
        progress = os.environ.get('QCENN_PROGRESS_FILE')
        if progress and not _AUDIT_WRITTEN and os.environ.get('QCENN_SAVE_FULL_READOUT_AUDIT', '1') != '0':
            path = Path(progress).parent / f'full-readout-first-s{seed}-n{shots}.json.gz'
            temporary = path.with_suffix('.tmp')
            with gzip.open(temporary, 'wt', encoding='utf8') as stream:
                json.dump({'seed': seed, 'shots': shots, 'full_counts': full,
                    'original_counts': self.counts, 'old_bit_to_measured_qubit': mappings,
                    'full_counts_sha256': identity}, stream)
            temporary.replace(path)
            from qiskit import qpy
            with path.with_suffix('.qpy').open('wb') as stream:
                qpy.dump(executed, stream)
            audit = str(path)
            _AUDIT_WRITTEN = True
        self.qcenn_readout = {'instrument': 'final_full_register_then_original_bit_marginal',
            'full_counts_sha256': identity, 'first_full_readout_audit': audit,
            'physical_measurements_per_shot': [c.num_qubits if m is not None else len(
                [i for i in c.data if i.operation.name == 'measure']) for c, m in zip(original, mappings, strict=True)],
            'execution_depth_max': max(c.depth() for c in executed),
            'original_classical_widths': [c.num_clbits for c in original],
            'mps_sampling_algorithm': 'mps_probabilities'}

    def get_counts(self, index=0):
        return self.counts[index]

    def get_memory(self, index=0):
        mapping = self.mappings[index]
        raw = self.result.get_memory(index)
        return [project(bits, mapping, self.original[index].num_clbits) for bits in raw] if mapping is not None else raw
