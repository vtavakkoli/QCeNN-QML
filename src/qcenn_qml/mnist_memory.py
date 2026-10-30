"""Balanced measured memory banks and fixed, label-independent image encoding.

All banks receive the same number of shots and contain the same number of raw
records. Pooling counts therefore preserves the uniform memory prior. Nearest
record readout ranks measured record frequencies, never CPU-computed distances.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .coherent import _compile_components, decode_postselection
from .quantum_memory import memory_circuit
from .study_utils import circuit_digest, digest


def preprocess(images, method='none'):
    if method not in ('none', 'center', 'deskew'):
        raise ValueError('Unknown fixed image preprocessing')
    images = np.asarray(images)
    if method == 'none':
        return images
    output = []
    for image in images:
        values = np.asarray(image, float)
        if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Finite nonnegative grayscale images required')
        height, width = values.shape
        y, x = np.indices(values.shape, dtype=float)
        mass = values.sum()
        if mass == 0:
            output.append(values.astype(np.uint8)); continue
        cy, cx = (values*y).sum()/mass, (values*x).sum()/mass
        shear = 0.
        if method == 'deskew':
            variance = (values*(y-cy)**2).sum()
            shear = float(np.clip((values*(x-cx)*(y-cy)).sum()/max(variance, 1e-12), -1, 1))
        # Inverse affine map with bilinear interpolation and zero padding.
        source_y = y - (height-1)/2 + cy
        source_x = x - (width-1)/2 + cx + shear*(source_y-cy)
        y0, x0 = np.floor(source_y).astype(int), np.floor(source_x).astype(int)
        result = np.zeros_like(values)
        for dy in (0, 1):
            for dx in (0, 1):
                yy, xx = y0+dy, x0+dx
                valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
                weight = (1-np.abs(source_y-yy))*(1-np.abs(source_x-xx))
                result[valid] += values[yy[valid], xx[valid]]*weight[valid]
        output.append(np.clip(np.rint(result), 0, 255).astype(np.uint8))
    return np.asarray(output)


@dataclass
class NativeMemory:
    circuits: list
    labels: list
    readout: str

    @property
    def num_qubits(self):
        return max(c.num_qubits for c in self.circuits)

    def depth(self):
        return max(c.depth() for c in self.circuits)

    def count_ops(self):
        return sum((Counter(c.count_ops()) for c in self.circuits), Counter())


def native_hash(native):
    if isinstance(native, NativeMemory):
        return digest({'banks': [circuit_digest(c) for c in native.circuits],
            'labels': native.labels, 'readout': native.readout})
    return circuit_digest(native)


def compile_memory(query, keys, labels, config, seed):
    keys = np.asarray(keys, float)
    readout = config.get('readout', 'class_sum')
    if readout not in ('class_sum', 'nearest_record'):
        raise ValueError('Unknown measured memory readout')
    labels = np.asarray(labels, int)
    if len(keys) <= 32 and readout == 'class_sum':
        return _compile_components(memory_circuit(query, keys, labels,
            rounds=config['rounds'], encoding=config['encoding'], damping=config['damping'], max_qubits=128), seed)
    if len(keys) > 32:
        counts = np.bincount(labels, minlength=10)
        if not np.all(counts == counts[0]) or counts[0] % 2:
            raise ValueError('Larger memories require equal even records per digit')
        # Two records per digit per bank; original seeded record order retained.
        groups = [np.concatenate([np.flatnonzero(labels == label)[i:i+2] for label in range(10)])
            for i in range(0, int(counts[0]), 2)]
        groups = [np.sort(group) for group in groups]
    else:
        groups = [np.arange(len(keys))]
    circuits, bank_labels = [], []
    for group in groups:
        circuits.append(_compile_components(memory_circuit(query, keys[group], labels[group],
            rounds=config['rounds'], encoding=config['encoding'], damping=config['damping'], max_qubits=128,
            measure_index=readout == 'nearest_record'), seed))
        bank_labels.append(labels[group].tolist())
    return NativeMemory(circuits, bank_labels, readout)


def decode_banks(counts, native, rounds):
    pooled, record_counts = Counter(), []
    if len(counts) != len(native.circuits) or len({sum(c.values()) for c in counts}) != 1:
        raise ValueError('Bank counts must have identical shot budgets')
    for bank, labels in zip(counts, native.labels, strict=True):
        records = [0]*len(labels)
        for bits, count in bank.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError('Nonnegative integer counts required')
            code = int(bits.replace(' ', ''), 2)
            pooled[format(code & ((1 << (4+rounds))-1), f'0{4+rounds}b')] += count
            if native.readout == 'nearest_record' and (code >> 4) & ((1 << rounds)-1) == (1 << rounds)-1:
                index, label = code >> (4+rounds), code & 15
                if index >= len(labels) or label != labels[index]:
                    raise ValueError('Measured label/index disagree with raw memory')
                records[index] += count
        record_counts.extend(zip(records, labels))
    row = decode_postselection(dict(pooled), 4, rounds, 10)
    if native.readout == 'nearest_record':
        # Deterministic class tie-break; abstention remains explicit.
        peak_by_class = [max((n for n, y in record_counts if y == label), default=0) for label in range(10)]
        row['prediction'] = int(np.argmax(peak_by_class)) if max(peak_by_class) else -1
        row['measured_record_peaks_by_class'] = peak_by_class
    row.update(bank_counts=counts, bank_labels=native.labels, measured_readout=native.readout)
    return row


def allocate_shots(native, anchor_cx, base_shots, mode):
    banks = len(native.circuits)
    cx = sum(int(c.count_ops().get('cx', 0)) for c in native.circuits)
    budget = anchor_cx*base_shots if mode == 'equal_cnot' else None
    per_bank = budget//cx if budget is not None else base_shots//banks
    if per_bank < 1:
        raise ValueError('Budget cannot pay for one shot in every memory bank')
    return per_bank*banks, budget
