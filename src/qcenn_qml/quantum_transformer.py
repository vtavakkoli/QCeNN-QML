"""A complete, tiny coherent fixed-point causal transformer block.

Two tokens, width two, one head, three-bit signed basis words (one fractional
bit). Q/K/V projections, quarter-turn RoPE, two-entry softmax, weighted vector,
output projection, residuals, RMSNorm and SwiGLU all execute reversibly. Only
final measurements leave the circuit. This is an intentionally quantized toy
block with fixed demonstration weights, not Smol weights or quantum training.
"""
import argparse
from functools import lru_cache
import math
from pathlib import Path

import numpy as np

from .coherent import execute, load_basis, mcx_clean, qrom
from .recurrent_mnist import provenance, write_json


BITS, FRACTION, SCALE = 3, 1, 2
WEIGHTS = {'q': [[1, 1], [1, -1]], 'k': [[1, -1], [1, 1]],
    'v': [[1, 0], [0, 1]], 'o': [[1, 0], [1, 1]],
    'gate': [[1, 1], [1, -1]], 'up': [[1, 0], [0, 1]],
    'down': [[1, 1], [-1, 1]]}
OPERATORS = ['rms_norm_1', 'qkv_projections', 'rotary_position_encoding',
    'causal_scaled_dot_product_softmax', 'coherent_weighted_sum_vector',
    'output_projection', 'attention_residual', 'rms_norm_2',
    'silu_gated_mlp', 'mlp_residual']


def signed(code, width=BITS):
    code = int(code) & ((1 << width) - 1)
    return code if code < 1 << (width - 1) else code - (1 << width)


class Arithmetic:
    """Gate arithmetic. Universal scalar lookup tables never depend on inputs."""
    def __init__(self):
        from qiskit import QuantumCircuit, QuantumRegister
        self.circuit = QuantumCircuit(QuantumRegister(12, 'input'))
        self.input = [list(range(i, i + BITS)) for i in range(0, 12, BITS)]
        self.accumulator = self.word(8, 'product_accumulator')
        self.norm = self.word(6, 'sum_of_squares')
        self.inverse_rms = self.word(4, 'inverse_rms')
        self.product = self.word(1, 'product_bit')[0]
        self.scratch = self.word(6, 'clean_mcx_work')
        self.functions = []

    def word(self, width=BITS, name='word'):
        from qiskit import QuantumRegister
        register = QuantumRegister(width, f'{name}_{self.circuit.num_qubits}')
        start = self.circuit.num_qubits
        self.circuit.add_register(register)
        return list(range(start, start + width))

    def add_power(self, control, output, power, inverse=False):
        if power >= len(output):
            return
        start = len(self.circuit.data)
        for bit in range(len(output) - 1, power, -1):
            mcx_clean(self.circuit, [control, *output[power:bit]], output[bit], self.scratch)
        self.circuit.cx(control, output[power])
        if inverse:
            operations = tuple(self.circuit.data[start:])
            del self.circuit.data[start:]
            for instruction in reversed(operations):
                self.circuit.append(instruction.operation.inverse(), instruction.qubits)

    def add_word(self, source, output, coefficient=1, unsigned=False):
        for bit, qubit in enumerate(source):
            value = coefficient * (-(1 << bit) if bit == len(source) - 1 and not unsigned else 1 << bit)
            for power in range(abs(value).bit_length()):
                if abs(value) >> power & 1:
                    self.add_power(qubit, output, power, inverse=value < 0)

    def product_add(self, left, right, output, *, right_unsigned=False, negative=False):
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                if i + j >= len(output):
                    continue
                if a == b:
                    self.circuit.cx(a, self.product)
                else:
                    self.circuit.ccx(a, b, self.product)
                sign = (i == len(left) - 1) ^ (j == len(right) - 1 and not right_unsigned) ^ negative
                self.add_power(self.product, output, i + j, inverse=sign)
                if a == b:
                    self.circuit.cx(a, self.product)
                else:
                    self.circuit.ccx(a, b, self.product)

    def multiply(self, left, right, *, shift=FRACTION, right_unsigned=False):
        output = self.word()
        start = len(self.circuit.data)
        self.product_add(left, right, self.accumulator, right_unsigned=right_unsigned)
        operations = tuple(self.circuit.data[start:])
        for bit, qubit in enumerate(output):
            self.circuit.cx(self.accumulator[bit + shift], qubit)
        for instruction in reversed(operations):
            self.circuit.append(instruction.operation.inverse(), instruction.qubits)
        return output

    def linear(self, inputs, matrix, *, shift=0, width=BITS):
        outputs = []
        for coefficients in matrix:
            output = self.word(width)
            if not shift:
                for source, coefficient in zip(inputs, coefficients, strict=True):
                    self.add_word(source, output, coefficient)
            else:
                start = len(self.circuit.data)
                for source, coefficient in zip(inputs, coefficients, strict=True):
                    self.add_word(source, self.accumulator, coefficient)
                operations = tuple(self.circuit.data[start:])
                for bit, qubit in enumerate(output):
                    self.circuit.cx(self.accumulator[bit + shift], qubit)
                for instruction in reversed(operations):
                    self.circuit.append(instruction.operation.inverse(), instruction.qubits)
            outputs.append(output)
        return outputs

    def function(self, source, output, table, name):
        qrom(self.circuit, source, output, table)
        self.functions.append({'name': name, 'input_bits': len(source),
            'output_bits': len(output), 'table_entries': len(table), 'input_data_dependent': False})

    def normalize(self, vector):
        start = len(self.circuit.data)
        for word in vector:
            self.product_add(word, word, self.norm)
        square_operations = tuple(self.circuit.data[start:])
        # Integer norm is sum(x_int**2); physical words have SCALE=2.
        table = [min(15, round(4 / math.sqrt(code / (2 * SCALE**2) + .25))) for code in range(64)]
        self.function(self.norm, self.inverse_rms, table, 'inverse_rms_epsilon_0.25')
        table_operations = tuple(self.circuit.data[start + len(square_operations):])
        outputs = [self.multiply(word, self.inverse_rms, shift=2, right_unsigned=True) for word in vector]
        for instruction in reversed(table_operations):
            self.circuit.append(instruction.operation.inverse(), instruction.qubits)
        for instruction in reversed(square_operations):
            self.circuit.append(instruction.operation.inverse(), instruction.qubits)
        return outputs


@lru_cache(maxsize=1)
def block_unitary():
    """|x,0,y> -> |x,0,y XOR block(x)>; workspace is uncomputed."""
    arithmetic = Arithmetic()
    circuit = arithmetic.circuit
    tokens = [arithmetic.input[:2], arithmetic.input[2:]]
    normalized = [arithmetic.normalize(vector) for vector in tokens]
    query = arithmetic.linear(normalized[1], WEIGHTS['q'], shift=1)
    keys = [arithmetic.linear(vector, WEIGHTS['k'], shift=1) for vector in normalized]
    values = [arithmetic.linear(vector, WEIGHTS['v']) for vector in normalized]
    # Fixed toy position angles: token 0 = 0, token 1 = pi/2.
    query = arithmetic.linear(query, [[0, -1], [1, 0]])
    keys[1] = arithmetic.linear(keys[1], [[0, -1], [1, 0]])
    difference = arithmetic.word(7, 'attention_score_difference')
    for feature in range(2):
        arithmetic.product_add(query[feature], keys[0][feature], difference)
        arithmetic.product_add(query[feature], keys[1][feature], difference, negative=True)
    weight = arithmetic.word(3, 'softmax_weight')
    sigmoid = [round(4 / (1 + math.exp(-signed(code, 7) / (SCALE**2 * math.sqrt(2))))) for code in range(128)]
    arithmetic.function(difference, weight, sigmoid, 'two_entry_scaled_softmax_weight')
    attended = [values[0], []]
    for feature in range(2):
        delta = arithmetic.linear([values[0][feature], values[1][feature]], [[1, -1]], width=4)[0]
        weighted = arithmetic.multiply(delta, weight, shift=2, right_unsigned=True)
        attended[1].append(arithmetic.linear([values[1][feature], weighted], [[1, 1]])[0])
    final = []
    for token, attention in zip(tokens, attended, strict=True):
        projected = arithmetic.linear(attention, WEIGHTS['o'], shift=1)
        residual = [arithmetic.linear([a, b], [[1, 1]])[0] for a, b in zip(token, projected, strict=True)]
        normalized_mlp = arithmetic.normalize(residual)
        gate = arithmetic.linear(normalized_mlp, WEIGHTS['gate'], shift=1)
        up = arithmetic.linear(normalized_mlp, WEIGHTS['up'])
        activated = []
        for word in gate:
            result = arithmetic.word()
            table = [round(signed(code) / (1 + math.exp(-signed(code) / SCALE))) & 7 for code in range(8)]
            arithmetic.function(word, result, table, 'silu')
            activated.append(result)
        gated = [arithmetic.multiply(a, b) for a, b in zip(activated, up, strict=True)]
        down = arithmetic.linear(gated, WEIGHTS['down'], shift=1)
        final.extend(arithmetic.linear([a, b], [[1, 1]])[0] for a, b in zip(residual, down, strict=True))
    computation = tuple(circuit.data)
    outputs = arithmetic.word(12, 'output')
    for source, target in zip([q for word in final for q in word], outputs, strict=True):
        circuit.cx(source, target)
    for instruction in reversed(computation):
        circuit.append(instruction.operation.inverse(), instruction.qubits)
    circuit.metadata = {'input_qubits': list(range(12)), 'output_qubits': outputs,
        'workspace_qubits': list(range(12, outputs[0])), 'operators': OPERATORS,
        'universal_scalar_functions': arithmetic.functions}
    return circuit


def reference_block(inputs):
    """Independent integer-array reference; never called by circuit construction."""
    def wrap(array):
        return np.array([signed(v) for v in np.asarray(array).reshape(-1)]).reshape(np.shape(array))
    def linear(vector, name):
        result = np.asarray(WEIGHTS[name]) @ vector
        return wrap(result if name in ('v', 'up') else result // 2)
    def norm(vector):
        inverse = min(15, round(4 / math.sqrt(float(vector @ vector) / (2 * SCALE**2) + .25)))
        return wrap(vector * inverse // 4)
    tokens = np.asarray(inputs, int)
    normalized = [norm(vector) for vector in tokens]
    q = linear(normalized[1], 'q'); q = wrap([-q[1], q[0]])
    k = [linear(vector, 'k') for vector in normalized]; k[1] = wrap([-k[1][1], k[1][0]])
    v = [linear(vector, 'v') for vector in normalized]
    difference = signed(int(q @ k[0] - q @ k[1]), 7)
    weight = round(4 / (1 + math.exp(-difference / (SCALE**2 * math.sqrt(2)))))
    attention = [v[0], wrap(v[1] + weight * (v[0] - v[1]) // 4)]
    result = []
    for token, attended in zip(tokens, attention, strict=True):
        residual = wrap(token + linear(attended, 'o'))
        normalized = norm(residual)
        gate, up = linear(normalized, 'gate'), linear(normalized, 'up')
        activated = wrap([round(int(x) / (1 + math.exp(-int(x) / SCALE))) for x in gate])
        hidden = wrap(activated * up // SCALE)
        result.append(wrap(residual + linear(hidden, 'down')).tolist())
    return result


def _pack(inputs):
    array = np.asarray(inputs)
    if array.shape != (2, 2) or not np.equal(array, array.astype(int)).all() or np.any(array < -4) or np.any(array > 3):
        raise ValueError('Expected a 2x2 array of signed three-bit integers (-4..3)')
    return sum((int(value) & 7) << (3 * i) for i, value in enumerate(array.flat))


def block_circuit(inputs, *, superposition=None, audit_workspace=True, max_qubits=256):
    from qiskit import QuantumCircuit
    unitary = block_unitary()
    if unitary.num_qubits > max_qubits:
        raise MemoryError(f'Tiny block needs {unitary.num_qubits} logical qubits; budget={max_qubits}')
    measured = [*unitary.metadata['input_qubits'], *unitary.metadata['output_qubits']]
    if audit_workspace:
        measured.extend(unitary.metadata['workspace_qubits'])
    circuit = QuantumCircuit(unitary.num_qubits, len(measured))
    initial = _pack(inputs)
    load_basis(circuit, range(12), initial)
    if superposition is not None:
        other = _pack(superposition)
        differing = [bit for bit in range(12) if (initial ^ other) >> bit & 1]
        if not differing:
            raise ValueError('Superposition inputs must differ')
        pivot = differing[0]
        circuit.h(pivot)
        if initial >> pivot & 1:
            circuit.x(pivot)
        for target in differing[1:]:
            circuit.cx(pivot, target)
        if initial >> pivot & 1:
            circuit.x(pivot)
    circuit.compose(unitary, inplace=True)
    circuit.measure(measured, range(len(measured)))
    circuit.metadata = {**unitary.metadata, 'audit_workspace': audit_workspace,
        'classical_hidden_state_updates': False, 'input_dependent_cpu_attention': False,
        'intermediate_measurements': False, 'actual_smol_weights': False,
        'basis_encoding_not_amplitude_nonlinearity': True}
    return circuit


def run(shots=256, seed=42, output_dir='results/quantum-transformer'):
    from qiskit import qpy
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Use a fresh transformer output directory')
    directory.mkdir(parents=True, exist_ok=True)
    source = provenance()
    examples = [[[1, -1], [2, 1]], [[-2, 1], [1, -1]]]
    cases = []
    for index, inputs in enumerate(examples):
        circuit = block_circuit(inputs, superposition=examples[1] if index == 0 else None)
        print(f'Coherent transformer case {index}: {circuit.num_qubits} qubits, {len(circuit.data)} logical operations', flush=True)
        counts, native, resources = execute([circuit], shots, seed + index)
        with (directory / f'block-{index}.qpy').open('wb') as stream:
            qpy.dump(native[0], stream)
        branches = [inputs, examples[1]] if index == 0 else [inputs]
        expected = {_pack(branch): _pack(reference_block(branch)) for branch in branches}
        correct = clean = 0
        for bits, count in counts[0].items():
            code = int(bits.replace(' ', ''), 2)
            original, output, workspace = code & 4095, (code >> 12) & 4095, code >> 24
            correct += count * (original in expected and output == expected[original])
            clean += count * (workspace == 0)
        row = {'inputs': branches, 'reference_outputs_integer_words': [reference_block(x) for x in branches],
            'counts': counts[0], 'correct_output_fraction': correct / shots,
            'clean_workspace_fraction': clean / shots, 'resources': resources,
            'coherent_superposition_input': index == 0}
        cases.append(row)
        write_json(directory / 'progress.json', {'completed_cases': cases})
    verified = all(r['correct_output_fraction'] == 1. and r['clean_workspace_fraction'] == 1. for r in cases)
    report = {'experiment': 'coherent_fixed_point_transformer_block',
        'status': 'VERIFIED' if verified else 'FAILED_REFERENCE_CHECK', 'physical_qpu_tested': False,
        'classical_optimizer': False, 'fitted_classical_weights': False,
        'classical_input_loading_compilation_and_count_decoding': True,
        'entire_host_pipeline_pure_quantum': False, 'quantum_neural_weight_training_implemented': False,
        'complete_small_causal_transformer_block': True, 'actual_smol_weights_executed': False,
        'configuration': {'tokens': 2, 'hidden_size': 2, 'heads': 1, 'mlp_width': 2,
            'word_bits': BITS, 'fractional_bits': FRACTION, 'softmax_fractional_bits': 2,
            'rms_epsilon': .25, 'rope_angles': [0, math.pi / 2], 'weights_integer_numerators': WEIGHTS,
            'projection_denominator': '2 except V and up, which use 1',
            'overflow': 'signed two-complement modular wrap', 'right_shift': 'floor toward negative infinity',
            'scalar_function_quantization': 'round to nearest, ties to even'},
        'operators': OPERATORS, 'cases': cases, 'source': source,
        'limitations': ['Fixed demonstration weights, not pretrained Smol weights or neural training',
            'Three-bit modular arithmetic and quarter-turn RoPE differ substantially from Smol precision and positions',
            'Universal scalar functions use exponentially sized finite-domain tables; loading and gates are charged',
            'Unitary evaluation of a classical function on basis registers is not a nonlinear map on arbitrary amplitudes',
            'Aer simulation; no QPU execution, scaling or advantage demonstrated']}
    write_json(directory / 'report.json', report)
    if not verified:
        raise RuntimeError('Coherent block disagrees with its independent reference or left dirty workspace')
    print(f'Transformer block status: {report["status"]}; report={directory / "report.json"}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='results/quantum-transformer')
    return run(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
