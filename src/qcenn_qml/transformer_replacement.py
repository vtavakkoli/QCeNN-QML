"""Measured replacement of a complete SMALL transformer language-model forward.

Four-token vocabulary, two-token context, one/two complete three-bit blocks,
quantum embedding lookup and quantum four-class projection/argmax readout.
An explicitly hybrid autoregressive host is tested separately. No Smol weights,
CPU attention/MLP inference, neural training or physical QPU are claimed.
"""
import argparse
from functools import lru_cache
from pathlib import Path

from .coherent import execute, load_basis, qrom
from .quantum_transformer import BITS, OPERATORS, block_unitary, reference_block
from .recurrent_mnist import provenance, write_json
from .resources import single_writer
from .study_utils import independent_seed, initialize, run_tasks


EMBEDDINGS = [[1, 1], [-2, 1], [1, -2], [-2, -2]]


def pack_tokens(tokens):
    if len(tokens) != 2 or any(isinstance(t, bool) or int(t) != t or not 0 <= t < 4 for t in tokens):
        raise ValueError('Expected two integer token IDs in 0..3')
    return int(tokens[0])+4*int(tokens[1])


def reference_forward(tokens, layers):
    """Independent classical comparator, never called by circuit construction."""
    pack_tokens(tokens)
    hidden = [EMBEDDINGS[int(t)] for t in tokens]
    for _ in range(layers):
        hidden = reference_block(hidden)
    # Four fixed logits: x+y, -x+y, x-y, -x-y. Their lowest-ID argmax
    # is exactly the pair of sign bits, including zero/tie cases.
    return [int(x < 0)+2*int(y < 0) for x, y in hidden]


@lru_cache(maxsize=2)
def model_unitary(layers=2):
    from qiskit import QuantumCircuit
    if layers not in (1, 2):
        raise ValueError('This measured feasibility model supports one or two blocks')
    block = block_unitary()
    workspace_size = len(block.metadata['workspace_qubits'])
    width = 4+12+workspace_size+12*layers+4
    circuit = QuantumCircuit(width)
    token_input = list(range(4))
    embedding = list(range(4, 16))
    workspace = list(range(16, 16+workspace_size))
    table = [(int(x)&7)+((int(y)&7) << BITS) for x, y in EMBEDDINGS]
    for token in range(2):
        qrom(circuit, token_input[2*token:2*token+2], embedding[6*token:6*token+6], table)
    hidden = embedding
    offset = 16+workspace_size
    for _ in range(layers):
        output = list(range(offset, offset+12)); offset += 12
        circuit.compose(block, [*hidden, *workspace, *output], inplace=True)
        hidden = output
    computation = tuple(circuit.data)
    prediction = list(range(offset, offset+4))
    for token in range(2):
        circuit.cx(hidden[6*token+2], prediction[2*token])
        circuit.cx(hidden[6*token+5], prediction[2*token+1])
    for item in reversed(computation):
        circuit.append(item.operation.inverse(), item.qubits)
    circuit.metadata = {'input_qubits': token_input, 'output_qubits': prediction,
        'workspace_qubits': list(range(4, offset)), 'layers': layers,
        'complete_forward_operators': ['quantum_token_embedding', *OPERATORS,
            'quantum_fixed_linear_vocabulary_projection_and_argmax'],
        'fixed_head_logits': ['x+y', '-x+y', 'x-y', '-x-y'],
        'head_tie_break': 'lowest token ID', 'intermediate_measurements': False,
        'classical_hidden_state_updates': False, 'actual_smol_weights': False}
    return circuit


def prepare_input(width, tokens=None, *, uniform=False, other=None):
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(width)
    if uniform:
        circuit.h(range(4))
    else:
        initial = pack_tokens(tokens)
        load_basis(circuit, range(4), initial)
        if other is not None:
            difference = initial ^ pack_tokens(other)
            if not difference:
                raise ValueError('Interference branches must differ')
            pivot = (difference & -difference).bit_length()-1
            circuit.h(pivot)
            if initial >> pivot & 1:
                circuit.x(pivot)
            for bit in range(4):
                if bit != pivot and difference >> bit & 1:
                    circuit.cx(pivot, bit)
            if initial >> pivot & 1:
                circuit.x(pivot)
    return circuit


def forward_circuit(tokens=None, *, layers=2, uniform=False):
    from qiskit import QuantumCircuit
    unitary = model_unitary(layers)
    measured = [*unitary.metadata['input_qubits'], *unitary.metadata['output_qubits'],
        *unitary.metadata['workspace_qubits']]
    circuit = QuantumCircuit(unitary.num_qubits, unitary.num_qubits)
    circuit.compose(prepare_input(unitary.num_qubits, tokens, uniform=uniform), inplace=True)
    circuit.compose(unitary, inplace=True)
    circuit.measure(measured, range(len(measured)))
    circuit.metadata = dict(unitary.metadata)
    return circuit


def phase_circuit(left, right, phase_bit, *, layers=2):
    unitary = model_unitary(layers)
    prepare = prepare_input(unitary.num_qubits, left, other=right)
    circuit = prepare.copy()
    circuit.compose(unitary, inplace=True)
    circuit.z(unitary.metadata['output_qubits'][phase_bit])
    circuit.compose(unitary.inverse(), inplace=True)
    circuit.compose(prepare.inverse(), inplace=True)
    circuit.measure_all()
    circuit.metadata = dict(unitary.metadata)
    return circuit


def _measure(circuit, shots, seed, directory, name):
    from qiskit import qpy
    counts, native, resources = execute([circuit], shots, seed)
    with (directory / (name+'.qpy')).open('wb') as stream:
        qpy.dump(native, stream)
    return counts[0], resources


def case_task(task):
    directory = Path(task['output_dir'])
    layers, shots, seed = task['layers'], task['shots'], task['measurement_seed']
    if task['kind'] == 'exhaustive_superposition':
        # Circuit built and executed before consulting the independent reference.
        counts, resources = _measure(forward_circuit(layers=layers, uniform=True), shots, seed, directory, 'forward')
        correct = clean = 0
        seen = set()
        for bits, count in counts.items():
            code = int(bits.replace(' ', ''), 2)
            original, output, work = code & 15, (code >> 4) & 15, code >> 8
            tokens = [original & 3, original >> 2]
            correct += count*(output == pack_tokens(reference_forward(tokens, layers)))
            clean += count*(work == 0)
            seen.add(original)
        row = {'counts': counts, 'resources': resources,
            'correct_output_fraction': correct/shots, 'clean_workspace_fraction': clean/shots,
            'observed_input_pairs': len(seen), 'required_input_pairs': 16,
            'passed': correct == clean == shots and len(seen) == 16}
    elif task['kind'] == 'phase_interference':
        pairs = [[a, b] for a in range(4) for b in range(4)]
        left = pairs[0]
        left_output = pack_tokens(reference_forward(left, layers))
        right = next(p for p in pairs if pack_tokens(reference_forward(p, layers)) != left_output)
        difference = left_output ^ pack_tokens(reference_forward(right, layers))
        phase_bit = (difference & -difference).bit_length()-1
        circuit = phase_circuit(left, right, phase_bit, layers=layers)
        counts, resources = _measure(circuit, shots, seed, directory, 'interference')
        input_difference = pack_tokens(left) ^ pack_tokens(right)
        expected = input_difference & -input_difference
        correct = sum(n for bits, n in counts.items() if int(bits.replace(' ', ''), 2) == expected)
        row = {'counts': counts, 'resources': resources, 'branches': [left, right],
            'phase_bit': phase_bit, 'expected_phase_kickback_bitstring': format(expected, f'0{circuit.num_qubits}b'),
            'interference_correct_fraction': correct/shots, 'passed': correct == shots}
    elif task['kind'] == 'hybrid_host_generation':
        context, generated, steps = [0, 1], [], []
        for step in range(2):
            counts, resources = _measure(forward_circuit(context, layers=layers), shots,
                independent_seed(seed, step), directory, f'host-step-{step}')
            decoded = decoded_outputs(counts)
            predicted = max(decoded, key=lambda key: (decoded[key], -key))
            reference = pack_tokens(reference_forward(context, layers))
            passed = decoded == {reference: shots} and all(int(bits.replace(' ', ''), 2) >> 8 == 0 for bits in counts)
            token = predicted >> 2
            steps.append({'context': context, 'measured_output_tokens': [predicted & 3, token],
                'classical_reference_tokens': [reference & 3, reference >> 2], 'counts': counts,
                'resources': resources, 'passed': passed})
            generated.append(token)
            context = [context[1], token]  # Explicitly HYBRID outer host update.
        row = {'hybrid_autoregressive_host': True, 'classical_between_token_updates': True,
            'cpu_attention_or_mlp_predictions': False, 'generated_token_ids': generated,
            'steps': steps, 'passed': all(step['passed'] for step in steps)}
    else:
        raise ValueError('Unknown replacement test kind')
    row.update(kind=task['kind'], layers=layers, measurement_seed=seed)
    write_json(directory / 'report.json', row)
    return row


def decoded_outputs(counts):
    from collections import Counter
    output = Counter()
    for bits, count in counts.items():
        output[(int(bits.replace(' ', ''), 2) >> 4) & 15] += count
    return dict(output)


@single_writer
def study(*, layers=(1, 2), measurement_seeds=(101, 202, 303), shots=1024,
        jobs=1, cpu_budget=1, output_dir='results/transformer-replacement', resume=False, smoke=False):
    layers, measurement_seeds = list(layers), list(measurement_seeds)
    if not layers or len(set(layers)) != len(layers) or any(layer not in (1, 2) for layer in layers):
        raise ValueError('Unique layer counts drawn from 1 and 2 required')
    if not measurement_seeds or len(set(measurement_seeds)) != len(measurement_seeds) or any(s < 0 for s in measurement_seeds) or shots < 256:
        raise ValueError('Unique nonnegative measurement seeds and at least 256 shots required')
    root = Path(output_dir)
    source = provenance()
    initialize(root, {'layers': layers, 'measurement_seeds': measurement_seeds,
        'shots': shots, 'embeddings': EMBEDDINGS, 'smoke_only': smoke}, source, resume)
    tasks = []
    for layer in layers:
        for kind in ('exhaustive_superposition', 'phase_interference', 'hybrid_host_generation'):
            for seed in (measurement_seeds if kind == 'exhaustive_superposition' else measurement_seeds[:1]):
                tasks.append({'id': f'l{layer}-{kind}-s{seed}', 'kind': kind, 'layers': layer,
                    'shots': shots, 'measurement_seed': seed, 'source_sha256': source['source_sha256']})
    report = {'experiment': 'complete_small_quantum_transformer_replacement', 'status': 'RUNNING',
        'scope': 'Four-token vocabulary, two-token context, width two, three-bit fixed toy weights',
        'actual_pretrained_smol_replaced': False, 'quantum_neural_weight_training_implemented': False,
        'physical_qpu_tested': False, 'entire_host_pipeline_pure_quantum': False,
        'forward_includes_quantum_embedding_and_vocabulary_argmax': True,
        'classical_attention_or_mlp_in_quantum_forward': False,
        'hybrid_generation_host_tested_separately': True, 'source': source, 'smoke_only': smoke}
    write_json(root / 'report.json', report)
    try:
        rows = run_tasks('qcenn_qml.transformer_replacement:case_task', tasks, jobs=jobs,
            cpu_budget=cpu_budget, directory=root / 'tasks')
    except BaseException as error:
        report.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=str(error))
        write_json(root / 'report.json', report)
        raise
    verified = all(r['passed'] for r in rows)
    report.update(status='VERIFIED' if verified else 'FAILED_REFERENCE_CHECK', cases=rows,
        configuration={'layers': layers, 'measurement_seeds': measurement_seeds, 'shots': shots,
            'vocabulary_size': 4, 'context_tokens': 2, 'word_bits': 3, 'embeddings': EMBEDDINGS},
        limitations=['Fixed toy weights and finite precision; full pretrained Smol is not executed',
            'Classical host loading, compilation, simulation and decoding remain',
            'Full-model forward is coherent; generation wrapper uses explicit classical token updates',
            'No measured model quality, neural training, quantum advantage or QPU result'])
    write_json(root / 'report.json', report)
    if not verified:
        raise RuntimeError('Small-model transformer replacement failed its independent checks')
    print(f'Transformer replacement VERIFIED: {len(rows)} cases; report={root / "report.json"}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--measurement-seeds', nargs='+', type=int, default=[101, 202, 303])
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--cpu-budget', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', default='results/transformer-replacement')
    args = vars(parser.parse_args(argv))
    if args['smoke']:
        args.update(layers=[1], measurement_seeds=[101], shots=256)
    return study(**args)


if __name__ == '__main__':
    main()
