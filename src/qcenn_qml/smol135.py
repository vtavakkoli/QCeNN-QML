"""Tiny quantum autoregressive memory inspired by SmolLM2 research.

NOT a conversion of SmolLM2-135M. Optional SmolLM2 teacher supplies classical
offline text only; inference here uses quantum gates and measurement counts.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from .quantum import basis_reflection, sample_counts, unit_vector, zero_reflection

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
EXAMPLE_CORPUS = (
    "the cat sat on the mat . the dog ran on the road . "
    "the cat ran on the road . the dog sat on the mat . "
    "the cat sat . the dog ran . the cat ran . the dog sat . "
) * 8


def tokenize(text):
    return re.findall(r"[a-z]+|[.,!?]", text.lower())


def create_vocabulary(tokens, size):
    if size < 2 or size > 8:
        raise ValueError("Prototype vocabulary must have 2 to 8 tokens")
    ranked = sorted(Counter(tokens).items(), key=lambda v: (-v[1], v[0]))
    return ["<unk>"] + [token for token, _ in ranked if token != "<unk>"][:size-1]


def encode_tokens(tokens, vocab):
    index = {word: i for i, word in enumerate(vocab)}
    return [index.get(word, 0) for word in tokens]


def context_integer(token_ids, bits):
    return sum(int(token) << (i * bits) for i, token in enumerate(token_ids))


def build_memory_amplitudes(tokens, vocab, order=1, smoothing=.05):
    """Classical DATA PREPARATION ONLY; prediction never reads a counts table."""
    if order not in (1, 2) or not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Invalid order or smoothing")
    bits = max(1, (len(vocab) - 1).bit_length())
    contexts, outputs = 1 << (bits * order), 1 << bits
    counts = np.zeros((contexts, outputs), dtype=float)
    counts[:, :len(vocab)] = smoothing
    ids = encode_tokens(tokens, vocab)
    for step in range(order, len(ids)):
        ctx = context_integer(ids[step-order:step], bits)
        counts[ctx, ids[step]] += 1
    if not np.any(counts):
        raise ValueError("Corpus does not contain any transitions")
    return unit_vector(np.sqrt(counts.reshape(-1))), bits, int(np.sum(counts > smoothing))


def memory_circuit(amplitudes, bits, order, query_context, amplify=0):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    if not 0 <= amplify <= 3:
        raise ValueError("amplify must be 0..3")
    width = bits * (order + 1)
    if len(amplitudes) != (1 << width):
        raise ValueError("Invalid quantum memory width")
    preparation = StatePreparation(amplitudes).definition
    circuit = QuantumCircuit(width, width)
    circuit.compose(preparation, qubits=list(range(width)), inplace=True)
    oracle = basis_reflection(width, range(bits, width), query_context)
    reflection = zero_reflection(width)
    for _ in range(amplify):
        circuit.compose(oracle, inplace=True)
        circuit.compose(preparation.inverse(), qubits=list(range(width)), inplace=True)
        circuit.compose(reflection, inplace=True)
        circuit.compose(preparation, qubits=list(range(width)), inplace=True)
    circuit.measure(range(width), range(width))
    return circuit


def sample_next_token(amplitudes, bits, order, context, vocab, *, shots, seed, amplify=0):
    target_context = context_integer(context, bits)
    circuit = memory_circuit(amplitudes, bits, order, target_context, amplify)
    count_list, compiled = sample_counts([circuit], shots, seed)
    histogram = np.zeros(len(vocab), dtype=int)
    for binary, count in count_list[0].items():
        value = int(binary.replace(" ", ""), 2)
        if value >> bits == target_context:
            output = value & ((1 << bits) - 1)
            if output < len(vocab):
                histogram[output] += int(count)
    accepted = int(histogram.sum())
    if not accepted:
        raise RuntimeError("No measured shots match this context; increase --shots")
    rng = np.random.default_rng(seed)
    chosen = int(rng.choice(len(vocab), p=histogram / accepted))
    diagnostics = {"postselected_shots": accepted,
                   "postselection_rate": accepted/shots,
                   "conditional_counts": histogram.tolist(),
                   "qubits": circuit.num_qubits,
                   "transpiled_depth": int(compiled[0].depth())}
    return chosen, diagnostics


def teacher_generated_corpus(prompt, samples, max_new_tokens, seed, model_id=MODEL_ID):
    """OFFLINE classical PyTorch teacher; not a quantum weight conversion."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    encoded = tokenizer(prompt or "The", return_tensors="pt")
    outputs = []
    with torch.no_grad():
        for _ in range(samples):
            sequence = model.generate(**encoded, do_sample=True, top_p=.9,
                                      temperature=.8, max_new_tokens=max_new_tokens,
                                      pad_token_id=tokenizer.eos_token_id)
            outputs.append(tokenizer.decode(sequence[0], skip_special_tokens=True))
    return "\n".join(outputs)


def generate(prompt="the cat", corpus_file=None, vocab_size=8, order=1,
             new_tokens=5, shots=4096, amplify=0, seed=42, teacher_samples=0,
             teacher_max_new_tokens=48, teacher_model=MODEL_ID,
             output_dir="results/smol135"):
    if new_tokens < 1 or shots < 1 or teacher_samples < 0:
        raise ValueError("Invalid generation budget")
    if corpus_file is not None:
        corpus = Path(corpus_file).read_text(encoding="utf8")
        origin = "user_corpus"
    elif teacher_samples:
        corpus = teacher_generated_corpus(prompt, teacher_samples,
                                          teacher_max_new_tokens, seed, teacher_model)
        origin = "classical_SmolLM2_135M_teacher_text"
    else:
        corpus = EXAMPLE_CORPUS
        origin = "built_in_toy_corpus"
    tokens = tokenize(corpus)
    if len(tokens) <= order:
        raise ValueError("Insufficient tokens for quantum memory")
    vocab = create_vocabulary(tokens, vocab_size)
    amplitudes, bits, transitions = build_memory_amplitudes(tokens, vocab, order)
    if bits * (order + 1) > 10:
        raise MemoryError("Qubit budget exceeded")
    seed_context = ([0] * order + encode_tokens(tokenize(prompt), vocab))[-order:]
    words, steps = [], []
    for i in range(new_tokens):
        choice, diag = sample_next_token(amplitudes, bits, order, seed_context[-order:],
                                          vocab, shots=shots, seed=seed+i, amplify=amplify)
        words.append(vocab[choice])
        steps.append({"step": i, "token": vocab[choice], **diag})
        seed_context.append(choice)
    report = {
        "experiment": "tiny_quantum_ngram_surrogate_NOT_SmolLM2_135M",
        "smol_reference_model": teacher_model if teacher_samples else MODEL_ID,
        "actual_SmolLM2_weights_on_quantum_circuit": False,
        "actual_SmolLM2_transformer_implemented": False,
        "classical_teacher_used": bool(teacher_samples), "corpus_origin": origin,
        "corpus_tokens": len(tokens), "vocabulary": vocab,
        "order": order, "qubits": bits*(order+1), "amplitude_count": len(amplitudes),
        "observed_transitions": transitions, "shots_per_token": shots,
        "amplification_rounds": amplify, "prompt": prompt,
        "generated_text": " ".join(words), "steps": steps,
        "backend": "qiskit_aer_shots", "physical_qpu_tested": False,
        "limitations": ["Quantum n-gram memory, not 135M transformer",
                        "Dense classical amplitude preparation and optional classical teacher",
                        "Classical sampling between token steps", "No quantum advantage claim"]
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if teacher_samples:
        (path / "teacher_corpus.txt").write_text(corpus, encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k != "steps"}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="the cat")
    parser.add_argument("--corpus-file")
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--order", type=int, choices=[1, 2], default=1)
    parser.add_argument("--new-tokens", type=int, default=5)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--amplify", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-samples", type=int, default=0)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=48)
    parser.add_argument("--teacher-model", default=MODEL_ID)
    parser.add_argument("--output-dir", default="results/smol135")
    return generate(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
