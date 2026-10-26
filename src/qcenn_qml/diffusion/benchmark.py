"""Reproducible toy masked-token quantum diffusion and controlled-noise audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from .core import (
    VOCAB, MASK, dataset, conditional_distribution, context_number,
    denoising_circuit, expected_clean_probability, counts_to_distribution
)


def execute(*, shots=256, beta=1.0, eta=0.65, steps=(0,1,3),
            examples=4, seed=42, readout_error=0.0,
            output_dir="results/diffusion"):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    if shots < 1 or examples < 1 or examples > 24:
        raise ValueError("Invalid shots or examples")
    if not np.isfinite(readout_error) or not 0 <= readout_error < 0.5:
        raise ValueError("Readout error must lie in [0,0.5)")
    if any(not isinstance(v,int) or v < 0 or v > 8 for v in steps):
        raise ValueError("Invalid denoising steps")
    train, test = dataset(seed)
    table = conditional_distribution(train)
    cases = test[:examples]
    noise = None
    if readout_error:
        noise = NoiseModel()
        flip = readout_error
        noise.add_all_qubit_readout_error(ReadoutError([[1-flip,flip],[flip,1-flip]]))
    backend = AerSimulator(method="matrix_product_state", noise_model=noise,
                           max_parallel_threads=1)
    all_rows = []
    configs = {}
    for round_count in steps:
        circuits = [
            denoising_circuit(left, target, right, table=table,
                              beta=beta,eta=eta,steps=round_count)
            for left,target,right in cases
        ]
        executable = transpile(circuits,backend,optimization_level=0,
                               seed_transpiler=seed)
        result = backend.run(executable,shots=shots,
                             seed_simulator=seed+round_count).result()
        if not result.success:
            raise RuntimeError(str(result.status))
        successes, totals, oracle_prob = 0, 0, []
        for i, (left,target,right) in enumerate(cases):
            counts = counts_to_distribution(result.get_counts(i),shots)
            successes += int(round(counts[target]*shots))
            totals += shots
            theory=expected_clean_probability(
                target,table[context_number(left,right)],beta,eta,round_count)
            oracle_prob.append(theory)
            all_rows.append({
                "steps":round_count, "context":[VOCAB[left],VOCAB[right]],
                "truth":VOCAB[target], "quantum_target_frequency":float(counts[target]),
                "quantum_distribution":counts.tolist(),
                "analytic_ideal_target_probability":theory,
                "top_measurement":VOCAB[int(np.argmax(counts))]
            })
        observed=successes/totals
        se=float(np.sqrt(observed*(1-observed)/totals))
        configs[str(round_count)]={
            "measured_correct_token_probability":observed,
            "normal_approx_95pct_interval":[max(0.0,observed-1.96*se),
                                            min(1.0,observed+1.96*se)],
            "independent_ideal_probability_mean":float(np.mean(oracle_prob)),
            "sequence_top1_accuracy":float(np.mean([
                row["top_measurement"]==row["truth"]
                for row in all_rows if row["steps"]==round_count
            ])),
            "samples":len(cases),"total_shots":totals,
            "compiled_depth_min":min(c.depth() for c in executable),
            "compiled_depth_max":max(c.depth() for c in executable),
            "compiled_max_qubits":max(c.num_qubits for c in executable)
        }
    report={
        "name":"QCeNN-DiffusionLM: tiny masked-token quantum denoising experiment",
        "status":"simulation_only_toy_corpus_not_a_trained_diffusion_LLM",
        "backend":"Qiskit Aer matrix_product_state with gate/reset operations",
        "quantum_channel":"context_conditioned_recurrent_ancilla_replacement",
        "actual_qcenn_mi_solver":False,"actual_smol135_weights_on_quantum_device":False,
        "physical_QPU_run":False,"learned_quantum_neural_weights":False,
        "vocabulary":list(VOCAB),"mask_token":VOCAB[MASK],
        "train_count":len(train),"heldout_count":len(test),
        "evaluation_count":len(cases), "seeds":{"data":seed,"simulator":seed},
        "forward_mask_probability_beta":beta,"reverse_replacement_probability_eta":eta,
        "readout_bitflip_probability":readout_error,
        "corruption_source":"engineered_ancilla_swap_reset_not_uncontrolled_hardware_noise",
        "shots_per_case":shots,"steps":list(steps),
        "results":configs,"cases":all_rows,
        "interpretation":["Higher correct-token probability vs fully masked input demonstrates conditional state replacement, not language understanding.",
                          "The model distribution comes from classically counted toy triples, encoded into quantum-controlled gates.",
                          "The ideal prediction uses an independently derived CPTP-channel formula; it does not drive inference.",
                          "Reset/dissipation can improve masked-token reconstruction but gate/readout noise is a separate effect.",
                          "The 95% intervals treat shots as independent with fixed cases and omit corpus uncertainty.",
                          "No comparison here establishes a quantum advantage or a new trained quantum LLM."]
    }
    path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
    (path/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    lines=["# QCeNN-DiffusionLM measured report","",
           "**Status:** toy 3-token masked denoising, Qiskit Aer simulation, not a full diffusion LLM.",
           "",f"Train/evaluation examples: {len(train)}/{len(cases)}; shots per case: {shots}.",
           f"Forward masking β={beta}; reverse replacement η={eta}; readout bit-flip={readout_error}.","",
           "| Denoise steps | Correct-token frequency | Ideal quantum-channel expectation | Top-1 case accuracy | Compiled depth range |",
           "|---:|---:|---:|---:|---:|"]
    for k,item in configs.items():
        lines.append(f"| {k} | {item['measured_correct_token_probability']:.4f} | "
                     f"{item['independent_ideal_probability_mean']:.4f} | "
                     f"{item['sequence_top1_accuracy']:.4f} | "
                     f"{item['compiled_depth_min']}–{item['compiled_depth_max']} |")
    lines+=["","The same target register is updated by quantum reset channels without measurement-conditioned classical hidden-state updates. The conditional distributions, however, are prepared from a classical toy count table. Real device-noise experiments and trained quantum denoising remain future work.",""]
    (path/"report.md").write_text("\n".join(lines),encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k!="cases"},indent=2))
    return report


def main(argv=None):
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if '--legacy-count-table' not in argv:
        from .matched import main as matched_main
        return matched_main(argv)
    argv.remove('--legacy-count-table')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots",type=int,default=256)
    parser.add_argument("--beta",type=float,default=1.0)
    parser.add_argument("--eta",type=float,default=.65)
    parser.add_argument("--steps",type=int,nargs="+",default=[0,1,3])
    parser.add_argument("--examples",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--readout-error",type=float,default=0.0)
    parser.add_argument("--output-dir",default="results/diffusion")
    return execute(**vars(parser.parse_args(argv)))

if __name__=="__main__":
    main()
