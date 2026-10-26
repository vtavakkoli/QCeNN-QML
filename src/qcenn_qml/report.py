"""Combine recorded experiment JSONs without inventing unexecuted results."""
import argparse
import json
from pathlib import Path


def build(mnist, smol, diffusion, output_dir, diffusion_noisy=None,
          recurrent=None,attention=None,smol_feasibility=None):
    sources = {"mnist":mnist, "smol":smol, "diffusion":diffusion}
    if diffusion_noisy:
        sources["diffusion_noisy"] = diffusion_noisy
    for kind,source in (("recurrent",recurrent),("attention",attention),("smol_feasibility",smol_feasibility)):
        if source:
            sources[kind]=source
    found,missing={},[]
    for kind,source in sources.items():
        path=Path(source)
        if path.exists():
            found[kind]=json.loads(path.read_text(encoding="utf8"))
        else:
            missing.append(str(path))
    result={
        "purpose":"QCeNN-QML reproducible cross-experiment audit",
        "found_experiments":list(found),"missing_experiments":missing,
        "physical_quantum_hardware_runs":0,
        "real_smol135_transformer_on_quantum_hardware":False,
        "experiments":found
    }
    target=Path(output_dir);target.mkdir(parents=True,exist_ok=True)
    (target/"combined.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    lines=["# QCeNN-QML measured experiment summary","",
           "Each result refers to quantum circuits simulated using Qiskit Aer, not physical quantum hardware.",""]
    if "mnist" in found:
        x=found["mnist"]
        lines+=["## MNIST baseline","",
                f"- Dataset: {x.get('dataset')}; original MNIST: {x.get('is_real_mnist')}.",
                f"- Accuracy: {x.get('accuracy'):.4f} on {len(x.get('predictions',[]))} selected examples.",
                f"- Qubits: {x.get('data_qubits')}; measured example compiled depth: {x.get('transpiled_example_depth')}.",
                "- Not a trained quantum neural network or QCeNN-MI recurrent classifier.",""]
    if "smol" in found:
        x=found["smol"]
        lines+=["## Quantum text memory","",
                f"- Generated text: {x.get('generated_text')}.",
                f"- Vocabulary: {len(x.get('vocabulary',[]))} tokens, {x.get('qubits')} qubits.",
                f"- Classical SmolLM2-135M teacher used: {x.get('classical_teacher_used')}.",
                "- Not the SmolLM2-135M pretrained transformer on a quantum device.",""]
    if "diffusion" in found:
        x=found["diffusion"]
        lines+=["## Quantum masked-token denoising","",
                f"- Train/test sizes: {x.get('train_count')}/{x.get('heldout_count')}; evaluated cases: {x.get('evaluation_count')}.",
                f"- Engineered masking beta={x.get('forward_mask_probability_beta')}; reverse collision eta={x.get('reverse_replacement_probability_eta')}.",
                "| Steps | Measured correct-token fraction | Ideal channel expectation |",
                "|---:|---:|---:|"]
        for k,item in x.get("results",{}).items():
            lines.append(f"| {k} | {item['measured_correct_token_probability']:.4f} | {item['independent_ideal_probability_mean']:.4f} |")
        lines+=["","- Not a trained quantum diffusion LLM or a reproduced upstream QCeNN-MI solver.",""]
    if "diffusion_noisy" in found:
        x=found["diffusion_noisy"]
        lines+=["## Readout bit-flip noise comparison","",
                f"- Simulated readout error per measured bit: {x.get('readout_bitflip_probability')}.",
                "| Steps | Measured clean-token frequency (noisy) | Ideal-readout result |",
                "|---:|---:|---:|"]
        # Prefer the paired noiseless run from the SAME file/cases. Never
        # compare broader text against an unrelated toy-corpus report.
        baseline=x.get('matched_results',{}).get('noiseless',{})
        if not baseline:
            previous=found.get('diffusion',{})
            keys=('vocabulary','train_count','heldout_count','evaluation_count','seeds',
                  'forward_mask_probability_beta','reverse_replacement_probability_eta','steps')
            if previous and all(previous.get(k)==x.get(k) for k in keys):
                baseline=previous.get('results',{})
        for k,item in x.get("results",{}).items():
            ref=baseline.get(k,{}).get("measured_correct_token_probability")
            lines.append(f"| {k} | {item['measured_correct_token_probability']:.4f} | "
                         + (f"{ref:.4f}" if ref is not None else "not run") + " |")
        lines+=["","Readout flips are a measurement-error model, not a beneficial denoising resource.",""]
    if 'recurrent' in found:
        x=found['recurrent']
        lines+=['## Quantum-memory MNIST','',
            f"- Tiny training fit gate required: {x.get('tiny_fit_gate_required')}.",
            f"- Mean evaluated test accuracy: {x.get('test_accuracy_mean')}.",
            '- Nonparametric quantum exemplar memory, with no classical optimizer.','']
    if 'attention' in found:
        x=found['attention']
        lines+=['## Reversible quantum attention','',
            f"- Measured component cases: {len(x.get('cases',[]))}.",
            '- Signed ternary dot products and softmax sampling are computed in gates.',
            '- Full pretrained Smol attention blocks and coherent full-precision vector output are unimplemented.','']
    if 'smol_feasibility' in found:
        x=found['smol_feasibility']
        lines+=['## Full Smol feasibility status','',
            f"- Status: {x.get('full_model_quantum_conversion_test')}.",
            f"- Explicit resident basis-encoding weight qubits: {x.get('chosen_basis_encoding_weight_qubits')}.",
            '- This is an encoding estimate, not a lower bound on all quantum implementations.','']
    if missing:lines+=["## Experiments not executed here",""]+["- "+p for p in missing]+[""]
    lines+=["## Conclusion","",
            "All experiments are feasibility demonstrations. They do not establish speedup, advantage, physically executed noise benefits, or a full 135M-parameter quantum language model.",""]
    (target/"combined.md").write_text("\n".join(lines),encoding="utf8")
    print(json.dumps({"summary":str(target/"combined.md"),"found":list(found),"missing":missing},indent=2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist",default="results/ci/mnist/report.json")
    parser.add_argument("--smol",default="results/ci/smol/report.json")
    parser.add_argument("--diffusion",default="results/ci/diffusion/report.json")
    parser.add_argument("--output-dir",default="results/ci/summary")
    parser.add_argument("--diffusion-noisy",default=None)
    parser.add_argument('--recurrent')
    parser.add_argument('--attention')
    parser.add_argument('--smol-feasibility')
    return build(**vars(parser.parse_args(argv)))

if __name__=="__main__":
    main()
