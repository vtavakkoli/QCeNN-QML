"""Combine recorded experiment JSONs without inventing unexecuted results."""
import argparse
import json
from pathlib import Path


def build(mnist, smol, diffusion, output_dir, diffusion_noisy=None,
          recurrent=None,attention=None,smol_feasibility=None,
          mnist_study=None,diffusion_study=None,transformer_block=None,transformer_replacement=None):
    sources = {key:value for key,value in (("mnist",mnist),("smol",smol),("diffusion",diffusion)) if value}
    if diffusion_noisy:
        sources["diffusion_noisy"] = diffusion_noisy
    for kind,source in (("recurrent",recurrent),("attention",attention),("smol_feasibility",smol_feasibility)):
        if source:
            sources[kind]=source
    for kind,source in (("mnist_study",mnist_study),("diffusion_study",diffusion_study),("transformer_block",transformer_block)):
        if source:
            sources[kind]=source
    if transformer_replacement:
        sources['transformer_replacement'] = transformer_replacement
    found,missing={},[]
    for kind,source in sources.items():
        paths=source if isinstance(source,(list,tuple)) else [source]
        for index,source_path in enumerate(paths):
            path=Path(source_path)
            key=kind if len(paths)==1 else f'{kind}_{index}'
            if path.exists():
                found[key]=json.loads(path.read_text(encoding="utf8"))
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
    for kind,x in found.items():
        if kind!='recurrent' and not kind.startswith('recurrent_'):
            continue
        lines+=[f'## Quantum-memory MNIST ({kind})','',
            f"- Tiny training fit gate required: {x.get('tiny_fit_gate_required')}.",
            f"- Mean evaluated test accuracy: {x.get('test_accuracy_mean')}.",
            '- Nonparametric quantum exemplar memory, with no classical optimizer.','']
        for run in x.get('runs',[]):
            gate=run.get('splits',{}).get('tiny_train',{}).get('fit_gate_passed')
            lines.append(f"- Seed {run['seed']}: fit gate {gate}; test evaluated {run.get('test_evaluated')}.")
        lines.append('')
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
    if 'mnist_study' in found:
        x=found['mnist_study']
        lines+=['## Staged MNIST study','',f"- Status: {x.get('status')}; test evaluated: {x.get('test_evaluated')}.",
            '- Independent data/measurement seeds; equal-shot and equal-CNOT comparisons; validation-only selection.','',
            '| Model | Budget | Test examples | Accuracy | Shot-weighted CX |','|---|---|---:|---:|---:|']
        for row in x.get('test_summary',[]):
            lines.append(f"| {row['model']} | {row['mode']} | {row['n']} | {row['accuracy']:.4f} | {row['cx_shot_weighted_total']} |")
        lines.append('')
    if 'diffusion_study' in found:
        x=found['diffusion_study']
        lines+=['## Separate multi-corpus diffusion study','',f"- Status: {x.get('status')}; recorded runs: {len(x.get('runs',[]))}.",
            '- Identical native circuits for noiseless/noisy readout; direct classical table evaluated separately.',
            '- Unknown words, unseen contexts and invalid noisy token codes retain their error mass.','']
    if 'transformer_block' in found:
        x=found['transformer_block']
        lines+=['## Complete small coherent transformer block','',f"- Status: {x.get('status')}.",
            '- Two tokens, width two, one head; three-bit fixed-point reversible arithmetic.',
            '- QKV, RoPE, causal softmax, coherent weighted vectors, output projection, RMSNorm, residuals and SwiGLU.',
            '- Fixed toy weights; no pretrained Smol weights, quantum neural training or hardware execution.','']
    if 'transformer_replacement' in found:
        x = found['transformer_replacement']
        lines += ['## Complete small quantum transformer replacement', '',
            f"- Status: {x.get('status')}; measured checks: {len(x.get('cases', []))}.",
            '- Quantum embedding, one/two complete fixed-point blocks and vocabulary argmax.',
            '- All 16 input pairs tested in superposition; phase interference and hybrid-host generation checked separately.',
            '- Four-token toy vocabulary and fixed weights; full pretrained Smol remains unimplemented.', '']
    if missing:lines+=["## Requested reports that are missing",""]+["- "+p for p in missing]+[""]
    lines+=["## Conclusion","",
            "All experiments are feasibility demonstrations. They do not establish speedup, advantage, physically executed noise benefits, or a full 135M-parameter quantum language model.",""]
    (target/"combined.md").write_text("\n".join(lines),encoding="utf8")
    print(json.dumps({"summary":str(target/"combined.md"),"found":list(found),"missing":missing},indent=2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist")
    parser.add_argument("--smol")
    parser.add_argument("--diffusion")
    parser.add_argument("--output-dir",default="results/ci/summary")
    parser.add_argument("--diffusion-noisy",default=None)
    parser.add_argument('--recurrent',nargs='+',help='One or more independently recorded MNIST seed reports')
    parser.add_argument('--attention')
    parser.add_argument('--smol-feasibility')
    parser.add_argument('--mnist-study')
    parser.add_argument('--diffusion-study')
    parser.add_argument('--transformer-block')
    parser.add_argument('--transformer-replacement')
    return build(**vars(parser.parse_args(argv)))

if __name__=="__main__":
    main()
