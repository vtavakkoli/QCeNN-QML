"""Operator-coverage/resource audit: a toy attention pass is not full SmolLM2."""
import argparse
from pathlib import Path
from .recurrent_mnist import write_json


CONFIG_SOURCE="https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json"
CONFIG={"hidden_size":576,"intermediate_size":1536,"num_hidden_layers":30,
    "num_attention_heads":9,"num_key_value_heads":3,"vocab_size":49152,
    "max_position_embeddings":8192,"tie_word_embeddings":True,
    "hidden_act":"silu","rms_norm_eps":1e-5,"rope_theta":100000,
    "torch_dtype":"bfloat16"}


def audit(attention_report=None,weight_bits=16,context_length=128,output_dir="results/smol-feasibility",transformer_report=None):
    import json
    if weight_bits<2 or not 1<=context_length<=CONFIG['max_position_embeddings']:
        raise ValueError("Invalid precision/context budget")
    h=CONFIG['hidden_size']; layers=CONFIG['num_hidden_layers']
    kv=CONFIG['num_key_value_heads']*(h//CONFIG['num_attention_heads'])
    attention_matrices=2*h*h+2*h*kv
    mlp_matrices=3*h*CONFIG['intermediate_size']
    parameters=CONFIG['vocab_size']*h+layers*(attention_matrices+mlp_matrices+2*h)+h
    component=json.loads(Path(attention_report).read_text()) if attention_report else None
    if component and component.get('experiment')!='signed_ternary_quantum_dot_softmax_attention':
        raise ValueError("Not a verified quantum-attention component report")
    measured=bool(component and component.get('cases'))
    block=json.loads(Path(transformer_report).read_text()) if transformer_report else None
    if block:
        from .quantum_transformer import OPERATORS
        if (block.get('experiment')!='coherent_fixed_point_transformer_block' or
                block.get('status')!='VERIFIED' or not block.get('cases') or
                set(block.get('operators',[]))!=set(OPERATORS) or
                any(row.get('correct_output_fraction')!=1 or row.get('clean_workspace_fraction')!=1
                    for row in block['cases'])):
            raise ValueError('Not a verified complete small quantum transformer block')
    rows={"ternary_signed_dot_product_attention_sampling":measured,
        "actual_pretrained_weight_loading":False,"full_precision_qkv_and_output_projections":False,
        "rotary_position_encoding":False,"rms_normalization":False,
        "coherent_weighted_sum_hidden_vector":False,"silu_gated_mlp":False,
        "full_vocabulary_head":False,"kv_cache":False,"thirty_layer_composition":False,
        "quantum_only_neural_weight_training":False,"physical_qpu_execution":False}
    report={"model":"HuggingFaceTB/SmolLM2-135M","configuration_source":CONFIG_SOURCE,
        "configuration_snapshot_checked":"2026-10-26","configuration":CONFIG,
        "full_model_quantum_conversion_test":"NOT_IMPLEMENTED_NOT_DEMONSTRATED",
        "actual_smol_weights_executed":False,"physical_qpu_tested":False,
        "attention_component_report":attention_report,"coverage":rows,
        "tiny_coherent_block_report":transformer_report,
        "complete_small_coherent_block_verified":bool(block),
        "tiny_block_operator_coverage":{name:bool(block) for name in (
            'qkv_projections','rotary_position_encoding','rms_normalization',
            'causal_softmax','coherent_weighted_sum_hidden_vector','output_projection',
            'residual_connections','silu_gated_mlp')},
        "tiny_block_scope":"Two tokens, width two, one head, three-bit modular arithmetic, fixed toy weights; not full Smol precision or pretrained weights",
        "parameters_from_shapes":parameters,"weight_bits":weight_bits,
        "chosen_basis_encoding_weight_qubits":parameters*weight_bits,
        "weight_bytes_at_chosen_precision":(parameters*weight_bits+7)//8,
        "dense_projection_and_mlp_MACs_per_token":layers*(attention_matrices+mlp_matrices),
        "vocabulary_head_MACs_per_token":CONFIG['vocab_size']*h,
        "total_dense_MACs_before_context_and_nonlinearities":layers*(attention_matrices+mlp_matrices)+CONFIG['vocab_size']*h,
        "additional_context_attention_MACs_per_token":2*layers*h*context_length,
        "context_length":context_length,
        "encoding_scope":"Explicit resident basis encoding; not a lower bound on all quantum encodings",
        "other_encoding_caveat":"Compiling weights into QROM can trade weight registers for large loading circuits; it does not provide a free oracle",
        "claim":"A complete tiny fixed-point block can be tested separately; full pretrained Smol conversion remains unimplemented and unexecuted",
        "limitations":["No 135M weight download or conversion was attempted by this audit",
            "Toy attention sampling alone is not a block; the separate tiny block does not establish full Smol precision or scaling",
            "MAC counts are classical-operation counts, not measured quantum-gate counts",
            "No quantum speedup or hardware feasibility follows from this resource ledger"]}
    directory=Path(output_dir); directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'report.json',report)
    (directory/'report.md').write_text(
        '# SmolLM2-135M quantum feasibility audit\n\n'
        '**Full pretrained model conversion: not implemented or demonstrated.**\n\n'
        f'Actual configuration: {layers} layers, width {h}, vocabulary {CONFIG["vocab_size"]}.\n\n'
        f'Parameter count from shapes: {parameters:,}. A {weight_bits}-bit resident basis encoding '
        f'would use {parameters*weight_bits:,} logical weight qubits, before workspace/error correction. '
        'This is one encoding estimate, not an impossibility theorem.\n\n'
        f'Dense projections and MLPs require {layers*(attention_matrices+mlp_matrices):,} scalar MACs per token, '
        'before context attention, nonlinearities, normalization, data access and readout.\n\n'
        'The tested quantum component computes small signed dot products and softmax attention samples. '
        'It does not execute Smol weights or produce a coherent full-precision hidden vector.\n\n'
        f'Complete tiny coherent block independently verified here: {bool(block)}. '
        'Its three-bit arithmetic, fixed weights and two-token context do not establish full Smol conversion.\n\n'
        f'Source: {CONFIG_SOURCE}\n',encoding='utf8')
    print(f"Smol full-model status: {report['full_model_quantum_conversion_test']}",flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention-report')
    parser.add_argument('--transformer-report')
    parser.add_argument('--weight-bits',type=int,default=16)
    parser.add_argument('--context-length',type=int,default=128)
    parser.add_argument('--output-dir',default='results/smol-feasibility')
    return audit(**vars(parser.parse_args(argv)))


if __name__=='__main__':
    main()
