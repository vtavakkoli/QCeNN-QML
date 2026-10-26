"""Verify extracted v0.3 evidence without rerunning expensive model circuits."""
import argparse
import hashlib
import json
from pathlib import Path

from qiskit import qpy
from qcenn_qml.coherent import decode_postselection


def verify(root):
    root=Path(root)
    totals={'raw_readouts':0,'qpy_examples':0,'source_files_verified':0}
    for path in sorted(root.rglob('report.json')):
        report=json.loads(path.read_text())
        for name,digest in report.get('source',{}).get('source_sha256',{}).items():
            candidates=[p/name for p in (root/'provenance-source').iterdir()]
            assert any(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
                       for p in candidates),(path,name,'source mismatch')
            totals['source_files_verified']+=1
        if report.get('experiment')=='coherent_recurrent_quantum_exemplar_memory':
            for run in report['runs']:
                for name,split in run['splits'].items():
                    rounds=1 if 'k0_reference' in name else report['configuration']['rounds']
                    for row in split['predictions']:
                        decoded=decode_postselection(row['counts'],4,rounds,10)
                        for key in ('prediction','histogram','invalid_value_shots','accepted_shots','total_shots'):
                            assert decoded[key]==row[key],(path,name,key)
                        totals['raw_readouts']+=1
                if report['configuration'].get('validation_only'):
                    assert not run['test_evaluated'] and 'test' not in run['splits']
        elif 'matched_results' in report and 'cases' in report:
            bits=(len(report['vocabulary'])-1).bit_length()
            for row in report['cases']:
                decoded=decode_postselection(row['counts'],bits,1,len(report['vocabulary']))
                for key in ('prediction','histogram','invalid_value_shots','accepted_shots','total_shots'):
                    assert decoded[key]==row[key],(path,key)
                totals['raw_readouts']+=1
        elif report.get('experiment')=='signed_ternary_quantum_dot_softmax_attention':
            for row in report['cases']:
                output_bits=max(1,(len(row['keys'])-1).bit_length())+2*len(row['values'][0])
                assert sum(row['counts'].values())==report['shots']
                assert sum(n for bits,n in row['counts'].items() if int(bits,2)>>output_bits)==row['accepted_shots']
                totals['raw_readouts']+=1
        elif 'sequence_counts' in report:
            assert sum(report['sequence_counts'].values())==report['accepted_shots']
            assert sum(report['counts'].values())==report['resources']['total_shots']
            totals['raw_readouts']+=1
    for path in sorted(root.rglob('*.qpy')):
        with path.open('rb') as stream:
            circuits=qpy.load(stream)
        assert circuits and all(c.num_qubits and c.num_clbits for c in circuits),path
        totals['qpy_examples']+=len(circuits)
    return totals


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',help='Directory extracted from raw-evidence.tar.gz')
    print(json.dumps(verify(parser.parse_args().directory),indent=2))
