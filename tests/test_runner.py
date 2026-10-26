"""Exercise real Bash scheduling/ZIP/error handling with cheap child processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import pytest


ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'scripts/run_quantum_corrections.sh'
NAMES=('mnist-42','mnist-43','mnist-44','diffusion','attention','text','smol-audit','summary')


@pytest.fixture
def fake_python(tmp_path):
    executable=tmp_path/'fake python'
    executable.write_text(f'#!{sys.executable}\n'+'''import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
if '-c' in args: sys.exit(0)
module=args[args.index('-m')+1]
out=Path(args[args.index('--output-dir')+1]); out.mkdir(parents=True,exist_ok=True)
started=time.time()
trace={'start':started,'pid':os.getpid(),'threads':int(os.environ['QCENN_THREADS']),'argv':args}
(out/'trace.json').write_text(json.dumps(trace))
print('running '+out.name,flush=True)
time.sleep(4 if os.environ.get('QCENN_TEST_SLOW')==out.name else float(os.environ.get('QCENN_TEST_SLEEP','.25')))
if os.environ.get('QCENN_TEST_FAIL')==out.name:
 (out/'partial.txt').write_text('partial output')
 sys.exit(9)
blocked=os.environ.get('QCENN_TEST_BLOCK')==out.name
report={'experiment':module,'runs':[{'seed':42,'test_evaluated':not blocked,
 'splits':{'tiny_train':{'fit_gate_passed':not blocked}}}]}
filename='combined.json' if module=='qcenn_qml.report' else 'report.json'
if os.environ.get('QCENN_TEST_MISSING')!=out.name:
 (out/filename).write_text(json.dumps(report))
trace['end']=time.time(); (out/'trace.json').write_text(json.dumps(trace))
print('finished '+out.name,flush=True)
sys.exit(2 if blocked else 0)
''')
    executable.chmod(0o755)
    return executable


def environment(fake_python,**extra):
    return {**os.environ,'QCENN_PYTHON':str(fake_python),'QCENN_CPUS':'4',
            'QCENN_JOBS':'2','QCENN_LATEST_FILE':str(fake_python.parent/'latest.txt'),**extra}


def run(tmp_path,fake_python,**extra):
    output=tmp_path/'run with spaces'
    process=subprocess.run(['bash',str(RUNNER),'--smoke',str(output)],cwd=ROOT,
        env=environment(fake_python,**extra),capture_output=True,text=True,timeout=30)
    return output,process


def test_parallel_runner_archives_each_job_and_preserves_dependencies(tmp_path,fake_python):
    output,process=run(tmp_path,fake_python)
    assert process.returncode==0,process.stdout+process.stderr
    assert 'FINISHED state=DONE exit=0' in process.stdout
    assert (output/'DONE').exists()
    traces={name:json.loads((output/f'experiments/{name}/trace.json').read_text()) for name in NAMES}
    assert max(traces['mnist-42']['start'],traces['mnist-43']['start']) < min(
        traces['mnist-42']['end'],traces['mnist-43']['end'])
    assert all(t['threads']==2 for t in traces.values())
    assert traces['smol-audit']['start']>=traces['attention']['end']
    assert traces['summary']['start']>=max(t['end'] for n,t in traces.items() if n!='summary')
    for name in NAMES:
        archive=output/f'archives/{name}.zip'
        assert archive.exists() and archive.with_suffix('.zip.sha256').exists()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            filename='combined.json' if name=='summary' else 'report.json'
            assert f'experiments/{name}/{filename}' in z.namelist()
            assert z.read(f'status/{name}.tsv').startswith(b'DONE\t0')
    args=traces['summary']['argv']
    assert all(str(output/f'experiments/mnist-{seed}/report.json') in args for seed in (42,43,44))
    status=subprocess.run(['bash',str(RUNNER),'--status'],env=environment(fake_python),capture_output=True,text=True)
    assert status.returncode==0 and 'State: DONE' in status.stdout
    repeated=subprocess.run(['bash',str(RUNNER),'--smoke',str(output)],
        env=environment(fake_python),capture_output=True,text=True)
    assert repeated.returncode!=0 # Never overwrite evidence from a previous run.


def test_blocked_fit_gate_is_not_reported_as_success(tmp_path,fake_python):
    output,process=run(tmp_path,fake_python,QCENN_TEST_BLOCK='mnist-42')
    assert process.returncode==2
    assert 'FINISHED state=DONE_WITH_BLOCKED_TESTS' in process.stdout
    assert not (output/'DONE').exists()
    assert (output/'DONE_WITH_BLOCKED_TESTS').exists()
    with zipfile.ZipFile(output/'archives/mnist-42.zip') as z:
        assert z.read('status/mnist-42.tsv').startswith(b'GATE_FAILED\t2')
    assert (output/'archives/diffusion.zip').exists()


def test_attention_audit_does_not_wait_for_slow_text(tmp_path,fake_python):
    output,process=run(tmp_path,fake_python,QCENN_TEST_SLOW='text')
    assert process.returncode==0,process.stdout+process.stderr
    audit=json.loads((output/'experiments/smol-audit/trace.json').read_text())
    text=json.loads((output/'experiments/text/trace.json').read_text())
    assert audit['start']<text['end']


def test_mnist_only_does_not_repeat_completed_other_experiments(tmp_path,fake_python):
    output=tmp_path/'mnist-only'
    result=subprocess.run(['bash',str(RUNNER),'--smoke','--mnist-only',str(output)],
        env=environment(fake_python),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert {p.name for p in (output/'archives').glob('*.zip')}=={
        'mnist-42.zip','mnist-43.zip','mnist-44.zip','summary.zip'}
    assert not (output/'experiments/diffusion').exists()


def test_failed_job_keeps_partial_output_and_other_experiments(tmp_path,fake_python):
    output,process=run(tmp_path,fake_python,QCENN_TEST_FAIL='attention')
    assert process.returncode==1
    assert 'FINISHED state=FAILED' in process.stdout
    assert not (output/'DONE').exists()
    with zipfile.ZipFile(output/'archives/attention.zip') as z:
        assert z.read('status/attention.tsv').startswith(b'FAILED\t9')
        assert z.read('experiments/attention/partial.txt')==b'partial output'
    with zipfile.ZipFile(output/'archives/smol-audit.zip') as z:
        assert z.read('status/smol-audit.tsv').startswith(b'SKIPPED\t1')
    assert (output/'archives/diffusion.zip').exists()


def test_zero_exit_without_report_is_a_failed_experiment(tmp_path,fake_python):
    output,process=run(tmp_path,fake_python,QCENN_TEST_MISSING='diffusion')
    assert process.returncode==1
    assert (output/'FAILED').exists()
    assert (output/'status/diffusion.tsv').read_text().startswith('FAILED\t1')
    assert 'expected report is missing' in (output/'logs/diffusion.log').read_text()


def test_termination_stops_active_children_and_records_interruption(tmp_path,fake_python):
    output=tmp_path/'interrupted'
    process=subprocess.Popen(['bash',str(RUNNER),'--smoke',str(output)],
        env=environment(fake_python,QCENN_TEST_SLEEP='30'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        trace=output/'experiments/mnist-42/trace.json'
        deadline=time.monotonic()+10
        while not trace.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert trace.exists()
        child=json.loads(trace.read_text())['pid']
        process.send_signal(signal.SIGTERM)
        stdout,stderr=process.communicate(timeout=10)
        assert process.returncode==143,stdout+stderr
        assert 'FINISHED state=INTERRUPTED' in stdout
        assert (output/'INTERRUPTED').exists()
        with pytest.raises(ProcessLookupError): os.kill(child,0)
    finally:
        if process.poll() is None:
            process.kill(); process.communicate(timeout=5)
