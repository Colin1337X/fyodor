"""Held-out evaluation must preserve exact training and recovery trajectories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('trainer', type=Path)
parser.add_argument('--evidence', type=Path)
args = parser.parse_args()
trainer = str(args.trainer.resolve())
evidence = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with tempfile.TemporaryDirectory(prefix='fyodor-eval-') as folder:
    root = Path(folder)
    corpus, sft, dpo = [root / name for name in ('corpus.txt','sft.tsv','dpo.tsv')]
    corpus.write_text('abc' * 30, encoding='utf-8')
    sft.write_text('a\tbc\nabc\tcabcabc\nc\tacba\n', encoding='utf-8')
    dpo.write_text('a\tbc\tc\nabc\tcabcabc\tccabc\nc\tacba\tabc\n', encoding='utf-8')
    validation = root / '\ud55c\uae00-\U0001f43e'
    validation.mkdir()
    eval_corpus, eval_sft, eval_dpo = [validation / p.name for p in (corpus,sft,dpo)]
    eval_corpus.write_text('abacbc' * 8, encoding='utf-8')
    eval_sft.write_text('ac\tbcabc\na\tc\n', encoding='utf-8')
    eval_dpo.write_text('ac\tbcabc\tcabc\na\tc\tb\n', encoding='utf-8')
    dimensions = ['--dimension','16','--ff','32','--layers','1','--heads','2','--kv-heads','1','--context','16']

    def run(name, steps, settings=(), resume=None, stop=False, failure=False):
        model, checkpoint, metrics = [root / (name + ext) for ext in ('.gguf','.ckpt','.csv')]
        command = [trainer, '--data', str(corpus), '--output', str(model), '--checkpoint', str(checkpoint),
                   '--metrics', str(metrics), '--steps', str(steps), '--accumulate','3', *dimensions, *settings]
        if resume:
            command += ['--resume',str(resume)]
        if stop:
            command += ['--control-stdin','1']
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding='utf-8')
            first = child.stdout.readline()
            assert first.startswith('evaluation_start step=0'), first
            child.stdin.write('S'); child.stdin.flush()
            output, error = child.communicate(timeout=30)
            output = first + output
            code = child.returncode
        else:
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=30)
            code, output, error = result.returncode, result.stdout, result.stderr
        record = {'name':name,'command':command,'exit_code':code,'stdout':output,'stderr':error}
        evidence.append(record)
        if failure:
            assert code != 0 and not model.exists() and not checkpoint.exists(), record
            return record, checkpoint, model
        assert code == 0, record
        record.update(model_sha256=sha(model),checkpoint_sha256=sha(checkpoint))
        with metrics.open(newline='') as stream:
            record['metrics'] = list(csv.DictReader(stream))
        record['evaluations'] = [dict(item.split('=',1) for item in line.split())
                                 for line in output.splitlines() if line.startswith('eval_step=')]
        return record, checkpoint, model

    _, _, base = run('base', 2)
    for mode, data, held in [('pretrain',corpus,eval_corpus),('cpt',corpus,eval_corpus),
                             ('sft',sft,eval_sft),('dpo',dpo,eval_dpo)]:
        settings = ['--mode',mode,'--data',str(data)]
        if mode != 'pretrain':
            settings += ['--base',str(base),'--rank','2']
        enabled = [*settings,'--eval-data',str(held),'--eval-every','2']
        plain, _, _ = run(mode+'-plain',5,settings)
        full, _, _ = run(mode+'-eval',5,enabled)
        first, checkpoint, _ = run(mode+'-first',2,enabled)
        resumed, _, _ = run(mode+'-resume',3,enabled,resume=checkpoint)
        for key in ('model_sha256','checkpoint_sha256'):
            assert plain[key] == full[key] == resumed[key], (mode,key)
        loss = lambda record: [row['loss'] for row in record['metrics']]
        assert loss(plain) == loss(full) == loss(first)+loss(resumed)
        assert [row['eval_step'] for row in full['evaluations']] == ['0','2','4','5']
        assert [row['eval_step'] for row in resumed['evaluations']] == ['2','4','5']
        for index, row in enumerate(resumed['evaluations']):
            assert row['validation_loss'] == full['evaluations'][index+1]['validation_loss']
        assert [row['step'] for row in full['metrics'] if row['validation_loss']] == ['2','4','5']
        for row in plain['metrics']:
            assert row['validation_loss'] == '' and row['eval_ms'] == ''
        limited, _, _ = run(mode+'-limited',5,[*enabled,'--eval-records','1'])
        assert all(row['eval_records'] == '1' for row in limited['evaluations'])
        assert limited['checkpoint_sha256'] == plain['checkpoint_sha256']
        changed = [*enabled,'--eval-records','1','--threads','3']
        toggled, _, _ = run(mode+'-toggled',3,changed,resume=checkpoint)
        assert toggled['checkpoint_sha256'] == full['checkpoint_sha256']

    for index, options in enumerate([['--eval-every','0'],['--eval-every','-1'],['--eval-records','-1'],
                                    ['--eval-every','1'],['--eval-records','1'],['--eval-data',str(root/'absent')]]):
        run('invalid-'+str(index),1,options,failure=True)
    malformed = validation / 'bad.tsv'; malformed.write_text('no tabs\n',encoding='utf-8')
    run('malformed',1,['--mode','sft','--base',str(base),'--data',str(sft),'--eval-data',str(malformed)],failure=True)
    empty = validation / 'empty.txt'; empty.write_bytes(b'')
    run('empty',1,['--eval-data',str(empty)],failure=True)

    # A long initial pass lets the controller stop at a deterministic published
    # boundary, without timing a sleep against child startup.
    long_data = validation / 'long.txt'; long_data.write_text('abc'*40000,encoding='utf-8')
    stopped, checkpoint, _ = run('stop-eval',5,['--eval-data',str(long_data)],stop=True)
    assert not stopped['evaluations'] and not stopped['metrics']
    assert 'stopping step=0' in stopped['stdout']
    resumed, _, _ = run('stop-resume',5,resume=checkpoint)
    plain, _, _ = run('stop-plain',5)
    assert resumed['checkpoint_sha256'] == plain['checkpoint_sha256']
    assert resumed['model_sha256'] == plain['model_sha256']

if args.evidence:
    args.evidence.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print(f'Evaluation: {len(evidence)} process runs passed; exact training, resume, masking, fixed subset and stop.')
