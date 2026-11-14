"""CSV learning rate follows the actual optimizer, including checkpoint resume."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

trainer=str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix='fyodor-rate-') as directory:
    root=Path(directory);data=root/'data.txt';data.write_text('abc'*24)
    def run(name, extra):
        metrics=root/(name+'.csv');checkpoint=root/(name+'.ckpt')
        command=[trainer,'--compute','cpu','--data',str(data),'--output',str(root/(name+'.gguf')),
                 '--checkpoint',str(checkpoint),'--metrics',str(metrics),'--steps','1',
                 '--dimension','8','--ff','16','--layers','1','--heads','2','--kv-heads','1',
                 '--context','8',*extra]
        subprocess.run(command,check=True,capture_output=True,text=True,timeout=30)
        with metrics.open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==1 and None not in rows[0],rows
        assert all(value is not None for value in rows[0].values()),rows
        assert abs(float(rows[0]['learning_rate'])-.002)<1e-8,rows
        assert rows[0]['device_used_bytes']=='',rows
        return rows[0],checkpoint
    initial,checkpoint=run('initial',['--lr','0.002'])
    resumed,_=run('resumed',['--resume',str(checkpoint)])
    assert initial['step']=='1' and resumed['step']=='2'
    assert initial['learning_rate']==resumed['learning_rate']
print('CSV learning rate and resumed optimizer rate agree; columns remain aligned')
