"""Verify NSIS bytes and exercise its backend over authenticated loopback HTTP."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
installer=root/'frontend/src-tauri/target/release/bundle/nsis/Fyodor_0.3.0_x64-setup.exe'
extracted=root/'build-train-profile/installer-split-attention-verified'
extracted.mkdir(exist_ok=False)
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
command=['C:/Program Files/7-Zip/7z.exe','x',str(installer),'-o'+str(extracted),'-y']
p=subprocess.run(command,capture_output=True,timeout=120)
(out/'installer-extract.log').write_bytes(p.stdout+p.stderr)
assert p.returncode==0
contents={}
for source in (root/'frontend/src-tauri/resources/backend').iterdir():
    if source.is_file():
        name='backend/'+source.name
        assert sha(source)==sha(extracted/name),name
        contents[name]=sha(source)
app=(root/'frontend/src-tauri/target/release/fyodor-desktop.exe').read_bytes()
marker=b'__TAURI_BUNDLE_TYPE_VAR_UNK'
assert app.count(marker)==1
assert (extracted/'fyodor-desktop.exe').read_bytes()==app.replace(marker,b'__TAURI_BUNDLE_TYPE_VAR_NSS',1)
contents['fyodor-desktop.exe']=sha(extracted/'fyodor-desktop.exe')
assert contents['backend/fyodor-backend.exe']==sha(root/'build-cuda/fyodor-backend.exe')
cli=[str(root/'.tools/w64devkit/bin/cmake.exe'),'-DTRAINER='+str(extracted/'backend/fyodor-train.exe'),'-DFIXTURES='+str(extracted),'-P',str(root/'backend/tests/TrainCli.cmake')]
p=subprocess.run(cli,capture_output=True,timeout=120)
(out/'packaged-cli.log').write_bytes(p.stdout+p.stderr)
assert p.returncode==0
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_COMPUTE='cuda',NYA_CPU_THREADS='6',NYA_CUDA_DEBUG='1')
config=extracted/'test-server.yaml'
config.write_text('server:\n  request_timeout_ms: 120000\n  worker_threads: 2\n  queue_capacity: 8\n')
calls=[]
with (out/'packaged-server.log').open('wb') as log:
    process=subprocess.Popen([str(extracted/'backend/fyodor-backend.exe'),'--config',str(config)],cwd=extracted,env=env,stdout=subprocess.PIPE,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        ready=queue.Queue()
        reader=threading.Thread(target=lambda:ready.put(process.stdout.readline()),daemon=True);reader.start()
        line=ready.get(timeout=30).decode().strip().split()
        assert len(line)==4 and line[0]=='FYODOR_READY' and line[1]=='127.0.0.1'
        host,port,token=line[1],int(line[2]),line[3]
        def request(method,path,body=None,authenticated=True):
            headers={'Content-Type':'application/json'}
            if authenticated:headers['Authorization']='Bearer '+token
            connection=http.client.HTTPConnection(host,port,timeout=120)
            connection.request(method,path,json.dumps(body) if body is not None else None,headers)
            response=connection.getresponse();data=json.loads(response.read());connection.close()
            calls.append({'method':method,'path':path,'status':response.status,'response':data})
            assert response.status==(200 if authenticated else 401),calls[-1]
            return data
        request('GET','/health',authenticated=False)
        request('GET','/health')
        loaded=request('POST','/model/load',{'path':str(root/'.tools/tinyllama-q4_k_m.gguf')})
        model_id=loaded['model']['id']
        assert loaded['model']['generation_supported']
        assert request('POST','/model/compute',{'model_id':model_id,'compute':'cuda'})['compute']=='cuda'
        prompt='The small library beside the river opens each morning. '*60+'The librarian'
        body={'model_id':model_id,'prompt':prompt,'max_tokens':8,'temperature':0,'seed':123}
        first=request('POST','/generate',body)
        assert 256<first['prompt_tokens']<2040 and first['generated_tokens']>0
        second=request('POST','/generate',body)
        assert first['text']==second['text'] and first['generated_tokens']==second['generated_tokens']
        runtime=request('GET','/runtime')
        assert runtime['models'][0]['compute']=='cuda'
        request('POST','/model/unload',{'model_id':model_id})
    finally:
        process.terminate();process.wait(timeout=30)
        process.stdout.close()
text=(out/'packaged-server.log').read_text(errors='replace')
assert 'CPU prefix replay required' not in text and 'resident plan declined' not in text and 'compilation failed' not in text
record={'installer':str(installer),'bytes':installer.stat().st_size,'sha256':sha(installer),'extraction_command':command,'verified_contents_sha256':contents,'desktop_comparison':'Identical after the one expected Tauri UNK -> NSS bundle marker replacement.','cli_command':cli,'cli_exit_code':0,'http_calls':calls,'installed':False,'interactive_ui_checked':False}
(out/'desktop-package.json').write_text(json.dumps(record,indent=2))
print('Installer contents, native trainer CLI and repeated long-prompt CUDA HTTP generation passed.',flush=True)
