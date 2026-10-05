"""Real HTTP/CLI interchange, auth boundaries, revisions and process restart."""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import uuid

binary, cli = [str(Path(p).resolve()) for p in sys.argv[1:3]]
model = str(Path(sys.argv[3]).resolve()) if len(sys.argv)>3 else None
with tempfile.TemporaryDirectory(prefix='fyodor-resource-http-') as folder:
    database = str(Path(folder) / '\u6587-workspace.db')
    def start(enabled=True, extra=()):
        env = dict(os.environ)
        env.pop('FYODOR_STORE_PATH', None)
        if enabled:
            env['FYODOR_STORE_PATH'] = database
        process = subprocess.Popen([binary, '--port', '0', *extra], cwd=folder, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        messages = queue.Queue()
        threading.Thread(target=lambda: messages.put(process.stdout.readline()), daemon=True).start()
        try:
            ready = messages.get(timeout=10).split()
            assert len(ready) == 4 and ready[0] == 'FYODOR_READY'
            return process, 'http://127.0.0.1:' + ready[2], ready[3]
        except BaseException:
            process.kill(); process.wait(); raise
    process, base, token = start()
    def request(operation, payload=None, status=200, authorized=True, origin=None, raw=None):
        headers = {'Content-Type':'application/json'}
        if authorized:
            headers['Authorization'] = 'Bearer ' + token
        if origin:
            headers['Origin'] = origin
        data = raw if raw is not None else json.dumps(payload or {}).encode()
        route=operation if operation.startswith('/') else '/resources/'+operation
        req = urllib.request.Request(base + '/api/v1' + route, data=data, headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = json.load(response)
            assert response.status == status, (response.status, body)
            return body
    def stop():
        try:
            req=urllib.request.Request(base+'/api/v1/shutdown',data=b'{}',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            urllib.request.urlopen(req,timeout=5).close()
            assert process.wait(timeout=5)==0
        finally:
            if process.poll() is None:
                process.kill();process.wait()
    uri='fyodor://writing/documents/'+str(uuid.uuid4())
    package=dict(schema=1,uri=uri,title='Document \u6587',content='<script>inert</script>\ntext',metadata={'large_integer':9007199254740993},provenance={})
    try:
        request('list',{'namespace':'work'},401,authorized=False)
        request('list',{'namespace':'work'},403,origin='https://untrusted.example')
        assert request('list',{'namespace':'work'})['resources']==[]
        assert request('put',{'namespace':'work','expected_revision':'0','resource':package})['revision']=='1'
        record=request('get',{'namespace':'work','uri':uri})
        assert record['resource']==package and record['revision']=='1'
        request('put',{'namespace':'work','expected_revision':1,'resource':package},400)
        request('put',{'namespace':'work','expected_revision':'01','resource':package},400)
        request('put',raw=b'{"namespace":"work","namespace":"other"}',status=400)
        args=[cli,'--store',database,'--namespace','work','resource']
        exported=subprocess.run([*args,'export',uri],capture_output=True,check=True)
        assert json.loads(exported.stdout)==package
        package['content']='updated through CLI'
        subprocess.run([*args,'import','1'],input=json.dumps(package).encode(),capture_output=True,check=True)
        assert request('get',{'namespace':'work','uri':uri})['revision']=='2'
        request('put',{'namespace':'work','expected_revision':'1','resource':package},409)
        assert [item['revision'] for item in request('history',{'namespace':'work','uri':uri})['revisions']]==['2','1']
        assert request('history',{'namespace':'work','uri':uri,'before':'2'})['revisions'][0]['revision']=='1'
        old=request('get',{'namespace':'work','uri':uri,'revision':'1'})
        assert old['resource']['content']=='<script>inert</script>\ntext' and old['deleted'] is False
        assert old['resource']['metadata']['large_integer']==9007199254740993
        for invalid in ('01','-1',1,'9223372036854775808'):
            request('get',{'namespace':'work','uri':uri,'revision':invalid},400)
            request('history',{'namespace':'work','uri':uri,'before':invalid},400)
        request('get',{'namespace':'work','uri':uri,'revision':'99'},404)
        assert json.loads(subprocess.run([*args,'export',uri,'1'],capture_output=True,check=True).stdout)==old['resource']
        assert [json.loads(row)['revision'] for row in subprocess.run([*args,'history',uri],capture_output=True,check=True).stdout.splitlines()]==['2','1']
        assert request('list',{'namespace':'work','scope':'writing'})['resources'][0]['uri']==uri
        request('list',{'namespace':'work','scope':'invalid'},400)

        # Folder mutations are shared with the CLI and retain native metadata precision.
        folder_uri='fyodor://writing/projects/'+str(uuid.uuid4())
        subfolder_uri='fyodor://writing/projects/'+str(uuid.uuid4())
        for project_uri in (folder_uri,subfolder_uri):
            request('put',{'namespace':'folders','expected_revision':'0','resource':dict(schema=1,uri=project_uri,title='Folder',content='',metadata={},provenance={})})
        folder_doc={**package,'uri':'fyodor://writing/documents/'+str(uuid.uuid4())}
        request('put',{'namespace':'folders','expected_revision':'0','resource':folder_doc})
        move={'namespace':'folders','uri':folder_doc['uri'],'expected_revision':'1','folder':folder_uri}
        request('move',{**move,'expected_revision':'0'},400)
        request('move',{**move,'folder':uri},400)
        assert request('move',move)['revision']=='2'
        request('move',move,409)
        request('delete',{'namespace':'folders','uri':folder_uri,'expected_revision':'1'},409)
        assert [item['uri'] for item in request('list',{'namespace':'folders','folder':folder_uri})['resources']]==[folder_doc['uri']]
        assert len(request('list',{'namespace':'folders','scope':'projects'})['resources'])==2
        folder_args=[cli,'--store',database,'--namespace','folders','resource']
        assert json.loads(subprocess.run([*folder_args,'folder',folder_uri],capture_output=True,check=True).stdout)['uri']==folder_doc['uri']
        subprocess.run([*folder_args,'move',subfolder_uri,'1',folder_uri],capture_output=True,check=True)
        request('move',{'namespace':'folders','uri':folder_uri,'expected_revision':'1','folder':subfolder_uri},409)
        exported_folder_doc=json.loads(subprocess.run([*folder_args,'export',folder_doc['uri']],capture_output=True,check=True).stdout)
        assert exported_folder_doc['metadata']=={**folder_doc['metadata'],'writing_parent':folder_uri}
        old_folder_doc=request('get',{'namespace':'folders','uri':folder_doc['uri'],'revision':'1'})['resource']
        assert old_folder_doc==folder_doc
        subprocess.run([*folder_args,'move',folder_doc['uri'],'2','root'],capture_output=True,check=True)
        assert request('get',{'namespace':'folders','uri':folder_doc['uri']})['resource']==folder_doc

        assert request('list',{'namespace':'other'})['resources']==[]
        def update(_):
            try:
                request('put',{'namespace':'work','expected_revision':'2','resource':package})
                return 'success'
            except AssertionError as error:
                assert error.args[0][0]==409, error
                return 'conflict'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(update,range(2)))==['conflict','success']
        if model:
            context_uri='fyodor://writing/notes/'+str(uuid.uuid4())
            principal=str(uuid.uuid4())
            context_package=dict(schema=1,uri=context_uri,title='Context source',content='abc',metadata={},provenance={})
            request('put',{'namespace':'work','expected_revision':'0','resource':context_package})
            model_id=request('/model/load',{'path':model})['model']['id']
            common={'namespace':'work','principal':principal}
            generation={**common,'resources':[context_uri],'byte_budget':3,'max_tokens':2,'model_id':model_id,'prompt':'a'}
            request('/context/generate',generation,401,authorized=False)
            request('/context/generate',generation,403)
            request('/context/generate',{**generation,'byte_budget':0},403)
            grant={**common,'uri':context_uri}
            assert request('/context/search',{**common,'query':'abc'})['resources']==[]
            assert request('/context/permissions',grant)['permissions']==0
            request('/context/permissions',{**grant,'permissions':1})
            assert request('/context/search',{**common,'query':'abc'})['resources']==[]
            request('/context/generate',{**generation,'layers':['retrieved']},403)
            request('/context/permissions',{**grant,'permissions':5})
            assert request('/context/search',{**common,'query':'ABC'})['resources'][0]['uri']==context_uri
            assert request('/context/search',{**common,'query':'%'})['resources']==[]
            assert request('/context/search',{**common,'principal':str(uuid.uuid4()),'query':'abc'})['resources']==[]
            request('/context/generate',{**generation,'resources':[context_uri]*65},400)
            request('/context/generate',{**generation,'resources':[context_uri]*2},400)
            for invalid_layers in ([], ['explicit','global'], ['unknown'], [1], None):
                request('/context/generate',{**generation,'layers':invalid_layers},400)
            completed=request('/context/generate',{**generation,'layers':['retrieved']})
            assert completed['context_bytes']==3 and completed['generated_tokens']>0
            saved={**common,'receipt_id':completed['receipt_id']}
            page=request('/context/receipts',common)
            assert [r['id'] for r in page['receipts']]==[completed['receipt_id']]
            assert request('/context/receipts',{**common,'after':completed['receipt_id']})['receipts']==[]
            assert request('/context/receipts',{**common,'principal':str(uuid.uuid4())})['receipts']==[]
            listed=subprocess.run([cli,'--store',database,'--namespace','work','context','receipts',principal],capture_output=True,check=True)
            assert json.loads(listed.stdout)['id']==completed['receipt_id']
            executed=request('/context/receipt',saved)
            assert executed['prompt']=='abc\n\na' and executed['output']==completed['text']
            sources=json.loads(executed['sources_json'])
            assert context_uri in executed['sources_json'] and sources[0]['revision']==1 and sources[0]['layer']=='retrieved'
            explained=subprocess.run([cli,'--store',database,'--namespace','work','context','explain',principal,'3','workspace='+context_uri],capture_output=True,check=True)
            assert json.loads(explained.stdout)['sources'][0]['layer']=='workspace'
            request('/context/receipt',{**saved,'principal':str(uuid.uuid4())},403)
            cli_receipt=subprocess.run([cli,'--store',database,'--namespace','work','context','receipt',principal,completed['receipt_id']],capture_output=True,check=True)
            assert json.loads(cli_receipt.stdout)['prompt']==executed['prompt']
            writing={**generation,'resources':[],'target':context_uri,'expected_revision':'1','mode':'continue'}
            request('/context/writing',{**writing,'expected_revision':'2'},409)
            request('/context/writing',{**writing,'mode':'unknown'},400)
            request('/context/writing',{**writing,'resources':[context_uri]},400)
            for mode,prefix in (('generate','Write:'),('rewrite','Rewrite:'),('continue','Continue:')):
                preview=request('/context/writing',{**writing,'mode':mode})
                receipt=request('/context/receipt',{**common,'receipt_id':preview['receipt_id']})
                assert receipt['prompt']=='abc\n\n'+prefix+'\na'
                assert request('get',{'namespace':'work','uri':context_uri})['revision']=='1'
            preview=json.loads(subprocess.run([cli,'--store',database,'--namespace','work','context','write',principal,model,context_uri,'1','continue','3','2','a'],capture_output=True,check=True).stdout)
            assert request('/context/receipt',{**common,'receipt_id':preview['receipt_id']})['prompt']=='abc\n\nContinue:\na'
            accepted={'namespace':'work','uri':context_uri,'expected_revision':'1','principal':principal,'receipt_id':preview['receipt_id'],'title':'Accepted','content':'abc'+preview['text']}
            request('update',{**accepted,'principal':str(uuid.uuid4())},403)
            request('update',{**accepted,'expected_revision':'2'},409)
            assert request('update',accepted)['revision']=='2'
            persisted=request('export',{'namespace':'work','uri':context_uri})
            link=persisted['provenance']['writing_generations'][0]
            assert link=={'receipt_id':preview['receipt_id'],'source_revision':'1','saved_revision':'2','mode':'continue','text_edited':False}
            preview=request('/context/writing',{**writing,'expected_revision':'2','mode':'rewrite'})
            saved_edit=subprocess.run([cli,'--store',database,'--namespace','work','context','accept',principal,context_uri,'2',preview['receipt_id'],'Edited','user edited output'],capture_output=True,check=True)
            assert json.loads(saved_edit.stdout)['revision']==3
            persisted=request('export',{'namespace':'work','uri':context_uri})
            assert len(persisted['provenance']['writing_generations'])==2
            assert persisted['provenance']['writing_generations'][1]['text_edited'] is True
            assert request('get',{'namespace':'work','uri':context_uri,'revision':'1'})['resource']['provenance']=={}
            request('/context/permissions',{**grant,'permissions':0})
            request('/context/writing',writing,403)
            request('/context/generate',generation,403)
            request('/context/receipt',saved,403)
            assert request('/context/search',{**common,'query':'abc'})['resources']==[]
            assert request('/context/receipts',common)['receipts']==[]
            request('delete',{'namespace':'work','uri':context_uri,'expected_revision':'3'})
            # Automatic project context uses the same service through HTTP and CLI.
            provider_parent='fyodor://writing/projects/'+str(uuid.uuid4())
            provider_target='fyodor://writing/documents/'+str(uuid.uuid4())
            for ref,text,metadata in ((provider_parent,'b',{}),(provider_target,'a',{'writing_parent':provider_parent})):
                request('put',{'namespace':'provider','expected_revision':'0','resource':dict(schema=1,uri=ref,title='Context',content=text,metadata=metadata,provenance={})})
            provider_common={**common,'namespace':'provider'}
            provider_request={**writing,**provider_common,'target':provider_target,'mode':'generate'}
            request('/context/permissions',{**provider_common,'uri':provider_target,'permissions':1})
            preview=request('/context/writing',provider_request)
            assert request('/context/receipt',{**provider_common,'receipt_id':preview['receipt_id']})['prompt']=='a\n\nWrite:\na'
            request('/context/permissions',{**provider_common,'uri':provider_parent,'permissions':1})
            preview=request('/context/writing',provider_request)
            provider_receipt=request('/context/receipt',{**provider_common,'receipt_id':preview['receipt_id']})
            assert provider_receipt['prompt']=='a\nb\n\nWrite:\na'
            assert [(s['uri'],s['layer']) for s in json.loads(provider_receipt['sources_json'])]==[(provider_target,'explicit'),(provider_parent,'workspace')]
            cli_preview=json.loads(subprocess.run([cli,'--store',database,'--namespace','provider','context','write',principal,model,provider_target,'1','generate','3','2','a'],capture_output=True,check=True).stdout)
            assert request('/context/receipt',{**provider_common,'receipt_id':cli_preview['receipt_id']})['prompt']==provider_receipt['prompt']
            request('/context/permissions',{**provider_common,'uri':provider_parent,'permissions':0})
            request('/context/receipt',{**provider_common,'receipt_id':preview['receipt_id']},403)
            request('/context/writing',{**provider_request,'resources':[provider_parent],'byte_budget':0},403)
            world_uri='fyodor://explore/worlds/'+str(uuid.uuid4())
            lore_uri=world_uri+'/lore/'+str(uuid.uuid4())
            for ref,text in ((world_uri,'world'),(lore_uri,'c')):
                request('put',{'namespace':'provider','expected_revision':'0','resource':dict(schema=1,uri=ref,title='Lore',content=text,metadata={},provenance={})})
            links={'namespace':'provider','uri':provider_target,'expected_revision':'1','resources':[lore_uri]}
            request('lore',{**links,'resources':[provider_parent]},400)
            request('lore',{**links,'resources':[lore_uri,lore_uri]},400)
            request('lore',{**links,'resources':[lore_uri]*33},400)
            assert request('lore',links)['revision']=='2'
            request('lore',links,409)
            assert request('list',{'namespace':'provider','scope':'lore'})['resources'][0]['uri']==lore_uri
            request('delete',{'namespace':'provider','uri':lore_uri,'expected_revision':'1'},409)
            preview=request('/context/writing',{**provider_request,'expected_revision':'2'})
            assert request('/context/receipt',{**provider_common,'receipt_id':preview['receipt_id']})['prompt']=='a\n\nWrite:\na'
            request('/context/permissions',{**provider_common,'uri':lore_uri,'permissions':1})
            preview=request('/context/writing',{**provider_request,'expected_revision':'2'})
            linked_receipt=request('/context/receipt',{**provider_common,'receipt_id':preview['receipt_id']})
            assert linked_receipt['prompt']=='a\nc\n\nWrite:\na'
            assert json.loads(linked_receipt['sources_json'])[1]['uri']==lore_uri
            exported=json.loads(subprocess.run([cli,'--store',database,'--namespace','provider','resource','export',provider_target],capture_output=True,check=True).stdout)
            assert exported['metadata']['writing_lore']==[lore_uri]
            assert 'writing_lore' not in request('get',{'namespace':'provider','uri':provider_target,'revision':'1'})['resource']['metadata']
            request('/context/permissions',{**provider_common,'uri':lore_uri,'permissions':0})
            request('/context/receipt',{**provider_common,'receipt_id':preview['receipt_id']},403)
            unlinked=subprocess.run([cli,'--store',database,'--namespace','provider','resource','lore',provider_target,'2'],capture_output=True,check=True)
            assert json.loads(unlinked.stdout)['revision']=='3'
            assert 'writing_lore' not in request('get',{'namespace':'provider','uri':provider_target})['resource']['metadata']
            request('delete',{'namespace':'provider','uri':lore_uri,'expected_revision':'1'})


    finally:
        stop()
    process,base,token=start()
    try:
        assert request('get',{'namespace':'work','uri':uri})['revision']=='3'
        request('update',{'namespace':'work','uri':uri,'expected_revision':'3','title':'Updated title','content':'Updated body'})
        assert request('export',{'namespace':'work','uri':uri})['metadata']['large_integer']==9007199254740993
        request('delete',{'namespace':'work','uri':uri,'expected_revision':'2'},409)
        request('delete',{'namespace':'work','uri':uri,'expected_revision':'4'})
        request('get',{'namespace':'work','uri':uri},404)
        tombstone=request('history',{'namespace':'work','uri':uri})['revisions'][0]
        assert tombstone['deleted'] is True and tombstone['revision']=='5'
        assert request('get',{'namespace':'work','uri':uri,'revision':'5'})['deleted'] is True
        request('update',{'namespace':'work','uri':uri,'expected_revision':'5','title':'restored','content':'text'},404)

        assert request('list',{'namespace':'work'})['resources']==[]
    finally:
        stop()
    process,base,token=start(False)
    try:
        request('list',{'namespace':'work'},403)
    finally:
        stop()
print('Resource HTTP authentication, CLI interchange, conflict and restart checks passed')
