"""Profile assistant host/assisted stage wall times in an isolated core object."""
import difflib,hashlib,json,os,re,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent;build=root/'build-cuda'
if (out/'profile.json').exists():raise SystemExit('Refusing to overwrite evidence')
production=(root/'backend/core/llm_cpu.c').read_text()
start=production.index('static int nya_llm_forward_impl(')
end=production.index('static int nya_llm_device_prefill(',start)
body=production[start:end]
def replace(old,new):
 global body
 assert body.count(old)==1,old
 body=body.replace(old,new,1)
replace('    size_t d = c->embedding_length, ple = c->per_layer_embedding_length;',
 '''    double stage_ms[9]={0}, stage_time=c->is_assistant?nya_runtime_monotonic_seconds():0;
#define MTP_STAGE(k) do {if(c->is_assistant) {double now=nya_runtime_monotonic_seconds();stage_ms[k]+=(now-stage_time)*1000;stage_time=now;}} while(0)
    size_t d = c->embedding_length, ple = c->per_layer_embedding_length;''')
markers=[
 ('    for (size_t layer = s->layer_begin; layer < layer_end; ++layer) {',0),
 ('        nya_compute_attention attention = ',1),
 ('        nya_llm_matvec(c->compute, s->residual, w->attention_output,',2),
 ('        nya_llm_rms_norm(s->normalized, s->activation, w->feed_forward_norm,',3),
 ('    if (c->is_assistant) nya_llm_matvec(c->compute, s->mtp_hidden,',5),
 ('    if (c->final_logit_softcap > 0.0f) {',6),
 ('    /* Reject non-finite model results before the checked suppression list is',7),
]
for marker,index in markers:
 indent=marker[:len(marker)-len(marker.lstrip())]
 replace(marker,indent+f'MTP_STAGE({index});\n'+marker)
replace('''        }
    }
    /* Prompt positions only need caches.''','''        }
        MTP_STAGE(4);
    }
    /* Prompt positions only need caches.''')
replace('''    return 0;
}

''','''    MTP_STAGE(8);
    if(c->is_assistant) fprintf(stderr,"MTP stages: setup=%.6f query=%.6f attention=%.6f attn_output=%.6f ffn=%.6f head=%.6f recurrence=%.6f softcap=%.6f validation=%.6f\\n",stage_ms[0],stage_ms[1],stage_ms[2],stage_ms[3],stage_ms[4],stage_ms[5],stage_ms[6],stage_ms[7],stage_ms[8]);
#undef MTP_STAGE
    return 0;
}

''')
source=production[:start]+body+production[end:]
needle='if (request->soft_token_spans != 0) {'
assert source.count(needle)==1
source=source.replace(needle,'if (1) { /* Diagnostic matched B512 prompt ingestion. */',1)
core=build/'mtp-stages-20261114-core.c';obj=core.with_suffix('.o');exe=build/'mtp-stages-20261114.exe'
core.write_text(source)
(out/'candidate.patch').write_text(''.join(difflib.unified_diff(production.splitlines(True),source.splitlines(True),fromfile='production/llm_cpu.c',tofile='diagnostic/llm_cpu.c')))
model=out/'model.c';model.write_bytes((out.parent/'mtp-cost-20261113/model.c').read_bytes())
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
flags=(build/'CMakeFiles/nya-engine.dir/flags.make').read_text()
defines=re.search(r'^C_DEFINES = (.*)$',flags,re.M)[1].split();cflags=re.search(r'^C_FLAGS = (.*)$',flags,re.M)[1].split()
commands=[[str(compiler),*defines,*cflags,'-I',str(root/'backend/core'),'@'+str(build/'CMakeFiles/nya-engine.dir/includes_C.rsp'),'-c',str(core),'-o',str(obj)],
 [str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(model),str(obj),str(build/'libnya-engine.a'),'-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]]
for command in commands:subprocess.run(command,env=env,check=True)
settings=json.loads((out.parent/'batch-logits-20261114/runtime-on/timing.json').read_text())['environment']
settings['NYA_CUDA_PROFILE']='0';env.update(settings)
meta={'accepted_llama_score':False,'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'commands':commands,'environment':settings,
      'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (core,obj,exe,model,build/'libnya-engine.a')},'runs':[]}
for context in ('short','long'):
 with (out/(context+'.stdout')).open('w') as stdout,(out/(context+'.stderr')).open('w') as stderr:
  result=subprocess.run([str(exe),context],env=env,stdout=stdout,stderr=stderr)
 meta['runs'].append({'context':context,'exit_code':result.returncode})
 (out/'profile.json').write_text(json.dumps(meta,indent=2)+'\n')
 assert result.returncode==0
 rows=[json.loads(line) for line in (out/(context+'.stdout')).read_text().splitlines() if line.startswith('{"round":')]
 assert len(rows)==12 and all(r['parity'] and r['generated']==32 for r in rows)
 print(context,'parity passed',flush=True)
subprocess.run([sys.executable,str(out/'summarize.py')],check=True)
