"""Build an isolated greedy early-rejection experiment; production is unchanged."""
import difflib,hashlib,json,os,re,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
build_name=sys.argv[1] if len(sys.argv)>1 else 'build-cuda'
assert build_name in ('build-cuda','build-cuda-c23')
build=root/build_name;suffix='-c23' if build_name.endswith('c23') else ''
if (out/f'probe-build{suffix}.json').exists():raise SystemExit('Refusing to overwrite build evidence')
compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
production=(root/'backend/core/llm_cpu.c').read_text()
source=production
def replace_once(before,after):
 global source
 assert source.count(before)==1,before
 source=source.replace(before,after,1)
replace_once('if (request->soft_token_spans != 0) {','if (1) { /* Standalone probe: matched batched prefill. */')
replace_once('int rejected = 0;', '''int rejected = 0, known_rejection = 0;
        const char *early_setting = getenv("NYA_SPECULATIVE_EARLY_REJECT");
        int early_enabled = early_setting && !strcmp(early_setting,"1");''')
replace_once('++out->draft_tokens;', '''++out->draft_tokens;
            /* The first target distribution is already available. Greedy
               sampling neither mutates logits nor advances RNG state. */
            if (early_enabled && proposed == 0 && r->temperature == 0.0f) {
                uint32_t best;
                if (nya_llm_sample(target_state.logits, vocabulary, 0.0f,
                    r->top_p, r->top_k, &rng, NULL, &best) != 0) goto invalid;
                if (token != best) {
                    known_rejection = 1;
                    ++proposed;
                    break;
                }
            }''')
replace_once('int batched=p_bank && nya_compute_plan_can_verify(target_state.device_plan);',
             'int batched=!known_rejection && p_bank && nya_compute_plan_can_verify(target_state.device_plan);')
path=build/'mtp-early-reject-core.c';obj=path.with_suffix('.o');exe=build/'mtp-early-reject.exe'
path.write_text(source)
flags=(build/'CMakeFiles/nya-engine.dir/flags.make').read_text()
defines=re.search(r'^C_DEFINES = (.*)$',flags,re.M)[1].split()
cflags=re.search(r'^C_FLAGS = (.*)$',flags,re.M)[1].split()
commands=[[str(compiler),*defines,*cflags,'-I',str(root/'backend/core'),'@'+str(build/'CMakeFiles/nya-engine.dir/includes_C.rsp'),'-c',str(path),'-o',str(obj)],
 [str(compiler),'-std=c23' if suffix else '-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(out.parent/'mtp-cost-20261113/model.c'),str(obj),str(build/'libnya-engine.a'),'-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]]
architecture=(root/'backend/tests/test_architectures.c').read_text()
architecture=architecture.replace('"../core/llm_cpu.c"','"mtp-early-reject-core.c"').replace('"../core/multimodal.h"','"multimodal.h"')
architecture_path=build/'mtp-early-reject-architectures.c'
architecture_path.write_text(architecture)
generation=(root/'backend/tests/test_generation.c').read_text()
needle='response.draft_tokens < 4) return fail("greedy draft rollback failed", error);'
assert generation.count(needle)==1
generation=generation.replace(needle,needle+'''
                const char *early_check=getenv("NYA_SPECULATIVE_EARLY_REJECT");
                if(early_check && !strcmp(early_check,"1") &&
                   (response.draft_tokens!=response.generated_tokens ||
                    response.target_steps!=response.prompt_tokens+response.generated_tokens-1))
                    return fail("early rejection did not eliminate known-wrong work",error);''')
generation_path=build/'mtp-early-reject-generation.c';generation_path.write_text(generation)
for test,inputs in [('generation',[str(generation_path),str(obj)]),('architectures',[str(architecture_path)])]:
 commands.append([str(compiler),*cflags,'-I','backend/include','-I','backend/core','-I','backend/tests',*inputs,str(build/'libnya-engine.a'),'-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(build/f'mtp-early-reject-{test}.exe')])
for command in commands:subprocess.run(command,env=env,check=True)
(out/f'candidate{suffix}.patch').write_text(''.join(difflib.unified_diff(production.splitlines(True),source.splitlines(True),fromfile='production/llm_cpu.c',tofile='probe/llm_cpu.c')))
manifest={'production_source_sha256':hashlib.sha256(production.encode()).hexdigest(),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'library_sha256':hashlib.sha256((build/'libnya-engine.a').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'commands':commands}
manifest['test_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (generation_path,architecture_path)}
(out/f'probe-build{suffix}.json').write_text(json.dumps(manifest,indent=2)+'\n')
