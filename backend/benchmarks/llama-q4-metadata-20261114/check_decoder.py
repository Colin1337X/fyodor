"""Compile the actual adapted metadata statements as C17/C23; no GPU work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'decoder.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite decoder evidence')
kernel = root / 'optcpp/matvec/llama_q4_metadata.cu'
text = kernel.read_text()
start = text.index('            unsigned short aux[2];')
end = text.index('            #pragma unroll', start)
decoder = text[start:end]
source = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static void check(const unsigned char *s) {
    unsigned short scales[6];
    for (unsigned i=0;i<6;++i) scales[i]=(unsigned short)(s[i*2]|(s[i*2+1]<<8));
    for (unsigned j=0;j<4;++j) {
DECODER
        for (unsigned k=0;k<2;++k) {
            unsigned g=j*2+k;
            unsigned sc=g<4 ? s[g]&63U : (s[g+4]&15U)|((s[g-4]>>6)<<4);
            unsigned mn=g<4 ? s[g+4]&63U : ((unsigned)s[g+4]>>4)|(((unsigned)s[g]>>6)<<4);
            assert(((aux[0]>>(k*8))&255U)==sc);
            assert(((aux[1]>>(k*8))&255U)==mn);
        }
    }
}
int main(void) {
    unsigned char s[12];
    for(unsigned a=0;a<256;++a) for(unsigned b=0;b<256;++b) {
        memset(s,(int)a,8); memset(s+8,(int)b,4); check(s);
    }
    uint32_t state=713;
    for(unsigned n=0;n<10000;++n) {
        for(unsigned i=0;i<12;++i) {
            state=state*1664525U+1013904223U; s[i]=(unsigned char)(state>>24);
        }
        check(s);
    }
    puts("65536 exhaustive byte pairs and 10000 independent blocks passed");
}
'''.replace('DECODER', decoder)
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = dict(os.environ, PATH=str(compiler.parent)+os.pathsep+os.environ['PATH'])
runs = []
with tempfile.TemporaryDirectory(prefix='fyodor-q4-decoder-') as temp:
    src = Path(temp) / 'decoder.c'; src.write_text(source)
    for standard in ('c17', 'c23'):
        exe = Path(temp) / (standard+'.exe')
        options = ['-std='+standard, '-O3', '-Wall', '-Wextra', '-Werror']
        build = subprocess.run([str(compiler), *options, str(src), '-o', str(exe)], env=env, capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stdout + build.stderr)
        run = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
        runs.append(dict(standard=standard, options=options, output=run.stdout, exit_code=run.returncode))
result = dict(passed=True, gpu_execution=False, accepted_benchmark=False, runs=runs,
              source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (kernel, Path(__file__), compiler)},
              limitation='Actual metadata statements tested on host only; CUDA execution remains unverified.')
with destination.open('x') as f: json.dump(result, f, indent=2)
print(json.dumps(runs, indent=2))
