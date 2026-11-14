"""Execute actual request/gather statements as C17/C23; no GPU claims."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'host.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite host evidence')
kernel = root / 'optcpp/graph/batch_request.cu'
body = kernel.read_text().replace('extern "C" __global__ ', '')
# C has a distinct struct tag namespace; no other kernel logic is changed.
body = body.replace('nya_graph_tokens tokens', 'struct nya_graph_tokens tokens')
source = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static struct { unsigned x; } blockIdx, threadIdx;
static float nya_weight(const unsigned char *w, unsigned long long i, unsigned type) {
    float value;
    assert(type == 0);
    memcpy(&value,w+i*sizeof(float),sizeof(value));
    return value;
}
KERNEL
int main(void) {
    static float weights[17][3840], output[4*3840+2];
    unsigned packet[7];
    const unsigned widths[]={1,31,256,257,3840};
    const unsigned positions[]={0,255,256,8191,8192,UINT32_MAX-4};
    for(unsigned t=0;t<17;++t) for(unsigned j=0;j<3840;++j)
        weights[t][j]=(float)((int)(t*3840+j)-30000)*0.125f;
    for(unsigned round=0;round<12;++round) {
        struct nya_graph_tokens tokens;
        for(unsigned q=0;q<4;++q) tokens.values[q]=(round*3+q*5)%17;
        unsigned position=positions[round%6];
        for(unsigned i=0;i<7;++i) packet[i]=0xdeadbeefU;
        /* Oversized launch must still have exactly one writer. */
        for(blockIdx.x=0;blockIdx.x<2;++blockIdx.x)
            for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
                nya_decode_batch_request(packet+1,tokens,position);
        assert(packet[0]==0xdeadbeefU && packet[6]==0xdeadbeefU);
        assert(packet[1]==tokens.values[0] && packet[2]==position);
        for(unsigned q=1;q<4;++q) assert(packet[q+2]==tokens.values[q]);
        for(unsigned wi=0;wi<5;++wi) for(unsigned count=2;count<=4;count+=2) {
            unsigned width=widths[wi], n=width*count;
            for(unsigned i=0;i<4*3840+2;++i) output[i]=123456.0f;
            for(blockIdx.x=0;blockIdx.x<(n+255)/256+1;++blockIdx.x)
                for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
                    nya_gather_graph_batch((const unsigned char *)weights,output+1,
                        width,sizeof(weights[0]),0,count,0.5f,packet+1);
            assert(output[0]==123456.0f);
            for(unsigned q=0;q<count;++q) for(unsigned j=0;j<width;++j)
                assert(output[1+q*width+j]==weights[tokens.values[q]][j]*0.5f);
            for(unsigned i=n+1;i<4*3840+2;++i) assert(output[i]==123456.0f);
        }
    }
    blockIdx.x=threadIdx.x=0;
    float untouched=42;
    for(unsigned count=0;count<8;++count) if(count!=2 && count!=4)
        nya_gather_graph_batch(NULL,&untouched,3840,0,0,count,1,NULL);
    nya_gather_graph_batch(NULL,&untouched,0,0,0,4,1,NULL);
    assert(untouched==42);
    puts("120 changing-token gather cases, request guards and invalid shapes passed");
}
'''.replace('KERNEL', body)
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = dict(os.environ, PATH=str(compiler.parent)+os.pathsep+os.environ['PATH'])
runs = []
with tempfile.TemporaryDirectory(prefix='fyodor-batch-request-') as temp:
    src = Path(temp) / 'request.c'
    src.write_text(source)
    for standard in ('c17', 'c23'):
        exe = Path(temp) / (standard+'.exe')
        options = ['-std='+standard, '-O3', '-Wall', '-Wextra', '-Werror']
        subprocess.run([str(compiler), *options, str(src), '-o', str(exe)], env=env, check=True)
        run = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
        runs.append(dict(standard=standard, options=options, output=run.stdout, exit_code=run.returncode))
result = dict(passed=True, gpu_execution=False, accepted_benchmark=False, runs=runs,
              generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (kernel, Path(__file__).resolve())})
destination.write_text(json.dumps(result, indent=2))
print('C17/C23 request/indexing checks passed; GPU replay remains untested')
