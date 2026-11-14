/* Exercise the actual generated C helper's admission and failure contract. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
typedef uintptr_t CUdeviceptr;
typedef struct {unsigned type;} nya_llm_tensor;
typedef struct {
    int qkv_input_reuse, reuse_interleaved_input, available, vendor;
    unsigned batch;
    unsigned long long qkv_reuse_hits;
    void *cuda;
    CUdeviceptr norm, blas_storage;
} nya_cuda_plan;
static int injected, observed, bad_input;
static int plan_interleaved_kernel(const nya_cuda_plan *p,unsigned type,size_t columns)
{ return p->available && columns==3840 && (type==12 || type==14); }
static size_t nya_blas_bytes(void *cuda,size_t rows,size_t columns,unsigned batch)
{ (void)rows;(void)columns;(void)batch;return (size_t)(*(const int *)cuda); }
static int plan_matvec(nya_cuda_plan *p,CUdeviceptr out,const nya_llm_tensor *t,CUdeviceptr in,size_t rows,size_t columns)
{
    (void)t;observed=p->reuse_interleaved_input;
    if(out!=456 || in!=123 || rows!=77 || columns!=3840)bad_input=1;
    return injected;
}
#include "helper.inc"
int main(void)
{
    unsigned cases=0;const unsigned batches[]={1,2,4,8},types[]={0,12,14};
    for(int enabled=0;enabled<2;++enabled)for(unsigned b=0;b<4;++b)
    for(unsigned t=0;t<3;++t)for(int available=0;available<2;++available)
    for(int vendor=0;vendor<2;++vendor)for(int ready_before=0;ready_before<2;++ready_before)
    for(int fail=0;fail<2;++fail){
        nya_cuda_plan p={0};p.qkv_input_reuse=enabled;p.batch=batches[b];p.available=available;
        p.vendor=vendor;p.cuda=&p.vendor;p.blas_storage=1;p.norm=123;
        nya_llm_tensor tensor={types[t]};int ready=ready_before;
        injected=fail?-1:0;observed=-1;bad_input=0;
        int eligible=enabled && (batches[b]==2 || batches[b]==4) && available && types[t]!=0 && !vendor;
        int result=plan_matvec_norm_reuse(&p,456,&tensor,77,3840,&ready);
        if(result!=injected || observed!=(eligible && ready_before) || p.reuse_interleaved_input || bad_input ||
           ready!=(ready_before || (eligible && !fail)) || p.qkv_reuse_hits!=(unsigned)(eligible && ready_before))return 1;
        ++cases;
    }
    /* Each new layer starts empty; one packing operation serves Q/K/V. */
    nya_cuda_plan p={0};p.qkv_input_reuse=1;p.batch=4;p.available=1;p.cuda=&p.vendor;p.norm=123;
    nya_llm_tensor tensor={12};injected=0;
    for(int layer=0;layer<2;++layer){
        int ready=0;
        for(int projection=0;projection<3;++projection){
            if(plan_matvec_norm_reuse(&p,456,&tensor,77,3840,&ready) || observed!=(projection!=0) || p.reuse_interleaved_input)return 2;
        }
    }
    if(p.qkv_reuse_hits!=4)return 3;
    printf("%u admission/failure cases and two independent QKV sequences passed\n",cases);
    return 0;
}
