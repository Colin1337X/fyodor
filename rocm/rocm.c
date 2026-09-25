#include "compute_backend.h"
#include "vendor.h"
#include "abi.h"
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct nya_rocm {
    nya_rocm_api api;
    void *hip_library, *blas_library, *input, *output;
    hipStream_t stream; rocblas_handle blas;
    size_t input_bytes, output_bytes;
    int device, failed;
    nya_vendor_cache cache;
    nya_compute_stats stats;
} nya_rocm;

static int checked(nya_rocm *c, long long status, const char *operation)
{
    if (!status) return 0;
    fprintf(stderr,"ROCm %s failed: status %lld; reverting to CPU execution\n",operation,status);
    c->failed=1; return -1;
}
static void release_weight(void *state, void *value)
{
    nya_rocm *c=state;
    if (value) (void)checked(c,c->api.deallocate(value),"free weight");
}
static int enter(nya_rocm *c, int *previous)
{
    if (checked(c,c->api.get_device(previous),"get device")) return -1;
    return checked(c,c->api.set_device(c->device),"select device");
}
static void destroy(void *state)
{
    nya_rocm *c=state;
    if (!c) return;
    int previous=0, selected=0;
    if (c->stream) {
        selected=!enter(c,&previous);
        {
            /* Even if selecting a lost device fails, attempt each owned
               resource release. A device-selection error must not suppress
               cleanup of independent BLAS/stream and host cache ownership. */
            (void)checked(c,c->api.synchronize(c->stream),"synchronize during cleanup");
            nya_vendor_clear(&c->cache,release_weight,c);
            if (c->input) (void)checked(c,c->api.deallocate(c->input),"free input");
            if (c->output) (void)checked(c,c->api.deallocate(c->output),"free output");
            if (c->blas) (void)checked(c,c->api.blas_destroy(c->blas),"destroy BLAS");
            (void)checked(c,c->api.stream_destroy(c->stream),"destroy stream");
            if (selected) (void)checked(c,c->api.set_device(previous),"restore device");
        }
    }
    nya_vendor_close(c->blas_library); nya_vendor_close(c->hip_library); free(c);
}
static int buffers(nya_rocm *c, size_t input, size_t output)
{
    if (input<=c->input_bytes && output<=c->output_bytes) return 0;
    size_t want_input=input>c->input_bytes ? input : c->input_bytes;
    size_t want_output=output>c->output_bytes ? output : c->output_bytes;
    if (want_input>SIZE_MAX-want_output ||
        nya_vendor_reserve(&c->cache,0,want_input+want_output,release_weight,c)) return checked(c,-1,"scratch budget");
    /* Every public matrix operation synchronizes before returning. Old scratch
       is therefore no longer in flight, and may be freed before growth. */
    if (want_input!=c->input_bytes) {
        if (c->input && checked(c,c->api.deallocate(c->input),"resize input")) return -1;
        c->input=NULL; c->input_bytes=0;
        if (checked(c,c->api.allocate(&c->input,want_input),"allocate input")) return -1;
        c->input_bytes=want_input;
    }
    if (want_output!=c->output_bytes) {
        if (c->output && checked(c,c->api.deallocate(c->output),"resize output")) return -1;
        c->output=NULL; c->output_bytes=0;
        if (checked(c,c->api.allocate(&c->output,want_output),"allocate output")) return -1;
        c->output_bytes=want_output;
    }
    return c->failed ? -1 : 0;
}
static nya_vendor_weight *weight(nya_rocm *c, const void *source, size_t rows, size_t cols, unsigned type, size_t bytes)
{
    nya_vendor_weight *w=nya_vendor_find(&c->cache,source,rows,cols,type);
    if (w) return w;
    if (nya_vendor_reserve(&c->cache,bytes,c->input_bytes+c->output_bytes,release_weight,c)) {
        (void)checked(c,-1,"weight budget"); return NULL;
    }
    w=calloc(1,sizeof(*w));
    float *expanded=type ? malloc(bytes) : NULL;
    if (!w || (type && !expanded)) {
        free(w); free(expanded); (void)checked(c,-1,"allocate host weight"); return NULL;
    }
    if (type) nya_vendor_decode(expanded,source,rows*cols,type);
    if (checked(c,c->api.allocate(&w->value,bytes),"allocate weight")) goto failure;
    if (checked(c,c->api.copy_async(w->value,type ? expanded : source,bytes,NYA_HIP_H2D,c->stream),"upload weight")) goto failure;
    ++c->stats.uploads;
    /* Decoding memory belongs to this call, so finish the async upload before
       freeing it. Weight uploads occur only on cache misses, outside steady
       repeated use when the model's expanded weights fit the cache budget. */
    ++c->stats.synchronizations;
    if (checked(c,c->api.synchronize(c->stream),"finish weight upload")) goto failure;
    free(expanded);
    w->source=source; w->rows=rows; w->columns=cols; w->type=type; w->bytes=bytes;
    nya_vendor_insert(&c->cache,w); return w;
failure:
    /* Drain a possibly submitted copy even when a later operation failed. */
    (void)checked(c,c->api.synchronize(c->stream),"drain failed upload");
    release_weight(c,w->value); free(w); free(expanded); return NULL;
}
static int matmul(void *state, const void *source, size_t rows, size_t cols,
    unsigned type, const float *x, float *y, size_t batch)
{
    nya_rocm *c=state; size_t wb,xb,yb;
    if (!c || c->failed || !source || !x || !y || nya_vendor_matrix(rows,cols,batch,type,&wb,&xb,&yb)) return -1;
    int previous, result=-1;
    if (enter(c,&previous)) return -1;
    if (buffers(c,xb,yb)) goto done;
    nya_vendor_weight *w=weight(c,source,rows,cols,type,wb);
    if (!w || c->failed) goto done;
    if (checked(c,c->api.copy_async(c->input,x,xb,NYA_HIP_H2D,c->stream),"upload input")) goto done;
    ++c->stats.uploads;
    const float alpha=1, beta=0;
    /* Stored W=[rows,cols], X=[batch,cols], Y=[batch,rows]. rocBLAS uses
       column-major matrices: transpose W, leave X/Y unchanged, preserving the
       original F32 reduction contract and supporting all batch/row tails. */
    if (checked(c,c->api.sgemm(c->blas,NYA_ROCBLAS_T,NYA_ROCBLAS_N,(int)rows,(int)batch,(int)cols,
        &alpha,w->value,(int)cols,c->input,(int)cols,&beta,c->output,(int)rows),"SGEMM")) goto done;
    ++c->stats.external_matmul_calls;
    if (checked(c,c->api.copy_async(y,c->output,yb,NYA_HIP_D2H,c->stream),"download output")) goto done;
    ++c->stats.downloads; result=0;
done:
    ++c->stats.synchronizations;
    if (checked(c,c->api.synchronize(c->stream),"finish matrix")) result=-1;
    if (checked(c,c->api.set_device(previous),"restore device")) result=-1;
    return result;
}
static int matvec(void *c, const void *w, size_t rows, size_t cols, unsigned type, const float *x, float *y)
{ return matmul(c,w,rows,cols,type,x,y,1); }
static int active(const void *state) { const nya_rocm *c=state; return c && !c->failed; }
static void stats(const void *state, nya_compute_stats *out)
{
    const nya_rocm *c=state; *out=c->stats;
    out->weights_bytes=c->cache.bytes; out->scratch_bytes=c->input_bytes+c->output_bytes;
    out->external_matmul_kind=3;
}
static void *create(void)
{
    nya_rocm *c=calloc(1,sizeof(*c)); if (!c) return NULL;
#ifdef _WIN32
    const char *hip_name="amdhip64.dll", *blas_name="rocblas.dll";
#else
    const char *hip_name="libamdhip64.so", *blas_name="librocblas.so";
#endif
    c->hip_library=nya_vendor_open(getenv("NYA_ROCM_HIP_LIBRARY"),hip_name);
    c->blas_library=nya_vendor_open(getenv("NYA_ROCM_BLAS_LIBRARY"),blas_name);
    if (!c->hip_library || !c->blas_library) goto unavailable;
#define HIP(field,name) if (nya_vendor_symbol(c->hip_library,name,&c->api.field,sizeof(c->api.field))) goto unavailable
#define BLAS(field,name) if (nya_vendor_symbol(c->blas_library,name,&c->api.field,sizeof(c->api.field))) goto unavailable
    HIP(init,"hipInit"); HIP(device_count,"hipGetDeviceCount"); HIP(get_device,"hipGetDevice"); HIP(set_device,"hipSetDevice");
    HIP(memory_info,"hipMemGetInfo"); HIP(allocate,"hipMalloc"); HIP(deallocate,"hipFree"); HIP(copy_async,"hipMemcpyAsync");
    HIP(stream_create,"hipStreamCreateWithFlags"); HIP(stream_destroy,"hipStreamDestroy"); HIP(synchronize,"hipStreamSynchronize");
    BLAS(blas_create,"rocblas_create_handle"); BLAS(blas_destroy,"rocblas_destroy_handle");
    BLAS(blas_stream,"rocblas_set_stream"); BLAS(blas_pointer_mode,"rocblas_set_pointer_mode"); BLAS(sgemm,"rocblas_sgemm");
#undef HIP
#undef BLAS
    int count=0, previous=0;
    if (checked(c,c->api.init(0),"initialize HIP") || checked(c,c->api.device_count(&count),"enumerate devices") || count<1) goto unavailable;
    const char *device=getenv("NYA_ROCM_DEVICE");
    if (device) {
        char *end; errno=0; unsigned long value=strtoul(device,&end,10);
        if (errno || device[0]<'0' || device[0]>'9' || *end || value>=(unsigned long)count) goto unavailable;
        c->device=(int)value;
    }
    if (enter(c,&previous)) goto unavailable;
    size_t available=0,total=0;
    int failed=checked(c,c->api.memory_info(&available,&total),"query memory") ||
        !(c->cache.limit=nya_vendor_budget(available,"NYA_ROCM_CACHE_MIB")) ||
        checked(c,c->api.stream_create(&c->stream,1),"create stream") ||
        checked(c,c->api.blas_create(&c->blas),"create BLAS") ||
        checked(c,c->api.blas_stream(c->blas,c->stream),"bind BLAS stream") ||
        checked(c,c->api.blas_pointer_mode(c->blas,rocblas_pointer_mode_host),"set host scalars");
    if (checked(c,c->api.set_device(previous),"restore initial device") || failed) goto unavailable;
    /* Verify real dispatch before an API provider switch can commit. A driver
       may enumerate a device even when its rocBLAS kernel package is missing. */
    static const float probe_w[]={1,2,3,4}, probe_x[]={1,1}; float probe_y[2];
    if (matvec(c,probe_w,2,2,0,probe_x,probe_y) || probe_y[0]!=3 || probe_y[1]!=7) goto unavailable;
    memset(&c->stats,0,sizeof(c->stats)); return c;
unavailable:
    fprintf(stderr,"ROCm unavailable: verify HIP, rocBLAS, device selection and memory budget; CPU remains available\n");
    destroy(c); return NULL;
}
const nya_backend_interface *nya_rocm_backend(void)
{
    static const nya_backend_interface api={.type=NYA_BACKEND_ROCM,.name="rocm",
        .capabilities=NYA_COMPUTE_MATVEC|NYA_COMPUTE_QUANTIZED|NYA_COMPUTE_MATMUL,
        .create=create,.destroy=destroy,.active=active,.matvec=matvec,.matmul=matmul,.context_stats=stats};
    return &api;
}
