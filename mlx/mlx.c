#include "compute_backend.h"
#include "vendor.h"
#include "abi.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* MLX C's default error callback exits the process. Install a nonfatal handler
   once per loaded module, before making any MLX call. A reference count keeps
   that module alive across independent model contexts. The atomic lock guards
   only module load/unload, never matrix execution or GPU synchronization. */
static struct { void *library; char *path; size_t references; nya_mlx_api api; } module;
static atomic_flag module_guard=ATOMIC_FLAG_INIT;
static _Thread_local int mlx_fault;
static void lock_module(void) { while (atomic_flag_test_and_set_explicit(&module_guard,memory_order_acquire)) {} }
static void unlock_module(void) { atomic_flag_clear_explicit(&module_guard,memory_order_release); }
static void error_handler(const char *message, void *unused)
{
    (void)unused; mlx_fault=1;
    fprintf(stderr,"MLX operation failed: %s\n",message ? message : "unspecified library error");
}
static int acquire(nya_mlx_api *api)
{
    const char *path=getenv("NYA_MLX_LIBRARY"); if (!path) path="";
    lock_module();
    if (module.references) {
        if (strcmp(module.path,path) || module.references==SIZE_MAX) { unlock_module(); return -1; }
        ++module.references; *api=module.api; unlock_module(); return 0;
    }
#ifdef _WIN32
    const char *name="mlxc.dll";
#elif defined(__APPLE__)
    const char *name="libmlxc.dylib";
#else
    const char *name="libmlxc.so";
#endif
    module.path=malloc(strlen(path)+1);
    if (!module.path) goto failed;
    strcpy(module.path,path);
    module.library=nya_vendor_open(path,name);
#define MLX(field,name) if (nya_vendor_symbol(module.library,name,&module.api.field,sizeof(module.api.field))) goto failed
    MLX(error_handler,"mlx_set_error_handler"); MLX(device_new,"mlx_device_new_type");
    MLX(device_available,"mlx_device_is_available"); MLX(device_free,"mlx_device_free");
    /* Ordinary MLX streams are thread-affine. Server requests may migrate
       between workers; this stream API permits serialized use on any thread. */
    MLX(stream_new,"mlx_stream_new_thread_unsafe"); MLX(stream_free,"mlx_stream_free"); MLX(synchronize,"mlx_synchronize");
    MLX(array_data,"mlx_array_new_data"); MLX(array_free,"mlx_array_free"); MLX(eval,"mlx_array_eval");
    MLX(data_f32,"mlx_array_data_float32"); MLX(transpose,"mlx_transpose"); MLX(contiguous,"mlx_contiguous");
    MLX(copy,"mlx_copy"); MLX(matmul,"mlx_matmul"); MLX(memory_limit,"mlx_get_memory_limit");
    MLX(active_memory,"mlx_get_active_memory"); MLX(cache_memory,"mlx_get_cache_memory");
#undef MLX
    module.api.error_handler(error_handler,NULL,NULL);
    module.references=1; *api=module.api; unlock_module(); return 0;
failed:
    nya_vendor_close(module.library); free(module.path); memset(&module,0,sizeof(module));
    unlock_module(); return -1;
}
static void relinquish(void)
{
    lock_module();
    if (--module.references==0) {
        nya_vendor_close(module.library); free(module.path); memset(&module,0,sizeof(module));
    }
    unlock_module();
}
typedef struct nya_mlx {
    nya_mlx_api api;
    nya_mlx_device gpu, cpu;
    nya_mlx_stream stream, host_stream;
    nya_vendor_cache cache;
    nya_compute_stats stats;
    int acquired, failed;
} nya_mlx;
static int checked(nya_mlx *c, int status, const char *operation)
{
    if (!status && !mlx_fault) return 0;
    fprintf(stderr,"MLX %s failed: status %d; reverting to CPU execution\n",operation,status);
    c->failed=1; return -1;
}
static void free_array(nya_mlx *c, nya_mlx_array value)
{ if (value.ctx) (void)checked(c,c->api.array_free(value),"release array"); }
static void release_weight(void *state, void *value)
{ nya_mlx_array array={value}; free_array(state,array); }
static void destroy(void *state)
{
    nya_mlx *c=state; if (!c) return;
    if (c->acquired) {
        mlx_fault=0;
        if (c->stream.ctx) (void)checked(c,c->api.synchronize(c->stream),"drain GPU stream");
        if (c->host_stream.ctx) (void)checked(c,c->api.synchronize(c->host_stream),"drain CPU stream");
        nya_vendor_clear(&c->cache,release_weight,c);
        if (c->stream.ctx) (void)checked(c,c->api.stream_free(c->stream),"free GPU stream");
        if (c->host_stream.ctx) (void)checked(c,c->api.stream_free(c->host_stream),"free CPU stream");
        if (c->gpu.ctx) (void)checked(c,c->api.device_free(c->gpu),"free GPU device");
        if (c->cpu.ctx) (void)checked(c,c->api.device_free(c->cpu),"free CPU device");
        relinquish();
    }
    free(c);
}
static nya_vendor_weight *weight(nya_mlx *c, const void *source, size_t rows, size_t cols,
    unsigned type, size_t bytes, size_t scratch)
{
    nya_vendor_weight *w=nya_vendor_find(&c->cache,source,rows,cols,type);
    if (w) return w;
    if (nya_vendor_reserve(&c->cache,bytes,scratch,release_weight,c)) {
        (void)checked(c,-1,"weight budget"); return NULL;
    }
    w=calloc(1,sizeof(*w));
    float *expanded=type ? malloc(bytes) : NULL;
    if (!w || (type && !expanded)) { free(w); free(expanded); return NULL; }
    if (type) nya_vendor_decode(expanded,source,rows*cols,type);
    int shape[]={(int)rows,(int)cols};
    nya_mlx_array a=c->api.array_data(type ? expanded : source,shape,2,NYA_MLX_F32), t={0}, packed={0};
    /* new_data copies its source. The transient F32 decode buffer can be
       released immediately; the cached MLX array retains independent storage. */
    free(expanded);
    if (!a.ctx || checked(c,0,"create weight") ||
        checked(c,c->api.transpose(&t,a,c->stream),"transpose weight") ||
        checked(c,c->api.contiguous(&packed,t,false,c->stream),"pack weight") ||
        checked(c,c->api.eval(packed),"evaluate weight")) goto failed;
    ++c->stats.uploads; ++c->stats.synchronizations;
    if (checked(c,c->api.synchronize(c->stream),"finish weight")) goto failed;
    free_array(c,t); free_array(c,a);
    if (c->failed) { free_array(c,packed); free(w); return NULL; }
    w->source=source; w->rows=rows; w->columns=cols; w->bytes=bytes; w->type=type; w->value=packed.ctx;
    nya_vendor_insert(&c->cache,w); return w;
failed:
    (void)checked(c,c->api.synchronize(c->stream),"drain failed weight");
    c->failed=1;
    free_array(c,packed); free_array(c,t); free_array(c,a); free(w); return NULL;
}
static int matmul(void *state, const void *source, size_t rows, size_t cols,
    unsigned type, const float *x, float *y, size_t batch)
{
    nya_mlx *c=state; size_t wb,xb,yb;
    if (!c || c->failed || !source || !x || !y || nya_vendor_matrix(rows,cols,batch,type,&wb,&xb,&yb)) return -1;
    mlx_fault=0;
    /* Account for both result arrays: the explicit CPU copy prevents treating
       a GPU-only data pointer as host memory on non-unified-memory MLX builds. */
    if (yb>(SIZE_MAX-xb)/2 || nya_vendor_reserve(&c->cache,0,xb+yb*2,release_weight,c)) return checked(c,-1,"scratch budget");
    nya_vendor_weight *w=weight(c,source,rows,cols,type,wb,xb+yb*2);
    if (!w || c->failed) return checked(c,-1,"prepare weight");
    nya_mlx_array matrix={w->value}, input={0}, result={0}, host={0}; int status=-1;
    int shape[]={(int)batch,(int)cols};
    input=c->api.array_data(x,shape,2,NYA_MLX_F32);
    if (!input.ctx || checked(c,0,"create input")) goto done;
    ++c->stats.uploads;
    if (checked(c,c->api.matmul(&result,input,matrix,c->stream),"matmul")) goto done;
    ++c->stats.external_matmul_calls;
    if (checked(c,c->api.copy(&host,result,c->host_stream),"copy result to CPU") ||
        checked(c,c->api.eval(host),"evaluate result") ||
        checked(c,c->api.synchronize(c->host_stream),"finish result")) goto done;
    ++c->stats.downloads; ++c->stats.synchronizations;
    const float *data=c->api.data_f32(host);
    if (!data || checked(c,0,"read result")) goto done;
    memcpy(y,data,yb); status=0;
done:
    if (status) {
        (void)checked(c,c->api.synchronize(c->stream),"drain failed matrix");
        (void)checked(c,c->api.synchronize(c->host_stream),"drain failed result");
        c->failed=1;
    }
    free_array(c,host); free_array(c,result); free_array(c,input);
    return c->failed ? -1 : status;
}
static int matvec(void *c, const void *w, size_t rows, size_t cols, unsigned type, const float *x, float *y)
{ return matmul(c,w,rows,cols,type,x,y,1); }
static int active(const void *state) { const nya_mlx *c=state; return c && !c->failed; }
static void stats(const void *state, nya_compute_stats *out)
{
    const nya_mlx *c=state; *out=c->stats; out->weights_bytes=c->cache.bytes; out->external_matmul_kind=4;
}
static void *create(void)
{
    nya_mlx *c=calloc(1,sizeof(*c)); if (!c) return NULL;
    if (acquire(&c->api)) goto unavailable;
    c->acquired=1; mlx_fault=0;
    c->gpu=c->api.device_new(NYA_MLX_GPU,0); c->cpu=c->api.device_new(NYA_MLX_CPU,0);
    bool available=false;
    if (!c->gpu.ctx || !c->cpu.ctx || checked(c,c->api.device_available(&available,c->gpu),"query GPU") || !available) goto unavailable;
    c->stream=c->api.stream_new(c->gpu); c->host_stream=c->api.stream_new(c->cpu);
    if (!c->stream.ctx || !c->host_stream.ctx || checked(c,0,"create streams")) goto unavailable;
    size_t limit=0,used=0,cached=0;
    if (checked(c,c->api.memory_limit(&limit),"query memory limit") ||
        checked(c,c->api.active_memory(&used),"query active memory") ||
        checked(c,c->api.cache_memory(&cached),"query allocator cache") ||
        used>limit || cached>limit-used || !(c->cache.limit=nya_vendor_budget(limit-used-cached,"NYA_MLX_CACHE_MIB"))) goto unavailable;
    static const float probe_w[]={1,2,3,4}, probe_x[]={1,1}; float probe_y[2];
    if (matvec(c,probe_w,2,2,0,probe_x,probe_y) || probe_y[0]!=3 || probe_y[1]!=7) goto unavailable;
    memset(&c->stats,0,sizeof(c->stats)); return c;
unavailable:
    fprintf(stderr,"MLX unavailable: verify the MLX C library, GPU and memory budget; CPU remains available\n");
    destroy(c); return NULL;
}
const nya_backend_interface *nya_mlx_backend(void)
{
    static const nya_backend_interface api={.type=NYA_BACKEND_MLX,.name="mlx",
        .capabilities=NYA_COMPUTE_MATVEC|NYA_COMPUTE_QUANTIZED|NYA_COMPUTE_MATMUL,
        .create=create,.destroy=destroy,.active=active,.matvec=matvec,.matmul=matmul,.context_stats=stats};
    return &api;
}
