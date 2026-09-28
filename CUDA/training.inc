/* Private resident training storage and matrix dispatch; no host tensor cache. */
#include "training_device_backend.h"
#include <stdatomic.h>
#include <errno.h>
#define NYA_TRAIN_BUFFERS 4096
typedef struct nya_cuda_train_buffer {
    nya_train_buffer id;
    CUdeviceptr memory;
    size_t bytes;
} nya_cuda_train_buffer;
typedef struct nya_cuda_train {
    nya_cuda_context *cuda;
    CUdeviceptr arena;
    CUfunction zero, dx[31], dw, check_finite;
    nya_train_device_stats stats;
    nya_cuda_train_buffer buffers[NYA_TRAIN_BUFFERS];
    nya_train_scope scope;
    size_t scope_bytes, scope_buffers;
    uint64_t fail_after;
    char error[160];
} nya_cuda_train;
static _Atomic uint64_t nya_train_next_id = 1;
static int nya_ct_error(nya_cuda_train *t, const char *message, int fatal)
{
    if (!t->stats.failed) snprintf(t->error,sizeof(t->error),"%s",message);
    if (fatal) t->stats.failed = 1;
    return -1;
}
/* Buffer and scope tokens share one never-reused namespace, including across
   devices. A recycled descriptor cannot make an expired handle valid again. */
static uint64_t nya_ct_id(nya_cuda_train *t)
{
    uint64_t id = atomic_load_explicit(&nya_train_next_id,memory_order_relaxed);
    do {
        if (id == UINT64_MAX) { nya_ct_error(t,"training handle space exhausted",0); return 0; }
    } while (!atomic_compare_exchange_weak_explicit(&nya_train_next_id,&id,id+1,memory_order_relaxed,memory_order_relaxed));
    return id;
}
static int nya_ct_enter(nya_cuda_train *t)
{
    if (t->stats.failed) return -1;
    if (t->cuda->push(t->cuda->context) != CUDA_SUCCESS)
        return nya_ct_error(t,"training context activation failed",1);
    t->error[0] = 0;
    return 0;
}
static int nya_ct_leave(nya_cuda_train *t, int result)
{
    CUcontext previous;
    if (t->cuda->pop(&previous) != CUDA_SUCCESS)
        return nya_ct_error(t,"training context restoration failed",1);
    return result;
}
static int nya_ct_sync(nya_cuda_train *t, CUstream stream)
{
    ++t->stats.synchronizations;
    return t->cuda->stream_sync(stream) == CUDA_SUCCESS ? 0 :
        nya_ct_error(t,"training stream failed",1);
}
static int nya_ct_launch(nya_cuda_train *t, CUfunction f, unsigned gx, unsigned gy,
    unsigned bx, unsigned by, void **args)
{
    if (t->stats.kernel_launches >= t->fail_after)
        return nya_ct_error(t,"injected training launch failure",1);
    if (t->cuda->launch(f,gx,gy,1,bx,by,1,0,t->cuda->stream,args,NULL) != CUDA_SUCCESS)
        return nya_ct_error(t,"training kernel launch failed",1);
    ++t->stats.kernel_launches;
    return 0;
}
static nya_cuda_train_buffer *nya_ct_find(nya_cuda_train *t, nya_train_buffer id)
{
    if (id) for (size_t i = 0; i < t->stats.buffers; ++i)
        if (t->buffers[i].id == id) return &t->buffers[i];
    return NULL;
}
static void nya_ct_destroy(void *context)
{
    nya_cuda_train *t = context;
    if (!t) return;
    if (t->cuda && t->arena && t->cuda->push(t->cuda->context) == CUDA_SUCCESS) {
        CUcontext previous;
        t->cuda->stream_sync(t->cuda->stream);
        t->cuda->deallocate(t->arena);
        t->cuda->pop(&previous);
    }
    nya_cuda_free(t->cuda);
    free(t);
}
static void *nya_ct_create(size_t capacity)
{
    nya_cuda_train *t = calloc(1,sizeof(*t));
    if (!t) return NULL;
    t->fail_after = UINT64_MAX;
    const char *fault = getenv("NYA_TRAIN_DEVICE_FAIL_LAUNCH_AFTER");
    if (fault) {
        char *end;
        errno = 0;
        unsigned long long value = strtoull(fault,&end,10);
        if (!*fault || *fault == '-' || *end || errno) goto fail;
        t->fail_after = value;
    }
    t->cuda = nya_cuda_create_for(1);
    if (!t->cuda || !capacity || capacity > t->cuda->cache_limit) goto fail;
    if (nya_ct_enter(t)) goto fail;
    int result = -1;
    static const unsigned types[] = {0,1,2,8,12,14,30};
    for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); ++i) {
        char name[48];
        snprintf(name,sizeof(name),"nya_train_dx_%u",types[i]);
        if (t->cuda->function_get(&t->dx[types[i]],t->cuda->module,name) != CUDA_SUCCESS) goto done;
    }
    if (t->cuda->function_get(&t->zero,t->cuda->module,"nya_train_zero") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->check_finite,t->cuda->module,"nya_train_check_finite") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->dw,t->cuda->module,"nya_train_dw") != CUDA_SUCCESS ||
        t->cuda->allocate(&t->arena,capacity) != CUDA_SUCCESS) goto done;
    t->stats.capacity_bytes = capacity;
    result = 0;
done:
    if (nya_ct_leave(t,result)) goto fail;
    return t;
fail:
    nya_ct_destroy(t);
    return NULL;
}
static const char *nya_ct_get_error(const void *context)
{ return ((const nya_cuda_train *)context)->error; }
static void nya_ct_stats(const void *context, nya_train_device_stats *stats)
{ *stats = ((const nya_cuda_train *)context)->stats; }
static int nya_ct_zero_buffer(nya_cuda_train *t, nya_cuda_train_buffer *b)
{
    unsigned long long bytes = b->bytes;
    void *args[] = {&b->memory,&bytes};
    size_t blocks = b->bytes/256 + (b->bytes%256 != 0);
    if (blocks > (size_t)t->cuda->grid_limit) return nya_ct_error(t,"training zero grid too large",0);
    return nya_ct_launch(t,t->zero,(unsigned)blocks,1,256,1,args);
}
static nya_train_buffer nya_ct_alloc(void *context, size_t bytes)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return 0;
    size_t used = t->stats.used_bytes;
    if (!bytes || used > SIZE_MAX-255 || t->stats.buffers == NYA_TRAIN_BUFFERS) {
        nya_ct_error(t,"invalid training allocation",0); return 0;
    }
    size_t offset = (used+255)&~(size_t)255;
    if (offset > t->stats.capacity_bytes || bytes > t->stats.capacity_bytes-offset ||
        bytes/256+(bytes%256 != 0) > (size_t)t->cuda->grid_limit) {
        nya_ct_error(t,"training arena budget exceeded",0); return 0;
    }
    uint64_t id = nya_ct_id(t);
    if (!id) return 0;
    if (nya_ct_enter(t)) return 0;
    nya_cuda_train_buffer b = {id,t->arena+offset,bytes};
    int result = nya_ct_zero_buffer(t,&b);
    if (nya_ct_leave(t,result)) return 0;
    t->buffers[t->stats.buffers++] = b;
    t->stats.used_bytes = offset+bytes;
    if (t->stats.used_bytes > t->stats.peak_bytes) t->stats.peak_bytes = t->stats.used_bytes;
    return id;
}
static nya_train_scope nya_ct_scratch_begin(void *context)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return 0;
    if (t->scope) { nya_ct_error(t,"training scratch scope already active",0); return 0; }
    nya_train_scope scope = nya_ct_id(t);
    if (!scope) return 0;
    t->scope = scope;
    t->scope_bytes = t->stats.used_bytes;
    t->scope_buffers = t->stats.buffers;
    t->error[0] = 0;
    return scope;
}
static int nya_ct_scratch_end(void *context, nya_train_scope scope)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return -1;
    if (!scope || scope != t->scope) return nya_ct_error(t,"invalid training scratch scope",0);
    /* No memory is freed and no device work is submitted here. cuLaunchKernel
       already copied queued argument values; every later arena use (including
       allocation zeroing) follows prior consumers on this device's one stream. */
    t->stats.used_bytes = t->scope_bytes;
    t->stats.buffers = t->scope_buffers;
    t->scope = 0;
    ++t->stats.scratch_resets;
    t->error[0] = 0;
    return 0;
}
static int nya_ct_transfer(void *context, nya_train_buffer id, size_t offset, void *host, size_t bytes, int upload)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return -1;
    nya_cuda_train_buffer *b = nya_ct_find(t,id);
    if (!b || !host || !bytes || offset > b->bytes || bytes > b->bytes-offset)
        return nya_ct_error(t,"invalid training transfer",0);
    if (nya_ct_enter(t)) return -1;
    int result = nya_ct_sync(t,t->cuda->stream);
    if (!result) {
        CUresult copied = upload ? t->cuda->upload(b->memory+offset,host,bytes) :
            t->cuda->download(host,b->memory+offset,bytes);
        if (copied != CUDA_SUCCESS) result = nya_ct_error(t,"training transfer failed",1);
        else if (upload) {
            ++t->stats.uploads; t->stats.upload_bytes += bytes;
            result = nya_ct_sync(t,NULL);
        } else { ++t->stats.downloads; t->stats.download_bytes += bytes; }
    }
    return nya_ct_leave(t,result);
}
static int nya_ct_zero(void *context, nya_train_buffer id)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return -1;
    nya_cuda_train_buffer *b = nya_ct_find(t,id);
    if (!b) return nya_ct_error(t,"invalid training buffer",0);
    if (nya_ct_enter(t)) return -1;
    return nya_ct_leave(t,nya_ct_zero_buffer(t,b));
}
static int nya_ct_finish(void *context)
{
    nya_cuda_train *t = context;
    if (nya_ct_enter(t)) return -1;
    return nya_ct_leave(t,nya_ct_sync(t,t->cuda->stream));
}
static int nya_ct_check_finite(void *context, nya_train_buffer status, nya_train_buffer data, size_t count, uint32_t tag)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return -1;
    nya_cuda_train_buffer *s = nya_ct_find(t,status), *b = nya_ct_find(t,data);
    if (!s || !b || status == data || s->bytes < sizeof(uint32_t) || !count ||
        count > b->bytes/sizeof(float) || !tag)
        return nya_ct_error(t,"invalid training finite-check descriptor",0);
    size_t blocks = count/256+(count%256 != 0);
    if (blocks > (size_t)t->cuda->grid_limit) return nya_ct_error(t,"training finite-check grid too large",0);
    unsigned long long n = count;
    unsigned label = tag;
    void *args[] = {&b->memory,&s->memory,&n,&label};
    if (nya_ct_enter(t)) return -1;
    return nya_ct_leave(t,nya_ct_launch(t,t->check_finite,(unsigned)blocks,1,256,1,args));
}
static int nya_ct_matrix(void *context, nya_train_buffer destination, nya_train_buffer a, nya_train_buffer b,
    unsigned type, size_t outputs, size_t inputs, size_t tokens, unsigned operation)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return -1;
    nya_cuda_train_buffer *dst = nya_ct_find(t,destination), *aa = nya_ct_find(t,a), *bb = nya_ct_find(t,b);
    if (!dst || !aa || !bb || destination == a || destination == b || !outputs || !inputs || !tokens ||
        outputs > UINT_MAX-63u || inputs > UINT_MAX-63u || tokens > UINT_MAX-63u || operation > 2)
        return nya_ct_error(t,"invalid training matrix descriptor",0);
    size_t block = 1, unit = 4;
    switch (type) {
        case 0: break;
        case 1: case 30: unit = 2; break;
        case 2: block = 32; unit = 18; break;
        case 8: block = 32; unit = 34; break;
        case 12: block = 256; unit = 144; break;
        case 14: block = 256; unit = 210; break;
        default: return nya_ct_error(t,"unsupported training weight type",0);
    }
    if (inputs%block || inputs/block > SIZE_MAX/unit || inputs > SIZE_MAX/4/tokens ||
        outputs > SIZE_MAX/4/tokens || outputs > SIZE_MAX/4/inputs)
        return nya_ct_error(t,"training matrix size overflow or alignment",0);
    size_t row_bytes = inputs/block*unit;
    if (outputs > SIZE_MAX/row_bytes) return nya_ct_error(t,"training weight size overflow",0);
    size_t xbytes = tokens*inputs*4, ybytes = tokens*outputs*4, wbytes = outputs*row_bytes;
    size_t need_dst = operation == 0 ? ybytes : operation == 1 ? xbytes : outputs*inputs*4;
    size_t need_a = operation == 2 ? xbytes : wbytes, need_b = operation == 0 ? xbytes : ybytes;
    if (dst->bytes < need_dst || aa->bytes < need_a || bb->bytes < need_b)
        return nya_ct_error(t,"training matrix buffer too small",0);
    unsigned tile = operation == 0 && tokens >= 32 && outputs >= 32 ? t->cuda->gemm_tile : 32;
    size_t gx = ((operation == 0 ? outputs : inputs)+tile-1)/tile;
    size_t gy = ((operation == 2 ? outputs : tokens)+tile-1)/tile;
    if (gx > (size_t)t->cuda->grid_limit || gy > 65535)
        return nya_ct_error(t,"training matrix grid too large",0);
    unsigned o = (unsigned)outputs, i = (unsigned)inputs, n = (unsigned)tokens;
    unsigned long long columns = inputs, stride = row_bytes;
    void *args[] = {&aa->memory,&bb->memory,&dst->memory,&columns,&stride,&o,&n};
    CUfunction kernel = operation == 0 ? (tile == 64 ? t->cuda->gemm_large[type] : t->cuda->gemm[type]) :
        operation == 1 ? t->dx[type] : t->dw;
    if (operation == 2) { args[3] = &i; args[4] = &o; args[5] = &n; }
    if (nya_ct_enter(t)) return -1;
    unsigned bx = tile == 64 ? 16 : 32;
    return nya_ct_leave(t,nya_ct_launch(t,kernel,(unsigned)gx,(unsigned)gy,bx,256/bx,args));
}
const nya_train_device_interface *nya_cuda_training_device(void)
{
    static const nya_train_device_interface api = {nya_ct_create,nya_ct_destroy,nya_ct_get_error,nya_ct_stats,
        nya_ct_alloc,nya_ct_transfer,nya_ct_zero,nya_ct_finish,nya_ct_matrix,
        nya_ct_scratch_begin,nya_ct_scratch_end,nya_ct_check_finite};
    return &api;
}
