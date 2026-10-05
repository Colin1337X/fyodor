/* Private resident training storage and matrix dispatch; no host tensor cache. */
#include "training_device_backend.h"
#include <stdatomic.h>
#include <errno.h>
#include <math.h>
#define NYA_TRAIN_BUFFERS 4096
typedef struct nya_cuda_train_buffer {
    nya_train_buffer id;
    CUdeviceptr memory;
    size_t bytes;
    size_t index_count, index_rows, index_groups;
} nya_cuda_train_buffer;
typedef struct nya_cuda_train {
    nya_cuda_context *cuda;
    CUdeviceptr arena;
    CUfunction zero, dx[31], dw, check_finite, unary, binary, rms, rms_dx, rms_dw, embedding, embedding_back, rope, attention, attention_prepare, attention_dq, attention_dkv;
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
        if (t->buffers[i].id == id && !t->buffers[i].index_count) return &t->buffers[i];
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
        t->cuda->function_get(&t->unary,t->cuda->module,"nya_train_unary") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->binary,t->cuda->module,"nya_train_binary") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->rms,t->cuda->module,"nya_train_rms") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->rms_dx,t->cuda->module,"nya_train_rms_dx") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->rms_dw,t->cuda->module,"nya_train_rms_dw") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->embedding,t->cuda->module,"nya_train_embedding") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->embedding_back,t->cuda->module,"nya_train_embedding_back") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->attention,t->cuda->module,"nya_train_attention") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->attention_prepare,t->cuda->module,"nya_train_attention_prepare") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->attention_dq,t->cuda->module,"nya_train_attention_dq") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->attention_dkv,t->cuda->module,"nya_train_attention_dkv") != CUDA_SUCCESS ||
        t->cuda->function_get(&t->rope,t->cuda->module,"nya_train_rope") != CUDA_SUCCESS ||
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
    nya_cuda_train_buffer b = {.id=id,.memory=t->arena+offset,.bytes=bytes};
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
static nya_train_indices nya_ct_indices(void *context, const nya_train_index_pack *pack)
{
    nya_cuda_train *t=context;
    if (t->stats.failed) return 0;
    if (!pack || !pack->data || !pack->count || !pack->rows || !pack->groups ||
        pack->count>(size_t)t->cuda->grid_limit || pack->groups>pack->count) {
        nya_ct_error(t,"invalid training token indices or host staging allocation failed",0); return 0;
    }
    nya_train_buffer id=nya_ct_alloc(t,pack->bytes);
    if (!id || nya_ct_transfer(t,id,0,pack->data,pack->bytes,1)) return 0;
    nya_cuda_train_buffer *b=nya_ct_find(t,id);
    b->index_count=pack->count; b->index_rows=pack->rows; b->index_groups=pack->groups;
    return id;
}
static int nya_ct_embedding(void *context, nya_train_buffer out, nya_train_buffer source,
    unsigned type, size_t columns, nya_train_indices indices, int backward)
{
    nya_cuda_train *t=context;
    if (t->stats.failed) return -1;
    nya_cuda_train_buffer *dst=nya_ct_find(t,out), *src=nya_ct_find(t,source), *map=NULL;
    if (indices) for (size_t k=0;k<t->stats.buffers;++k)
        if (t->buffers[k].id==indices && t->buffers[k].index_count) { map=&t->buffers[k]; break; }
    if (!dst || !src || !map || out==source || !columns || (backward!=0 && backward!=1))
        return nya_ct_error(t,"invalid training embedding descriptor",0);
    size_t block=1, unit=4;
    switch (type) {
        case 0: break;
        case 1: case 30: unit=2; break;
        case 2: block=32; unit=18; break;
        case 8: block=32; unit=34; break;
        case 12: block=256; unit=144; break;
        case 14: block=256; unit=210; break;
        default: return nya_ct_error(t,"unsupported training embedding type",0);
    }
    if ((backward && type) || columns%block || columns/block>SIZE_MAX/unit ||
        columns>SIZE_MAX/4/map->index_count)
        return nya_ct_error(t,"training embedding size overflow or alignment",0);
    size_t stride=columns/block*unit;
    if (map->index_rows>SIZE_MAX/stride)
        return nya_ct_error(t,"training embedding table size overflow",0);
    size_t table_bytes=map->index_rows*stride, token_bytes=map->index_count*columns*4;
    if (dst->bytes<(backward?table_bytes:token_bytes) || src->bytes<(backward?token_bytes:table_bytes))
        return nya_ct_error(t,"training embedding buffer too small",0);
    unsigned long long cols=columns, row_bytes=stride, count=map->index_count, groups=map->index_groups;
    void *args[]={&src->memory,&map->memory,&dst->memory,&cols,backward?&count:&row_bytes,backward?(void *)&groups:(void *)&type};
    if (nya_ct_enter(t)) return -1;
    return nya_ct_leave(t,nya_ct_launch(t,backward?t->embedding_back:t->embedding,
        (unsigned)(backward?groups:count),1,256,1,args));
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
static int nya_ct_unary(void *context, nya_train_buffer destination, nya_train_buffer x, nya_train_buffer dy,
    size_t count, unsigned operation, double scalar, int backward)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return -1;
    nya_cuda_train_buffer *dst = nya_ct_find(t,destination), *a = nya_ct_find(t,x), *g = nya_ct_find(t,dy);
    if (!dst || !a || destination == x || destination == dy || !count || count > dst->bytes/4 || count > a->bytes/4 ||
        (backward && (!g || count > g->bytes/4)) || (!backward && dy) ||
        (backward != 0 && backward != 1) || operation > NYA_TRAIN_UNARY_SOFTCAP ||
        !isfinite(scalar) || (operation == NYA_TRAIN_UNARY_SOFTCAP && scalar <= 0))
        return nya_ct_error(t,"invalid training unary descriptor",0);
    size_t blocks = count/256+(count%256 != 0);
    if (blocks > (size_t)t->cuda->grid_limit) return nya_ct_error(t,"training unary grid too large",0);
    unsigned long long n = count;
    CUdeviceptr gradient = g ? g->memory : 0;
    void *args[] = {&a->memory,&gradient,&dst->memory,&n,&operation,&scalar,&backward};
    if (nya_ct_enter(t)) return -1;
    return nya_ct_leave(t,nya_ct_launch(t,t->unary,(unsigned)blocks,1,256,1,args));
}
static nya_cuda_train_buffer *nya_ct_view(nya_cuda_train *t, nya_train_view view, size_t *count)
{
    nya_cuda_train_buffer *buffer = nya_ct_find(t,view.buffer);
    if (!buffer || !view.rows || !view.columns || view.rows > SIZE_MAX/view.columns) return NULL;
    *count = view.rows*view.columns;
    return *count <= buffer->bytes/4 ? buffer : NULL;
}
static int nya_ct_attention(void *context, nya_train_buffer out_q, nya_train_buffer out_k, nya_train_buffer out_v,
    nya_train_buffer state, nya_train_buffer dy, nya_train_buffer workspace, nya_train_attention_desc desc, int backward)
{
    nya_cuda_train *t=context;
    if (t->stats.failed) return -1;
    size_t nq=0,nk=0,nv=0;
    nya_cuda_train_buffer *q=nya_ct_view(t,desc.q,&nq), *k=nya_ct_view(t,desc.k,&nk), *v=nya_ct_view(t,desc.v,&nv);
    nya_cuda_train_buffer *saved=nya_ct_find(t,state), *groups=nya_ct_find(t,desc.groups);
    nya_cuda_train_buffer *gradient=nya_ct_find(t,dy), *work=nya_ct_find(t,workspace);
    size_t n=desc.q.rows,h=desc.heads,kh=desc.kv_heads,dim=desc.dimension;
    if (!q || !k || !v || !saved || !h || !kh || !dim || h%kh || h>SIZE_MAX/dim || kh>SIZE_MAX/dim ||
        desc.q.columns!=h*dim || desc.k.columns!=kh*dim || desc.v.columns!=desc.k.columns ||
        desc.k.rows!=n || desc.v.rows!=n || !isfinite(desc.scale) || desc.scale<=0 ||
        h>SIZE_MAX/n || n*h>saved->bytes/16 || n*h>(size_t)t->cuda->grid_limit ||
        (desc.groups && (!groups || n>groups->bytes/4)) ||
        state==desc.q.buffer || state==desc.k.buffer || state==desc.v.buffer || state==desc.groups ||
        (backward!=0 && backward!=1)) return nya_ct_error(t,"invalid training attention descriptor",0);
    size_t work_bytes=nya_train_attention_workspace_bytes(n,h);
    if (!backward) {
        if (!out_q || out_k || out_v || dy || workspace) return nya_ct_error(t,"invalid training attention forward",0);
    } else {
        if ((!out_q && !out_k && !out_v) || !gradient || nq>gradient->bytes/4 ||
            !work || !work_bytes || work_bytes>work->bytes || workspace==state || workspace==dy ||
            workspace==desc.q.buffer || workspace==desc.k.buffer || workspace==desc.v.buffer || workspace==desc.groups || dy==state)
            return nya_ct_error(t,"invalid training attention backward state",0);
    }
    nya_train_buffer ids[]={out_q,out_k,out_v};
    nya_cuda_train_buffer *outputs[3]={NULL,NULL,NULL};
    for (size_t j=0;j<3;++j) if (ids[j]) {
        outputs[j]=nya_ct_find(t,ids[j]);
        if (!outputs[j] || (j?nk:nq)>outputs[j]->bytes/4 || ids[j]==desc.q.buffer || ids[j]==desc.k.buffer ||
            ids[j]==desc.v.buffer || ids[j]==desc.groups || ids[j]==state || ids[j]==dy || ids[j]==workspace)
            return nya_ct_error(t,"invalid training attention output",0);
        for (size_t prior=0;prior<j;++prior) if (ids[j]==ids[prior])
            return nya_ct_error(t,"aliased training attention gradients",0);
    }
    unsigned long long tokens=n,heads=h,kv_heads=kh,dimension=dim,window=desc.window;
    CUdeviceptr gp=groups?groups->memory:0,oq=outputs[0]?outputs[0]->memory:0;
    double scale=desc.scale;
    if (nya_ct_enter(t)) return -1;
    int result=0;
    if (!backward) {
        void *args[]={&q->memory,&k->memory,&v->memory,&gp,&oq,&saved->memory,&tokens,&heads,&kv_heads,&dimension,&window,&scale};
        result=nya_ct_launch(t,t->attention,(unsigned)(n*h),1,256,1,args);
    } else {
        CUdeviceptr ok=outputs[1]?outputs[1]->memory:0,ov=outputs[2]?outputs[2]->memory:0;
        int scores=out_q || out_k;
        for (size_t first=0;first<n && !result;first+=16) {
            unsigned long long start=first,count=n-first<16?n-first:16;
            void *prepare[]={&q->memory,&k->memory,&v->memory,&gp,&gradient->memory,&saved->memory,&work->memory,
                &tokens,&heads,&kv_heads,&dimension,&window,&scale,&start,&scores};
            result=nya_ct_launch(t,t->attention_prepare,(unsigned)(count*h),1,256,1,prepare);
            if (!result && out_q) {
                void *args[]={&k->memory,&work->memory,&oq,&tokens,&heads,&kv_heads,&dimension,&start};
                result=nya_ct_launch(t,t->attention_dq,(unsigned)(count*h),1,256,1,args);
            }
            if (!result && (out_k || out_v)) {
                void *args[]={&q->memory,&gradient->memory,&work->memory,&ok,&ov,&tokens,&heads,&kv_heads,&dimension,&start,&count};
                result=nya_ct_launch(t,t->attention_dkv,(unsigned)(n*kh),1,256,1,args);
            }
        }
    }
    return nya_ct_leave(t,result);
}
static int nya_ct_rope(void *context, nya_train_buffer destination, nya_train_view source,
    size_t heads, size_t dimension, nya_train_buffer frequencies, int split_half, int backward)
{
    nya_cuda_train *t=context;
    if (t->stats.failed) return -1;
    size_t count=0;
    nya_cuda_train_buffer *src=nya_ct_view(t,source,&count), *dst=nya_ct_find(t,destination);
    nya_cuda_train_buffer *freq=nya_ct_find(t,frequencies);
    if (!src || !dst || !freq || !heads || !dimension || dimension%2 ||
        heads>SIZE_MAX/dimension || heads*dimension!=source.columns ||
        dimension/2>freq->bytes/4 || count>dst->bytes/4 ||
        destination==source.buffer || destination==frequencies ||
        (split_half!=0 && split_half!=1) || (backward!=0 && backward!=1))
        return nya_ct_error(t,"invalid training RoPE descriptor",0);
    size_t pairs=count/2, blocks=pairs/256+(pairs%256!=0);
    if (blocks>(size_t)t->cuda->grid_limit) return nya_ct_error(t,"training RoPE grid too large",0);
    unsigned long long n=pairs, columns=source.columns, dim=dimension;
    void *args[]={&src->memory,&freq->memory,&dst->memory,&n,&columns,&dim,&split_half,&backward};
    if (nya_ct_enter(t)) return -1;
    return nya_ct_leave(t,nya_ct_launch(t,t->rope,(unsigned)blocks,1,256,1,args));
}
static int nya_ct_binary(void *context, nya_train_buffer out_a, nya_train_buffer out_b,
    nya_train_view a, nya_train_view b, nya_train_buffer dy, unsigned operation, int backward)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return -1;
    size_t na = 0, nb = 0;
    nya_cuda_train_buffer *aa = nya_ct_view(t,a,&na), *bb = nya_ct_view(t,b,&nb);
    nya_cuda_train_buffer *oa = nya_ct_find(t,out_a), *ob = nya_ct_find(t,out_b), *g = nya_ct_find(t,dy);
    if (!aa || !bb || operation > NYA_TRAIN_BINARY_MUL || (backward != 0 && backward != 1) ||
        (a.rows != b.rows && a.rows != 1 && b.rows != 1) ||
        (a.columns != b.columns && a.columns != 1 && b.columns != 1))
        return nya_ct_error(t,"invalid training binary descriptor",0);
    size_t rows = a.rows > b.rows ? a.rows : b.rows, cols = a.columns > b.columns ? a.columns : b.columns;
    if (rows > SIZE_MAX/cols || rows*cols > SIZE_MAX/4)
        return nya_ct_error(t,"training binary size overflow",0);
    size_t count = rows*cols;
    if (!backward) {
        if (!oa || out_b || dy || count > oa->bytes/4 || out_a == a.buffer || out_a == b.buffer)
            return nya_ct_error(t,"invalid training binary output",0);
    } else {
        if ((!out_a && !out_b) || !g || count > g->bytes/4 ||
            (out_a && (!oa || na > oa->bytes/4 || out_a == a.buffer || out_a == b.buffer || out_a == dy)) ||
            (out_b && (!ob || nb > ob->bytes/4 || out_b == a.buffer || out_b == b.buffer || out_b == dy)) ||
            (out_a == out_b && (a.rows != b.rows || a.columns != b.columns)))
            return nya_ct_error(t,"invalid training binary gradients",0);
        count = out_a ? na : 0;
        if (out_b && nb > count) count = nb;
    }
    size_t blocks = count/256+(count%256 != 0);
    if (blocks > (size_t)t->cuda->grid_limit) return nya_ct_error(t,"training binary grid too large",0);
    unsigned long long ar = a.rows, ac = a.columns, br = b.rows, bc = b.columns, r = rows, c = cols;
    CUdeviceptr gradient = g ? g->memory : 0, dst_a = oa ? oa->memory : 0, dst_b = ob ? ob->memory : 0;
    void *args[] = {&aa->memory,&bb->memory,&gradient,&dst_a,&dst_b,&ar,&ac,&br,&bc,&r,&c,&operation,&backward};
    if (nya_ct_enter(t)) return -1;
    return nya_ct_leave(t,nya_ct_launch(t,t->binary,(unsigned)blocks,1,256,1,args));
}
static int nya_ct_rms_norm(void *context, nya_train_buffer out, nya_train_buffer dweight, nya_train_buffer inverse,
    nya_train_view x, nya_train_buffer weight, nya_train_buffer dy, float epsilon, int backward)
{
    nya_cuda_train *t = context;
    if (t->stats.failed) return -1;
    size_t count = 0;
    nya_cuda_train_buffer *a = nya_ct_view(t,x,&count), *w = nya_ct_find(t,weight), *g = nya_ct_find(t,dy);
    nya_cuda_train_buffer *dst = nya_ct_find(t,out), *dw = nya_ct_find(t,dweight), *inv = nya_ct_find(t,inverse);
    if (!a || !inv || x.rows > inv->bytes/8 || inverse == x.buffer || inverse == weight || inverse == dy ||
        (weight && (!w || x.columns > w->bytes/4)) || (backward != 0 && backward != 1) ||
        (out && (!dst || count > dst->bytes/4 || out == x.buffer || out == weight || out == dy || out == inverse)))
        return nya_ct_error(t,"invalid training RMSNorm descriptor",0);
    if (!backward) {
        if (!out || dweight || dy || !isfinite(epsilon) || epsilon <= 0)
            return nya_ct_error(t,"invalid training RMSNorm forward",0);
    } else {
        if ((!out && !dweight) || !g || count > g->bytes/4 ||
            (dweight && (!weight || !dw || x.columns > dw->bytes/4 || dweight == x.buffer ||
                dweight == weight || dweight == dy || dweight == inverse)) ||
            (out && out == dweight && x.rows != 1))
            return nya_ct_error(t,"invalid training RMSNorm gradients",0);
    }
    size_t blocks = x.columns/256+(x.columns%256 != 0);
    if (((!backward || out) && x.rows > (size_t)t->cuda->grid_limit) ||
        (dweight && blocks > (size_t)t->cuda->grid_limit))
        return nya_ct_error(t,"training RMSNorm grid too large",0);
    unsigned long long rows = x.rows, columns = x.columns;
    CUdeviceptr wp = w ? w->memory : 0;
    double eps = epsilon;
    if (nya_ct_enter(t)) return -1;
    int result = 0;
    if (!backward) {
        void *args[] = {&a->memory,&wp,&dst->memory,&inv->memory,&columns,&eps};
        result = nya_ct_launch(t,t->rms,(unsigned)rows,1,256,1,args);
    } else {
        if (out) {
            void *args[] = {&a->memory,&wp,&g->memory,&inv->memory,&dst->memory,&columns};
            result = nya_ct_launch(t,t->rms_dx,(unsigned)rows,1,256,1,args);
        }
        if (!result && dweight) {
            void *args[] = {&a->memory,&g->memory,&inv->memory,&dw->memory,&rows,&columns};
            result = nya_ct_launch(t,t->rms_dw,(unsigned)blocks,1,256,1,args);
        }
    }
    return nya_ct_leave(t,result);
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
        nya_ct_scratch_begin,nya_ct_scratch_end,nya_ct_check_finite,nya_ct_unary,nya_ct_binary,nya_ct_rms_norm,
        nya_ct_indices,nya_ct_embedding,nya_ct_rope,nya_ct_attention};
    return &api;
}
