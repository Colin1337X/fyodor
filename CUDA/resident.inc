/* Host-side CUDA graph lowering. Included by cuda.c so driver types and entry
   points remain entirely inside the removable CUDA directory. The model binding
   is read-only; every live request owns one scratch allocation and KV prefix. */
#include "../backend/core/llm_internal.h"
#include <math.h>
#include <errno.h>
#include "q4_byte_pack.h"
#include "kv_layout.h"
typedef struct nya_cuda_timing { CUevent start, end; unsigned kind, matrix; size_t attention; } nya_cuda_timing;
typedef struct nya_cuda_matrix_profile {
    size_t rows, columns, batch;
    unsigned type, path;
    unsigned long long calls;
    double milliseconds;
} nya_cuda_matrix_profile;

typedef struct nya_cuda_attention_profile { unsigned long long kernels; double milliseconds; } nya_cuda_attention_profile;

typedef struct nya_cuda_plan {
    nya_cuda_context *cuda;
    const nya_llm_context *model;
    CUdeviceptr storage, x, norm, residual, gate, up, q, scores, keys, values, logits, frequencies;
    CUdeviceptr blas_storage, attention_parts, interleaved_input;
    size_t interleaved_columns;
    CUdeviceptr request, decode_key, decode_value, active_request;
    size_t blas_expansion;
    CUdeviceptr *tensors, *byte_tensors;
    CUdeviceptr byte_storage;
    size_t byte_weights;
#ifdef NYA_PORTABLE_ATTENTION
    CUfunction portable_attention;
    int use_portable_attention;
#endif
#ifdef NYA_FLASH_DECODING_PP
    CUfunction fdpp_partial, fdpp_combine;
    int use_fdpp, trace_fdpp, fdpp_short_split;
#ifdef NYA_FLASH_DECODING_PP_SPECIALIZED
    CUfunction fdpp_partial_64, fdpp_partial_128, fdpp_direct_64, fdpp_direct_128;
#endif
#endif
    CUfunction gather, gather_batch, norm_op, add_norm, rope, rope_pair, vector, attention;
    CUfunction dot_add[2], qkv[2], decode_request, attention_prefill, attention_split, attention_combine;
    size_t capacity, bytes, valid, batch, maximum_batch, logits_batch;
    nya_cuda_kv_layout *kv_layout;
    size_t kv_high_water, hidden_valid, hidden_offset;
    CUfunction rope_ring, attention_ring, attention_ring_split;
    CUfunction wide_value[4];
    int use_wide_value, use_wide_batch;
    unsigned long long borrowed_wide_calls, batched_wide_calls, batched_logits_calls, interleaved_calls;
    int failed;
    int gemma_qkv, gemma_gate;
    nya_compute_stats stats;
    CUgraphExec executable[2];
    unsigned long long graph_kernels[2];
    int use_graphs, split_attention, wide_split_attention, fuse_gate_up, fuse_qkv, fuse_residual, fuse_rope_pair;
    size_t fail_after;
    size_t allocations, fail_allocation_after;
    nya_cuda_timing *timings;
    size_t timing_count, profile_from, profiled_batches, profiled_tokens;
    int profile_active;
    double milliseconds[6];
    nya_cuda_matrix_profile *matrix_profiles;
    unsigned matrix_profile, matrix_profile_count;
    nya_cuda_attention_profile *attention_profiles;
    size_t profile_layer;
} nya_cuda_plan;

static void plan_profile_flush(nya_cuda_plan *p)
{
    if (!p->timing_count) return;
    p->cuda->stream_sync(p->cuda->stream);
    for (size_t i = 0; i < p->timing_count; ++i) {
        nya_cuda_timing *t = &p->timings[i]; float elapsed;
        if (p->cuda->event_elapsed(&elapsed, t->start, t->end) == CUDA_SUCCESS) {
            p->milliseconds[t->kind] += elapsed;
            if(t->attention) {
                nya_cuda_attention_profile *a=&p->attention_profiles[t->attention-1];
                ++a->kernels;a->milliseconds+=elapsed;
            }
            if (t->matrix) {
                nya_cuda_matrix_profile *m = &p->matrix_profiles[t->matrix - 1];
                ++m->calls; m->milliseconds += elapsed;
            }
        }
        p->cuda->event_free(t->start); p->cuda->event_free(t->end);
    }
    p->timing_count = 0;
}

static CUdeviceptr plan_tensor(const nya_cuda_plan *p, const nya_llm_tensor *t)
{ return t == NULL ? 0 : p->tensors[(size_t)(t - p->model->tensors)]; }

static void cuda_plan_free(void *opaque)
{
    nya_cuda_plan *p = opaque;
    if (p == NULL) return;
    CUcontext previous;
    if (p->storage && p->cuda->push(p->cuda->context) == CUDA_SUCCESS) {
        p->cuda->stream_sync(p->cuda->stream);
        plan_profile_flush(p);
        for (size_t i = 0; i < 2; ++i) if (p->executable[i]) p->cuda->graph_exec_free(p->executable[i]);
        p->cuda->deallocate(p->storage);
        if (p->blas_storage) p->cuda->deallocate(p->blas_storage);
        if (p->byte_storage) p->cuda->deallocate(p->byte_storage);
        p->cuda->pop(&previous);
    }
    if (p->timings) fprintf(stderr, "CUDA kernel GPU milliseconds: matmul=%.3f norm=%.3f rope=%.3f vector=%.3f attention=%.3f gather=%.3f (profiling disables graphs)\n",
        p->milliseconds[0], p->milliseconds[1], p->milliseconds[2], p->milliseconds[3], p->milliseconds[4], p->milliseconds[5]);
    if(p->timings) fprintf(stderr,"CUDA profile scope: from_position=%zu batches=%zu tokens=%zu (batches starting below the threshold excluded)\n",
        p->profile_from,p->profiled_batches,p->profiled_tokens);
    if (p->matrix_profiles) {
        static const char *paths[] = {"dot", "reference", "native-gemm", "vendor-gemm", "overflow", "gate-up-q4", "qkv", "dot-add", "small-batch-dot", "small-batch-weight-reuse", "interleaved-weight-reuse"};
        for (unsigned i = 0; i < 64; ++i) {
            const nya_cuda_matrix_profile *m = &p->matrix_profiles[i];
            if (m->calls) fprintf(stderr, "CUDA matrix profile: path=%s type=%u rows=%zu columns=%zu batch=%zu calls=%llu milliseconds=%.6f\n",
                paths[m->path], m->type, m->rows, m->columns, m->batch, m->calls, m->milliseconds);
        }
    }
    if(p->attention_profiles) for(size_t i=0;i<p->model->block_count;++i) {
        const nya_cuda_attention_profile *a=&p->attention_profiles[i];
        const nya_llm_layer *w=&p->model->layers[i];
        if(a->kernels) fprintf(stderr,"CUDA attention profile: layer=%zu width=%u heads=%u kvheads=%u window=%u batch_scope=all_selected kernels=%llu milliseconds=%.6f\n",
            i,w->head_dimension,w->head_count,w->kv_head_count,w->sliding_window,a->kernels,a->milliseconds);
    }
    const char *mtp_trace=getenv("NYA_MTP_TRACE");
    if(p->borrowed_wide_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA borrowed wide-value attention calls: %llu\n",p->borrowed_wide_calls);
    if(p->batched_wide_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA batched wide-value attention calls: %llu\n",p->batched_wide_calls);
    if(p->interleaved_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA interleaved weight-reuse calls: %llu\n",p->interleaved_calls);
    if(p->batched_logits_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA batched output projection calls: %llu\n",p->batched_logits_calls);
    free(p->attention_profiles);free(p->kv_layout); free(p->matrix_profiles); free(p->timings); free(p->byte_tensors); free(p->tensors); free(p);
}

static int plan_add(size_t *total, size_t count)
{
    if (count > SIZE_MAX / sizeof(float) || *total > SIZE_MAX - count * sizeof(float)) return -1;
    *total += count * sizeof(float); return 0;
}

static void *plan_decline(const char *reason)
{
    if (getenv("NYA_CUDA_DEBUG")) fprintf(stderr, "CUDA resident plan declined: %s; selecting CPU reference execution\n", reason);
    return NULL;
}

/* Fault injection follows the real partial-allocation cleanup path without
   consuming unrelated VRAM. The limit counts successful request allocations. */
static CUresult plan_allocate(nya_cuda_plan *p, CUdeviceptr *out, size_t bytes)
{
    if (p->allocations >= p->fail_allocation_after) return CUDA_ERROR_OUT_OF_MEMORY;
    CUresult result = p->cuda->allocate(out, bytes);
    if (result == CUDA_SUCCESS) ++p->allocations;
    else fprintf(stderr, "CUDA resident allocation of %zu bytes failed: driver status %d\n", bytes, (int)result);
    return result;
}

/* A packed tensor is usable only for its complete, block-aligned matrix. */
static int plan_byte_shape(const nya_llm_tensor *t, size_t rows, size_t cols)
{
    return t && t->type==12 && t->dimension_count==2 && rows && cols &&
        cols%256==0 && cols/256<=SIZE_MAX/160 &&
        t->dimensions[0]==cols && t->dimensions[1]==rows &&
        rows<=SIZE_MAX/(cols/256*144) && t->data_size==rows*(cols/256*144);
}
static CUdeviceptr plan_byte_tensor(nya_cuda_plan *p, const nya_llm_tensor *t,
    size_t rows, size_t cols)
{
    if (!p->byte_tensors || !plan_byte_shape(t,rows,cols)) return 0;
    return p->byte_tensors[(size_t)(t-p->model->tensors)];
}
static void plan_byte_mark(CUdeviceptr *table, const nya_llm_context *m,
    const nya_llm_tensor *t)
{
    if (t && plan_byte_shape(t,(size_t)t->dimensions[1],(size_t)t->dimensions[0]))
        table[(size_t)(t-m->tensors)]=1;
}

/* Optional plan-owned copy: native weights remain available for prefill and
   fallback. All packing/allocation happens before graph capture, never decode. */
static void plan_byte_weights(nya_cuda_plan *p)
{
    nya_cuda_context *c=p->cuda;
    const nya_llm_context *m=p->model;
    int gemma_gate=p->gemma_gate && c->gemma_gate && p->fuse_gate_up && m->is_gemma && m->embedding_length==3840;
    if (!c->byte_dot && !gemma_gate && (!c->byte_gate || !p->fuse_gate_up || m->is_gemma)) return;
    CUdeviceptr *table=calloc(m->tensor_count,sizeof(*table));
    if (!table) return;
    for (size_t layer=0;layer<m->block_count;++layer) {
        const nya_llm_layer *w=&m->layers[layer];
        const nya_llm_tensor *pair[]={w->feed_forward_gate,w->feed_forward_up};
        if (c->byte_dot) {
            const nya_llm_tensor *projections[]={w->query,w->key,w->value,w->attention_output,
                w->feed_forward_gate,w->feed_forward_up,w->feed_forward_down};
            for (unsigned j=0;j<sizeof(projections)/sizeof(projections[0]);++j)
                plan_byte_mark(table,m,projections[j]);
        }
        if (gemma_gate && plan_byte_shape(pair[0],w->hidden_length,3840) &&
            plan_byte_shape(pair[1],w->hidden_length,3840)) {
            plan_byte_mark(table,m,pair[0]); plan_byte_mark(table,m,pair[1]);
        }
        int eligible=c->byte_gate && p->fuse_gate_up && !m->is_gemma;
        for (unsigned j=0;j<2;++j) {
            const nya_llm_tensor *t=pair[j];
            if (!t || t->type!=12 || t->dimension_count!=2 || t->dimensions[0]!=2048 ||
                !t->dimensions[1] || t->dimensions[1]>SIZE_MAX/1152 ||
                t->data_size!=(size_t)t->dimensions[1]*1152) eligible=0;
        }
        if (eligible) for (unsigned j=0;j<2;++j) table[(size_t)(pair[j]-m->tensors)]=1;
    }
    if (c->byte_dot) plan_byte_mark(table,m,m->output);
    size_t bytes=0, largest=0;
    for (size_t i=0;i<m->tensor_count;++i) if (table[i]) {
        size_t amount=nya_q4_byte_size(m->tensors[i].data_size);
        if (!amount || amount>SIZE_MAX-bytes) { free(table); return; }
        bytes+=amount; if (amount>largest) largest=amount;
    }
    size_t free_bytes,total_bytes;
    if (!bytes || c->cached_bytes>c->cache_limit || p->bytes>c->cache_limit-c->cached_bytes ||
        c->memory_info(&free_bytes,&total_bytes)!=CUDA_SUCCESS) { free(table); return; }
    size_t reserve=free_bytes/10;
    if (reserve<256U*1024U*1024U) reserve=256U*1024U*1024U;
    if (free_bytes<=reserve) { free(table); return; }
    size_t budget=c->cache_limit-c->cached_bytes-p->bytes;
    if (budget>free_bytes-reserve) budget=free_bytes-reserve;
    if (bytes>budget) {
        const char *partial=getenv("NYA_CUDA_GEMMA_GATE_PARTIAL");
        if (!gemma_gate || !partial || strcmp(partial,"1")) { free(table); return; }
        /* Opt-in: retain complete gate/up pairs, never a half-fusable layer.
           Native matrices remain resident for every unselected layer. */
        memset(table,0,m->tensor_count*sizeof(*table));
        bytes=0; largest=0;
        for (size_t layer=0;layer<m->block_count;++layer) {
            const nya_llm_layer *w=&m->layers[layer];
            const nya_llm_tensor *g=w->feed_forward_gate,*u=w->feed_forward_up;
            if (!plan_byte_shape(g,w->hidden_length,3840) ||
                !plan_byte_shape(u,w->hidden_length,3840)) continue;
            size_t gi=(size_t)(g-m->tensors),ui=(size_t)(u-m->tensors);
            size_t ga=table[gi]?0:nya_q4_byte_size(g->data_size);
            size_t ua=(table[ui] || gi==ui)?0:nya_q4_byte_size(u->data_size);
            if (ga>budget-bytes || ua>budget-bytes-ga) continue;
            table[gi]=1; table[ui]=1; bytes+=ga+ua;
            if (ga>largest) largest=ga;
            if (ua>largest) largest=ua;
        }
        if (!bytes) { free(table); return; }
    }
    void *host=malloc(largest);
    CUdeviceptr storage=0;
    if (!host || plan_allocate(p,&storage,bytes)!=CUDA_SUCCESS) { free(host); free(table); return; }
    size_t offset=0; int success=1;
    for (size_t i=0;i<m->tensor_count;++i) if (table[i]) {
        const nya_llm_tensor *t=&m->tensors[i];
        size_t amount=nya_q4_byte_size(t->data_size);
        if (nya_q4_byte_pack(host,largest,t->data,t->data_size) ||
            c->upload(storage+offset,host,amount)!=CUDA_SUCCESS) { success=0; break; }
        table[i]=storage+offset; offset+=amount;
    }
    free(host);
    if (!success) { c->deallocate(storage); free(table); return; }
    p->byte_storage=storage; p->byte_tensors=table; p->byte_weights=bytes;
    p->stats.weights_bytes+=bytes; p->bytes+=bytes;
    if (getenv("NYA_CUDA_DEBUG")) fprintf(stderr,"CUDA aligned Q4 gate copy: %zu extra device bytes\n",bytes);
}

static void *cuda_plan_create(void *opaque, const nya_llm_context *m, size_t capacity)
{
    nya_cuda_context *c = opaque;
    /* Routed experts, PLE, MTP and multimodal overlays retain their audited
       reference execution until their complete graphs have resident lowering. */
    if (m == NULL || m->is_assistant || m->expert_count || m->per_layer_embedding_length ||
        capacity == 0 || capacity > m->context_length || capacity > UINT32_MAX || m->cache_width > SIZE_MAX / capacity) return NULL;
    size_t d = m->embedding_length, h = m->maximum_hidden_length, q = m->maximum_query_length;
    size_t heads = 0, rope_count = 0, total = 0, weights = 0, maximum_kv = 0, partial_count = 0;
    const char *split_setting = getenv("NYA_CUDA_SPLIT_ATTENTION");
    int split_attention = !c->reference && capacity > 256 && (!split_setting || strcmp(split_setting,"0"));
    int use_fdpp = 0;
#ifdef NYA_FLASH_DECODING_PP
    const char *fdpp_setting = getenv("NYA_CUDA_FLASH_DECODING_PP");
    use_fdpp = !c->reference && fdpp_setting && !strcmp(fdpp_setting,"1");
    if (!c->reference && fdpp_setting && !strcmp(fdpp_setting,"short")) use_fdpp = 2;
#endif
    const char *wide_split_setting=getenv("NYA_CUDA_WIDE_SPLIT");
    int wide_split=wide_split_setting && !strcmp(wide_split_setting,"1");
    int has_split_attention = 0;
    const char *ring_setting=getenv("NYA_CUDA_RING_KV");
    int ring_kv=!c->reference && ring_setting && !strcmp(ring_setting,"1");
    int all_tiled = !c->reference && !ring_kv;
    for (size_t i = 0; i < m->block_count; ++i) {
        if (m->layers[i].head_dimension != 64 && m->layers[i].head_dimension != 128) all_tiled = 0;
        if (m->layers[i].head_count > heads) heads = m->layers[i].head_count;
        if (!m->layers[i].head_dimension || m->layers[i].kv_head_count > SIZE_MAX / m->layers[i].head_dimension) return NULL;
        size_t kv = m->layers[i].kv_head_count * m->layers[i].head_dimension;
        if (kv > maximum_kv) maximum_kv = kv;
        if (split_attention && (m->layers[i].head_dimension == 64 || m->layers[i].head_dimension == 128 ||
            (wide_split && (m->layers[i].head_dimension == 256 || m->layers[i].head_dimension == 512))) &&
            (!m->layers[i].sliding_window || m->layers[i].sliding_window > 256)) has_split_attention = 1;
        if ((use_fdpp && m->layers[i].head_dimension <= 256) ||
            (split_attention && (m->layers[i].head_dimension == 64 || m->layers[i].head_dimension == 128 ||
            (wide_split && (m->layers[i].head_dimension == 256 || m->layers[i].head_dimension == 512))) &&
            (!m->layers[i].sliding_window || m->layers[i].sliding_window > 256))) {
            size_t width = m->layers[i].head_dimension+2;
            if (m->layers[i].head_count > SIZE_MAX/width/4) return NULL;
            size_t partials = m->layers[i].head_count*width*4;
            if (partials > partial_count) partial_count = partials;
        }
        if (plan_add(&rope_count, m->layers[i].head_dimension / 2) != 0) return NULL;
    }
    if (heads > SIZE_MAX / capacity) return NULL;
    size_t batch_limit = 512;
    const char *batch_setting = getenv("NYA_CUDA_BATCH");
    if (batch_setting) {
        char *end; unsigned long value = strtoul(batch_setting, &end, 10);
        if (batch_setting[0] < '1' || batch_setting[0] > '9' || *end || value > 512) return NULL;
        batch_limit = (size_t)value;
    }
    size_t batch = capacity < batch_limit ? capacity : batch_limit;
    const char *logits_setting=getenv("NYA_CUDA_BATCH_LOGITS");
    int batch_logits=!c->reference && logits_setting && !strcmp(logits_setting,"1");
    size_t logits_batch=batch_logits?(batch<4?batch:4):1;
    if(m->tokenizer.vocabulary_size>SIZE_MAX/logits_batch) return NULL;
    if (heads * capacity > SIZE_MAX / batch || h > SIZE_MAX / batch || q > SIZE_MAX / batch || d > SIZE_MAX / batch) return NULL;
    /* Tiled prefill never materializes global scores. Only its fewer-than-eight
       query fallback needs score rows; decode uses the first row of that same
       persistent allocation. Mixed/unsupported widths and reference mode still
       reserve a complete batch. Account this before admitting model weights. */
    const char *interleaved_setting=getenv("NYA_CUDA_INTERLEAVED_REUSE");
    int interleaved=!c->reference && interleaved_setting && !strcmp(interleaved_setting,"1") &&
        c->interleave_four && (c->interleaved_reuse[12] || c->interleaved_reuse[14]);
    size_t interleaved_columns=d>h?d:h;
    if(q>interleaved_columns)interleaved_columns=q;
    if(interleaved && interleaved_columns>(SIZE_MAX-3)/4)return NULL;
    size_t interleaved_count=interleaved?4*interleaved_columns+3:0;
    size_t score_batch = all_tiled && batch > 7 ? 7 : batch;
    size_t kv_count=m->cache_width*capacity;
    if(ring_kv && (maximum_kv>SIZE_MAX/batch || nya_cuda_kv_layout_build(m,capacity,batch,NULL,&kv_count))) return NULL;
    size_t counts[] = {d * batch, (d > q ? d : q) * batch, d * batch, h * batch, h * batch, q * batch, heads * capacity * score_batch,
        kv_count, kv_count, m->tokenizer.vocabulary_size*logits_batch, rope_count / sizeof(float),
        maximum_kv*(ring_kv?batch:1), maximum_kv*(ring_kv?batch:1), 2, partial_count, batch>=4?interleaved_count:0};
    for (size_t i = 0; i < sizeof(counts)/sizeof(counts[0]); ++i) if (plan_add(&total, counts[i]) != 0) return NULL;
    for (size_t i = 0; i < m->tensor_count; ++i) {
        if (weights > SIZE_MAX - m->tensors[i].data_size) return NULL;
        weights += m->tensors[i].data_size;
    }
    if (m->tensor_count > NYA_CUDA_WEIGHT_LIMIT || weights > c->cache_limit)
        return plan_decline("model exceeds tensor-cache entries or configured weight budget");
    while (total > c->cache_limit - weights && batch > 1) {
        size_t smaller = batch / 2;
        for (size_t i = 0; i < 6; ++i) counts[i] = counts[i] / batch * smaller;
        counts[6] = heads * capacity * (all_tiled && smaller > 7 ? 7 : smaller);
        batch = smaller; total = 0;
        logits_batch=batch_logits?(batch<4?batch:4):1;
        counts[9]=m->tokenizer.vocabulary_size*logits_batch;
        counts[15]=batch>=4?interleaved_count:0;
        if(ring_kv) {
            if(nya_cuda_kv_layout_build(m,capacity,batch,NULL,&kv_count)) return NULL;
            counts[7]=counts[8]=kv_count;
            counts[11]=counts[12]=maximum_kv*batch;
        }
        for (size_t i = 0; i < sizeof(counts)/sizeof(counts[0]); ++i) if (plan_add(&total, counts[i])) return NULL;
    }
    if (total > c->cache_limit - weights) return plan_decline("weights plus bounded KV/scratch exceed the memory budget");
    nya_cuda_plan *p = calloc(1, sizeof(*p));
    if (p == NULL) return NULL;
    p->cuda = c; p->model = m; p->capacity = capacity; p->bytes = total;
    p->batch = 1; p->maximum_batch = batch;
    p->logits_batch=logits_batch;
    p->split_attention = has_split_attention;
    p->wide_split_attention = wide_split;
#ifdef NYA_FLASH_DECODING_PP
    p->use_fdpp = use_fdpp;
    const char *fdpp_trace = getenv("NYA_CUDA_FDPP_TRACE");
    p->trace_fdpp = fdpp_trace && !strcmp(fdpp_trace,"1");
#endif
    const char *graphs = getenv("NYA_CUDA_GRAPHS");
    p->use_graphs = graphs == NULL || strcmp(graphs, "0") != 0;
#ifdef NYA_FLASH_DECODING_PP
    if (p->trace_fdpp) p->use_graphs = 0;
#endif
    const char *residual = getenv("NYA_CUDA_RESIDUAL");
    p->fuse_residual = !residual || strcmp(residual,"0");
#ifdef NYA_PORTABLE_ATTENTION
    const char *portable_attention = getenv("NYA_CUDA_PORTABLE_ATTENTION");
    p->use_portable_attention = portable_attention && !strcmp(portable_attention,"1");
#endif
    const char *rope_pair = getenv("NYA_CUDA_ROPE_PAIR");
    p->fuse_rope_pair = !rope_pair || strcmp(rope_pair,"0");
    const char *qkv = getenv("NYA_CUDA_QKV");
    p->fuse_qkv = !qkv || strcmp(qkv,"0");
    const char *gemma_qkv = getenv("NYA_CUDA_GEMMA_QKV");
    p->gemma_qkv = gemma_qkv && !strcmp(gemma_qkv,"1");
    const char *gemma_gate = getenv("NYA_CUDA_GEMMA_GATE");
    p->gemma_gate = gemma_gate && !strcmp(gemma_gate,"1");
    const char *gate_up = getenv("NYA_CUDA_GATE_UP");
    p->fuse_gate_up = !gate_up || strcmp(gate_up,"0");
    const char *profile = getenv("NYA_CUDA_PROFILE");
    const char *profile_from=getenv("NYA_CUDA_PROFILE_FROM");
    int profile_valid=1;
    if(profile_from && profile && !strcmp(profile,"1")) {
        char *end;errno=0;unsigned long long first=strtoull(profile_from,&end,10);
        if(profile_from[0]<'0' || profile_from[0]>'9' || *end || errno || first>SIZE_MAX) {
            profile_valid=0;fprintf(stderr,"CUDA profiling disabled: invalid NYA_CUDA_PROFILE_FROM\n");
        } else p->profile_from=(size_t)first;
    }
    if (profile && !strcmp(profile, "1") && profile_valid) {
        p->timings = calloc(4096, sizeof(*p->timings));
        if (p->timings) {
            p->use_graphs = 0;
            p->matrix_profiles = calloc(64, sizeof(*p->matrix_profiles));
            p->attention_profiles = calloc(m->block_count,sizeof(*p->attention_profiles));
        }
    }
    p->fail_after = SIZE_MAX;
    p->fail_allocation_after = SIZE_MAX;
    const char *allocation_failure = getenv("NYA_CUDA_FAIL_ALLOCATION_AFTER");
    if (allocation_failure) {
        char *end; unsigned long long at = strtoull(allocation_failure, &end, 10);
        if (allocation_failure[0] >= '0' && allocation_failure[0] <= '9' && !*end && at <= SIZE_MAX)
            p->fail_allocation_after = (size_t)at;
    }
    /* Deterministic fault injection exercises complete-prefix replay without
       exhausting the user's GPU or depending on driver-specific OOM timing. */
    const char *failure_at = getenv("NYA_CUDA_FAIL_AFTER");
    if (failure_at != NULL) {
        char *end; unsigned long long at = strtoull(failure_at, &end, 10);
        if (failure_at[0] >= '0' && failure_at[0] <= '9' && !*end && at <= SIZE_MAX) p->fail_after = (size_t)at;
    }
    p->stats.weights_bytes = weights;
    p->stats.kv_bytes = kv_count * sizeof(float) * 2;
    p->stats.scratch_bytes = total - p->stats.kv_bytes;
    p->stats.prefill_batch = batch;
    p->tensors = calloc(m->tensor_count, sizeof(*p->tensors));
    if (p->tensors == NULL) { free(p->attention_profiles);free(p->matrix_profiles); free(p->timings); free(p); return NULL; }
    CUcontext previous;
    if (c->push(c->context) != CUDA_SUCCESS) { free(p->attention_profiles);free(p->matrix_profiles); free(p->timings); free(p->tensors); free(p); return NULL; }
    size_t start_weights = c->weight_count, free_bytes, total_bytes;
#ifdef NYA_WIDE_VALUE
    const char *wide_value=getenv("NYA_CUDA_WIDE_VALUE");
    if(!c->reference && wide_value && !strcmp(wide_value,"1")) {
        static const char *names[]={"nya_attention_vec16","nya_attention_vec16_split","nya_attention_vec16_ring","nya_attention_vec16_ring_split"};
        p->use_wide_value=1;
        for(unsigned i=0;i<4;++i) if(c->function_get(&p->wide_value[i],c->module,names[i])!=CUDA_SUCCESS) p->use_wide_value=0;
        const char *wide_batch=getenv("NYA_CUDA_WIDE_BATCH");
        p->use_wide_batch=p->use_wide_value && wide_batch && !strcmp(wide_batch,"1");
    }
#endif
    if(ring_kv) {
        p->kv_layout=calloc(m->block_count,sizeof(*p->kv_layout));
        if(!p->kv_layout || nya_cuda_kv_layout_build(m,capacity,batch,p->kv_layout,&kv_count)) goto failure;
        if(c->function_get(&p->rope_ring,c->module,"nya_rope_ring")!=CUDA_SUCCESS ||
            c->function_get(&p->attention_ring,c->module,"nya_attention_ring")!=CUDA_SUCCESS ||
            c->function_get(&p->attention_ring_split,c->module,"nya_attention_ring_split")!=CUDA_SUCCESS) goto failure;
    }
#ifdef NYA_PORTABLE_ATTENTION
    if (c->function_get(&p->portable_attention,c->module,"nya_attention_portable_f32") != CUDA_SUCCESS) goto failure;
#ifdef NYA_FLASH_DECODING_PP
    const char *fdpp_partial_name = "nya_fdpp_partial", *fdpp_combine_name = "nya_fdpp_combine";
    const char *fdpp_short_split = getenv("NYA_CUDA_FDPP_SHORT_SPLIT");
    p->fdpp_short_split = fdpp_short_split && !strcmp(fdpp_short_split,"1");
#ifdef NYA_FLASH_DECODING_PP_WARP
    const char *fdpp_portable = getenv("NYA_CUDA_FDPP_PORTABLE");
    if (!fdpp_portable || strcmp(fdpp_portable,"1")) {
        fdpp_partial_name = "nya_fdpp_partial_warp";
        fdpp_combine_name = "nya_fdpp_combine_warp";
    }
#endif
    if (c->function_get(&p->fdpp_partial,c->module,fdpp_partial_name) != CUDA_SUCCESS ||
        c->function_get(&p->fdpp_combine,c->module,fdpp_combine_name) != CUDA_SUCCESS) goto failure;
#ifdef NYA_FLASH_DECODING_PP_SPECIALIZED
    p->fdpp_partial_64 = p->fdpp_partial_128 = p->fdpp_partial;
    const char *fdpp_generic = getenv("NYA_CUDA_FDPP_GENERIC");
    if ((!fdpp_generic || strcmp(fdpp_generic,"1")) && !strcmp(fdpp_partial_name,"nya_fdpp_partial_warp")) {
        if (c->function_get(&p->fdpp_partial_64,c->module,"nya_fdpp_w64_u4_r32") != CUDA_SUCCESS ||
            c->function_get(&p->fdpp_partial_128,c->module,"nya_fdpp_w128_u1_r32") != CUDA_SUCCESS) goto failure;
        const char *fdpp_direct = getenv("NYA_CUDA_FDPP_DIRECT");
        if (!fdpp_direct || strcmp(fdpp_direct,"0")) {
            if (c->function_get(&p->fdpp_direct_64,c->module,"nya_fdpp_w64_direct") != CUDA_SUCCESS ||
                c->function_get(&p->fdpp_direct_128,c->module,"nya_fdpp_w128_direct") != CUDA_SUCCESS) goto failure;
        }
    }
#endif
#endif
#endif
    size_t missing = 0;
    for (size_t i = 0; i < m->tensor_count; ++i) {
        const nya_llm_tensor *t = &m->tensors[i];
        for (size_t j = 0; j < c->weight_count; ++j) {
            if (c->weights[j].source == t->data && c->weights[j].bytes == t->data_size) { p->tensors[i] = c->weights[j].memory; break; }
        }
        if (!p->tensors[i]) missing += t->data_size;
    }
    if (c->memory_info(&free_bytes, &total_bytes) != CUDA_SUCCESS) goto failure;
    size_t reserve = free_bytes / 10;
    if (reserve < 256U * 1024U * 1024U) reserve = 256U * 1024U * 1024U;
    if (free_bytes <= reserve || total > free_bytes - reserve || missing > free_bytes - reserve - total) goto failure;
    /* Reserve KV/scratch before weights. A plan is accepted as a whole, so OOM
       cannot silently turn individual layers into PCIe/CPU round trips. */
    if (plan_allocate(p, &p->storage, total) != CUDA_SUCCESS) goto failure;
    /* Optional matrix scratch is admitted after the mandatory request budget.
       Failure leaves the resident native path intact, rather than rejecting a
       model which otherwise fits. Reserve it once; never allocate per layer. */
    size_t expansion = 0;
    for (size_t i = 0; i < m->block_count; ++i) {
        const nya_llm_layer *w = &m->layers[i];
        const nya_llm_tensor *matrices[] = {w->query,w->key,w->value,w->attention_output,
            w->feed_forward_gate,w->feed_forward_up,w->feed_forward_down};
        for (size_t j = 0; j < sizeof(matrices)/sizeof(matrices[0]); ++j) {
            const nya_llm_tensor *t = matrices[j];
            if (!t || t->dimension_count != 2) continue;
            size_t needed = nya_blas_bytes(c, (size_t)t->dimensions[1], (size_t)t->dimensions[0], batch);
            if (needed > expansion) expansion = needed;
        }
    }
    size_t optional = expansion ? expansion + NYA_BLAS_WORKSPACE : 0;
    if (optional && optional <= c->cache_limit - weights - total &&
        optional <= free_bytes - reserve - total - missing &&
        plan_allocate(p, &p->blas_storage, optional) == CUDA_SUCCESS) {
        p->blas_expansion = expansion;
        p->stats.scratch_bytes += optional;
        p->stats.external_matmul_bytes = optional;
        p->stats.external_matmul_kind = 1;
#ifdef NYA_ENABLE_CUTLASS
        if (c->cutlass.library) p->stats.external_matmul_kind = 2;
#endif
        p->bytes += optional;
    }
    CUdeviceptr *buffers[] = {&p->x, &p->norm, &p->residual, &p->gate, &p->up, &p->q,
        &p->scores, &p->keys, &p->values, &p->logits, &p->frequencies,
        &p->decode_key, &p->decode_value, &p->request, &p->attention_parts, &p->interleaved_input};
    size_t offset = 0;
    for (size_t i = 0; i < sizeof(counts)/sizeof(counts[0]); ++i) {
        *buffers[i] = p->storage + offset; offset += counts[i] * sizeof(float);
    }
    if(counts[15]) {
        /* Three reserved floats cover alignment padding without escaping the
           admitted scratch allocation. Conversion and dot share one stream. */
        p->interleaved_input=(p->interleaved_input+15)&~(CUdeviceptr)15;
        p->interleaved_columns=(counts[15]-3)/4;
    } else p->interleaved_input=0;
    for (size_t i = 0; i < m->tensor_count; ++i) {
        if (p->tensors[i]) continue;
        if (c->weight_count == NYA_CUDA_WEIGHT_LIMIT) goto failure;
        const nya_llm_tensor *t = &m->tensors[i];
        nya_cuda_weight *w = &c->weights[c->weight_count];
        if (plan_allocate(p, &w->memory, t->data_size) != CUDA_SUCCESS) goto failure;
        w->source = t->data; w->bytes = t->data_size; w->type = t->type;
        w->columns = (size_t)t->dimensions[0]; w->rows = 1;
        for (uint32_t dim = 1; dim < t->dimension_count; ++dim) w->rows *= (size_t)t->dimensions[dim];
        ++c->weight_count; c->cached_bytes += t->data_size;
        if (c->upload(w->memory, t->data, t->data_size) != CUDA_SUCCESS) goto failure;
        p->tensors[i] = w->memory;
    }
    float *freq = malloc(rope_count);
    if (freq == NULL) goto failure;
    offset = 0;
    for (size_t i = 0; i < m->block_count; ++i) {
        const nya_llm_layer *w = &m->layers[i];
        size_t pairs = w->head_dimension / 2;
        float base = w->sliding_window ? m->sliding_rope_base : m->rope_frequency_base;
        for (size_t j = 0; j < pairs; ++j) {
            float factor = w->rope_factors ? nya_llm_tensor_value(w->rope_factors, j) : 1;
            freq[offset++] = powf(base, -(float)j / (float)pairs) / factor;
        }
    }
    int uploaded = c->upload(p->frequencies, freq, rope_count) == CUDA_SUCCESS;
    free(freq);
    if (!uploaded || c->stream_sync(NULL) != CUDA_SUCCESS) goto failure;
    if (c->function_get(&p->gather, c->module, "nya_gather") != CUDA_SUCCESS ||
        c->function_get(&p->decode_request, c->module, "nya_decode_request") != CUDA_SUCCESS ||
        c->function_get(&p->gather_batch, c->module, "nya_gather_batch") != CUDA_SUCCESS ||
        c->function_get(&p->dot_add[0], c->module, "nya_add_12") != CUDA_SUCCESS ||
        c->function_get(&p->dot_add[1], c->module, "nya_add_14") != CUDA_SUCCESS ||
        c->function_get(&p->qkv[0], c->module, "nya_qkv_12") != CUDA_SUCCESS ||
        c->function_get(&p->qkv[1], c->module, "nya_qkv_14") != CUDA_SUCCESS ||
        c->function_get(&p->norm_op, c->module, "nya_norm") != CUDA_SUCCESS ||
        c->function_get(&p->add_norm, c->module, "nya_add_norm") != CUDA_SUCCESS ||
        c->function_get(&p->rope, c->module, "nya_rope") != CUDA_SUCCESS ||
        c->function_get(&p->rope_pair, c->module, "nya_rope_pair") != CUDA_SUCCESS ||
        c->function_get(&p->vector, c->module, "nya_vector") != CUDA_SUCCESS ||
        c->function_get(&p->attention, c->module, c->reference ? "nya_attention_reference" : "nya_attention") != CUDA_SUCCESS ||
        c->function_get(&p->attention_prefill, c->module, "nya_attention_prefill") != CUDA_SUCCESS ||
        c->function_get(&p->attention_split, c->module, "nya_attention_split") != CUDA_SUCCESS ||
        c->function_get(&p->attention_combine, c->module, "nya_attention_combine") != CUDA_SUCCESS) goto failure;
    if (c->integer_q6_add) p->dot_add[1] = c->integer_q6_add;
    if (c->integer_add) {
        p->dot_add[0] = c->integer_add;
        p->qkv[0] = c->integer_qkv[0];
        p->qkv[1] = c->integer_qkv[1];
    }
    plan_byte_weights(p);
    if (c->pop(&previous) != CUDA_SUCCESS) { p->failed = 1; cuda_plan_free(p); return NULL; }
    return p;
failure:
    if (getenv("NYA_CUDA_DEBUG")) fprintf(stderr, "CUDA resident allocation/setup failed; rolling back new allocations\n");
    /* Roll back only allocations introduced by this attempted plan. Existing
       cached weights may still belong to another session on this model. */
    while (c->weight_count > start_weights) {
        nya_cuda_weight *w = &c->weights[--c->weight_count];
        c->deallocate(w->memory); c->cached_bytes -= w->bytes; memset(w, 0, sizeof(*w));
    }
    c->pop(&previous); cuda_plan_free(p); return NULL;
}

static int plan_grid_shared(nya_cuda_plan *p, CUfunction f, size_t gx, size_t gy, unsigned bx, unsigned by, unsigned shared_bytes, void **args)
{
    ++p->stats.kernel_launches;
    if (gx > (size_t)p->cuda->grid_limit || gy > 65535) return -1;
    nya_cuda_timing *timing = NULL;
    if (p->timings && p->profile_active) {
        if (p->timing_count == 4096) plan_profile_flush(p);
        timing = &p->timings[p->timing_count];
        timing->start = NULL; timing->end = NULL;
        timing->matrix = p->matrix_profile;timing->attention=0;
        if (p->cuda->event_create(&timing->start, 0) != CUDA_SUCCESS) return -1;
        if (p->cuda->event_create(&timing->end, 0) != CUDA_SUCCESS) { p->cuda->event_free(timing->start); return -1; }
        timing->kind = f == p->norm_op || f == p->add_norm ? 1U : f == p->rope || f == p->rope_ring || f == p->rope_pair ? 2U : f == p->vector ? 3U : f == p->attention_ring || f == p->attention_ring_split || f == p->attention || f == p->attention_prefill || f == p->attention_split || f == p->attention_combine ? 4U : f == p->gather || f == p->gather_batch ? 5U : 0U;
#ifdef NYA_PORTABLE_ATTENTION
        if (f == p->portable_attention) timing->kind = 4U;
#ifdef NYA_FLASH_DECODING_PP
        if (f == p->fdpp_partial || f == p->fdpp_combine) timing->kind = 4U;
#ifdef NYA_FLASH_DECODING_PP_SPECIALIZED
        if (f == p->fdpp_partial_64 || f == p->fdpp_partial_128 ||
            f == p->fdpp_direct_64 || f == p->fdpp_direct_128) timing->kind = 4U;
#endif
#endif
#endif
        for(unsigned i=0;i<4;++i) if(f==p->wide_value[i]) timing->kind=4U;
        if(timing->kind==4 && p->attention_profiles) timing->attention=p->profile_layer;
        ++p->timing_count;
        if (p->cuda->event_record(timing->start, p->cuda->stream) != CUDA_SUCCESS) return -1;
    }
    CUresult launched = p->cuda->launch(f, (unsigned)gx, (unsigned)gy, 1, bx, by, 1, shared_bytes, p->cuda->stream, args, NULL);
    int result = launched == CUDA_SUCCESS ? 0 : -1;
    if (result) fprintf(stderr, "CUDA resident kernel launch failed: driver status %d\n", (int)launched);
    if (timing && p->cuda->event_record(timing->end, p->cuda->stream) != CUDA_SUCCESS) result = -1;
    return result;
}
static int plan_grid(nya_cuda_plan *p, CUfunction f, size_t gx, size_t gy, unsigned bx, unsigned by, void **args)
{ return plan_grid_shared(p, f, gx, gy, bx, by, 0, args); }
static int plan_launch(nya_cuda_plan *p, CUfunction f, size_t blocks, void **args)
{ return plan_grid(p, f, blocks, 1, 256, 1, args); }
static CUfunction plan_interleaved_kernel(const nya_cuda_plan *p,unsigned type,size_t columns)
{
    return p->batch==4 && p->interleaved_input && columns<=p->interleaved_columns && type<31 ?
        p->cuda->interleaved_reuse[type]:NULL;
}
static int plan_matvec_impl(nya_cuda_plan *p, CUdeviceptr out, const nya_llm_tensor *t, CUdeviceptr in, size_t rows, size_t columns)
{
    CUdeviceptr matrix = plan_tensor(p, t);
    unsigned long long cols = columns, stride = t->data_size / rows;
    unsigned type = t->type;
    if (p->batch > 1) {
        if (p->blas_storage && nya_blas_bytes(p->cuda, rows, columns, p->batch)) {
            /* cuBLAS may dispatch more than one internal kernel. Count logical
               expansion/GEMM dispatches; the documentation identifies this
               explicitly instead of claiming an exact vendor kernel count. */
            size_t chunk = p->blas_expansion / sizeof(float) / columns;
            if (type != 0) p->stats.kernel_launches += (rows + chunk - 1)/chunk;
            p->stats.external_matmul_calls += (rows + chunk - 1)/chunk;
            nya_cuda_timing *timing = NULL;
            if (p->timings && p->profile_active) {
                if (p->timing_count == 4096) plan_profile_flush(p);
                timing = &p->timings[p->timing_count];
                timing->start = NULL; timing->end = NULL; timing->kind = 0;
                timing->matrix = p->matrix_profile;timing->attention=0;
                if (p->cuda->event_create(&timing->start, 0) != CUDA_SUCCESS) return -1;
                if (p->cuda->event_create(&timing->end, 0) != CUDA_SUCCESS) { p->cuda->event_free(timing->start); return -1; }
                ++p->timing_count;
                if (p->cuda->event_record(timing->start, p->cuda->stream) != CUDA_SUCCESS) return -1;
            }
            int result = nya_blas_multiply(p->cuda, matrix, in, out, rows, columns, type, (size_t)stride,
                p->batch, p->blas_storage + NYA_BLAS_WORKSPACE, p->blas_expansion, p->blas_storage,
                &p->stats.cutlass_matmul_calls);
            if (timing && p->cuda->event_record(timing->end, p->cuda->stream) != CUDA_SUCCESS) return -1;
            return result;
        }
        unsigned r = (unsigned)rows, batch = (unsigned)p->batch;
        void *gemm_args[] = {&matrix, &in, &out, &cols, &stride, &r, &batch};
        CUfunction interleaved=plan_interleaved_kernel(p,type,columns);
        if(interleaved) {
            void *convert_args[]={&in,&p->interleaved_input,&cols};
            if(plan_launch(p,p->cuda->interleave_four,(columns+255)/256,convert_args))return -1;
            gemm_args[1]=&p->interleaved_input;
            int result=plan_grid(p,interleaved,(r+7)/8,1,256,1,gemm_args);
            if(!result)++p->interleaved_calls;
            return result;
        }
        unsigned grid_y;
        CUfunction narrow=nya_cuda_narrow_kernel(p->cuda,type,batch,&grid_y);
        if(narrow) return plan_grid(p,narrow,(r+7)/8,grid_y,256,1,gemm_args);
        unsigned tile = batch >= 32 && r >= 32 ? p->cuda->gemm_tile : 32, bx = tile == 64 ? 16 : 32;
        CUfunction kernel = tile == 64 ? p->cuda->gemm_large[type] : p->cuda->gemm[type];
        return plan_grid(p, kernel, (r + tile-1) / tile, (batch + tile-1) / tile, bx, 256/bx, gemm_args);
    }
    void *args[] = {&matrix, &in, &out, &cols, &stride, &type};
    if (p->cuda->reference) return plan_launch(p, p->cuda->function, rows, args);
    unsigned long long row_count = rows;
    args[5] = &row_count;
    CUfunction function=p->cuda->dot[type];
    CUdeviceptr packed=p->cuda->byte_dot ? plan_byte_tensor(p,t,rows,columns) : 0;
    if (packed) { matrix=packed; stride=columns/256*160; function=p->cuda->byte_dot; }
    return plan_launch(p, function, (rows + 7) / 8, args);
}
/* One profile entry denotes one dispatched operation; gate/up represents
   both projections and their activation, so its count is one per layer. */
static unsigned plan_matrix_slot(nya_cuda_plan *p, size_t rows, size_t columns,
    unsigned type, unsigned path)
{
    if (!p->matrix_profiles || !p->profile_active) return 0;
    unsigned slot = 0;
    for (; slot < p->matrix_profile_count; ++slot) {
        const nya_cuda_matrix_profile *m = &p->matrix_profiles[slot];
        if (m->rows == rows && m->columns == columns && m->batch == p->batch && m->type == type && m->path == path) break;
    }
    if (slot == p->matrix_profile_count && slot < 63) {
        p->matrix_profiles[slot] = (nya_cuda_matrix_profile){rows, columns, p->batch, type, path, 0, 0};
        ++p->matrix_profile_count;
    } else if (slot == 63) p->matrix_profiles[slot].path = 4;
    return slot + 1;
}
static int plan_matvec(nya_cuda_plan *p, CUdeviceptr out, const nya_llm_tensor *t, CUdeviceptr in, size_t rows, size_t columns)
{
    if (!p->matrix_profiles) return plan_matvec_impl(p,out,t,in,rows,columns);
    unsigned grid_y;
    CUfunction narrow=nya_cuda_narrow_kernel(p->cuda,t->type,(unsigned)p->batch,&grid_y);
    unsigned path = p->batch == 1 ? (unsigned)(p->cuda->reference != 0) :
        p->blas_storage && nya_blas_bytes(p->cuda, rows, columns, p->batch) ? 3U :
        plan_interleaved_kernel(p,t->type,columns) ? 10U : narrow ? (grid_y==1 ? 9U : 8U) : 2U;
    p->matrix_profile = plan_matrix_slot(p,rows,columns,t->type,path);
    int result = plan_matvec_impl(p,out,t,in,rows,columns);
    p->matrix_profile = 0;
    return result;
}
static int plan_gate_up(nya_cuda_plan *p, const nya_llm_tensor *gate,
    const nya_llm_tensor *up, size_t rows, size_t columns)
{
    CUdeviceptr g=plan_tensor(p,gate), u=plan_tensor(p,up);
    unsigned long long cols=columns, stride=gate->data_size/rows, count=rows;
    void *args[]={&g,&u,&p->norm,&p->gate,&cols,&stride,&count};
    p->matrix_profile=plan_matrix_slot(p,rows,columns,12,5);
    CUfunction function=p->cuda->gate_up_q4;
    if (p->cuda->gate_up_2048 && columns==2048 && stride==1152 && up->data_size/rows==1152)
        function=p->cuda->gate_up_2048;
    if (p->cuda->byte_gate && p->byte_tensors && columns==2048 && stride==1152 && up->data_size/rows==1152) {
        CUdeviceptr packed_gate=p->byte_tensors[(size_t)(gate-p->model->tensors)];
        CUdeviceptr packed_up=p->byte_tensors[(size_t)(up-p->model->tensors)];
        if (packed_gate && packed_up) { g=packed_gate; u=packed_up; stride=1280; function=p->cuda->byte_gate; }
    }
    int result=plan_launch(p,function,(rows+7)/8,args);
    p->matrix_profile=0;
    return result;
}
static int plan_matvec_add(nya_cuda_plan *p, CUdeviceptr out, const nya_llm_tensor *t,
    CUdeviceptr in, size_t rows, size_t columns)
{
    CUdeviceptr matrix=plan_tensor(p,t);
    unsigned long long cols=columns, stride=t->data_size/rows, count=rows;
    void *args[]={&matrix,&in,&out,&cols,&stride,&count};
    p->matrix_profile=plan_matrix_slot(p,rows,columns,t->type,7);
    CUfunction function=p->dot_add[t->type==14];
    CUdeviceptr packed=p->cuda->byte_add ? plan_byte_tensor(p,t,rows,columns) : 0;
    if (packed) { matrix=packed; stride=columns/256*160; function=p->cuda->byte_add; }
    int result=plan_launch(p,function,(rows+7)/8,args);
    p->matrix_profile=0;
    return result;
}
static int plan_qkv(nya_cuda_plan *p, const nya_llm_layer *w, CUdeviceptr key,
    CUdeviceptr value, size_t qrows, size_t krows, size_t columns)
{
    CUdeviceptr q=plan_tensor(p,w->query), k=plan_tensor(p,w->key), v=plan_tensor(p,w->value);
    unsigned long long cols=columns, qs=w->query->data_size/qrows, vs=w->value->data_size/krows;
    unsigned long long qr=qrows, kr=krows;
    void *args[]={&q,&k,&v,&p->norm,&p->q,&key,&value,&cols,&qs,&vs,&qr,&kr};
    p->matrix_profile=plan_matrix_slot(p,qrows+2*krows,columns,w->value->type,6);
    unsigned q6=w->value->type==14;
    CUfunction function=p->qkv[q6];
    if (p->cuda->byte_qkv[q6]) {
        CUdeviceptr pq=plan_byte_tensor(p,w->query,qrows,columns);
        CUdeviceptr pk=plan_byte_tensor(p,w->key,krows,columns);
        CUdeviceptr pv=q6 ? v : plan_byte_tensor(p,w->value,krows,columns);
        if (pq && pk && pv) {
            q=pq; k=pk; v=pv; qs=columns/256*160;
            if (!q6) vs=qs;
            function=p->cuda->byte_qkv[q6];
        }
    }
    int result=plan_launch(p,function,(qrows+7)/8+2*((krows+7)/8),args);
    p->matrix_profile=0;
    return result;
}
static int plan_norm(nya_cuda_plan *p, CUdeviceptr out, CUdeviceptr in, const nya_llm_tensor *t, size_t n, size_t rows)
{
    CUdeviceptr weight = plan_tensor(p, t);
    unsigned type = t ? t->type : 0;
    unsigned long long length = n;
    float eps = p->model->norm_epsilon;
    void *args[] = {&in, &out, &weight, &length, &type, &eps};
    return plan_launch(p, p->norm_op, rows * p->batch, args);
}
static int plan_add_norm(nya_cuda_plan *p, CUdeviceptr out, CUdeviceptr x, CUdeviceptr residual,
    const nya_llm_tensor *t, size_t n)
{
    CUdeviceptr weight = plan_tensor(p,t);
    unsigned type = t ? t->type : 0;
    unsigned long long length = n;
    float epsilon = p->model->norm_epsilon;
    void *args[] = {&x,&out,&residual,&weight,&length,&type,&epsilon};
    return plan_launch(p,p->add_norm,p->batch,args);
}
static int plan_vector(nya_cuda_plan *p, CUdeviceptr out, CUdeviceptr x, CUdeviceptr y, size_t n, unsigned kind, float scale)
{
    unsigned long long length = n * p->batch;
    void *args[] = {&out, &x, &y, &length, &kind, &scale};
    return plan_launch(p, p->vector, ((size_t)length + 255) / 256, args);
}
static int plan_rope(nya_cuda_plan *p, CUdeviceptr x, CUdeviceptr frequencies, unsigned hd, unsigned heads, unsigned position,
    CUdeviceptr keys, CUdeviceptr values, CUdeviceptr value, unsigned ring_slots)
{
    unsigned gemma = (unsigned)p->model->is_gemma;
    unsigned batch = (unsigned)p->batch;
    void *args[] = {&x, &frequencies, &hd, &heads, &position, &gemma, &batch, &p->active_request, &keys, &values, &value, &ring_slots};
    return plan_launch(p, ring_slots ? p->rope_ring : p->rope, ((size_t)hd / 2 * heads * batch + 255) / 256, args);
}

static int plan_rope_pair(nya_cuda_plan *p, CUdeviceptr key, CUdeviceptr frequencies,
    unsigned hd, unsigned heads, unsigned kvheads, unsigned position,
    CUdeviceptr keys, CUdeviceptr values, CUdeviceptr value)
{
    void *args[] = {&p->q, &key, &frequencies, &hd, &heads, &kvheads, &position,
        &p->active_request, &keys, &values, &value};
    return plan_launch(p, p->rope_pair, ((size_t)hd / 2 * (heads + (size_t)kvheads) + 255) / 256, args);
}

static int plan_read_logits_row(nya_cuda_plan *p,float *output,size_t row)
{
    const nya_llm_context *m=p->model;nya_cuda_context *c=p->cuda;
    ++p->stats.synchronizations;
    if(c->stream_sync(c->stream)!=CUDA_SUCCESS) return -1;
    ++p->stats.downloads;
    if(row>=p->logits_batch) return -1;
    if(c->download(output,p->logits+row*m->tokenizer.vocabulary_size*sizeof(float),m->tokenizer.vocabulary_size*sizeof(float))!=CUDA_SUCCESS) return -1;
    for(size_t i=0;i<m->tokenizer.vocabulary_size;++i) if(!isfinite(output[i])) return -1;
    for(size_t i=0;i<m->tokenizer.suppressed_count;++i) output[m->tokenizer.suppressed_tokens[i]]=-INFINITY;
    return 0;
}
static int plan_read_logits(nya_cuda_plan *p,float *output)
{ return plan_read_logits_row(p,output,0); }

static int cuda_plan_batch(void *opaque, const uint32_t *tokens, size_t count, size_t position, int logits, float *output)
{
    nya_cuda_plan *p = opaque;
    const nya_llm_context *m = p->model;
    nya_cuda_context *c = p->cuda;
    if (p->failed || count == 0 || count > p->maximum_batch || position >= p->capacity || count > p->capacity - position ||
        (position != 0 && position > p->valid) || (logits && output == NULL)) return -1;
    for (size_t i = 0; i < count; ++i) if (tokens[i] >= m->tokenizer.vocabulary_size) return -1;
    if (position >= p->fail_after || count > p->fail_after - position) { p->failed = 1; return -1; }
    if(p->kv_layout && position) for(size_t i=0;i<m->block_count;++i) {
        size_t slots=p->kv_layout[i].slots,window=m->layers[i].sliding_window;
        size_t first=window && position>=window ? position-window+1 : 0;
        if(slots<p->capacity && !nya_cuda_kv_retained(p->kv_high_water,slots,first)) return -1;
    }
    p->batch = count;
    p->profile_active=p->timings && position>=p->profile_from;
    CUcontext previous;
    if (c->push(c->context) != CUDA_SUCCESS) { p->failed = 1; return -1; }
    /* Single-token/logits execution retains at most two fixed topologies:
       ordinary attention and four-way partitioned attention after 256 tokens.
       Reset/rewind select the matching cached graph; neither graph embeds a
       stale token or position. Prefill remains independently batched. */
    int capturing = 0;
    int separate_long_graph = p->split_attention;
#ifdef NYA_FLASH_DECODING_PP
    /* Selective FDPP changes topology at this boundary even when native
       split attention is disabled or every layer uses a small window. */
    if (p->use_fdpp == 2) separate_long_graph = 1;
#endif
    unsigned graph_index = separate_long_graph && count == 1 && position >= 256 ? 1U : 0U;
    CUgraph graph = NULL;
    unsigned long long capture_start = 0;
    p->active_request = 0;
    if (p->use_graphs && count == 1 && logits) {
        unsigned token = tokens[0], pos = (unsigned)position;
        void *args[] = {&p->request, &token, &pos};
        if (plan_grid(p, p->decode_request, 1, 1, 1, 1, args)) goto failure;
        p->active_request = p->request;
        if (p->executable[graph_index]) {
            if (c->graph_launch(p->executable[graph_index], c->stream) != CUDA_SUCCESS) goto failure;
            p->stats.kernel_launches += p->graph_kernels[graph_index];
            ++p->stats.graph_replays;
            goto read_logits;
        }
        capture_start = p->stats.kernel_launches;
        if (c->capture_begin(c->stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) != CUDA_SUCCESS) goto failure;
        capturing = 1;
    }
    size_t d = m->embedding_length, freq_offset = 0;
    CUdeviceptr embedding = plan_tensor(p, m->token_embedding);
    unsigned long long n = d, row_bytes = m->token_embedding->data_size / m->tokenizer.vocabulary_size;
    unsigned type = m->token_embedding->type;
    float scale = m->is_gemma ? sqrtf((float)d) : 1;
#define RUN(expression) do { if ((expression) != 0) goto failure; } while (0)
    if (count > 1) {
        struct { uint32_t values[512]; } packet = {{0}};
        memcpy(packet.values,tokens,count*sizeof(*tokens));
        unsigned batch = (unsigned)count;
        void *gather[] = {&embedding,&p->x,&n,&row_bytes,&type,&packet,&batch,&scale};
        RUN(plan_launch(p,p->gather_batch,(d*count+255)/256,gather));
    } else {
        CUdeviceptr x = p->x;
        uint32_t token = tokens[0];
        void *gather[] = {&embedding, &x, &n, &row_bytes, &type, &token, &scale, &p->active_request};
        RUN(plan_launch(p, p->gather, (d + 255) / 256, gather));
    }
    for (size_t layer = 0; layer < m->block_count; ++layer) {
        const nya_llm_layer *w = &m->layers[layer];
        size_t hd = w->head_dimension, heads = w->head_count, kvheads = w->kv_head_count;
        size_t qdim = hd * heads, kvdim = hd * kvheads, h = w->hidden_length;
        size_t cache_offset=p->kv_layout?p->kv_layout[layer].offset:w->cache_offset*p->capacity;
        unsigned ring_slots=p->kv_layout && p->kv_layout[layer].slots<p->capacity ? (unsigned)p->kv_layout[layer].slots : 0;
        CUdeviceptr keys = p->keys + cache_offset * sizeof(float);
        CUdeviceptr values = p->values + cache_offset * sizeof(float);
        CUdeviceptr key = keys + position * kvdim * sizeof(float), value = values + position * kvdim * sizeof(float);
        if (p->active_request || ring_slots) { key = p->decode_key; value = p->decode_value; }
        CUdeviceptr frequencies = p->frequencies + freq_offset * sizeof(float);
        freq_offset += hd / 2;
        RUN(plan_norm(p, p->norm, p->x, w->attention_norm, d, 1));
        int rope_pair = !ring_slots && count == 1 && p->fuse_rope_pair && !m->is_gemma &&
            !c->reference && w->kv_source == layer &&
            kvdim <= UINT32_MAX && qdim <= UINT32_MAX - kvdim;
        int qkv = count == 1 && p->fuse_qkv && (!m->is_gemma || p->gemma_qkv) && !c->reference &&
            w->kv_source == layer && w->key && w->value && w->query->type == 12 &&
            w->key->type == 12 && (w->value->type == 12 || w->value->type == 14) &&
            w->query->data_size/qdim == w->key->data_size/kvdim;
        if (qkv) RUN(plan_qkv(p,w,key,value,qdim,kvdim,d));
        else RUN(plan_matvec(p, p->q, w->query, p->norm, qdim, d));
        if (m->is_gemma) RUN(plan_norm(p, p->q, p->q, w->query_norm, hd, heads));
        if (w->kv_source == layer) {
            if (!qkv) {
            RUN(plan_matvec(p, key, w->key, p->norm, kvdim, d));
            if (w->value) RUN(plan_matvec(p, value, w->value, p->norm, kvdim, d));
            else RUN(plan_vector(p, value, key, 0, kvdim, 0, 1));
            }
            if (m->is_gemma) {
                RUN(plan_norm(p, key, key, w->key_norm, hd, kvheads));
                RUN(plan_norm(p, value, value, NULL, hd, kvheads));
            }
            if (rope_pair) RUN(plan_rope_pair(p, key, frequencies, (unsigned)hd, (unsigned)heads,
                (unsigned)kvheads, (unsigned)position,
                p->active_request ? keys : 0, p->active_request ? values : 0, value));
            else RUN(plan_rope(p, key, frequencies, (unsigned)hd, (unsigned)kvheads, (unsigned)position,
                p->active_request || ring_slots ? keys : 0, p->active_request || ring_slots ? values : 0, value, ring_slots));
        }
        if (!rope_pair) RUN(plan_rope(p, p->q, frequencies, (unsigned)hd, (unsigned)heads, (unsigned)position, 0, 0, 0, 0));
        unsigned width = (unsigned)hd, nh = (unsigned)heads, nk = (unsigned)kvheads;
        unsigned window = w->sliding_window;
        unsigned end = (unsigned)position, capacity = (unsigned)p->capacity;
        scale = m->is_gemma ? 1 : 1 / sqrtf((float)hd);
        p->profile_layer=layer+1;
        void *attention[] = {&p->q, &keys, &values, &p->norm, &p->scores, &width, &nh, &nk, &window, &end, &capacity, &scale, &p->active_request};
        /* The short graph only addresses positions 0..255. Those physical
           rows are still linear when the ring has at least 256 slots, so keep
           the existing short FDPP/portable math and performance. The separate
           long topology is mandatory: graph_index==0 alone is not a bound. */
        int linear_ring_decode=count==1 && separate_long_graph && !graph_index && ring_slots>=256;
        /* Narrow verification owns independent score/output rows. Split partials
           are single-query storage, so batching must use the unsplit kernel. */
        int wide_batch=p->use_wide_batch && (count==2 || count==4) && position>=256;
        if(p->use_wide_value && ((count==1 && graph_index) || wide_batch) && (width==256 || width==512)) {
            int split=count==1 && p->split_attention && p->wide_split_attention && (!window || window>256);
            void *wide[]={&p->q,&keys,&values,&p->norm,&p->scores,&width,&nh,&nk,&window,&end,&capacity,&scale,&p->active_request,&ring_slots};
            if(split) wide[3]=&p->attention_parts;
            RUN(plan_grid(p,p->wide_value[(ring_slots?2:0)+(split?1:0)],heads,split?4:count,256,1,wide));
            if(wide_batch) ++p->batched_wide_calls;
            if(split) {
                void *combine[]={&p->attention_parts,&p->norm,&width};
                RUN(plan_grid(p,p->attention_combine,heads,1,width,1,combine));
            }
        } else if(ring_slots && !linear_ring_decode) {
            void *ring[]={&p->q,&keys,&values,&p->norm,&p->scores,&width,&nh,&nk,&window,&end,&capacity,&scale,&p->active_request,&ring_slots};
            if(p->split_attention && graph_index && (width==64 || width==128 || (p->wide_split_attention && (width==256 || width==512))) && window>256) {
                ring[3]=&p->attention_parts;
                RUN(plan_grid(p,p->attention_ring_split,heads,4,256,1,ring));
                void *combine[]={&p->attention_parts,&p->norm,&width};
                RUN(plan_grid(p,p->attention_combine,heads,1,width>128?width:128,1,combine));
            } else RUN(plan_grid(p,p->attention_ring,heads,count,256,1,ring));
        } else
#ifdef NYA_FLASH_DECODING_PP
        if (p->use_fdpp && (p->use_fdpp != 2 || !graph_index) && count == 1 && width <= 256) {
            /* Partition choice is fixed within each captured graph. Device
               request positions determine bounds on every graph replay. */
            unsigned partitions = graph_index || p->fdpp_short_split || p->use_fdpp == 2 ? 4U : 1U;
#ifdef NYA_FLASH_DECODING_PP_SPECIALIZED
            CUfunction direct = width == 64 ? p->fdpp_direct_64 : width == 128 ? p->fdpp_direct_128 : NULL;
            if (partitions == 1 && direct && !p->trace_fdpp) {
                void *args[]={&p->q,&keys,&values,&p->norm,&width,&nh,&nk,&window,&end,
                    &scale,&p->active_request,&partitions,&p->scores,&capacity};
                RUN(plan_grid(p,direct,heads,1,256,1,args));
            } else
#endif
            {
            void *partial[]={&p->q,&keys,&values,&p->attention_parts,&width,&nh,&nk,
                &window,&end,&scale,&p->active_request,&partitions};
            CUfunction partial_function = p->fdpp_partial;
#ifdef NYA_FLASH_DECODING_PP_SPECIALIZED
            if (width == 64) partial_function = p->fdpp_partial_64;
            else if (width == 128) partial_function = p->fdpp_partial_128;
#endif
            RUN(plan_grid(p,partial_function,heads,partitions,256,1,partial));
            if (p->trace_fdpp) {
                size_t partial_bytes = heads*partitions*(width+2)*sizeof(float);
                float *host_parts = malloc(partial_bytes);
                if (!host_parts) goto failure;
                CUresult traced = c->stream_sync(c->stream);
                if (traced == CUDA_SUCCESS) traced = c->download(host_parts,p->attention_parts,partial_bytes);
                unsigned retries = 0;
                if (traced == CUDA_SUCCESS) for (size_t head = 0; head < heads; ++head) {
                    int retry = 0;
                    for (unsigned part = 0; part < partitions; ++part)
                        retry |= host_parts[(head*partitions+part)*(width+2)+width+1] != 0;
                    retries += retry != 0;
                }
                free(host_parts);
                if (traced != CUDA_SUCCESS) goto failure;
                ++p->stats.downloads; ++p->stats.synchronizations;
                fprintf(stderr,"FDPP trace: layer=%zu position=%zu heads=%zu retry_heads=%u partitions=%u\n",
                    layer,position,heads,retries,partitions);
            }
            void *combine[]={&p->q,&keys,&values,&p->norm,&p->attention_parts,&width,&nh,&nk,
                &window,&end,&scale,&p->active_request,&partitions,&p->scores,&capacity};
            RUN(plan_grid(p,p->fdpp_combine,heads,1,256,1,combine));
            }
        } else
#endif
#ifdef NYA_PORTABLE_ATTENTION
        if (p->use_portable_attention && !c->reference && width <= 512) {
            unsigned batch_count=(unsigned)count;
            void *portable[]={&p->q,&keys,&values,&p->norm,&width,&nh,&nk,&window,&end,&batch_count,&scale,&p->active_request};
            RUN(plan_grid(p,p->portable_attention,heads,count,256,1,portable));
        } else
#endif
        if (!p->cuda->reference && count >= 8 && (width == 64 || width == 128)) {
            unsigned batch_count = (unsigned)count;
            void *tiled[] = {&p->q, &keys, &values, &p->norm, &width, &nh, &nk, &window, &end, &batch_count, &scale};
            RUN(plan_grid_shared(p, p->attention_prefill, heads, (count+7)/8, 256, 1,
                73*width*(unsigned)sizeof(float), tiled));
            ++p->stats.tiled_attention_calls;
        } else if (p->split_attention && graph_index && (width == 64 || width == 128 || (p->wide_split_attention && (width == 256 || width == 512))) && (!window || window > 256)) {
            attention[3] = &p->attention_parts;
            RUN(plan_grid(p, p->attention_split, heads, 4, 256, 1, attention));
            void *combine[] = {&p->attention_parts, &p->norm, &width};
            RUN(plan_grid(p, p->attention_combine, heads, 1, width > 128 ? width : 128, 1, combine));
        } else RUN(plan_grid(p, p->attention, heads, count, 256, 1, attention));
        p->profile_layer=0;
        int add_attention = count == 1 && d > 256 && p->fuse_residual && !m->is_gemma &&
            !c->reference && (w->attention_output->type == 12 || w->attention_output->type == 14);
        if (add_attention) {
            RUN(plan_matvec_add(p,p->x,w->attention_output,p->norm,d,qdim));
            RUN(plan_norm(p,p->norm,p->x,w->feed_forward_norm,d,1));
        } else {
            RUN(plan_matvec(p, p->residual, w->attention_output, p->norm, d, qdim));
            if (m->is_gemma) RUN(plan_norm(p, p->residual, p->residual, w->attention_post_norm, d, 1));
            /* One-block decode states benefit from dropping a launch. Wider
               residuals lose the parallelism of the separate vector grid; paired
               measurements did not justify that tradeoff or batched fusion. */
            if (count == 1 && d <= 256) RUN(plan_add_norm(p, p->norm, p->x, p->residual, w->feed_forward_norm, d));
            else {
                RUN(plan_vector(p, p->x, p->x, p->residual, d, 1, 1));
                RUN(plan_norm(p, p->norm, p->x, w->feed_forward_norm, d, 1));
            }
        }
        CUdeviceptr gemma_gate=0, gemma_up=0;
        if (count==1 && p->fuse_gate_up && m->is_gemma && !c->reference &&
            p->gemma_gate && c->gemma_gate && d==3840) {
            gemma_gate=plan_byte_tensor(p,w->feed_forward_gate,h,d);
            gemma_up=plan_byte_tensor(p,w->feed_forward_up,h,d);
        }
        if (gemma_gate && gemma_up) {
            unsigned long long cols=d, stride=2400, rows=h;
            void *args[]={&gemma_gate,&gemma_up,&p->norm,&p->gate,&cols,&stride,&rows};
            p->matrix_profile=plan_matrix_slot(p,h,d,12,5);
            RUN(plan_launch(p,c->gemma_gate,(h+7)/8,args));
            p->matrix_profile=0;
        } else
        if (count == 1 && p->fuse_gate_up && !m->is_gemma && !c->reference &&
            w->feed_forward_gate->type == 12 && w->feed_forward_up->type == 12 &&
            w->feed_forward_gate->data_size == w->feed_forward_up->data_size) {
            RUN(plan_gate_up(p,w->feed_forward_gate,w->feed_forward_up,h,d));
        } else {
            RUN(plan_matvec(p, p->gate, w->feed_forward_gate, p->norm, h, d));
            RUN(plan_matvec(p, p->up, w->feed_forward_up, p->norm, h, d));
            RUN(plan_vector(p, p->gate, p->gate, p->up, h, m->is_gemma ? 3U : 2U, 1));
        }
        int add_down = count == 1 && d > 256 && p->fuse_residual && !m->is_gemma &&
            !c->reference && (w->feed_forward_down->type == 12 || w->feed_forward_down->type == 14);
        if (add_down) RUN(plan_matvec_add(p,p->x,w->feed_forward_down,p->gate,d,h));
        else {
            RUN(plan_matvec(p, p->residual, w->feed_forward_down, p->gate, d, h));
            if (m->is_gemma) RUN(plan_norm(p, p->residual, p->residual, w->ffn_post_norm, d, 1));
            RUN(plan_vector(p, p->x, p->x, p->residual, d, 1, 1));
        }
        if (w->output_scale) RUN(plan_vector(p, p->x, p->x, 0, d, 0, nya_llm_tensor_value(w->output_scale, 0)));
    }
    /* Opt-in verification projections reserve at most four vocabulary rows.
       Other counts retain serial projection and ordinary plans reserve one.
       Preserve per-row readback validation, suppression and transfer statistics.
       No verification batch captures a single-token decode graph. */
    if(logits==2 && count>1) {
        if((count==2 || count==4) && count<=p->logits_batch) {
            RUN(plan_norm(p,p->norm,p->x,m->output_norm,d,1));
            RUN(plan_matvec(p,p->logits,m->output,p->norm,m->tokenizer.vocabulary_size,d));
            if(m->final_logit_softcap>0) RUN(plan_vector(p,p->logits,p->logits,0,m->tokenizer.vocabulary_size,5,m->final_logit_softcap));
            for(size_t i=0;i<count;++i) RUN(plan_read_logits_row(p,output+i*m->tokenizer.vocabulary_size,i));
            ++p->batched_logits_calls;
            p->batch=1;
            goto batch_done;
        }
        p->batch=1;
        for(size_t i=0;i<count;++i) {
            RUN(plan_norm(p,p->norm,p->x+i*d*sizeof(float),m->output_norm,d,1));
            RUN(plan_matvec(p,p->logits,m->output,p->norm,m->tokenizer.vocabulary_size,d));
            if(m->final_logit_softcap>0) RUN(plan_vector(p,p->logits,p->logits,0,m->tokenizer.vocabulary_size,5,m->final_logit_softcap));
            RUN(plan_read_logits(p,output+i*m->tokenizer.vocabulary_size));
        }
        goto batch_done;
    }
    if (logits) {
        p->batch = 1;
        RUN(plan_norm(p, p->norm, p->x + (count - 1) * d * sizeof(float), m->output_norm, d, 1));
        RUN(plan_matvec(p, p->logits, m->output, p->norm, m->tokenizer.vocabulary_size, d));
        if (m->final_logit_softcap > 0) RUN(plan_vector(p, p->logits, p->logits, 0, m->tokenizer.vocabulary_size, 5, m->final_logit_softcap));
        if (capturing) {
            CUresult ended = c->capture_end(c->stream, &graph); capturing = 0;
            if (ended != CUDA_SUCCESS) goto failure;
            if (c->graph_instantiate(&p->executable[graph_index], graph, 0) != CUDA_SUCCESS) goto failure;
            p->graph_kernels[graph_index] = p->stats.kernel_launches - capture_start;
            ++p->stats.graph_captures;
            c->graph_free(graph); graph = NULL;
            if (c->graph_launch(p->executable[graph_index], c->stream) != CUDA_SUCCESS) goto failure;
            ++p->stats.graph_replays;
        }
read_logits:
        RUN(plan_read_logits(p,output));
    }
batch_done:
    if(p->profile_active) {++p->profiled_batches;p->profiled_tokens+=count;}
    p->valid = position + count;
    p->hidden_valid=p->valid;p->hidden_offset=(count-1)*m->embedding_length*sizeof(float);
    if(!position || p->valid>p->kv_high_water) p->kv_high_water=p->valid;
    if (c->pop(&previous) == CUDA_SUCCESS) return 0;
    p->failed = 1; return -1;
failure:
    if (capturing) { c->capture_end(c->stream, &graph); capturing = 0; }
    if (graph) c->graph_free(graph);
    fprintf(stderr, "CUDA resident execution failed at token position %zu; CPU prefix replay required\n", position);
    c->stream_sync(c->stream); c->pop(&previous); p->failed = 1; return -1;
#undef RUN
}

static int cuda_plan_token(void *p, uint32_t token, size_t position, int logits, float *output)
{ return cuda_plan_batch(p, &token, 1, position, logits, output); }

static int cuda_plan_prefill(void *opaque, const uint32_t *tokens, size_t count, size_t position, float *output)
{
    nya_cuda_plan *p = opaque;
    if (tokens == NULL || count == 0 || position > p->capacity || count > p->capacity - position) return -1;
    for (size_t i = 0; i < count; ) {
        size_t chunk = count - i < p->maximum_batch ? count - i : p->maximum_batch;
        if (cuda_plan_batch(p, tokens + i, chunk, position + i, i + chunk == count, output)) return -1;
        i += chunk;
    }
    return 0;
}
static int cuda_plan_verify(void *opaque,const uint32_t *tokens,size_t count,
    size_t position,float *output,size_t elements)
{
    nya_cuda_plan *p=opaque;size_t vocabulary=p->model->tokenizer.vocabulary_size;
    if(!tokens || !output || !count || !vocabulary || count>SIZE_MAX/vocabulary ||
       count*vocabulary>SIZE_MAX/sizeof(float) || elements<count*vocabulary ||
       position>p->capacity || count>p->capacity-position) return -1;
    for(size_t i=0;i<count;++i) if(tokens[i]>=vocabulary) return -1;
    for(size_t i=0;i<count;) {
        size_t chunk=count-i<p->maximum_batch?count-i:p->maximum_batch;
        if(cuda_plan_batch(p,tokens+i,chunk,position+i,2,output+i*vocabulary)) return -1;
        i+=chunk;
    }
    return 0;
}
static int cuda_plan_rewind(void *opaque,size_t position)
{
    nya_cuda_plan *p=opaque;
    if(!p || p->failed || position>p->valid) return -1;
    if(position==p->valid) return 0;
    if(position && p->kv_layout) for(size_t i=0;i<p->model->block_count;++i) {
        size_t window=p->model->layers[i].sliding_window;
        size_t first=window && position>=window ? position-window+1 : 0;
        size_t slots=p->kv_layout[i].slots;
        if(slots<p->capacity && !nya_cuda_kv_retained(p->kv_high_water,slots,first)) return -1;
    }
    p->valid=position;
    /* Physical writes survive a rewind; never make overwritten rows readable. */
    if(!position) p->kv_high_water=0;
    return 0;
}
static int cuda_plan_hidden(void *opaque,float *out,size_t elements)
{
    nya_cuda_plan *p=opaque;
    if(!p || p->failed || !out || !p->valid || p->valid!=p->hidden_valid || elements!=p->model->embedding_length) return -1;
    nya_cuda_context *c=p->cuda;CUcontext previous;
    if(c->push(c->context)!=CUDA_SUCCESS)return -1;
    size_t batch=p->batch;p->batch=1;
    int bad=plan_norm(p,p->norm,p->x+p->hidden_offset,p->model->output_norm,elements,1);
    p->batch=batch;
    ++p->stats.synchronizations;
    if(!bad && c->stream_sync(c->stream)!=CUDA_SUCCESS)bad=1;
    if(!bad) {++p->stats.downloads;if(c->download(out,p->norm,elements*sizeof(float))!=CUDA_SUCCESS)bad=1;}
    if(c->pop(&previous)!=CUDA_SUCCESS)bad=1;
    if(!bad)for(size_t i=0;i<elements;++i)if(!isfinite(out[i]))bad=1;
    if(bad)p->failed=1;
    return bad?-1:0;
}
static int cuda_plan_borrow_attention(void *opaque,size_t layer,size_t prefix,
    const float *query,size_t heads,size_t requested_window,float *out,size_t elements)
{
    nya_cuda_plan *p=opaque;
    if(!p || p->failed || !query || !out || !prefix || prefix>p->valid || layer>=p->model->block_count || requested_window>UINT32_MAX)return -1;
    const nya_llm_layer *w=&p->model->layers[layer];
    if(!heads || heads>w->head_count || heads%w->kv_head_count || elements!=heads*w->head_dimension)return -1;
    for(size_t i=0;i<elements;++i)if(!isfinite(query[i]))return -1;
    size_t first=requested_window && prefix>requested_window?prefix-requested_window:0;
    size_t slots=p->kv_layout?p->kv_layout[layer].slots:p->capacity;
    if(!nya_cuda_kv_retained(p->kv_high_water,slots,first))return -1;
    size_t base=p->kv_layout?p->kv_layout[layer].offset:w->cache_offset*p->capacity;
    CUdeviceptr keys=p->keys+base*sizeof(float),values=p->values+base*sizeof(float),request=0;
    unsigned width=w->head_dimension,nh=(unsigned)heads,nk=w->kv_head_count;
    unsigned window=(unsigned)requested_window,end=(unsigned)(prefix-1),capacity=(unsigned)p->capacity,ring_slots=(unsigned)slots;
    float scale=1;nya_cuda_context *c=p->cuda;CUcontext previous;
    if(c->push(c->context)!=CUDA_SUCCESS)return -1;
    int bad=0;++p->stats.uploads;
    if(c->upload(p->q,query,elements*sizeof(float))!=CUDA_SUCCESS || c->stream_sync(NULL)!=CUDA_SUCCESS)bad=1;
    ++p->stats.synchronizations;
    void *args[]={&p->q,&keys,&values,&p->norm,&p->scores,&width,&nh,&nk,&window,&end,&capacity,&scale,&request,&ring_slots};
    CUfunction attention=p->kv_layout?p->attention_ring:p->attention;
    /* Borrowing uses its explicit frozen prefix, never the live decode request.
       Keep the unsplit reduction contract and existing buffers. This independent
       opt-in allows measuring assistant attention without changing target math. */
    if(p->use_wide_value && prefix>256 && (width==256 || width==512)) {
        const char *wide_borrow=getenv("NYA_CUDA_MTP_WIDE_VALUE");
        if(wide_borrow && !strcmp(wide_borrow,"1")) attention=p->wide_value[p->kv_layout?2:0];
    }
    if(!bad) {
        if(plan_grid(p,attention,heads,1,256,1,args))bad=1;
        else if(attention==p->wide_value[0] || attention==p->wide_value[2]) ++p->borrowed_wide_calls;
    }
    ++p->stats.synchronizations;
    if(!bad && c->stream_sync(c->stream)!=CUDA_SUCCESS)bad=1;
    if(!bad) {++p->stats.downloads;if(c->download(out,p->norm,elements*sizeof(float))!=CUDA_SUCCESS)bad=1;}
    if(c->pop(&previous)!=CUDA_SUCCESS)bad=1;
    if(!bad)for(size_t i=0;i<elements;++i)if(!isfinite(out[i]))bad=1;
    if(bad)p->failed=1;
    return bad?-1:0;
}
static void cuda_plan_stats(const void *opaque, nya_compute_stats *stats)
{ const nya_cuda_plan *p = opaque; *stats = p->stats; }

/* Opt-in diagnostics only. No snapshot operation is inserted into the inference
   graph or timed path. The provider owns all layout and prefix validity checks. */
static int cuda_plan_read_kv(void *opaque, size_t layer, size_t position, size_t count,
    float *keys, float *values, size_t elements)
{
    nya_cuda_plan *p = opaque;
    if (!p || p->failed || !keys || !values || !count || layer >= p->model->block_count ||
        position > p->valid || count > p->valid-position) return -1;
    const nya_llm_layer *w = &p->model->layers[layer];
    size_t width = w->head_dimension*w->kv_head_count;
    if (!width || count > SIZE_MAX/width || elements != count*width || elements > SIZE_MAX/sizeof(float)) return -1;
    size_t bytes = sizeof(float);
    size_t slots=p->kv_layout?p->kv_layout[layer].slots:p->capacity;
    if(slots<p->capacity && !nya_cuda_kv_retained(p->kv_high_water,slots,position)) return -1;
    size_t base=p->kv_layout?p->kv_layout[layer].offset:w->cache_offset*p->capacity;
    size_t slot=position%slots,first=count<slots-slot?count:slots-slot;
    size_t offset=(base+slot*width)*bytes;
    nya_cuda_context *c = p->cuda;
    CUcontext previous;
    if (c->push(c->context) != CUDA_SUCCESS) return -1;
    ++p->stats.synchronizations;
    int result = -1;
    CUresult status = c->stream_sync(c->stream);
    for(unsigned part=0;part<2 && status==CUDA_SUCCESS;++part) {
        size_t rows=part?count-first:first;
        if(!rows) continue;
        size_t host=part?first*width:0,source=part?base*bytes:offset;
        status=c->download(keys+host,p->keys+source,rows*width*bytes);
        if(status==CUDA_SUCCESS) {++p->stats.downloads;status=c->download(values+host,p->values+source,rows*width*bytes);}
        if(status==CUDA_SUCCESS) ++p->stats.downloads;
    }
    if(status==CUDA_SUCCESS) result=0;
    if (status != CUDA_SUCCESS) fprintf(stderr,"CUDA KV snapshot failed: driver status %d\n",(int)status);
    status = c->pop(&previous);
    if (status != CUDA_SUCCESS) { fprintf(stderr,"CUDA KV snapshot context restore failed: %d\n",(int)status); result = -1; }
    return result;
}
