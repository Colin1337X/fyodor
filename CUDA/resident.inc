/* Host-side CUDA graph lowering. Included by cuda.c so driver types and entry
   points remain entirely inside the removable CUDA directory. The model binding
   is read-only; every live request owns one scratch allocation and KV prefix. */
#include "../backend/core/llm_internal.h"
#include <math.h>
typedef struct nya_cuda_timing { CUevent start, end; unsigned kind; } nya_cuda_timing;

typedef struct nya_cuda_plan {
    nya_cuda_context *cuda;
    const nya_llm_context *model;
    CUdeviceptr storage, x, norm, residual, gate, up, q, scores, keys, values, logits, frequencies;
    CUdeviceptr blas_storage, attention_parts;
    CUdeviceptr request, decode_key, decode_value, active_request;
    size_t blas_expansion;
    CUdeviceptr *tensors;
    CUfunction gather, gather_batch, norm_op, add_norm, rope, vector, attention;
    CUfunction decode_request, attention_prefill, attention_split, attention_combine;
    size_t capacity, bytes, valid, batch, maximum_batch;
    int failed;
    nya_compute_stats stats;
    CUgraphExec executable[2];
    unsigned long long graph_kernels[2];
    int use_graphs, split_attention;
    size_t fail_after;
    size_t allocations, fail_allocation_after;
    nya_cuda_timing *timings;
    size_t timing_count;
    double milliseconds[6];
} nya_cuda_plan;

static void plan_profile_flush(nya_cuda_plan *p)
{
    if (!p->timing_count) return;
    p->cuda->stream_sync(p->cuda->stream);
    for (size_t i = 0; i < p->timing_count; ++i) {
        nya_cuda_timing *t = &p->timings[i]; float elapsed;
        if (p->cuda->event_elapsed(&elapsed, t->start, t->end) == CUDA_SUCCESS) p->milliseconds[t->kind] += elapsed;
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
        p->cuda->pop(&previous);
    }
    if (p->timings) fprintf(stderr, "CUDA kernel GPU milliseconds: matmul=%.3f norm=%.3f rope=%.3f vector=%.3f attention=%.3f gather=%.3f (profiling disables graphs)\n",
        p->milliseconds[0], p->milliseconds[1], p->milliseconds[2], p->milliseconds[3], p->milliseconds[4], p->milliseconds[5]);
    free(p->timings); free(p->tensors); free(p);
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
    int all_tiled = !c->reference;
    for (size_t i = 0; i < m->block_count; ++i) {
        if (m->layers[i].head_dimension != 64 && m->layers[i].head_dimension != 128) all_tiled = 0;
        if (m->layers[i].head_count > heads) heads = m->layers[i].head_count;
        if (!m->layers[i].head_dimension || m->layers[i].kv_head_count > SIZE_MAX / m->layers[i].head_dimension) return NULL;
        size_t kv = m->layers[i].kv_head_count * m->layers[i].head_dimension;
        if (kv > maximum_kv) maximum_kv = kv;
        if (split_attention && (m->layers[i].head_dimension == 64 || m->layers[i].head_dimension == 128) &&
            (!m->layers[i].sliding_window || m->layers[i].sliding_window > 256)) {
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
    if (heads * capacity > SIZE_MAX / batch || h > SIZE_MAX / batch || q > SIZE_MAX / batch || d > SIZE_MAX / batch) return NULL;
    /* Tiled prefill never materializes global scores. Only its fewer-than-eight
       query fallback needs score rows; decode uses the first row of that same
       persistent allocation. Mixed/unsupported widths and reference mode still
       reserve a complete batch. Account this before admitting model weights. */
    size_t score_batch = all_tiled && batch > 7 ? 7 : batch;
    size_t counts[] = {d * batch, (d > q ? d : q) * batch, d * batch, h * batch, h * batch, q * batch, heads * capacity * score_batch,
        m->cache_width * capacity, m->cache_width * capacity, m->tokenizer.vocabulary_size, rope_count / sizeof(float),
        maximum_kv, maximum_kv, 2, partial_count};
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
        for (size_t i = 0; i < sizeof(counts)/sizeof(counts[0]); ++i) if (plan_add(&total, counts[i])) return NULL;
    }
    if (total > c->cache_limit - weights) return plan_decline("weights plus bounded KV/scratch exceed the memory budget");
    nya_cuda_plan *p = calloc(1, sizeof(*p));
    if (p == NULL) return NULL;
    p->cuda = c; p->model = m; p->capacity = capacity; p->bytes = total;
    p->batch = 1; p->maximum_batch = batch;
    p->split_attention = partial_count != 0;
    const char *graphs = getenv("NYA_CUDA_GRAPHS");
    p->use_graphs = graphs == NULL || strcmp(graphs, "0") != 0;
    const char *profile = getenv("NYA_CUDA_PROFILE");
    if (profile && !strcmp(profile, "1")) {
        p->timings = calloc(4096, sizeof(*p->timings));
        if (p->timings) p->use_graphs = 0;
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
    p->stats.kv_bytes = m->cache_width * capacity * sizeof(float) * 2;
    p->stats.scratch_bytes = total - p->stats.kv_bytes;
    p->stats.prefill_batch = batch;
    p->tensors = calloc(m->tensor_count, sizeof(*p->tensors));
    if (p->tensors == NULL) { free(p->timings); free(p); return NULL; }
    CUcontext previous;
    if (c->push(c->context) != CUDA_SUCCESS) { free(p->timings); free(p->tensors); free(p); return NULL; }
    size_t start_weights = c->weight_count, free_bytes, total_bytes;
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
        &p->decode_key, &p->decode_value, &p->request, &p->attention_parts};
    size_t offset = 0;
    for (size_t i = 0; i < sizeof(counts)/sizeof(counts[0]); ++i) {
        *buffers[i] = p->storage + offset; offset += counts[i] * sizeof(float);
    }
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
        c->function_get(&p->norm_op, c->module, "nya_norm") != CUDA_SUCCESS ||
        c->function_get(&p->add_norm, c->module, "nya_add_norm") != CUDA_SUCCESS ||
        c->function_get(&p->rope, c->module, "nya_rope") != CUDA_SUCCESS ||
        c->function_get(&p->vector, c->module, "nya_vector") != CUDA_SUCCESS ||
        c->function_get(&p->attention, c->module, c->reference ? "nya_attention_reference" : "nya_attention") != CUDA_SUCCESS ||
        c->function_get(&p->attention_prefill, c->module, "nya_attention_prefill") != CUDA_SUCCESS ||
        c->function_get(&p->attention_split, c->module, "nya_attention_split") != CUDA_SUCCESS ||
        c->function_get(&p->attention_combine, c->module, "nya_attention_combine") != CUDA_SUCCESS) goto failure;
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
    if (p->timings) {
        if (p->timing_count == 4096) plan_profile_flush(p);
        timing = &p->timings[p->timing_count];
        timing->start = NULL; timing->end = NULL;
        if (p->cuda->event_create(&timing->start, 0) != CUDA_SUCCESS) return -1;
        if (p->cuda->event_create(&timing->end, 0) != CUDA_SUCCESS) { p->cuda->event_free(timing->start); return -1; }
        timing->kind = f == p->norm_op || f == p->add_norm ? 1U : f == p->rope ? 2U : f == p->vector ? 3U : f == p->attention || f == p->attention_prefill || f == p->attention_split || f == p->attention_combine ? 4U : f == p->gather || f == p->gather_batch ? 5U : 0U;
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
static int plan_matvec(nya_cuda_plan *p, CUdeviceptr out, const nya_llm_tensor *t, CUdeviceptr in, size_t rows, size_t columns)
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
            if (p->timings) {
                if (p->timing_count == 4096) plan_profile_flush(p);
                timing = &p->timings[p->timing_count];
                timing->start = NULL; timing->end = NULL; timing->kind = 0;
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
        unsigned tile = batch >= 32 && r >= 32 ? p->cuda->gemm_tile : 32, bx = tile == 64 ? 16 : 32;
        CUfunction kernel = tile == 64 ? p->cuda->gemm_large[type] : p->cuda->gemm[type];
        return plan_grid(p, kernel, (r + tile-1) / tile, (batch + tile-1) / tile, bx, 256/bx, gemm_args);
    }
    void *args[] = {&matrix, &in, &out, &cols, &stride, &type};
    if (p->cuda->reference) return plan_launch(p, p->cuda->function, rows, args);
    unsigned long long row_count = rows;
    args[5] = &row_count;
    return plan_launch(p, p->cuda->dot[type], (rows + 7) / 8, args);
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
    CUdeviceptr keys, CUdeviceptr values, CUdeviceptr value)
{
    unsigned gemma = (unsigned)p->model->is_gemma;
    unsigned batch = (unsigned)p->batch;
    void *args[] = {&x, &frequencies, &hd, &heads, &position, &gemma, &batch, &p->active_request, &keys, &values, &value};
    return plan_launch(p, p->rope, ((size_t)hd / 2 * heads * batch + 255) / 256, args);
}

static int cuda_plan_batch(void *opaque, const uint32_t *tokens, size_t count, size_t position, int logits, float *output)
{
    nya_cuda_plan *p = opaque;
    const nya_llm_context *m = p->model;
    nya_cuda_context *c = p->cuda;
    if (p->failed || count == 0 || count > p->maximum_batch || position >= p->capacity || count > p->capacity - position ||
        (position != 0 && position > p->valid) || (logits && output == NULL)) return -1;
    for (size_t i = 0; i < count; ++i) if (tokens[i] >= m->tokenizer.vocabulary_size) return -1;
    if (position >= p->fail_after || count > p->fail_after - position) { p->failed = 1; return -1; }
    p->batch = count;
    CUcontext previous;
    if (c->push(c->context) != CUDA_SUCCESS) { p->failed = 1; return -1; }
    /* Single-token/logits execution retains at most two fixed topologies:
       ordinary attention and four-way partitioned attention after 256 tokens.
       Reset/rewind select the matching cached graph; neither graph embeds a
       stale token or position. Prefill remains independently batched. */
    int capturing = 0;
    unsigned graph_index = p->split_attention && count == 1 && position >= 256 ? 1U : 0U;
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
        CUdeviceptr keys = p->keys + w->cache_offset * p->capacity * sizeof(float);
        CUdeviceptr values = p->values + w->cache_offset * p->capacity * sizeof(float);
        CUdeviceptr key = keys + position * kvdim * sizeof(float), value = values + position * kvdim * sizeof(float);
        if (p->active_request) { key = p->decode_key; value = p->decode_value; }
        CUdeviceptr frequencies = p->frequencies + freq_offset * sizeof(float);
        freq_offset += hd / 2;
        RUN(plan_norm(p, p->norm, p->x, w->attention_norm, d, 1));
        RUN(plan_matvec(p, p->q, w->query, p->norm, qdim, d));
        if (m->is_gemma) RUN(plan_norm(p, p->q, p->q, w->query_norm, hd, heads));
        if (w->kv_source == layer) {
            RUN(plan_matvec(p, key, w->key, p->norm, kvdim, d));
            if (w->value) RUN(plan_matvec(p, value, w->value, p->norm, kvdim, d));
            else RUN(plan_vector(p, value, key, 0, kvdim, 0, 1));
            if (m->is_gemma) {
                RUN(plan_norm(p, key, key, w->key_norm, hd, kvheads));
                RUN(plan_norm(p, value, value, NULL, hd, kvheads));
            }
            RUN(plan_rope(p, key, frequencies, (unsigned)hd, (unsigned)kvheads, (unsigned)position,
                p->active_request ? keys : 0, p->active_request ? values : 0, value));
        }
        RUN(plan_rope(p, p->q, frequencies, (unsigned)hd, (unsigned)heads, (unsigned)position, 0, 0, 0));
        unsigned width = (unsigned)hd, nh = (unsigned)heads, nk = (unsigned)kvheads;
        unsigned window = w->sliding_window;
        unsigned end = (unsigned)position, capacity = (unsigned)p->capacity;
        scale = m->is_gemma ? 1 : 1 / sqrtf((float)hd);
        void *attention[] = {&p->q, &keys, &values, &p->norm, &p->scores, &width, &nh, &nk, &window, &end, &capacity, &scale, &p->active_request};
        if (!p->cuda->reference && count >= 8 && (width == 64 || width == 128)) {
            unsigned batch_count = (unsigned)count;
            void *tiled[] = {&p->q, &keys, &values, &p->norm, &width, &nh, &nk, &window, &end, &batch_count, &scale};
            RUN(plan_grid_shared(p, p->attention_prefill, heads, (count+7)/8, 256, 1,
                73*width*(unsigned)sizeof(float), tiled));
            ++p->stats.tiled_attention_calls;
        } else if (graph_index && (width == 64 || width == 128) && (!window || window > 256)) {
            attention[3] = &p->attention_parts;
            RUN(plan_grid(p, p->attention_split, heads, 4, 256, 1, attention));
            void *combine[] = {&p->attention_parts, &p->norm, &width};
            RUN(plan_grid(p, p->attention_combine, heads, 1, 128, 1, combine));
        } else RUN(plan_grid(p, p->attention, heads, count, 256, 1, attention));
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
        RUN(plan_matvec(p, p->gate, w->feed_forward_gate, p->norm, h, d));
        RUN(plan_matvec(p, p->up, w->feed_forward_up, p->norm, h, d));
        RUN(plan_vector(p, p->gate, p->gate, p->up, h, m->is_gemma ? 3U : 2U, 1));
        RUN(plan_matvec(p, p->residual, w->feed_forward_down, p->gate, d, h));
        if (m->is_gemma) RUN(plan_norm(p, p->residual, p->residual, w->ffn_post_norm, d, 1));
        RUN(plan_vector(p, p->x, p->x, p->residual, d, 1, 1));
        if (w->output_scale) RUN(plan_vector(p, p->x, p->x, 0, d, 0, nya_llm_tensor_value(w->output_scale, 0)));
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
        ++p->stats.synchronizations;
        if (c->stream_sync(c->stream) != CUDA_SUCCESS) goto failure;
        ++p->stats.downloads;
        if (c->download(output, p->logits, m->tokenizer.vocabulary_size * sizeof(float)) != CUDA_SUCCESS) goto failure;
        for (size_t i = 0; i < m->tokenizer.vocabulary_size; ++i) if (!isfinite(output[i])) goto failure;
        for (size_t i = 0; i < m->tokenizer.suppressed_count; ++i) output[m->tokenizer.suppressed_tokens[i]] = -INFINITY;
    }
    p->valid = position + count;
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
    size_t offset = (w->cache_offset*p->capacity+position*width)*bytes;
    nya_cuda_context *c = p->cuda;
    CUcontext previous;
    if (c->push(c->context) != CUDA_SUCCESS) return -1;
    ++p->stats.synchronizations;
    int result = -1;
    CUresult status = c->stream_sync(c->stream);
    if (status == CUDA_SUCCESS) {
        status = c->download(keys,p->keys+offset,elements*bytes);
        if (status == CUDA_SUCCESS) ++p->stats.downloads;
    }
    if (status == CUDA_SUCCESS) {
        status = c->download(values,p->values+offset,elements*bytes);
        if (status == CUDA_SUCCESS) { ++p->stats.downloads; result = 0; }
    }
    if (status != CUDA_SUCCESS) fprintf(stderr,"CUDA KV snapshot failed: driver status %d\n",(int)status);
    status = c->pop(&previous);
    if (status != CUDA_SUCCESS) { fprintf(stderr,"CUDA KV snapshot context restore failed: %d\n",(int)status); result = -1; }
    return result;
}
