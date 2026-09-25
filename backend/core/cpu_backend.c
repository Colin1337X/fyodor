#include "compute_backend.h"
#include "llm_internal.h"
#include "cpu_kernels.h"
#include "thread.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

typedef struct nya_cpu_context nya_cpu_context;
typedef struct nya_cpu_worker { nya_cpu_context *pool; size_t index; nya_thread thread; } nya_cpu_worker;
struct nya_cpu_context {
    nya_mutex mutex;
    nya_condition work, done;
    nya_cpu_worker workers[63];
    size_t count, pending;
    unsigned long long epoch;
    int stopping;
    nya_cpu_dot_function dot;
    nya_cpu_decode_function decode;
    nya_cpu_gemm_function gemm;
    nya_cpu_attention_function attention;
    const nya_compute_attention *attention_job;
    float *scores;
    size_t score_capacity;
    int errors[64];
    float *decoded_rows;
    float *packed_input;
    size_t packed_capacity, packed_stride;
    size_t decoded_columns;
    const char *isa;
    const unsigned char *weights;
    const float *input;
    float *output;
    size_t rows, columns, row_bytes, batch;
    unsigned type;
};

/* Jobs partition whole rows. Weights/input are immutable; each worker writes a
   disjoint output slice. The caller participates as worker zero and joins the
   epoch before returning, preserving synchronous matvec semantics. */
static void cpu_rows(nya_cpu_context *c, size_t index, size_t workers)
{
    size_t begin = c->rows / workers * index + (c->rows % workers < index ? c->rows % workers : index);
    size_t end = begin + c->rows / workers + (index < c->rows % workers);
    if (c->attention_job) {
        const nya_compute_attention *a = c->attention_job;
        size_t stride = a->kv_heads*a->width;
        float *scores = c->scores+index*c->score_capacity;
        for (size_t job = begin; job < end; ++job) {
            size_t token = job/a->heads, head = job%a->heads, end_at = a->position+token;
            size_t first = a->window && end_at >= a->window ? end_at-a->window+1 : 0;
            size_t offset = first*stride+(head/(a->heads/a->kv_heads))*a->width;
            if (c->attention(a->query+job*a->width,a->keys+offset,a->values+offset,a->output+job*a->width,
                scores,a->width,stride,end_at-first+1,a->scale)) { c->errors[index] = 1; break; }
        }
        return;
    }
    if (c->gemm && c->batch >= 8) {
        float *decoded = c->decoded_rows + index*c->decoded_columns*2;
        for (size_t row = begin; row < end; row += 2) {
            size_t count = end-row < 2 ? end-row : 2;
            for (size_t r = 0; r < count; ++r)
                c->decode(c->weights+(row+r)*c->row_bytes,decoded+r*c->columns,c->columns,c->type);
            if (count == 1) memset(decoded+c->columns,0,c->columns*sizeof(float));
            c->gemm(decoded,c->packed_input,c->output+row,c->columns,count,c->batch,c->packed_stride,c->rows);
        }
        return;
    }
    if (c->dot) {
        for (size_t row = begin; row < end; ++row) {
            const unsigned char *weights = c->weights + row*c->row_bytes;
            unsigned type = c->type;
            if (c->batch > 1 && c->decode && c->decoded_rows) {
                /* Decode one row once, reuse it across the prompt tile, then
                   recycle the same per-worker scratch for the next row. */
                float *decoded = c->decoded_rows + index*c->decoded_columns*2;
                c->decode(weights, decoded, c->columns, type);
                weights = (const unsigned char *)decoded; type = 0;
            }
            for (size_t token = 0; token < c->batch; ++token)
                c->output[token*c->rows+row] = c->dot(weights, c->input+token*c->columns, c->columns, type);
        }
    } else if (end > begin) {
        nya_llm_tensor t = {0}; t.data = c->weights + begin*c->row_bytes; t.type = c->type;
        for (size_t token = 0; token < c->batch; ++token)
            nya_llm_matvec(NULL, c->output+token*c->rows+begin, &t, c->input+token*c->columns, c->columns, end-begin);
    }
}
static int cpu_worker(void *opaque)
{
    nya_cpu_worker *w = opaque;
    nya_cpu_context *c = w->pool;
    unsigned long long seen = 0;
    nya_mutex_lock(&c->mutex);
    for (;;) {
        while (!c->stopping && c->epoch == seen) nya_condition_wait(&c->work, &c->mutex);
        if (c->stopping) break;
        seen = c->epoch;
        nya_mutex_unlock(&c->mutex);
        cpu_rows(c, w->index, c->count + 1);
        nya_mutex_lock(&c->mutex);
        if (--c->pending == 0) nya_condition_signal(&c->done);
    }
    nya_mutex_unlock(&c->mutex);
    return 0;
}
static size_t cpu_core_count(void)
{
#ifdef _WIN32
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &bytes);
    unsigned char *data = bytes ? malloc(bytes) : NULL;
    size_t cores = 0;
    if (data && GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(void *)data, &bytes)) {
        for (size_t offset = 0; bytes - offset >= offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor); ) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX entry = (void *)(data + offset);
            if (entry->Size == 0 || entry->Size > bytes-offset) break;
            ++cores; offset += entry->Size;
        }
    }
    free(data);
    return cores ? cores : 1;
#else
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    return cores > 0 ? (size_t)cores : 1;
#endif
}
static void *cpu_create(void)
{
    nya_cpu_context *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    if (nya_mutex_init(&c->mutex)) { free(c); return NULL; }
    if (nya_condition_init(&c->work)) { nya_mutex_destroy(&c->mutex); free(c); return NULL; }
    if (nya_condition_init(&c->done)) { nya_condition_destroy(&c->work); nya_mutex_destroy(&c->mutex); free(c); return NULL; }
    c->dot = nya_cpu_select_dot(&c->isa);
    c->decode = nya_cpu_select_decode();
    const char *attention_mode = getenv("NYA_CPU_ATTENTION");
    if (!attention_mode || strcmp(attention_mode,"reference")) c->attention = nya_cpu_select_attention();
    const char *gemm_mode = getenv("NYA_CPU_GEMM");
    if (!gemm_mode || strcmp(gemm_mode,"dot")) c->gemm = nya_cpu_select_gemm();
    size_t count = cpu_core_count();
    const char *setting = getenv("NYA_CPU_THREADS");
    if (setting) { char *end; unsigned long n = strtoul(setting, &end, 10);
        count = setting[0] >= '1' && setting[0] <= '9' && !*end && n <= 64 ? (size_t)n : 1; }
    if (count > 64) count = 64;
    for (size_t i = 1; i < count; ++i) {
        nya_cpu_worker *w = &c->workers[c->count]; w->pool = c; w->index = i;
        if (nya_thread_create(&w->thread, cpu_worker, w)) break;
        ++c->count;
    }
    return c;
}
static void cpu_destroy(void *opaque)
{
    nya_cpu_context *c = opaque;
    nya_mutex_lock(&c->mutex); c->stopping = 1; nya_condition_broadcast(&c->work); nya_mutex_unlock(&c->mutex);
    for (size_t i = 0; i < c->count; ++i) nya_thread_join(&c->workers[i].thread);
    nya_condition_destroy(&c->work); nya_condition_destroy(&c->done); nya_mutex_destroy(&c->mutex); free(c->decoded_rows); free(c->packed_input); free(c->scores); free(c);
}
static int cpu_active(const void *p) { return p != NULL; }
static void cpu_dispatch(nya_cpu_context *c, int parallel)
{
    if (!c->count || !parallel) { cpu_rows(c,0,1); return; }
    nya_mutex_lock(&c->mutex);
    c->pending = c->count; ++c->epoch;
    nya_condition_broadcast(&c->work); nya_mutex_unlock(&c->mutex);
    cpu_rows(c,0,c->count+1);
    nya_mutex_lock(&c->mutex);
    while (c->pending) nya_condition_wait(&c->done,&c->mutex);
    nya_mutex_unlock(&c->mutex);
}

static int cpu_attention(void *opaque, const nya_compute_attention *a)
{
    nya_cpu_context *c = opaque;
    if (!c->attention || !a || !a->query || !a->keys || !a->values || !a->output || !a->batch ||
        !a->heads || !a->kv_heads || a->heads%a->kv_heads || !a->width || !a->capacity || !isfinite(a->scale) ||
        a->position >= a->capacity || a->batch > a->capacity-a->position ||
        a->heads > SIZE_MAX/a->width || a->batch > SIZE_MAX/sizeof(float)/(a->heads*a->width) ||
        a->capacity > SIZE_MAX/sizeof(float)/(a->kv_heads*a->width)) return -1;
    size_t span = a->position+a->batch;
    if (a->window && span > a->window) span = a->window;
    if (span > c->score_capacity) {
        if (span > SIZE_MAX/sizeof(float)/(c->count+1)) return -1;
        float *next = malloc(span*(c->count+1)*sizeof(float));
        if (!next) return -1;
        free(c->scores); c->scores = next; c->score_capacity = span;
    }
    c->attention_job = a; c->rows = a->batch*a->heads;
    memset(c->errors,0,sizeof(c->errors));
    /* Tiny heads are faster locally; larger causal tiles distribute independent
       query/head jobs over the same persistent pool used by projections. */
    cpu_dispatch(c,c->rows >= (c->count+1)*2 && span >= 16);
    c->attention_job = NULL;
    for (size_t i = 0; i <= c->count; ++i) if (c->errors[i]) return -1;
    return 0;
}
static int cpu_matmul(void *opaque, const void *w, size_t rows, size_t columns, unsigned type, const float *x, float *y, size_t batch)
{
    nya_cpu_context *c = opaque;
    size_t block = 1, bytes = 4;
    switch (type) {
        case 0: break;
        case 1: case 30: bytes = 2; break;
        case 2: block = 32; bytes = 18; break;
        case 8: block = 32; bytes = 34; break;
        case 12: block = 256; bytes = 144; break;
        case 14: block = 256; bytes = 210; break;
        default: return -1;
    }
    if (!w || !x || !y || !rows || !columns || !batch || columns % block || batch > SIZE_MAX / sizeof(float) ||
        columns > SIZE_MAX / sizeof(float) / batch || rows > SIZE_MAX / sizeof(float) / batch || columns / block > SIZE_MAX / bytes) return -1;
    size_t stride = columns / block * bytes;
    if (rows > SIZE_MAX / stride) return -1;
    if (batch > 1 && c->decode && columns > c->decoded_columns) {
        if (columns > SIZE_MAX / sizeof(float) / 2 / (c->count+1)) return -1;
        float *next = malloc(columns * 2 * (c->count+1) * sizeof(float));
        if (next) { free(c->decoded_rows); c->decoded_rows = next; c->decoded_columns = columns; }
        else return -1;
    }
    if (batch >= 8 && c->gemm) {
        /* Pad to the widest ISA. All padding is initialized so a final SIMD
           load is valid even for a partial prompt tile. Allocate before
           publishing the job; workers only read this shared immutable pack. */
        if (batch > SIZE_MAX-15) return -1;
        size_t padded = (batch+15)/16*16;
        if (columns > SIZE_MAX / sizeof(float) / padded) return -1;
        size_t elements = columns*padded;
        if (elements > c->packed_capacity) {
            float *next = malloc(elements*sizeof(float));
            if (!next) return -1;
            free(c->packed_input); c->packed_input = next; c->packed_capacity = elements;
        }
        c->packed_stride = padded;
        for (size_t k = 0; k < columns; ++k) {
            for (size_t t = 0; t < batch; ++t) c->packed_input[k*padded+t] = x[t*columns+k];
            for (size_t t = batch; t < padded; ++t) c->packed_input[k*padded+t] = 0;
        }
    }
    c->weights = w; c->input = x; c->output = y; c->rows = rows; c->columns = columns; c->row_bytes = stride; c->type = type; c->batch = batch;
    /* Dispatch overhead dominates tiny projections. Avoid waking the pool for
       small fixtures, norms or skinny output heads. */
    cpu_dispatch(c,rows >= (c->count+1)*2 && columns >= 64 && rows >= 65536/columns);
    return 0;
}
static int cpu_matvec(void *p, const void *w, size_t r, size_t c, unsigned t, const float *x, float *y)
{ return cpu_matmul(p, w, r, c, t, x, y, 1); }
const nya_backend_interface *nya_cpu_backend(void)
{
    static const nya_backend_interface api = {NYA_BACKEND_CPU, "cpu", NYA_COMPUTE_MATVEC | NYA_COMPUTE_QUANTIZED | NYA_COMPUTE_MATMUL,
        cpu_create, cpu_destroy, cpu_active, cpu_matvec, NULL, NULL, NULL, NULL, NULL, cpu_matmul, cpu_attention, NULL, NULL};
    return &api;
}
