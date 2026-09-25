#include "training.h"
#include "training_internal.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum nya_train_op { TR_INPUT, TR_LEAF, TR_ADD, TR_MUL, TR_SCALE, TR_LINEAR,
    TR_GELU, TR_SILU, TR_RMS, TR_EMBED, TR_LOGP, TR_DPO, TR_RESHAPE, TR_ROPE, TR_ATTENTION, TR_MAPPED_LINEAR,
    TR_SOFTCAP, TR_SLICE };

struct nya_train_parameter {
    size_t rows, columns, count;
    float *data, *gradient, *moment, *variance;
};

struct nya_train_tensor {
    nya_train_graph *graph;
    struct nya_train_tensor *previous, *a, *b, *c;
    enum nya_train_op operation;
    size_t rows, columns, count;
    float *data, *gradient;
    uint32_t *indices;
    unsigned char *mask;
    float *saved;
    size_t dimensions[4];
    const nya_llm_tensor *mapped;
    double scalar;
    int borrowed;
};

struct nya_train_graph {
    nya_train_tensor *last;
    size_t used, limit;
    int backward;
    char error[160];
};

static void tr_error(nya_train_graph *g, const char *text)
{
    if (g != NULL && g->error[0] == '\0') snprintf(g->error, sizeof(g->error), "%s", text);
}

void nya_train_graph_fail(nya_train_graph *g, const char *text) { tr_error(g, text); }

/* Every intermediate allocation is charged to a caller-set graph budget. Failed
   construction stays on the graph's cleanup list, so short-circuited operation
   chains cannot leak partially allocated nodes. */
static void *tr_alloc(nya_train_graph *g, size_t count, size_t width)
{
    if (g == NULL || g->error[0] != '\0') return NULL;
    if (count == 0 || width == 0 || count > SIZE_MAX / width || count * width > g->limit - g->used) {
        tr_error(g, "training graph memory budget exceeded"); return NULL;
    }
    void *p = calloc(count, width);
    if (p == NULL) { tr_error(g, "training graph allocation failed"); return NULL; }
    g->used += count * width;
    return p;
}

static nya_train_tensor *tr_node(nya_train_graph *g, size_t rows, size_t columns, int gradient, int borrowed)
{
    if (g == NULL) return NULL;
    if (g->backward || rows == 0 || columns == 0 || rows > SIZE_MAX / columns) {
        tr_error(g, "invalid tensor shape or graph already used for backward"); return NULL;
    }
    nya_train_tensor *t = (nya_train_tensor *)tr_alloc(g, 1, sizeof(*t));
    if (t == NULL) return NULL;
    t->graph = g; t->rows = rows; t->columns = columns; t->count = rows * columns;
    t->previous = g->last; g->last = t; t->borrowed = borrowed;
    if (!borrowed) {
        t->data = (float *)tr_alloc(g, t->count, sizeof(float));
        if (gradient) t->gradient = (float *)tr_alloc(g, t->count, sizeof(float));
        if (t->data == NULL || (gradient && t->gradient == NULL)) return NULL;
    }
    return t;
}

static nya_train_tensor *tr_finite(nya_train_tensor *t)
{
    if (t != NULL) for (size_t i = 0; i < t->count; ++i) if (!isfinite(t->data[i])) {
        tr_error(t->graph, "training operation produced a non-finite value"); return NULL;
    }
    return t;
}

nya_train_graph *nya_train_graph_create(size_t limit)
{
    if (limit < sizeof(nya_train_graph)) return NULL;
    nya_train_graph *g = (nya_train_graph *)calloc(1, sizeof(*g));
    if (g != NULL) { g->limit = limit; g->used = sizeof(*g); }
    return g;
}

void nya_train_graph_free(nya_train_graph *g)
{
    if (g == NULL) return;
    nya_train_tensor *t = g->last;
    while (t != NULL) {
        nya_train_tensor *previous = t->previous;
        if (!t->borrowed) { free(t->data); free(t->gradient); }
        free(t->indices); free(t->mask); free(t->saved); free(t); t = previous;
    }
    free(g);
}

const char *nya_train_error(const nya_train_graph *g) { return g == NULL ? "missing training graph" : g->error; }
size_t nya_train_memory_used(const nya_train_graph *g) { return g == NULL ? 0 : g->used; }

nya_train_parameter *nya_train_parameter_create(size_t rows, size_t columns, const float *initial)
{
    if (rows == 0 || columns == 0 || rows > SIZE_MAX / columns || rows * columns > SIZE_MAX / sizeof(float)) return NULL;
    size_t count = rows * columns;
    if (initial != NULL) for (size_t i = 0; i < count; ++i) if (!isfinite(initial[i])) return NULL;
    nya_train_parameter *p = (nya_train_parameter *)calloc(1, sizeof(*p));
    if (p == NULL) return NULL;
    p->rows = rows; p->columns = columns; p->count = count;
    p->data = (float *)calloc(count, sizeof(float)); p->gradient = (float *)calloc(count, sizeof(float));
    p->moment = (float *)calloc(count, sizeof(float)); p->variance = (float *)calloc(count, sizeof(float));
    if (p->data == NULL || p->gradient == NULL || p->moment == NULL || p->variance == NULL) {
        nya_train_parameter_free(p); return NULL;
    }
    if (initial != NULL) memcpy(p->data, initial, count * sizeof(float));
    return p;
}

void nya_train_parameter_free(nya_train_parameter *p)
{
    if (p == NULL) return;
    free(p->data); free(p->gradient); free(p->moment); free(p->variance); free(p);
}
float *nya_train_parameter_data(nya_train_parameter *p) { return p == NULL ? NULL : p->data; }
const float *nya_train_parameter_gradient(const nya_train_parameter *p) { return p == NULL ? NULL : p->gradient; }
void nya_train_zero_grad(nya_train_parameter *p) { if (p != NULL) memset(p->gradient, 0, p->count * sizeof(float)); }

nya_train_tensor *nya_train_input(nya_train_graph *g, size_t rows, size_t columns, const float *data)
{
    if (data == NULL) { tr_error(g, "missing training input"); return NULL; }
    nya_train_tensor *t = tr_node(g, rows, columns, 0, 0);
    if (t == NULL) return NULL;
    memcpy(t->data, data, t->count * sizeof(float));
    return tr_finite(t);
}

nya_train_tensor *nya_train_leaf(nya_train_graph *g, nya_train_parameter *p)
{
    if (p == NULL) { tr_error(g, "missing training parameter"); return NULL; }
    nya_train_tensor *t = tr_node(g, p->rows, p->columns, 1, 1);
    if (t == NULL) return NULL;
    t->operation = TR_LEAF; t->data = p->data; t->gradient = p->gradient;
    return tr_finite(t);
}
const float *nya_train_data(const nya_train_tensor *t) { return t == NULL ? NULL : t->data; }
size_t nya_train_rows(const nya_train_tensor *t) { return t == NULL ? 0 : t->rows; }
size_t nya_train_columns(const nya_train_tensor *t) { return t == NULL ? 0 : t->columns; }

/* Broadcast indexing is also used during backward: all expanded rows sum into
   the original scalar/vector gradient instead of overwriting one another. */
static size_t tr_index(const nya_train_tensor *t, size_t row, size_t col)
{
    return (t->rows == 1 ? 0 : row) * t->columns + (t->columns == 1 ? 0 : col);
}
static nya_train_tensor *tr_binary(nya_train_tensor *a, nya_train_tensor *b, enum nya_train_op op)
{
    if (a == NULL || b == NULL) return NULL;
    if (a->graph != b->graph || (a->rows != b->rows && a->rows != 1 && b->rows != 1) ||
        (a->columns != b->columns && a->columns != 1 && b->columns != 1)) {
        tr_error(a->graph, "incompatible binary tensor shapes or graphs"); return NULL;
    }
    size_t rows = a->rows > b->rows ? a->rows : b->rows, columns = a->columns > b->columns ? a->columns : b->columns;
    nya_train_tensor *t = tr_node(a->graph, rows, columns, a->gradient != NULL || b->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = a; t->b = b; t->operation = op;
    for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < columns; ++j) {
        float x = a->data[tr_index(a, i, j)], y = b->data[tr_index(b, i, j)];
        t->data[i * columns + j] = op == TR_ADD ? x + y : x * y;
    }
    return tr_finite(t);
}
nya_train_tensor *nya_train_add(nya_train_tensor *a, nya_train_tensor *b) { return tr_binary(a, b, TR_ADD); }
nya_train_tensor *nya_train_mul(nya_train_tensor *a, nya_train_tensor *b) { return tr_binary(a, b, TR_MUL); }

static nya_train_tensor *tr_unary(nya_train_tensor *a, enum nya_train_op op, double scalar)
{
    if (a == NULL) return NULL;
    if (!isfinite(scalar)) { tr_error(a->graph, "non-finite operation parameter"); return NULL; }
    nya_train_tensor *t = tr_node(a->graph, a->rows, a->columns, a->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = a; t->operation = op; t->scalar = scalar;
    for (size_t i = 0; i < a->count; ++i) {
        double x = a->data[i];
        if (op == TR_SCALE) t->data[i] = (float)(x * scalar);
        else if (op == TR_SOFTCAP) t->data[i] = (float)(scalar * tanh(x / scalar));
        else if (op == TR_GELU) t->data[i] = (float)(0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))));
        else t->data[i] = (float)(x * (x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x))));
    }
    return tr_finite(t);
}
nya_train_tensor *nya_train_scale(nya_train_tensor *a, float scale) { return tr_unary(a, TR_SCALE, scale); }
nya_train_tensor *nya_train_gelu(nya_train_tensor *a) { return tr_unary(a, TR_GELU, 0); }
nya_train_tensor *nya_train_silu(nya_train_tensor *a) { return tr_unary(a, TR_SILU, 0); }
nya_train_tensor *nya_train_softcap(nya_train_tensor *a, float cap)
{
    if (a == NULL) return NULL;
    if (!isfinite(cap) || cap <= 0) { tr_error(a->graph,"invalid softcap"); return NULL; }
    return tr_unary(a,TR_SOFTCAP,cap);
}

nya_train_tensor *nya_train_slice_columns(nya_train_tensor *a, size_t first, size_t count)
{
    if (a == NULL) return NULL;
    if (first >= a->columns || count == 0 || count > a->columns-first) {
        tr_error(a->graph,"invalid column slice"); return NULL;
    }
    nya_train_tensor *t = tr_node(a->graph,a->rows,count,a->gradient != NULL,0);
    if (t == NULL) return NULL;
    t->operation = TR_SLICE; t->a = a; t->dimensions[0] = first;
    for (size_t i = 0; i < a->rows; ++i)
        memcpy(t->data+i*count,a->data+i*a->columns+first,count*sizeof(float));
    return t;
}

nya_train_tensor *nya_train_linear(nya_train_tensor *x, nya_train_tensor *w)
{
    if (x == NULL || w == NULL) return NULL;
    if (x->graph != w->graph || x->columns != w->columns) { tr_error(x->graph, "invalid linear weight shape"); return NULL; }
    nya_train_tensor *t = tr_node(x->graph, x->rows, w->rows, x->gradient != NULL || w->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = x; t->b = w; t->operation = TR_LINEAR;
    for (size_t n = 0; n < x->rows; ++n) for (size_t o = 0; o < w->rows; ++o) {
        double sum = 0.0;
        for (size_t i = 0; i < x->columns; ++i) sum += (double)x->data[n * x->columns + i] * w->data[o * w->columns + i];
        t->data[n * w->rows + o] = (float)sum;
    }
    return tr_finite(t);
}

nya_train_tensor *nya_train_linear_mapped(nya_train_tensor *x, const nya_llm_tensor *w)
{
    if (x == NULL) return NULL;
    if (w == NULL || w->data == NULL || w->dimension_count != 2 || w->dimensions[0] != x->columns ||
        w->dimensions[1] == 0 || w->dimensions[1] > SIZE_MAX) {
        tr_error(x->graph, "invalid frozen matrix shape"); return NULL;
    }
    /* The internal caller passes a loader-bound tensor. Recheck its type and
       row alignment so an unsupported block cannot reach a decoder fallback. */
    size_t block = 1, block_bytes = 4;
    switch (w->type) {
        case NYA_LLM_TENSOR_F32: break;
        case NYA_LLM_TENSOR_F16: case NYA_LLM_TENSOR_BF16: block_bytes = 2; break;
        case NYA_LLM_TENSOR_Q4_0: block = 32; block_bytes = 18; break;
        case NYA_LLM_TENSOR_Q8_0: block = 32; block_bytes = 34; break;
        case NYA_LLM_TENSOR_Q4_K: block = 256; block_bytes = 144; break;
        case NYA_LLM_TENSOR_Q6_K: block = 256; block_bytes = 210; break;
        default: tr_error(x->graph, "unsupported frozen matrix quantization"); return NULL;
    }
    size_t rows = (size_t)w->dimensions[1];
    if (x->columns % block != 0 || x->columns / block > SIZE_MAX / block_bytes ||
        rows > SIZE_MAX / (x->columns / block * block_bytes) ||
        rows * (x->columns / block * block_bytes) != w->data_size) {
        tr_error(x->graph, "frozen matrix extent or block alignment is invalid"); return NULL;
    }
    nya_train_tensor *t = tr_node(x->graph, x->rows, rows, x->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = x; t->mapped = w; t->operation = TR_MAPPED_LINEAR;
    for (size_t n = 0; n < x->rows; ++n) nya_llm_matvec(NULL, t->data + n * rows, w, x->data + n * x->columns, x->columns, rows);
    return tr_finite(t);
}

nya_train_tensor *nya_train_embedding_mapped(nya_train_graph *g, const nya_llm_tensor *w, const uint32_t *ids, size_t count)
{
    if (w == NULL || !w->bound || w->data == NULL || w->dimension_count != 2 || w->dimensions[0] > SIZE_MAX || ids == NULL || count == 0) {
        tr_error(g, "invalid frozen embedding table"); return NULL;
    }
    for (size_t i = 0; i < count; ++i) if (ids[i] >= w->dimensions[1]) { tr_error(g, "frozen embedding ID out of range"); return NULL; }
    nya_train_tensor *t = tr_node(g, count, (size_t)w->dimensions[0], 0, 0);
    if (t == NULL) return NULL;
    for (size_t i = 0; i < count; ++i) for (size_t j = 0; j < t->columns; ++j)
        t->data[i * t->columns + j] = nya_llm_tensor_value(w, (size_t)ids[i] * t->columns + j);
    return tr_finite(t);
}

nya_train_tensor *nya_train_constant_mapped(nya_train_graph *g, const nya_llm_tensor *w)
{
    if (w == NULL || !w->bound || w->data == NULL || w->dimension_count != 1 || w->dimensions[0] > SIZE_MAX) {
        tr_error(g, "invalid frozen normalization vector"); return NULL;
    }
    nya_train_tensor *t = tr_node(g, 1, (size_t)w->dimensions[0], 0, 0);
    if (t == NULL) return NULL;
    for (size_t i = 0; i < t->count; ++i) t->data[i] = nya_llm_tensor_value(w, i);
    return tr_finite(t);
}

nya_train_tensor *nya_train_rms_norm(nya_train_tensor *x, nya_train_tensor *w, float epsilon)
{
    if (x == NULL) return NULL;
    if (!isfinite(epsilon) || epsilon <= 0 || (w != NULL && (w->graph != x->graph || w->rows != 1 || w->columns != x->columns))) {
        tr_error(x->graph, "invalid RMSNorm parameters"); return NULL;
    }
    nya_train_tensor *t = tr_node(x->graph, x->rows, x->columns, x->gradient != NULL || (w != NULL && w->gradient != NULL), 0);
    if (t == NULL) return NULL;
    t->a = x; t->b = w; t->operation = TR_RMS; t->scalar = epsilon;
    for (size_t row = 0; row < x->rows; ++row) {
        double square = 0.0;
        for (size_t j = 0; j < x->columns; ++j) { double v = x->data[row * x->columns + j]; square += v * v; }
        double inverse = 1.0 / sqrt(square / (double)x->columns + epsilon);
        for (size_t j = 0; j < x->columns; ++j) t->data[row * x->columns + j] =
            (float)((double)x->data[row * x->columns + j] * inverse * (w == NULL ? 1.0 : w->data[j]));
    }
    return tr_finite(t);
}

nya_train_tensor *nya_train_embedding(nya_train_tensor *table, const uint32_t *ids, size_t count)
{
    if (table == NULL) return NULL;
    if (ids == NULL || count == 0) { tr_error(table->graph, "invalid embedding IDs"); return NULL; }
    for (size_t i = 0; i < count; ++i) if (ids[i] >= table->rows) { tr_error(table->graph, "embedding ID out of range"); return NULL; }
    nya_train_tensor *t = tr_node(table->graph, count, table->columns, table->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = table; t->operation = TR_EMBED;
    t->indices = (uint32_t *)tr_alloc(t->graph, count, sizeof(uint32_t));
    if (t->indices == NULL) return NULL;
    memcpy(t->indices, ids, count * sizeof(uint32_t));
    for (size_t i = 0; i < count; ++i) memcpy(t->data + i * t->columns, table->data + (size_t)ids[i] * t->columns, t->columns * sizeof(float));
    return tr_finite(t);
}

nya_train_tensor *nya_train_reshape(nya_train_tensor *x, size_t rows, size_t columns)
{
    if (x == NULL) return NULL;
    if (rows == 0 || columns == 0 || rows > SIZE_MAX / columns || rows * columns != x->count) {
        tr_error(x->graph, "reshape changes the tensor element count"); return NULL;
    }
    nya_train_tensor *t = tr_node(x->graph, rows, columns, x->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = x; t->operation = TR_RESHAPE;
    memcpy(t->data, x->data, x->count * sizeof(float));
    return t;
}

nya_train_tensor *nya_train_rope(nya_train_tensor *x, size_t heads, size_t dimension,
    const float *frequencies, int split_half)
{
    if (x == NULL) return NULL;
    if (heads == 0 || dimension == 0 || dimension % 2 != 0 || heads > SIZE_MAX / dimension ||
        heads * dimension != x->columns || frequencies == NULL || (split_half != 0 && split_half != 1)) {
        tr_error(x->graph, "invalid rotary embedding dimensions"); return NULL;
    }
    for (size_t i = 0; i < dimension / 2; ++i) if (!isfinite(frequencies[i])) {
        tr_error(x->graph, "non-finite rotary frequency"); return NULL;
    }
    nya_train_tensor *t = tr_node(x->graph, x->rows, x->columns, x->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = x; t->operation = TR_ROPE; t->dimensions[0] = heads;
    t->dimensions[1] = dimension; t->dimensions[2] = (size_t)split_half;
    t->saved = (float *)tr_alloc(x->graph, dimension / 2, sizeof(float));
    if (t->saved == NULL) return NULL;
    memcpy(t->saved, frequencies, dimension / 2 * sizeof(float));
    for (size_t n = 0; n < x->rows; ++n) for (size_t h = 0; h < heads; ++h) for (size_t j = 0; j < dimension / 2; ++j) {
        size_t i = n * x->columns + h * dimension + (split_half ? j : 2 * j);
        size_t k = i + (split_half ? dimension / 2 : 1);
        double angle = (double)n * frequencies[j], co = cos(angle), si = sin(angle);
        t->data[i] = (float)((double)x->data[i] * co - (double)x->data[k] * si);
        t->data[k] = (float)((double)x->data[k] * co + (double)x->data[i] * si);
    }
    return tr_finite(t);
}

nya_train_tensor *nya_train_attention(nya_train_tensor *q, nya_train_tensor *k,
    nya_train_tensor *v, size_t heads, size_t kv_heads, size_t dimension,
    float scale, size_t window, const uint32_t *groups)
{
    if (q == NULL || k == NULL || v == NULL) return NULL;
    if (q->graph != k->graph || q->graph != v->graph || heads == 0 || kv_heads == 0 || heads % kv_heads != 0 ||
        dimension == 0 || heads > SIZE_MAX / dimension || heads * dimension != q->columns ||
        kv_heads * dimension != k->columns || k->columns != v->columns || q->rows != k->rows || k->rows != v->rows ||
        !isfinite(scale) || scale <= 0 || q->rows > SIZE_MAX / q->rows || q->rows * q->rows > SIZE_MAX / heads) {
        tr_error(q->graph, "invalid grouped-query attention shapes"); return NULL;
    }
    nya_train_tensor *t = tr_node(q->graph, q->rows, q->columns,
        q->gradient != NULL || k->gradient != NULL || v->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = q; t->b = k; t->c = v; t->operation = TR_ATTENTION; t->scalar = scale;
    t->dimensions[0] = heads; t->dimensions[1] = kv_heads; t->dimensions[2] = dimension;
    size_t n = q->rows;
    t->saved = (float *)tr_alloc(q->graph, n * n * heads, sizeof(float));
    if (t->saved == NULL) return NULL;
    /* Keep one probability matrix per query head for the softmax Jacobian.
       Masked entries remain exactly zero. The graph budget bounds N^2 storage. */
    for (size_t row = 0; row < n; ++row) for (size_t head = 0; head < heads; ++head) {
        float *probability = t->saved + (head * n + row) * n;
        size_t first = window != 0 && row >= window ? row - window + 1 : 0;
        size_t kh = head / (heads / kv_heads);
        double maximum = -INFINITY, mass = 0.0;
        for (size_t col = first; col < n; ++col) {
            if (col > row && !(window != 0 && groups != NULL && groups[row] != 0 && groups[row] == groups[col])) continue;
            double score = 0.0;
            for (size_t j = 0; j < dimension; ++j) score +=
                (double)q->data[row * q->columns + head * dimension + j] * k->data[col * k->columns + kh * dimension + j];
            score *= scale;
            if (!isfinite(score) || fabs(score) > FLT_MAX) { tr_error(q->graph, "attention score overflow"); return NULL; }
            probability[col] = (float)score;
            if (probability[col] > maximum) maximum = probability[col];
        }
        for (size_t col = 0; col < n; ++col) {
            if (col < first || (col > row && !(window != 0 && groups != NULL && groups[row] != 0 && groups[row] == groups[col]))) {
                probability[col] = 0.0f; continue;
            }
            probability[col] = (float)exp((double)probability[col] - maximum); mass += probability[col];
        }
        for (size_t col = first; col < n; ++col) probability[col] = (float)(probability[col] / mass);
        for (size_t j = 0; j < dimension; ++j) {
            double sum = 0.0;
            for (size_t col = first; col < n; ++col) sum += (double)probability[col] * v->data[col * v->columns + kh * dimension + j];
            t->data[row * t->columns + head * dimension + j] = (float)sum;
        }
    }
    return tr_finite(t);
}

static nya_train_tensor *tr_logprob(nya_train_tensor *x, const uint32_t *labels,
    const unsigned char *mask, size_t count, int mean_loss)
{
    if (x == NULL) return NULL;
    size_t active = 0;
    if (count != x->rows || labels == NULL) { tr_error(x->graph, "label count differs from logit rows"); return NULL; }
    for (size_t n = 0; n < count; ++n) if (mask == NULL || mask[n]) {
        if (labels[n] >= x->columns) { tr_error(x->graph, "training label out of range"); return NULL; }
        ++active;
    }
    if (active == 0) { tr_error(x->graph, "loss has no unmasked labels"); return NULL; }
    nya_train_tensor *t = tr_node(x->graph, 1, 1, x->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = x; t->operation = TR_LOGP; t->scalar = mean_loss ? -1.0 / (double)active : 1.0;
    t->indices = (uint32_t *)tr_alloc(t->graph, count, sizeof(uint32_t));
    t->mask = (unsigned char *)tr_alloc(t->graph, count, 1);
    if (t->indices == NULL || t->mask == NULL) return NULL;
    memcpy(t->indices, labels, count * sizeof(uint32_t));
    double total = 0.0;
    for (size_t n = 0; n < count; ++n) {
        t->mask[n] = mask == NULL || mask[n];
        if (!t->mask[n]) continue;
        const float *row = x->data + n * x->columns;
        double maximum = row[0], sum = 0.0;
        for (size_t j = 1; j < x->columns; ++j) if (row[j] > maximum) maximum = row[j];
        for (size_t j = 0; j < x->columns; ++j) sum += exp((double)row[j] - maximum);
        total += ((double)row[labels[n]] - maximum) - log(sum);
    }
    t->data[0] = (float)(total * t->scalar);
    return tr_finite(t);
}
nya_train_tensor *nya_train_cross_entropy(nya_train_tensor *x, const uint32_t *labels, const unsigned char *mask, size_t count)
{ return tr_logprob(x, labels, mask, count, 1); }
nya_train_tensor *nya_train_logprob(nya_train_tensor *x, const uint32_t *labels, const unsigned char *mask, size_t count)
{ return tr_logprob(x, labels, mask, count, 0); }

nya_train_tensor *nya_train_dpo(nya_train_tensor *chosen, nya_train_tensor *rejected,
    double ref_chosen, double ref_rejected, float beta)
{
    if (chosen == NULL || rejected == NULL) return NULL;
    if (chosen->graph != rejected->graph || chosen->count != 1 || rejected->count != 1 ||
        !isfinite(ref_chosen) || !isfinite(ref_rejected) || !isfinite(beta) || beta <= 0) {
        tr_error(chosen->graph, "invalid DPO scalar inputs"); return NULL;
    }
    nya_train_tensor *t = tr_node(chosen->graph, 1, 1, chosen->gradient != NULL || rejected->gradient != NULL, 0);
    if (t == NULL) return NULL;
    t->a = chosen; t->b = rejected; t->operation = TR_DPO;
    double margin = (double)beta * (((double)chosen->data[0] - rejected->data[0]) - (ref_chosen - ref_rejected));
    if (!isfinite(margin)) { tr_error(t->graph, "DPO margin overflow"); return NULL; }
    /* softplus(-margin) and its derivative remain finite at extreme margins. */
    t->data[0] = (float)(fmax(-margin, 0.0) + log1p(exp(-fabs(margin))));
    t->scalar = -(double)beta * (margin >= 0 ? exp(-margin) / (1.0 + exp(-margin)) : 1.0 / (1.0 + exp(margin)));
    return tr_finite(t);
}

int nya_train_backward(nya_train_tensor *loss)
{
    if (loss == NULL) return -1;
    nya_train_graph *g = loss->graph;
    if (g->backward || g->error[0] != '\0' || loss->count != 1 || loss->gradient == NULL) {
        tr_error(g, "backward requires an unused graph and a differentiable scalar loss"); return -1;
    }
    g->backward = 1; loss->gradient[0] += 1.0f;
    /* Construction order is a topological ordering: an operation can only
       reference existing nodes. Reverse traversal therefore handles branches
       and shared parameters without recursion or an extra sorting allocation. */
    for (nya_train_tensor *t = g->last; t != NULL; t = t->previous) {
        if (t->gradient == NULL) continue;
        for (size_t i = 0; i < t->count; ++i) if (!isfinite(t->gradient[i])) { tr_error(g, "non-finite training gradient"); return -1; }
        nya_train_tensor *a = t->a, *b = t->b;
        switch (t->operation) {
        case TR_INPUT: case TR_LEAF: break;
        case TR_SLICE:
            if (a->gradient != NULL) for (size_t row = 0; row < t->rows; ++row)
                for (size_t col = 0; col < t->columns; ++col)
                    a->gradient[row*a->columns+t->dimensions[0]+col] += t->gradient[row*t->columns+col];
            break;
        case TR_RESHAPE:
            if (a->gradient != NULL) for (size_t i = 0; i < t->count; ++i) a->gradient[i] += t->gradient[i];
            break;
        case TR_ROPE: {
            size_t heads = t->dimensions[0], dimension = t->dimensions[1]; int split = t->dimensions[2] != 0;
            if (a->gradient != NULL) for (size_t n = 0; n < a->rows; ++n) for (size_t h = 0; h < heads; ++h)
                for (size_t j = 0; j < dimension / 2; ++j) {
                    size_t i = n * a->columns + h * dimension + (split ? j : 2 * j), k = i + (split ? dimension / 2 : 1);
                    double angle = (double)n * t->saved[j], co = cos(angle), si = sin(angle);
                    a->gradient[i] += (float)((double)t->gradient[i] * co + (double)t->gradient[k] * si);
                    a->gradient[k] += (float)((double)t->gradient[k] * co - (double)t->gradient[i] * si);
                }
            break;
        }
        case TR_ATTENTION: {
            nya_train_tensor *v = t->c;
            size_t n = a->rows, heads = t->dimensions[0], kv_heads = t->dimensions[1], dimension = t->dimensions[2];
            for (size_t row = 0; row < n; ++row) for (size_t head = 0; head < heads; ++head) {
                size_t kh = head / (heads / kv_heads);
                const float *probability = t->saved + (head * n + row) * n;
                const float *dy = t->gradient + row * t->columns + head * dimension;
                double average = 0.0;
                for (size_t col = 0; col < n; ++col) if (probability[col] != 0.0f) {
                    double dp = 0.0;
                    for (size_t j = 0; j < dimension; ++j) dp += (double)dy[j] * v->data[col * v->columns + kh * dimension + j];
                    average += probability[col] * dp;
                }
                for (size_t col = 0; col < n; ++col) if (probability[col] != 0.0f) {
                    double dp = 0.0;
                    for (size_t j = 0; j < dimension; ++j) dp += (double)dy[j] * v->data[col * v->columns + kh * dimension + j];
                    double ds = probability[col] * (dp - average) * t->scalar;
                    for (size_t j = 0; j < dimension; ++j) {
                        size_t qi = row * a->columns + head * dimension + j, ki = col * b->columns + kh * dimension + j;
                        if (a->gradient != NULL) a->gradient[qi] += (float)(ds * b->data[ki]);
                        if (b->gradient != NULL) b->gradient[ki] += (float)(ds * a->data[qi]);
                        if (v->gradient != NULL) v->gradient[ki] += probability[col] * dy[j];
                    }
                }
            }
            break;
        }
        case TR_ADD: case TR_MUL:
            for (size_t row = 0; row < t->rows; ++row) for (size_t col = 0; col < t->columns; ++col) {
                size_t ia = tr_index(a, row, col), ib = tr_index(b, row, col);
                float dy = t->gradient[row * t->columns + col];
                if (a->gradient != NULL) a->gradient[ia] += dy * (t->operation == TR_ADD ? 1.0f : b->data[ib]);
                if (b->gradient != NULL) b->gradient[ib] += dy * (t->operation == TR_ADD ? 1.0f : a->data[ia]);
            }
            break;
        case TR_SCALE: case TR_GELU: case TR_SILU: case TR_SOFTCAP:
            if (a->gradient != NULL) for (size_t i = 0; i < t->count; ++i) {
                double x = a->data[i], derivative;
                if (t->operation == TR_SCALE) derivative = t->scalar;
                else if (t->operation == TR_SOFTCAP) {
                    double z = tanh(x/t->scalar); derivative = 1.0-z*z;
                }
                else if (t->operation == TR_GELU) {
                    double z = tanh(0.7978845608028654 * (x + 0.044715 * x * x * x));
                    derivative = 0.5 * (1.0 + z) + 0.5 * x * (1.0 - z * z) * 0.7978845608028654 * (1.0 + 0.134145 * x * x);
                } else {
                    double sigmoid = x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
                    derivative = sigmoid * (1.0 + x * (1.0 - sigmoid));
                }
                a->gradient[i] += (float)((double)t->gradient[i] * derivative);
            }
            break;
        case TR_LINEAR:
            for (size_t n = 0; n < a->rows; ++n) for (size_t o = 0; o < b->rows; ++o) {
                float dy = t->gradient[n * b->rows + o];
                for (size_t i = 0; i < a->columns; ++i) {
                    if (a->gradient != NULL) a->gradient[n * a->columns + i] += dy * b->data[o * b->columns + i];
                    if (b->gradient != NULL) b->gradient[o * b->columns + i] += dy * a->data[n * a->columns + i];
                }
            }
            break;
        case TR_MAPPED_LINEAR:
            /* Backward multiplies by W, not W^T. Decode a scalar only as it is
               used; no dense copy or gradient of the frozen base is allocated. */
            if (a->gradient != NULL) for (size_t o = 0; o < t->columns; ++o) for (size_t i = 0; i < a->columns; ++i) {
                float weight = nya_llm_tensor_value(t->mapped, o * a->columns + i);
                for (size_t n = 0; n < a->rows; ++n) a->gradient[n * a->columns + i] += t->gradient[n * t->columns + o] * weight;
            }
            break;
        case TR_RMS:
            for (size_t row = 0; row < a->rows; ++row) {
                double square = 0.0, dot = 0.0;
                for (size_t j = 0; j < a->columns; ++j) {
                    size_t i = row * a->columns + j;
                    square += (double)a->data[i] * a->data[i];
                    dot += (double)t->gradient[i] * (b == NULL ? 1.0 : b->data[j]) * a->data[i];
                }
                double inverse = 1.0 / sqrt(square / (double)a->columns + t->scalar);
                for (size_t j = 0; j < a->columns; ++j) {
                    size_t i = row * a->columns + j;
                    if (a->gradient != NULL) a->gradient[i] += (float)(inverse * ((double)t->gradient[i] * (b == NULL ? 1.0 : b->data[j]) -
                        (double)a->data[i] * inverse * inverse * dot / (double)a->columns));
                    if (b != NULL && b->gradient != NULL) b->gradient[j] += (float)((double)t->gradient[i] * a->data[i] * inverse);
                }
            }
            break;
        case TR_EMBED:
            if (a->gradient != NULL) for (size_t row = 0; row < t->rows; ++row) for (size_t j = 0; j < t->columns; ++j)
                a->gradient[(size_t)t->indices[row] * t->columns + j] += t->gradient[row * t->columns + j];
            break;
        case TR_LOGP:
            if (a->gradient != NULL) for (size_t row = 0; row < a->rows; ++row) {
                if (!t->mask[row]) continue;
                double maximum = a->data[row * a->columns], sum = 0.0;
                for (size_t j = 1; j < a->columns; ++j) if (a->data[row * a->columns + j] > maximum) maximum = a->data[row * a->columns + j];
                for (size_t j = 0; j < a->columns; ++j) sum += exp((double)a->data[row * a->columns + j] - maximum);
                for (size_t j = 0; j < a->columns; ++j) {
                    double probability = exp((double)a->data[row * a->columns + j] - maximum) / sum;
                    a->gradient[row * a->columns + j] += (float)((double)t->gradient[0] * t->scalar * ((j == t->indices[row] ? 1.0 : 0.0) - probability));
                }
            }
            break;
        case TR_DPO:
            if (a->gradient != NULL) a->gradient[0] += (float)((double)t->gradient[0] * t->scalar);
            if (b->gradient != NULL) b->gradient[0] -= (float)((double)t->gradient[0] * t->scalar);
            break;
        }
    }
    return 0;
}

void nya_train_adamw_defaults(nya_train_adamw *o)
{
    if (o == NULL) return;
    *o = (nya_train_adamw){0.0003f, 0.9f, 0.999f, 1e-8f, 0.01f, 1.0f, 0};
}

int nya_train_adamw_step(nya_train_adamw *o, nya_train_parameter *const *parameters,
    size_t count, char *error, size_t capacity)
{
    const char *failure = "invalid AdamW settings or parameter list";
    size_t total = 0;
    double square = 0.0;
    float *next = NULL;
    if (o == NULL || parameters == NULL || count == 0 || o->step == UINT64_MAX ||
        !isfinite(o->learning_rate) || o->learning_rate <= 0 || !isfinite(o->beta1) || o->beta1 < 0 || o->beta1 >= 1 ||
        !isfinite(o->beta2) || o->beta2 < 0 || o->beta2 >= 1 || !isfinite(o->epsilon) || o->epsilon <= 0 ||
        !isfinite(o->weight_decay) || o->weight_decay < 0 || !isfinite(o->max_grad_norm) || o->max_grad_norm < 0) goto fail;
    for (size_t p = 0; p < count; ++p) {
        if (parameters[p] == NULL || parameters[p]->count > SIZE_MAX - total) goto fail;
        for (size_t j = 0; j < p; ++j) if (parameters[j] == parameters[p]) goto fail;
        total += parameters[p]->count;
        for (size_t i = 0; i < parameters[p]->count; ++i) {
            double v = parameters[p]->gradient[i];
            if (!isfinite(v)) { failure = "AdamW rejected a non-finite gradient"; goto fail; }
            square += v * v;
        }
    }
    failure = "AdamW staging allocation failed";
    if (total > SIZE_MAX / (3 * sizeof(float))) goto fail;
    next = (float *)malloc(total * 3 * sizeof(float));
    if (next == NULL) goto fail;
    double norm = sqrt(square), clip = o->max_grad_norm > 0 && norm > o->max_grad_norm ? o->max_grad_norm / norm : 1.0;
    double correction1 = 1.0 - pow(o->beta1, (double)(o->step + 1));
    double correction2 = 1.0 - pow(o->beta2, (double)(o->step + 1));
    size_t offset = 0;
    for (size_t p = 0; p < count; ++p) for (size_t i = 0; i < parameters[p]->count; ++i, ++offset) {
        nya_train_parameter *w = parameters[p];
        double grad = w->gradient[i] * clip;
        double m = (double)o->beta1 * w->moment[i] + (1.0 - o->beta1) * grad;
        double v = (double)o->beta2 * w->variance[i] + (1.0 - o->beta2) * grad * grad;
        double value = (double)w->data[i] * (1.0 - (double)o->learning_rate * o->weight_decay) -
            o->learning_rate * (m / correction1) / (sqrt(v / correction2) + o->epsilon);
        if (!isfinite(m) || !isfinite(v) || !isfinite(value) || fabs(m) > FLT_MAX || v < 0 || v > FLT_MAX || fabs(value) > FLT_MAX) {
            failure = "AdamW rejected a non-finite or overflowing update"; goto fail;
        }
        next[offset] = (float)value; next[total + offset] = (float)m; next[2 * total + offset] = (float)v;
    }
    offset = 0;
    for (size_t p = 0; p < count; ++p) {
        nya_train_parameter *w = parameters[p];
        memcpy(w->data, next + offset, w->count * sizeof(float));
        memcpy(w->moment, next + total + offset, w->count * sizeof(float));
        memcpy(w->variance, next + 2 * total + offset, w->count * sizeof(float));
        offset += w->count;
    }
    ++o->step; free(next);
    return 0;
fail:
    free(next);
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", failure);
    return -1;
}

/* Checkpoints use explicit bytes, never struct dumps: host padding, enum size,
   pointer values and endianness cannot leak into the on-disk representation.
   FNV-1a detects accidental corruption; it is not an authentication mechanism. */
static int tr_checkpoint_bytes(FILE *file, int writing, unsigned char *bytes, size_t count, uint64_t *hash)
{
    if ((writing ? fwrite(bytes, 1, count, file) : fread(bytes, 1, count, file)) != count) return -1;
    if (hash != NULL) for (size_t i = 0; i < count; ++i) { *hash ^= bytes[i]; *hash *= UINT64_C(1099511628211); }
    return 0;
}

static int tr_checkpoint_u64(FILE *file, int writing, uint64_t *value, uint64_t *hash)
{
    unsigned char bytes[8];
    if (writing) for (size_t i = 0; i < 8; ++i) bytes[i] = (unsigned char)(*value >> (8 * i));
    if (tr_checkpoint_bytes(file, writing, bytes, 8, hash) != 0) return -1;
    if (!writing) {
        *value = 0;
        for (size_t i = 0; i < 8; ++i) *value |= (uint64_t)bytes[i] << (8 * i);
    }
    return 0;
}

static int tr_checkpoint_float(FILE *file, int writing, float *value, uint64_t *hash)
{
    uint32_t bits;
    unsigned char bytes[4];
    if (writing) {
        memcpy(&bits, value, 4);
        for (size_t i = 0; i < 4; ++i) bytes[i] = (unsigned char)(bits >> (8 * i));
    }
    if (tr_checkpoint_bytes(file, writing, bytes, 4, hash) != 0) return -1;
    if (!writing) {
        bits = 0;
        for (size_t i = 0; i < 4; ++i) bits |= (uint32_t)bytes[i] << (8 * i);
        memcpy(value, &bits, 4);
    }
    return isfinite(*value) ? 0 : -1;
}

static int tr_optimizer_valid(const nya_train_adamw *o)
{
    return o != NULL && isfinite(o->learning_rate) && o->learning_rate > 0 &&
        isfinite(o->beta1) && o->beta1 >= 0 && o->beta1 < 1 && isfinite(o->beta2) && o->beta2 >= 0 && o->beta2 < 1 &&
        isfinite(o->epsilon) && o->epsilon > 0 && isfinite(o->weight_decay) && o->weight_decay >= 0 &&
        isfinite(o->max_grad_norm) && o->max_grad_norm >= 0;
}

static int tr_checkpoint(FILE *file, int writing, nya_train_adamw *optimizer,
    nya_train_parameter *const *parameters, size_t count)
{
    static const unsigned char expected_magic[8] = {'N','Y','A','T','R','N',1,0};
    unsigned char magic[8];
    size_t total = 0, offset = 0;
    uint64_t hash = UINT64_C(14695981039346656037), checksum, stored_count = count;
    nya_train_adamw temporary = {0};
    float *staging = NULL;
    int result = -1;
    if (file == NULL || optimizer == NULL || parameters == NULL || count == 0 ||
        sizeof(float) != 4 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) return -1;
    for (size_t p = 0; p < count; ++p) {
        if (parameters[p] == NULL || parameters[p]->count > SIZE_MAX - total) return -1;
        for (size_t j = 0; j < p; ++j) if (parameters[p] == parameters[j]) return -1;
        total += parameters[p]->count;
    }
    if (total > SIZE_MAX / (4 * sizeof(float))) return -1;
    if (writing) {
        if (!tr_optimizer_valid(optimizer)) return -1;
        temporary = *optimizer;
        memcpy(magic, expected_magic, 8);
    } else {
        staging = (float *)malloc(total * 4 * sizeof(float));
        if (staging == NULL) return -1;
    }
    if (tr_checkpoint_bytes(file, writing, magic, 8, &hash) != 0 || memcmp(magic, expected_magic, 8) != 0 ||
        tr_checkpoint_u64(file, writing, &stored_count, &hash) != 0 || stored_count != count ||
        tr_checkpoint_u64(file, writing, &temporary.step, &hash) != 0) goto cleanup;
    float *settings[] = {&temporary.learning_rate, &temporary.beta1, &temporary.beta2,
        &temporary.epsilon, &temporary.weight_decay, &temporary.max_grad_norm};
    for (size_t i = 0; i < 6; ++i) if (tr_checkpoint_float(file, writing, settings[i], &hash) != 0) goto cleanup;
    if (!tr_optimizer_valid(&temporary)) goto cleanup;
    for (size_t p = 0; p < count; ++p) {
        nya_train_parameter *w = parameters[p];
        uint64_t rows = w->rows, columns = w->columns;
        if (tr_checkpoint_u64(file, writing, &rows, &hash) != 0 || tr_checkpoint_u64(file, writing, &columns, &hash) != 0 ||
            rows != w->rows || columns != w->columns) goto cleanup;
        float *arrays[] = {w->data, w->gradient, w->moment, w->variance};
        for (size_t state = 0; state < 4; ++state) for (size_t i = 0; i < w->count; ++i) {
            float value = writing ? arrays[state][i] : 0.0f;
            if (tr_checkpoint_float(file, writing, &value, &hash) != 0 || (state == 3 && value < 0)) goto cleanup;
            if (!writing) staging[(offset + i) * 4 + state] = value;
        }
        offset += w->count;
    }
    checksum = hash;
    if (tr_checkpoint_u64(file, writing, &checksum, NULL) != 0 || checksum != hash) goto cleanup;
    if (writing) { if (fflush(file) != 0) goto cleanup; }
    else {
        if (fgetc(file) != EOF || ferror(file)) goto cleanup;
        offset = 0;
        for (size_t p = 0; p < count; ++p) {
            nya_train_parameter *w = parameters[p];
            float *arrays[] = {w->data, w->gradient, w->moment, w->variance};
            for (size_t state = 0; state < 4; ++state) for (size_t i = 0; i < w->count; ++i)
                arrays[state][i] = staging[(offset + i) * 4 + state];
            offset += w->count;
        }
        *optimizer = temporary;
    }
    result = 0;
cleanup:
    free(staging);
    return result;
}

int nya_train_checkpoint_write(FILE *file, const nya_train_adamw *optimizer,
    nya_train_parameter *const *parameters, size_t count)
{
    if (optimizer == NULL) return -1;
    nya_train_adamw copy = *optimizer;
    return tr_checkpoint(file, 1, &copy, parameters, count);
}

int nya_train_checkpoint_read(FILE *file, nya_train_adamw *optimizer,
    nya_train_parameter *const *parameters, size_t count)
{
    return tr_checkpoint(file, 0, optimizer, parameters, count);
}
