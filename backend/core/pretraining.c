#include "pretraining.h"
#include "training_internal.h"
#include "model.h"

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct nya_decoder_weight {
    char name[128];
    size_t rows, columns;
    int vector;
    const nya_llm_tensor *frozen;
    nya_train_parameter *weight, *a, *b;
} nya_decoder_weight;

struct nya_train_decoder {
    nya_train_decoder_config config;
    nya_decoder_weight *weights;
    size_t weight_count, parameter_count, parameter_bytes, rank;
    float lora_scale;
    nya_train_parameter **parameters;
    const nya_llm_context *source;
    uint64_t random;
    float *frequencies;
    size_t frequency_stride;
};

static void decoder_error(char *error, size_t capacity, const char *message)
{
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

void nya_train_decoder_defaults(nya_train_decoder_config *c)
{
    if (c == NULL) return;
    *c = (nya_train_decoder_config){259,64,128,2,4,2,256,1e-5f,10000.0f,0.02f,42,256U*1024U*1024U};
}

static double decoder_uniform(nya_train_decoder *m)
{
    m->random ^= m->random >> 12; m->random ^= m->random << 25; m->random ^= m->random >> 27;
    uint64_t word = m->random * UINT64_C(0x2545F4914F6CDD1D);
    return ((double)(word >> 11) + 0.5) / 9007199254740992.0;
}

/* Initialization belongs to the model, not a global RNG. Box-Muller starts
   every trainable matrix from reproducible Gaussian weights; norm scales are
   exactly one. LoRA B starts at zero so the initial policy equals its base. */
static nya_train_parameter *decoder_parameter(nya_train_decoder *m, size_t rows, size_t columns, int initialize)
{
    if (rows == 0 || columns == 0 || rows > SIZE_MAX / columns || rows * columns > (SIZE_MAX - 64) / (4 * sizeof(float))) return NULL;
    size_t cost = rows * columns * 4 * sizeof(float) + 64;
    if (cost > m->config.parameter_memory_limit - m->parameter_bytes) return NULL;
    nya_train_parameter *p = nya_train_parameter_create(rows, columns, NULL);
    if (p == NULL) return NULL;
    m->parameter_bytes += cost; m->parameters[m->parameter_count++] = p;
    float *values = nya_train_parameter_data(p);
    for (size_t i = 0; i < rows * columns; ++i) {
        if (initialize == 1) values[i] = 1.0f;
        else if (initialize == 2) values[i] = (float)(sqrt(-2.0 * log(decoder_uniform(m))) *
            cos(6.2831853071795864769 * decoder_uniform(m)) * m->config.initializer_std);
    }
    return p;
}

static int decoder_weight(nya_train_decoder *m, size_t index, const char *name,
    size_t rows, size_t columns, int vector)
{
    nya_decoder_weight *w = &m->weights[index];
    snprintf(w->name, sizeof(w->name), "%s", name);
    w->rows = rows; w->columns = columns; w->vector = vector;
    if (m->source != NULL) {
        for (size_t i = 0; i < m->source->tensor_count; ++i)
            if (strcmp(m->source->tensors[i].name, name) == 0) w->frozen = &m->source->tensors[i];
        if (w->frozen == NULL) return -1;
    }
    if (m->rank == 0) {
        w->weight = decoder_parameter(m,rows,columns,m->source == NULL ? (vector ? 1 : 2) : 0);
        if (w->weight == NULL) return -1;
        if (w->frozen != NULL) {
            float *values = nya_train_parameter_data(w->weight);
            for (size_t i = 0; i < rows * columns; ++i) {
                values[i] = nya_llm_tensor_value(w->frozen,i);
                if (!isfinite(values[i])) return -1;
            }
        }
    } else if (!vector && index != 0) {
        w->a = decoder_parameter(m,m->rank,columns,2);
        w->b = decoder_parameter(m,rows,m->rank,0);
        if (w->a == NULL || w->b == NULL) return -1;
    }
    return 0;
}

static nya_train_decoder *decoder_create(const nya_train_decoder_config *c,
    const nya_llm_context *source, size_t rank, float alpha, char *error, size_t capacity)
{
    if (c == NULL || c->vocabulary_size < 3 || c->vocabulary_size > 1000000 ||
        c->embedding_length == 0 || c->embedding_length > 16384 || c->feed_forward_length == 0 || c->feed_forward_length > 65536 ||
        c->block_count == 0 || c->block_count > 256 || c->head_count == 0 || c->kv_head_count == 0 ||
        c->embedding_length % c->head_count != 0 || (c->embedding_length / c->head_count) % 2 != 0 ||
        c->head_count % c->kv_head_count != 0 || c->context_length == 0 || c->context_length > 1048576 ||
        !isfinite(c->norm_epsilon) || c->norm_epsilon <= 0 || !isfinite(c->rope_base) || c->rope_base < 1 ||
        !isfinite(c->initializer_std) || c->initializer_std <= 0 || c->initializer_std > 1 ||
        c->parameter_memory_limit == 0 || rank > 256 || (rank != 0 && (source == NULL || !isfinite(alpha) || alpha <= 0))) {
        decoder_error(error,capacity,"invalid dense decoder configuration"); return NULL;
    }
    nya_train_decoder *m = (nya_train_decoder *)calloc(1,sizeof(*m));
    if (m == NULL) { decoder_error(error,capacity,"decoder allocation failed"); return NULL; }
    m->config = *c; m->source = source; m->rank = rank; m->lora_scale = rank == 0 ? 0 : alpha / (float)rank;
    m->random = c->seed == 0 ? UINT64_C(0x9E3779B97F4A7C15) : c->seed;
    m->weight_count = (size_t)c->block_count * 9 + 3;
    m->weights = (nya_decoder_weight *)calloc(m->weight_count,sizeof(*m->weights));
    m->parameters = (nya_train_parameter **)calloc(m->weight_count * 2,sizeof(*m->parameters));
    size_t d = c->embedding_length, h = c->feed_forward_length, hd = d / c->head_count, kv = hd * c->kv_head_count;
    m->frequencies = (float *)malloc(hd / 2 * sizeof(float));
    if (m->weights == NULL || m->parameters == NULL || m->frequencies == NULL) goto failure;
    for (size_t j = 0; j < hd / 2; ++j) m->frequencies[j] = powf(c->rope_base,-2.0f*(float)j/(float)hd);
    if (decoder_weight(m,0,"token_embd.weight",c->vocabulary_size,d,0) != 0 ||
        decoder_weight(m,1,"output_norm.weight",1,d,1) != 0 ||
        decoder_weight(m,2,"output.weight",c->vocabulary_size,d,0) != 0) goto failure;
    for (size_t layer = 0; layer < c->block_count; ++layer) {
        const char *names[] = {"attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_down", "ffn_up"};
        size_t rows[] = {1,d,kv,kv,d,1,h,d,h};
        size_t columns[] = {d,d,d,d,d,d,d,h,d};
        for (size_t i = 0; i < 9; ++i) {
            char name[64]; snprintf(name,sizeof(name),"blk.%zu.%s.weight",layer,names[i]);
            if (decoder_weight(m,3+layer*9+i,name,rows[i],columns[i],i==0||i==5) != 0) goto failure;
        }
    }
    return m;
failure:
    decoder_error(error,capacity,"decoder weights are unavailable, invalid, or exceed the parameter memory budget");
    nya_train_decoder_free(m); return NULL;
}

nya_train_decoder *nya_train_decoder_create(const nya_train_decoder_config *c, char *error, size_t capacity)
{
    return decoder_create(c,NULL,0,0,error,capacity);
}

/* Shared-KV readers do not execute their retained K/V tensors. Freeze those
   unused tensors, and frequency factors, instead of allocating optimizer state
   for values outside the differentiable graph. Pointer identity also keeps
   tied input/output embeddings represented by exactly one parameter. */
static int decoder_gemma_used(const nya_llm_context *c, const nya_llm_tensor *t)
{
    if (t == c->token_embedding || t == c->output || t == c->output_norm ||
        t == c->ple_embedding || t == c->ple_projection || t == c->ple_norm) return 1;
    for (size_t i = 0; i < c->block_count; ++i) {
        const nya_llm_layer *w = &c->layers[i];
        if (t == w->attention_norm || t == w->query || t == w->query_norm ||
            t == w->attention_output || t == w->attention_post_norm || t == w->feed_forward_norm ||
            t == w->feed_forward_gate || t == w->feed_forward_up || t == w->feed_forward_down ||
            t == w->ffn_post_norm || t == w->ple_gate || t == w->ple_projection ||
            t == w->ple_norm || t == w->output_scale) return 1;
        if (w->kv_source == i && (t == w->key || t == w->value || t == w->key_norm)) return 1;
    }
    return 0;
}

static nya_train_decoder *decoder_gemma_create(const nya_train_decoder_config *config,
    const nya_llm_context *source, size_t rank, float alpha, char *error, size_t capacity)
{
    if (rank > 256 || config->parameter_memory_limit == 0 ||
        (rank != 0 && (!isfinite(alpha) || alpha <= 0))) {
        decoder_error(error,capacity,"invalid Gemma training rank, alpha or memory budget"); return NULL;
    }
    nya_train_decoder *m = (nya_train_decoder *)calloc(1,sizeof(*m));
    if (m == NULL) { decoder_error(error,capacity,"Gemma training allocation failed"); return NULL; }
    m->config = *config; m->source = source; m->rank = rank; m->random = config->seed;
    m->lora_scale = rank == 0 ? 0 : alpha/(float)rank;
    m->weight_count = source->tensor_count;
    m->weights = (nya_decoder_weight *)calloc(m->weight_count,sizeof(*m->weights));
    m->parameters = (nya_train_parameter **)calloc(m->weight_count*2,sizeof(*m->parameters));
    for (size_t i = 0; i < source->block_count; ++i)
        if (source->layers[i].head_dimension/2 > m->frequency_stride) m->frequency_stride = source->layers[i].head_dimension/2;
    m->frequencies = (float *)calloc(m->frequency_stride*source->block_count,sizeof(float));
    if (m->weights == NULL || m->parameters == NULL || m->frequencies == NULL) goto failure;
    for (size_t i = 0; i < m->weight_count; ++i) {
        const nya_llm_tensor *t = &source->tensors[i]; nya_decoder_weight *w = &m->weights[i];
        if (t->dimension_count == 0 || t->dimension_count > 2 || t->dimensions[0] > SIZE_MAX ||
            (t->dimension_count == 2 && t->dimensions[1] > SIZE_MAX)) goto failure;
        w->columns = (size_t)t->dimensions[0]; w->rows = t->dimension_count == 1 ? 1 : (size_t)t->dimensions[1];
        w->vector = t->dimension_count == 1; w->frozen = t;
        snprintf(w->name,sizeof(w->name),"%s",t->name);
        if (!decoder_gemma_used(source,t)) continue;
        if (rank == 0) {
            w->weight = decoder_parameter(m,w->rows,w->columns,0);
            if (w->weight == NULL) goto failure;
            float *data = nya_train_parameter_data(w->weight);
            for (size_t j = 0; j < w->rows*w->columns; ++j) {
                data[j] = nya_llm_tensor_value(t,j); if (!isfinite(data[j])) goto failure;
            }
        } else if (!w->vector && t != source->token_embedding && t != source->ple_embedding) {
            /* Tied heads share the frozen embedding matrix. Adapting only the
               head would break that tie when merging, so they remain frozen. */
            w->a = decoder_parameter(m,rank,w->columns,2);
            w->b = decoder_parameter(m,w->rows,rank,0);
            if (w->a == NULL || w->b == NULL) goto failure;
        }
    }
    for (size_t i = 0; i < source->block_count; ++i) {
        const nya_llm_layer *w = &source->layers[i]; size_t pairs = w->head_dimension/2;
        float base = w->sliding_window != 0 ? source->sliding_rope_base : source->rope_frequency_base;
        for (size_t j = 0; j < pairs; ++j) {
            float factor = w->rope_factors == NULL ? 1 : nya_llm_tensor_value(w->rope_factors,j);
            if (!isfinite(factor) || factor <= 0) goto failure;
            float frequency = powf(base,-(float)j/(float)pairs)/factor;
            if (!isfinite(frequency)) goto failure;
            m->frequencies[i*m->frequency_stride+j] = frequency;
        }
    }
    return m;
failure:
    decoder_error(error,capacity,"Gemma training weights are invalid or exceed the parameter budget");
    nya_train_decoder_free(m); return NULL;
}

nya_train_decoder *nya_train_decoder_from_model(nya_model *model, size_t rank, float alpha,
    size_t memory_limit, char *error, size_t capacity)
{
    const nya_llm_context *source = model == NULL ? NULL : (const nya_llm_context *)model->generation_context;
    if (source == NULL || !model->generation_supported || source->is_assistant || source->expert_count != 0 ||
        (!source->is_gemma && source->output == source->token_embedding)) {
        decoder_error(error,capacity,"training requires a native dense Gemma 4 or an untied dense LLaMA; MoE/MTP training is not implemented"); return NULL;
    }
    nya_train_decoder_config c;
    nya_train_decoder_defaults(&c);
    c.vocabulary_size = source->tokenizer.vocabulary_size; c.embedding_length = source->embedding_length;
    c.feed_forward_length = source->feed_forward_length; c.block_count = source->block_count;
    c.head_count = source->head_count; c.kv_head_count = source->key_value_head_count; c.context_length = source->context_length;
    c.norm_epsilon = source->norm_epsilon; c.rope_base = source->rope_frequency_base; c.parameter_memory_limit = memory_limit;
    return source->is_gemma ? decoder_gemma_create(&c,source,rank,alpha,error,capacity) :
        decoder_create(&c,source,rank,alpha,error,capacity);
}

void nya_train_decoder_free(nya_train_decoder *m)
{
    if (m == NULL) return;
    for (size_t i = 0; i < m->parameter_count; ++i) nya_train_parameter_free(m->parameters[i]);
    free(m->parameters); free(m->weights); free(m->frequencies); free(m);
}

nya_train_parameter *const *nya_train_decoder_parameters(nya_train_decoder *m, size_t *count)
{
    if (count != NULL) *count = m == NULL ? 0 : m->parameter_count;
    return m == NULL ? NULL : m->parameters;
}

const nya_train_decoder_config *nya_train_decoder_configuration(const nya_train_decoder *m)
{
    return m == NULL ? NULL : &m->config;
}

int nya_train_decoder_tokenize(nya_train_decoder *m, const char *text,
    uint32_t **tokens, size_t *count, char *error, size_t capacity)
{
    if (tokens != NULL) *tokens = NULL;
    if (count != NULL) *count = 0;
    if (m == NULL || text == NULL || tokens == NULL || count == NULL) return -1;
    if (m->source != NULL) return nya_llm_tokenize(m->source,text,tokens,count,error,capacity);
    size_t length = strlen(text);
    if (!nya_llm_text_is_utf8(text,length)) { decoder_error(error,capacity,"training text is not valid UTF-8"); return -1; }
    if (m->config.vocabulary_size != 259 || length >= SIZE_MAX/sizeof(uint32_t)) {
        decoder_error(error,capacity,"byte tokenization requires vocabulary_size=259 and a bounded input"); return -1;
    }
    uint32_t *ids = (uint32_t *)malloc((length+1)*sizeof(*ids));
    if (ids == NULL) { decoder_error(error,capacity,"token allocation failed"); return -1; }
    ids[0] = 1;
    for (size_t i = 0; i < length; ++i) ids[i+1] = (uint32_t)(unsigned char)text[i]+3;
    *tokens = ids; *count = length+1; return 0;
}

static nya_train_tensor *decoder_norm(nya_train_decoder *m, nya_train_graph *g, nya_train_tensor *x, size_t index)
{
    nya_decoder_weight *w = &m->weights[index];
    nya_train_tensor *scale = w->weight != NULL ? nya_train_leaf(g,w->weight) : nya_train_constant_mapped(g,w->frozen);
    return nya_train_rms_norm(x,scale,m->config.norm_epsilon);
}

static nya_train_tensor *decoder_linear(nya_train_decoder *m, nya_train_graph *g, nya_train_tensor *x, size_t index)
{
    nya_decoder_weight *w = &m->weights[index];
    if (w->weight != NULL) return nya_train_linear(x,nya_train_leaf(g,w->weight));
    nya_train_tensor *base = nya_train_linear_mapped(x,w->frozen);
    if (w->a == NULL) return base;
    nya_train_tensor *adapter = nya_train_linear(nya_train_linear(x,nya_train_leaf(g,w->a)),nya_train_leaf(g,w->b));
    return nya_train_add(base,nya_train_scale(adapter,m->lora_scale));
}

/* Gemma entries preserve the source tensor directory order. All tensor
   bindings below originate in that same validated array. This gives O(1)
   lookup without names or duplicate leaves for shared weight storage. */
static size_t decoder_index(const nya_train_decoder *m, const nya_llm_tensor *t)
{
    return (size_t)(t-m->source->tensors);
}

static nya_train_tensor *decoder_gemma_linear(nya_train_decoder *m, nya_train_graph *g,
    nya_train_tensor *x, const nya_llm_tensor *t)
{
    return decoder_linear(m,g,x,decoder_index(m,t));
}

static nya_train_tensor *decoder_gemma_norm(nya_train_decoder *m, nya_train_graph *g,
    nya_train_tensor *x, const nya_llm_tensor *t)
{
    return t == NULL ? nya_train_rms_norm(x,NULL,m->config.norm_epsilon) : decoder_norm(m,g,x,decoder_index(m,t));
}

static nya_train_tensor *decoder_gemma_embedding(nya_train_decoder *m, nya_train_graph *g,
    const nya_llm_tensor *t, const uint32_t *tokens, size_t count)
{
    nya_decoder_weight *w = &m->weights[decoder_index(m,t)];
    return w->weight == NULL ? nya_train_embedding_mapped(g,t,tokens,count) :
        nya_train_embedding(nya_train_leaf(g,w->weight),tokens,count);
}

static nya_train_tensor *decoder_gemma_head_norm(nya_train_decoder *m, nya_train_graph *g,
    nya_train_tensor *x, const nya_llm_tensor *weight, size_t tokens, size_t heads, size_t width)
{
    /* RMSNorm operates separately on each head, not across concatenated heads.
       Reshape preserves order and autograd routes the shared scale gradients. */
    return nya_train_reshape(decoder_gemma_norm(m,g,nya_train_reshape(x,tokens*heads,width),weight),tokens,heads*width);
}

static nya_train_tensor *decoder_gemma_forward(nya_train_decoder *m, nya_train_graph *g,
    const uint32_t *tokens, size_t count)
{
    const nya_llm_context *c = m->source;
    size_t d = c->embedding_length, ple = c->per_layer_embedding_length;
    nya_train_tensor *keys[256] = {0}, *values[256] = {0};
    nya_train_tensor *x = nya_train_scale(decoder_gemma_embedding(m,g,c->token_embedding,tokens,count),sqrtf((float)d));
    nya_train_tensor *ple_input = NULL;
    if (ple != 0) {
        size_t packed = ple*c->block_count;
        nya_train_tensor *projection = nya_train_scale(decoder_gemma_linear(m,g,x,c->ple_projection),1.0f/sqrtf((float)d));
        projection = nya_train_reshape(decoder_gemma_norm(m,g,nya_train_reshape(projection,count*c->block_count,ple),c->ple_norm),count,packed);
        nya_train_tensor *identity = nya_train_scale(decoder_gemma_embedding(m,g,c->ple_embedding,tokens,count),sqrtf((float)ple));
        ple_input = nya_train_scale(nya_train_add(projection,identity),0.7071067811865475f);
    }
    for (size_t layer = 0; layer < c->block_count; ++layer) {
        const nya_llm_layer *w = &c->layers[layer];
        size_t hd = w->head_dimension, heads = w->head_count, kv_heads = w->kv_head_count;
        const float *freq = m->frequencies+layer*m->frequency_stride;
        nya_train_tensor *n = decoder_gemma_norm(m,g,x,w->attention_norm);
        nya_train_tensor *q = decoder_gemma_head_norm(m,g,decoder_gemma_linear(m,g,n,w->query),w->query_norm,count,heads,hd);
        q = nya_train_rope(q,heads,hd,freq,1);
        if (w->kv_source == layer) {
            nya_train_tensor *raw_key = decoder_gemma_linear(m,g,n,w->key);
            nya_train_tensor *raw_value = w->value == NULL ? raw_key : decoder_gemma_linear(m,g,n,w->value);
            keys[layer] = nya_train_rope(decoder_gemma_head_norm(m,g,raw_key,w->key_norm,count,kv_heads,hd),kv_heads,hd,freq,1);
            values[layer] = decoder_gemma_head_norm(m,g,raw_value,NULL,count,kv_heads,hd);
        } else {
            /* Borrow the earlier layer's differentiable tensors. Every reader
               contributes gradients to the same KV source through the DAG. */
            keys[layer] = keys[w->kv_source]; values[layer] = values[w->kv_source];
        }
        nya_train_tensor *attention = nya_train_attention(q,keys[layer],values[layer],heads,kv_heads,hd,1,w->sliding_window,NULL);
        nya_train_tensor *branch = decoder_gemma_norm(m,g,decoder_gemma_linear(m,g,attention,w->attention_output),w->attention_post_norm);
        x = nya_train_add(x,branch);
        n = decoder_gemma_norm(m,g,x,w->feed_forward_norm);
        nya_train_tensor *gate = nya_train_gelu(decoder_gemma_linear(m,g,n,w->feed_forward_gate));
        nya_train_tensor *up = decoder_gemma_linear(m,g,n,w->feed_forward_up);
        branch = decoder_gemma_linear(m,g,nya_train_mul(gate,up),w->feed_forward_down);
        x = nya_train_add(x,decoder_gemma_norm(m,g,branch,w->ffn_post_norm));
        if (ple != 0) {
            gate = nya_train_gelu(decoder_gemma_linear(m,g,x,w->ple_gate));
            gate = nya_train_mul(gate,nya_train_slice_columns(ple_input,layer*ple,ple));
            branch = decoder_gemma_linear(m,g,gate,w->ple_projection);
            x = nya_train_add(x,decoder_gemma_norm(m,g,branch,w->ple_norm));
        }
        if (w->output_scale != NULL) {
            nya_decoder_weight *scale = &m->weights[decoder_index(m,w->output_scale)];
            nya_train_tensor *scalar = scale->weight == NULL ? nya_train_constant_mapped(g,scale->frozen) : nya_train_leaf(g,scale->weight);
            x = nya_train_mul(x,scalar);
        }
    }
    nya_train_tensor *logits = decoder_gemma_linear(m,g,decoder_gemma_norm(m,g,x,c->output_norm),c->output);
    /* Suppression is a generation policy. Training returns the finite model
       logits over its full vocabulary, as required by the language-model loss. */
    return c->final_logit_softcap > 0 ? nya_train_softcap(logits,c->final_logit_softcap) : logits;
}

nya_train_tensor *nya_train_decoder_forward(nya_train_decoder *m, nya_train_graph *g, const uint32_t *tokens, size_t count)
{
    if (m == NULL || g == NULL || tokens == NULL || count == 0 || count > m->config.context_length) {
        nya_train_graph_fail(g,"invalid decoder training sequence"); return NULL;
    }
    if (m->source != NULL && m->source->is_gemma) return decoder_gemma_forward(m,g,tokens,count);
    size_t d = m->config.embedding_length, heads = m->config.head_count, kv_heads = m->config.kv_head_count, hd = d / heads;
    nya_decoder_weight *embedding = &m->weights[0];
    nya_train_tensor *x = embedding->weight != NULL ? nya_train_embedding(nya_train_leaf(g,embedding->weight),tokens,count) :
        nya_train_embedding_mapped(g,embedding->frozen,tokens,count);
    for (size_t layer = 0; layer < m->config.block_count; ++layer) {
        size_t base = 3 + layer * 9;
        nya_train_tensor *n = decoder_norm(m,g,x,base);
        nya_train_tensor *q = nya_train_rope(decoder_linear(m,g,n,base+1),heads,hd,m->frequencies,0);
        nya_train_tensor *k = nya_train_rope(decoder_linear(m,g,n,base+2),kv_heads,hd,m->frequencies,0);
        nya_train_tensor *v = decoder_linear(m,g,n,base+3);
        nya_train_tensor *attention = nya_train_attention(q,k,v,heads,kv_heads,hd,1.0f/sqrtf((float)hd),0,NULL);
        x = nya_train_add(x,decoder_linear(m,g,attention,base+4));
        n = decoder_norm(m,g,x,base+5);
        nya_train_tensor *gate = nya_train_silu(decoder_linear(m,g,n,base+6));
        nya_train_tensor *up = decoder_linear(m,g,n,base+8);
        x = nya_train_add(x,decoder_linear(m,g,nya_train_mul(gate,up),base+7));
    }
    return decoder_linear(m,g,decoder_norm(m,g,x,1),2);
}

static int decoder_u32(FILE *f, uint32_t value)
{
    unsigned char bytes[4]; for (size_t i = 0; i < 4; ++i) bytes[i] = (unsigned char)(value >> (8*i));
    return fwrite(bytes,1,4,f)==4 ? 0 : -1;
}
static int decoder_u64(FILE *f, uint64_t value)
{
    unsigned char bytes[8]; for (size_t i = 0; i < 8; ++i) bytes[i] = (unsigned char)(value >> (8*i));
    return fwrite(bytes,1,8,f)==8 ? 0 : -1;
}
static int decoder_string(FILE *f, const char *text)
{
    size_t count = strlen(text);
    return decoder_u64(f,count)==0 && fwrite(text,1,count,f)==count ? 0 : -1;
}
static int decoder_float(FILE *f, float value)
{
    uint32_t bits; memcpy(&bits,&value,4); return decoder_u32(f,bits);
}
static int decoder_meta_int(FILE *f, const char *key, uint32_t value)
{
    return decoder_string(f,key)==0 && decoder_u32(f,4)==0 && decoder_u32(f,value)==0 ? 0 : -1;
}
static int decoder_meta_float(FILE *f, const char *key, float value)
{
    return decoder_string(f,key)==0 && decoder_u32(f,6)==0 && decoder_float(f,value)==0 ? 0 : -1;
}
static int decoder_meta_string(FILE *f, const char *key, const char *value)
{
    return decoder_string(f,key)==0 && decoder_u32(f,8)==0 && decoder_string(f,value)==0 ? 0 : -1;
}
static int decoder_meta_array(FILE *f, const char *key, uint32_t type, uint64_t count)
{
    return decoder_string(f,key)==0 && decoder_u32(f,9)==0 && decoder_u32(f,type)==0 && decoder_u64(f,count)==0 ? 0 : -1;
}

static int decoder_padding(FILE *f, uint64_t bytes, uint32_t alignment)
{
    uint64_t remainder = bytes % alignment;
    for (uint64_t i = remainder; i != 0 && i < alignment; ++i) if (fputc(0,f) == EOF) return -1;
    return 0;
}

/* Emit merged values in bounded little-endian batches. Frozen tensors are
   copied separately in their original representation; no whole-model dense
   temporary is needed even when adapting a multi-gigabyte quantized model. */
static int decoder_write_weight(nya_train_decoder *m, FILE *f, const nya_decoder_weight *w)
{
    const float *full = nya_train_parameter_data(w->weight), *a = nya_train_parameter_data(w->a), *b = nya_train_parameter_data(w->b);
    unsigned char bytes[4096]; size_t used = 0;
    for (size_t row = 0; row < w->rows; ++row) for (size_t col = 0; col < w->columns; ++col) {
        size_t index = row*w->columns+col;
        double value = full == NULL ? nya_llm_tensor_value(w->frozen,index) : full[index];
        if (a != NULL) for (size_t rank = 0; rank < m->rank; ++rank)
            value += (double)m->lora_scale*b[row*m->rank+rank]*a[rank*w->columns+col];
        if (!isfinite(value) || fabs(value) > FLT_MAX) return -1;
        float output = (float)value; uint32_t bits; memcpy(&bits,&output,4);
        for (size_t j = 0; j < 4; ++j) bytes[used++] = (unsigned char)(bits >> (8*j));
        if (used == sizeof(bytes)) { if (fwrite(bytes,1,used,f) != used) return -1; used = 0; }
    }
    return used == 0 || fwrite(bytes,1,used,f) == used ? 0 : -1;
}

static int decoder_export_gemma(nya_train_decoder *m, FILE *f)
{
    const nya_llm_context *c = m->source; uint64_t offset = 0, header = c->metadata_end;
    uint32_t alignment = c->tensor_alignment;
    if (alignment == 0 || c->metadata_end > c->mapping.size) return -1;
    /* general.file_type summarizes the old quantization. Omit that optional
       summary after mixed-precision export; the rebuilt tensor directory is
       authoritative. Every other metadata value is preserved verbatim. */
    if (c->file_type_begin != 0) {
        size_t first = c->file_type_begin, end = c->file_type_end;
        if (first < 24 || end < first || end > c->metadata_end || c->gguf_metadata_count == 0 ||
            fwrite(c->mapping.data,1,16,f) != 16 || decoder_u64(f,c->gguf_metadata_count-1) != 0 ||
            fwrite(c->mapping.data+24,1,first-24,f) != first-24 ||
            fwrite(c->mapping.data+end,1,c->metadata_end-end,f) != c->metadata_end-end) return -1;
        header -= end-first;
    } else if (fwrite(c->mapping.data,1,c->metadata_end,f) != c->metadata_end) return -1;
    for (size_t i = 0; i < m->weight_count; ++i) {
        const nya_decoder_weight *w = &m->weights[i]; const nya_llm_tensor *t = w->frozen;
        int changed = w->weight != NULL || w->a != NULL;
        uint64_t bytes = changed ? (uint64_t)w->rows*w->columns*4 : t->data_size;
        if (decoder_string(f,t->name) != 0 || decoder_u32(f,t->dimension_count) != 0) return -1;
        for (size_t j = 0; j < t->dimension_count; ++j) if (decoder_u64(f,t->dimensions[j]) != 0) return -1;
        if (decoder_u32(f,changed ? NYA_LLM_TENSOR_F32 : t->type) != 0 || decoder_u64(f,offset) != 0 ||
            offset > UINT64_MAX-bytes-(alignment-1)) return -1;
        offset = (offset+bytes+alignment-1)/alignment*alignment;
        header += 8+strlen(t->name)+4+8*t->dimension_count+4+8;
    }
    if (decoder_padding(f,header,alignment) != 0) return -1;
    for (size_t i = 0; i < m->weight_count; ++i) {
        const nya_decoder_weight *w = &m->weights[i]; uint64_t bytes;
        if (w->weight != NULL || w->a != NULL) {
            bytes = (uint64_t)w->rows*w->columns*4;
            if (decoder_write_weight(m,f,w) != 0) return -1;
        } else {
            bytes = w->frozen->data_size;
            for (size_t begin = 0; begin < w->frozen->data_size;) {
                size_t size = w->frozen->data_size-begin; if (size > 1048576) size = 1048576;
                if (fwrite(w->frozen->data+begin,1,size,f) != size) return -1;
                begin += size;
            }
        }
        if (decoder_padding(f,bytes,alignment) != 0) return -1;
    }
    return fflush(f) == 0 ? 0 : -1;
}

int nya_train_decoder_export(nya_train_decoder *m, FILE *f, char *error, size_t capacity)
{
    if (m == NULL || f == NULL || (m->source == NULL && m->config.vocabulary_size != 259)) {
        decoder_error(error,capacity,"export requires an imported tokenizer or the 259-token byte vocabulary"); return -1;
    }
    if (ftell(f) != 0) { decoder_error(error,capacity,"GGUF export requires a binary stream positioned at its beginning"); return -1; }
    if (m->source != NULL && m->source->is_gemma) {
        if (decoder_export_gemma(m,f) == 0) return 0;
        decoder_error(error,capacity,"Gemma GGUF export failed; discard the incomplete output stream"); return -1;
    }
    const nya_train_decoder_config *c = &m->config;
    const nya_llm_tokenizer *tok = m->source == NULL ? NULL : &m->source->tokenizer;
    uint64_t offset = 0;
#define WRITE(x) do { if ((x) != 0) goto failure; } while (0)
    if (fwrite("GGUF",1,4,f)!=4) goto failure;
    WRITE(decoder_u32(f,3)); WRITE(decoder_u64(f,m->weight_count));
    WRITE(decoder_u64(f,20U+(tok != NULL && tok->suppressed_count != 0 ? 1U : 0U)));
    WRITE(decoder_meta_string(f,"general.architecture","llama"));
    WRITE(decoder_meta_int(f,"llama.context_length",c->context_length));
    WRITE(decoder_meta_int(f,"llama.embedding_length",c->embedding_length));
    WRITE(decoder_meta_int(f,"llama.feed_forward_length",c->feed_forward_length));
    WRITE(decoder_meta_int(f,"llama.block_count",c->block_count));
    WRITE(decoder_meta_int(f,"llama.attention.head_count",c->head_count));
    WRITE(decoder_meta_int(f,"llama.attention.head_count_kv",c->kv_head_count));
    WRITE(decoder_meta_int(f,"llama.rope.dimension_count",c->embedding_length/c->head_count));
    WRITE(decoder_meta_float(f,"llama.attention.layer_norm_rms_epsilon",c->norm_epsilon));
    WRITE(decoder_meta_float(f,"llama.rope.freq_base",c->rope_base));
    WRITE(decoder_meta_string(f,"tokenizer.ggml.model","llama"));
    WRITE(decoder_meta_array(f,"tokenizer.ggml.tokens",8,c->vocabulary_size));
    for (uint32_t i = 0; i < c->vocabulary_size; ++i) {
        char piece[16];
        if (tok != NULL) WRITE(decoder_string(f,tok->pieces[i]));
        else if (i < 3) { const char *special[] = {"<unk>","<s>","</s>"}; WRITE(decoder_string(f,special[i])); }
        else { snprintf(piece,sizeof(piece),"<0x%02X>",(unsigned int)(i-3)); WRITE(decoder_string(f,piece)); }
    }
    WRITE(decoder_meta_array(f,"tokenizer.ggml.scores",6,c->vocabulary_size));
    for (uint32_t i = 0; i < c->vocabulary_size; ++i) WRITE(decoder_float(f,tok == NULL ? 0 : tok->scores[i]));
    WRITE(decoder_meta_array(f,"tokenizer.ggml.token_type",5,c->vocabulary_size));
    for (uint32_t i = 0; i < c->vocabulary_size; ++i) WRITE(decoder_u32(f,tok == NULL ? (i == 0 ? 2U : i < 3 ? 3U : 6U) : (uint32_t)tok->types[i]));
    WRITE(decoder_meta_int(f,"tokenizer.ggml.bos_token_id",tok == NULL ? 1 : tok->beginning_token));
    WRITE(decoder_meta_int(f,"tokenizer.ggml.eos_token_id",tok == NULL ? 2 : tok->end_token));
    WRITE(decoder_meta_int(f,"tokenizer.ggml.unknown_token_id",tok == NULL ? 0 : tok->unknown_token));
    const char *flags[] = {"tokenizer.ggml.add_bos_token","tokenizer.ggml.add_eos_token","tokenizer.ggml.add_space_prefix"};
    int values[] = {tok == NULL ? 1 : tok->add_beginning_token,tok == NULL ? 0 : tok->add_end_token,tok == NULL ? 0 : tok->add_space_prefix};
    for (size_t i = 0; i < 3; ++i) { WRITE(decoder_string(f,flags[i])); WRITE(decoder_u32(f,7)); if (fputc(values[i]!=0,f)==EOF) goto failure; }
    if (tok != NULL && tok->suppressed_count != 0) {
        WRITE(decoder_meta_array(f,"tokenizer.ggml.suppress_tokens",4,tok->suppressed_count));
        for (size_t i = 0; i < tok->suppressed_count; ++i) WRITE(decoder_u32(f,tok->suppressed_tokens[i]));
    }
    for (size_t i = 0; i < m->weight_count; ++i) {
        const nya_decoder_weight *w = &m->weights[i];
        uint64_t bytes = (uint64_t)w->rows * w->columns * 4;
        WRITE(decoder_string(f,w->name)); WRITE(decoder_u32(f,w->vector ? 1 : 2)); WRITE(decoder_u64(f,w->columns));
        if (!w->vector) WRITE(decoder_u64(f,w->rows));
        WRITE(decoder_u32(f,0)); WRITE(decoder_u64(f,offset));
        if (offset > UINT64_MAX-bytes-31) goto failure;
        offset = (offset+bytes+31)/32*32;
    }
    /* Only the small header uses ftell; payloads can exceed Windows LONG_MAX.
       Each row is streamed directly from trainable or mapped base weights. */
    long header = ftell(f); if (header < 0) goto failure;
    for (size_t i = (size_t)header % 32; i != 0 && i < 32; ++i) if (fputc(0,f)==EOF) goto failure;
    for (size_t i = 0; i < m->weight_count; ++i) {
        const nya_decoder_weight *w = &m->weights[i];
        const float *full = nya_train_parameter_data(w->weight), *a = nya_train_parameter_data(w->a), *b = nya_train_parameter_data(w->b);
        for (size_t row = 0; row < w->rows; ++row) for (size_t col = 0; col < w->columns; ++col) {
            size_t index = row*w->columns+col;
            double value = full == NULL ? nya_llm_tensor_value(w->frozen,index) : full[index];
            if (a != NULL) for (size_t rank = 0; rank < m->rank; ++rank)
                value += (double)m->lora_scale*b[row*m->rank+rank]*a[rank*w->columns+col];
            if (!isfinite(value) || fabs(value) > FLT_MAX) goto failure;
            float output = (float)value;
            WRITE(decoder_float(f,output));
        }
        size_t remainder = (size_t)((uint64_t)w->rows*w->columns*4 % 32);
        for (size_t j = remainder; j != 0 && j < 32; ++j) if (fputc(0,f)==EOF) goto failure;
    }
    if (fflush(f) != 0) goto failure;
    return 0;
failure:
    decoder_error(error,capacity,"GGUF export failed; discard the incomplete output stream"); return -1;
#undef WRITE
}
