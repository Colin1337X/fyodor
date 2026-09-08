#include "llm_internal.h"

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One request owns all mutable transformer buffers and its complete KV cache. */
typedef struct nya_llm_run_state {
    float *activation;
    float *normalized;
    float *residual;
    float *feed_forward_gate;
    float *feed_forward_up;
    float *query;
    float *attention;
    float *logits;
    float *key_cache;
    float *value_cache;
    float *rope_frequencies;
    float *rope_cosines;
    float *rope_sines;
    float *ple_inputs, *ple_gate, *expert_input, *expert_output, *router_probabilities;
    uint32_t *expert_indices;
    size_t rope_stride;
    float *mtp_input, *mtp_hidden;
    uint32_t mtp_sources[256];
    size_t sequence_capacity;
    /* Layer-at-a-time prefill prepares all keys before evaluating vision
       attention. These borrowed overrides exist only during that operation. */
    const float *embedded_input, *layer_input;
    size_t layer_begin, layer_end, attention_extent;
    int prepare_kv_only;
} nya_llm_run_state;

/* Sampling candidates are sorted from greatest to least probability. */
typedef struct nya_llm_probability {
    float probability;
    uint32_t token;
} nya_llm_probability;

/* Store one bounded CPU provider error. */
static void nya_llm_cpu_error(char *error, size_t capacity, const char *format, ...)
{
    va_list arguments;

    if (error == NULL || capacity == 0) return;
    va_start(arguments, format);
    vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
    error[capacity - 1] = '\0';
}

/* Checked size multiplication protects every scratch and cache allocation. */
static int nya_llm_cpu_product(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left) return -1;
    *result = left * right;
    return 0;
}

/* Convert one IEEE binary16 value into the host float representation. */
static float nya_llm_half_to_float(const unsigned char *bytes)
{
    uint16_t half;
    uint32_t sign;
    int32_t exponent;
    uint32_t fraction;
    uint32_t bits;
    float value;

    half = (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
    sign = (uint32_t)(half & 0x8000U) << 16;
    exponent = (int32_t)((half >> 10) & 0x1FU);
    fraction = half & 0x03FFU;

    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((fraction & 0x0400U) == 0) {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03FFU;
            bits = sign | ((uint32_t)(exponent + 112) << 23) | (fraction << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000U | (fraction << 13);
    } else {
        bits = sign | ((uint32_t)(exponent + 112) << 23) | (fraction << 13);
    }

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Read one supported vector or matrix scalar for small non-matmul operations. */
static float nya_llm_k_quant_value(uint32_t type, const unsigned char *block, size_t index)
{
    if (type == NYA_LLM_TENSOR_Q4_K) {
        size_t group = index / 32;
        const unsigned char *s = block + 4;
        unsigned int scale = group < 4 ? s[group] & 63U : (s[group + 4] & 15U) | ((unsigned int)(s[group - 4] >> 6) << 4);
        unsigned int minimum = group < 4 ? s[group + 4] & 63U : (s[group + 4] >> 4) | ((unsigned int)(s[group] >> 6) << 4);
        unsigned int packed = block[16 + group / 2 * 32 + index % 32];
        unsigned int q = (packed >> (group % 2 * 4)) & 15U;
        return nya_llm_half_to_float(block) * (float)scale * (float)q - nya_llm_half_to_float(block + 2) * (float)minimum;
    } else {
        size_t half = index / 128, quarter = index % 128 / 32, lane = index % 32;
        unsigned int lo = block[half * 64 + quarter % 2 * 32 + lane];
        unsigned int hi = block[128 + half * 32 + lane];
        int q = (int)(((lo >> (quarter / 2 * 4)) & 15U) | (((hi >> (quarter * 2)) & 3U) << 4)) - 32;
        int8_t scale;
        memcpy(&scale, block + 192 + half * 8 + quarter * 2 + lane / 16, 1);
        return nya_llm_half_to_float(block + 208) * (float)scale * (float)q;
    }
}

float nya_llm_tensor_value(const nya_llm_tensor *tensor, size_t index)
{
    if (tensor->type == NYA_LLM_TENSOR_Q4_K || tensor->type == NYA_LLM_TENSOR_Q6_K) {
        size_t bytes = tensor->type == NYA_LLM_TENSOR_Q4_K ? 144 : 210;
        return nya_llm_k_quant_value(tensor->type, tensor->data + index / 256 * bytes, index % 256);
    }
    if (tensor->type == NYA_LLM_TENSOR_F32) {
        float value;

        memcpy(&value, tensor->data + index * 4, sizeof(value));
        return value;
    }
    if (tensor->type == NYA_LLM_TENSOR_F16) {
        return nya_llm_half_to_float(tensor->data + index * 2);
    }
    if (tensor->type == NYA_LLM_TENSOR_BF16) {
        const unsigned char *p = tensor->data + index * 2;
        uint32_t bits = ((uint32_t)p[0] | ((uint32_t)p[1] << 8)) << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
    if (tensor->type == NYA_LLM_TENSOR_Q8_0) {
        size_t block;
        size_t offset;
        const unsigned char *data;
        float scale;
        int8_t quantized;

        block = index / 32;
        offset = index % 32;
        data = tensor->data + block * 34;
        scale = nya_llm_half_to_float(data);
        memcpy(&quantized, data + 2 + offset, 1);
        return scale * quantized;
    }
    if (tensor->type == NYA_LLM_TENSOR_Q4_0) {
        size_t block;
        size_t offset;
        const unsigned char *data;
        unsigned char packed;
        int quantized;

        block = index / 32;
        offset = index % 32;
        data = tensor->data + block * 18;
        packed = data[2 + offset % 16];
        quantized = offset < 16 ? (packed & 0x0FU) : (packed >> 4);
        return nya_llm_half_to_float(data) * (float)(quantized - 8);
    }
    return 0.0f;
}

/* Copy one matrix row, such as a token embedding, into float scratch memory. */
static void nya_llm_copy_row(float *output, const nya_llm_tensor *tensor, size_t row, size_t columns)
{
    size_t column;
    size_t base;

    base = row * columns;
    for (column = 0; column < columns; ++column) {
        output[column] = nya_llm_tensor_value(tensor, base + column);
    }
}

/* Multiply one supported GGML matrix by a dense float vector. */
void nya_llm_matvec(
    nya_compute_context *compute,
    float *output,
    const nya_llm_tensor *matrix,
    const float *input,
    size_t columns,
    size_t rows
)
{
    size_t row;

    /* The provider selects supported storage types. CUDA keeps quantized
       matrices compressed; Vulkan currently accepts F32. A declined dispatch
       falls through to a complete CPU kernel for exactly the same graph. */
    if (compute != NULL && nya_compute_matvec_typed(compute, matrix->data,
        rows, columns, matrix->type, input, output) == 0) return;
    if (matrix->type == NYA_LLM_TENSOR_F32) {
        for (row = 0; row < rows; ++row) {
            const unsigned char *row_weights;
            size_t column;
            float sum;

            row_weights = matrix->data + row * columns * sizeof(float);
            sum = 0.0f;
            /* The mapped file is byte storage. memcpy avoids effective-type
               and alignment assumptions while optimizing to a scalar load. */
            for (column = 0; column < columns; ++column) {
                float weight;
                memcpy(&weight, row_weights + column * sizeof(float), sizeof(weight));
                sum += weight * input[column];
            }
            output[row] = sum;
        }
        return;
    }

    if (matrix->type == NYA_LLM_TENSOR_F16 || matrix->type == NYA_LLM_TENSOR_BF16) {
        for (row = 0; row < rows; ++row) {
            const unsigned char *row_weights;
            size_t column;
            float sum;

            row_weights = matrix->data + row * columns * 2;
            sum = 0.0f;
            for (column = 0; column < columns; ++column) {
                if (matrix->type == NYA_LLM_TENSOR_BF16) {
                    const unsigned char *p = row_weights + column * 2;
                    uint32_t bits = ((uint32_t)p[0] | ((uint32_t)p[1] << 8)) << 16;
                    float weight;
                    memcpy(&weight, &bits, sizeof(weight));
                    sum += weight * input[column];
                } else sum += nya_llm_half_to_float(row_weights + column * 2) * input[column];
            }
            output[row] = sum;
        }
        return;
    }

    if (matrix->type == NYA_LLM_TENSOR_Q8_0) {
        size_t row_bytes;

        row_bytes = columns / 32 * 34;
        for (row = 0; row < rows; ++row) {
            const unsigned char *row_weights;
            size_t block;
            float sum;

            row_weights = matrix->data + row * row_bytes;
            sum = 0.0f;
            for (block = 0; block < columns / 32; ++block) {
                const unsigned char *block_data;
                float scale;
                size_t item;

                block_data = row_weights + block * 34;
                scale = nya_llm_half_to_float(block_data);
                for (item = 0; item < 32; ++item) {
                    int8_t quantized;

                    memcpy(&quantized, block_data + 2 + item, 1);
                    sum += scale * (float)quantized * input[block * 32 + item];
                }
            }
            output[row] = sum;
        }
        return;
    }

    if (matrix->type == NYA_LLM_TENSOR_Q4_K || matrix->type == NYA_LLM_TENSOR_Q6_K) {
        size_t block_bytes = matrix->type == NYA_LLM_TENSOR_Q4_K ? 144 : 210;
        size_t blocks = columns / 256;
        for (row = 0; row < rows; ++row) {
            const unsigned char *row_data = matrix->data + row * blocks * block_bytes;
            float sum = 0.0f;
            for (size_t b = 0; b < blocks; ++b) {
                const unsigned char *block = row_data + b * block_bytes;
                const float *x = input + b * 256;
                if (matrix->type == NYA_LLM_TENSOR_Q4_K) {
                    float d = nya_llm_half_to_float(block), minimum = nya_llm_half_to_float(block + 2);
                    const unsigned char *scales = block + 4;
                    for (size_t g = 0; g < 8; ++g) {
                        unsigned int sc = g < 4 ? scales[g] & 63U : (scales[g + 4] & 15U) | ((unsigned int)(scales[g - 4] >> 6) << 4);
                        unsigned int mn = g < 4 ? scales[g + 4] & 63U : (scales[g + 4] >> 4) | ((unsigned int)(scales[g] >> 6) << 4);
                        float factor = d * (float)sc, bias = minimum * (float)mn;
                        const unsigned char *packed = block + 16 + g / 2 * 32;
                        unsigned int shift = (unsigned int)(g % 2 * 4);
                        for (size_t j = 0; j < 32; ++j) sum += (factor * (float)((packed[j] >> shift) & 15U) - bias) * x[g * 32 + j];
                    }
                } else {
                    float d = nya_llm_half_to_float(block + 208);
                    for (size_t g = 0; g < 16; ++g) {
                        size_t half = g / 8, quarter = g % 8 / 2, lane_base = g % 2 * 16;
                        int8_t sc;
                        memcpy(&sc, block + 192 + g, 1);
                        float factor = d * (float)sc;
                        for (size_t j = 0; j < 16; ++j) {
                            size_t lane = lane_base + j;
                            unsigned int lo = block[half * 64 + quarter % 2 * 32 + lane];
                            unsigned int hi = block[128 + half * 32 + lane];
                            int q = (int)(((lo >> (quarter / 2 * 4)) & 15U) | (((hi >> (quarter * 2)) & 3U) << 4)) - 32;
                            sum += factor * (float)q * x[g * 16 + j];
                        }
                    }
                }
            }
            output[row] = sum;
        }
        return;
    }

    /* Q4_0 stores the first sixteen values in low nibbles and the rest in high nibbles. */
    {
        size_t row_bytes;

        row_bytes = columns / 32 * 18;
        for (row = 0; row < rows; ++row) {
            const unsigned char *row_weights;
            size_t block;
            float sum;

            row_weights = matrix->data + row * row_bytes;
            sum = 0.0f;
            for (block = 0; block < columns / 32; ++block) {
                const unsigned char *block_data;
                float scale;
                size_t item;

                block_data = row_weights + block * 18;
                scale = nya_llm_half_to_float(block_data);
                for (item = 0; item < 16; ++item) {
                    unsigned char packed;
                    float low;
                    float high;

                    packed = block_data[2 + item];
                    low = (float)((int)(packed & 0x0FU) - 8);
                    high = (float)((int)(packed >> 4) - 8);
                    sum += scale * low * input[block * 32 + item];
                    sum += scale * high * input[block * 32 + item + 16];
                }
            }
            output[row] = sum;
        }
    }
}

/* Apply RMS normalization and its learned scale. */
static void nya_llm_rms_norm(
    float *output,
    const float *input,
    const nya_llm_tensor *weight,
    size_t length,
    float epsilon
)
{
    size_t index;
    float sum;
    float scale;

    sum = 0.0f;
    for (index = 0; index < length; ++index) sum += input[index] * input[index];
    scale = 1.0f / sqrtf(sum / (float)length + epsilon);
    for (index = 0; index < length; ++index) {
        output[index] = input[index] * scale * (weight == NULL ? 1.0f : nya_llm_tensor_value(weight, index));
    }
}

/* Normalize one non-empty score vector in place with a stable softmax. */
static int nya_llm_softmax(float *values, size_t length)
{
    size_t index;
    float maximum;
    float sum;

    if (length == 0) return -1;
    maximum = values[0];
    for (index = 1; index < length; ++index) {
        if (values[index] > maximum) maximum = values[index];
    }
    sum = 0.0f;
    for (index = 0; index < length; ++index) {
        values[index] = expf(values[index] - maximum);
        sum += values[index];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) return -1;
    for (index = 0; index < length; ++index) values[index] /= sum;
    return 0;
}

/* Allocate every mutable buffer from model dimensions and this request's sequence. */
static int nya_llm_state_create(
    const nya_llm_context *context,
    size_t sequence_capacity,
    nya_llm_run_state *state,
    char *error,
    size_t error_capacity
)
{
    size_t dimension;
    size_t hidden;
    size_t vocabulary;
    size_t cache_count;
    size_t cache_bytes;
    size_t attention_bytes;
    size_t rope_pairs;
    size_t index;

    memset(state, 0, sizeof(*state));
    dimension = context->embedding_length;
    hidden = context->maximum_hidden_length;
    vocabulary = context->tokenizer.vocabulary_size;
    rope_pairs = 0;
    for (index = 0; index < context->block_count; ++index)
        if (context->layers[index].head_dimension / 2 > rope_pairs)
            rope_pairs = context->layers[index].head_dimension / 2;
    if (sequence_capacity == 0 ||
        nya_llm_cpu_product(sequence_capacity, sizeof(float), &attention_bytes) != 0 ||
        nya_llm_cpu_product(context->cache_width, sequence_capacity, &cache_count) != 0 ||
        nya_llm_cpu_product(cache_count, sizeof(float), &cache_bytes) != 0 ||
        dimension > SIZE_MAX / sizeof(float) || hidden > SIZE_MAX / sizeof(float) ||
        vocabulary > SIZE_MAX / sizeof(float)) {
        nya_llm_cpu_error(error, error_capacity, "generation dimensions overflow addressable memory");
        return -1;
    }

    state->activation = (float *)malloc(dimension * sizeof(float));
    state->normalized = (float *)malloc((dimension > context->maximum_query_length ? dimension : context->maximum_query_length) * sizeof(float));
    state->residual = (float *)malloc(dimension * sizeof(float));
    state->feed_forward_gate = (float *)malloc(hidden * sizeof(float));
    state->feed_forward_up = (float *)malloc(hidden * sizeof(float));
    state->query = (float *)malloc(context->maximum_query_length * sizeof(float));
    /* Attention heads execute serially, so their scores can share one sequence
       buffer. KV entries need no zero fill: forward writes each position before
       any head attends to it, and never reads a future position. */
    state->attention = (float *)malloc(attention_bytes);
    state->logits = (float *)malloc(vocabulary * sizeof(float));
    state->key_cache = (float *)malloc(cache_bytes == 0 ? 1 : cache_bytes);
    state->value_cache = (float *)malloc(cache_bytes == 0 ? 1 : cache_bytes);
    state->rope_frequencies = (float *)calloc(rope_pairs * 2, sizeof(float));
    state->rope_cosines = (float *)malloc(rope_pairs * 2 * sizeof(float));
    state->rope_sines = (float *)malloc(rope_pairs * 2 * sizeof(float));
    state->rope_stride = rope_pairs;
    state->sequence_capacity = sequence_capacity;
    if (context->is_assistant) {
        state->mtp_input = (float *)malloc((size_t)context->backbone_length * 2 * sizeof(float));
        state->mtp_hidden = (float *)malloc((size_t)context->backbone_length * sizeof(float));
        if (state->mtp_input == NULL || state->mtp_hidden == NULL) {
            nya_llm_cpu_error(error, error_capacity, "MTP scratch allocation failed");
            return -1;
        }
    }
    if (context->per_layer_embedding_length != 0) {
        state->ple_inputs = (float *)malloc((size_t)context->per_layer_embedding_length * context->block_count * sizeof(float));
        state->ple_gate = (float *)malloc((size_t)context->per_layer_embedding_length * sizeof(float));
        if (state->ple_inputs == NULL || state->ple_gate == NULL) {
            nya_llm_cpu_error(error, error_capacity, "PLE scratch allocation failed"); return -1;
        }
    }
    if (context->expert_count != 0) {
        state->expert_input = (float *)malloc(dimension * sizeof(float));
        state->expert_output = (float *)malloc(dimension * sizeof(float));
        state->router_probabilities = (float *)malloc(context->expert_count * sizeof(float));
        state->expert_indices = (uint32_t *)malloc(context->experts_used * sizeof(uint32_t));
        if (state->expert_input == NULL || state->expert_output == NULL ||
            state->router_probabilities == NULL || state->expert_indices == NULL) {
            nya_llm_cpu_error(error, error_capacity, "MoE scratch allocation failed"); return -1;
        }
    }

    if (state->activation == NULL || state->normalized == NULL || state->residual == NULL ||
        state->feed_forward_gate == NULL || state->feed_forward_up == NULL ||
        state->query == NULL || state->attention == NULL || state->logits == NULL ||
        state->key_cache == NULL || state->value_cache == NULL || state->rope_frequencies == NULL ||
        state->rope_cosines == NULL || state->rope_sines == NULL) {
        nya_llm_cpu_error(error, error_capacity, "generation scratch or KV-cache allocation failed");
        return -1;
    }
    /* Frequencies depend only on the model. Compute them once per request;
       trigonometric values will be shared by every head/layer at each position. */
    for (index = 0; index < context->block_count; ++index) {
        const nya_llm_layer *w = &context->layers[index];
        size_t offset = w->sliding_window != 0 ? rope_pairs : 0;
        size_t pairs = w->head_dimension / 2;
        float base = w->sliding_window != 0 ? context->sliding_rope_base : context->rope_frequency_base;
        for (size_t j = 0; j < pairs; ++j) {
            float factor = w->rope_factors == NULL ? 1.0f : nya_llm_tensor_value(w->rope_factors, j);
            if (!(factor > 0.0f) || !isfinite(factor)) {
                nya_llm_cpu_error(error, error_capacity, "invalid proportional RoPE factor");
                return -1;
            }
            state->rope_frequencies[offset + j] = powf(base, -(float)j / (float)pairs) / factor;
        }
    }
    return 0;
}

/* Release request-local transformer memory. */
static void nya_llm_state_free(nya_llm_run_state *state)
{
    free(state->activation);
    free(state->normalized);
    free(state->residual);
    free(state->feed_forward_gate);
    free(state->feed_forward_up);
    free(state->query);
    free(state->attention);
    free(state->logits);
    free(state->key_cache);
    free(state->value_cache);
    free(state->rope_frequencies);
    free(state->rope_cosines);
    free(state->rope_sines);
    free(state->ple_inputs);
    free(state->ple_gate);
    free(state->expert_input);
    free(state->expert_output);
    free(state->router_probabilities);
    free(state->expert_indices);
    free(state->mtp_input);
    free(state->mtp_hidden);
    memset(state, 0, sizeof(*state));
}

/* Rotate adjacent query or key components with standard unscaled LLaMA RoPE. */
static void nya_llm_rope(
    float *vector,
    size_t head_count,
    size_t head_dimension,
    const float *cosines,
    const float *sines
)
{
    size_t head;

    for (head = 0; head < head_count; ++head) {
        size_t component;
        float *head_vector;

        head_vector = vector + head * head_dimension;
        for (component = 0; component < head_dimension; component += 2) {
            float cosine;
            float sine;
            float first;
            float second;

            cosine = cosines[component / 2];
            sine = sines[component / 2];
            first = head_vector[component];
            second = head_vector[component + 1];
            head_vector[component] = first * cosine - second * sine;
            head_vector[component + 1] = first * sine + second * cosine;
        }
    }
}

/* Gemma uses GELU(tanh), including in its PLE branch; LLaMA uses SiLU.
   Keep this choice explicit rather than inferring it from a tensor name. */
static float nya_llm_gelu(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

static void nya_llm_gated_activation(float *gate, const float *up, size_t n, int gemma)
{
    for (size_t j = 0; j < n; ++j) {
        float x = gate[j];
        gate[j] = (gemma ? nya_llm_gelu(x) : x / (1.0f + expf(-x))) * up[j];
    }
}

/* Unlike LLaMA's interleaved pairs, Gemma rotates corresponding components in
   the two halves of a head. Large stored frequency factors leave the prescribed
   components effectively unrotated without permuting the shared K/V values. */
static void nya_llm_gemma_rope(float *v, size_t heads, size_t width, const float *co, const float *si)
{
    size_t half = width / 2;
    for (size_t h = 0; h < heads; ++h) {
        float *p = v + h * width;
        for (size_t j = 0; j < half; ++j) {
            float a = p[j], b = p[j + half];
            p[j] = a * co[j] - b * si[j];
            p[j + half] = b * co[j] + a * si[j];
        }
    }
}

static void nya_llm_head_norm(float *v, size_t heads, size_t width, const nya_llm_tensor *w, float eps)
{
    for (size_t h = 0; h < heads; ++h) nya_llm_rms_norm(v + h * width, v + h * width, w, width, eps);
}

/* Slice an expert in the outer dimension. All extents and byte sizes were
   validated at load time. Views borrow storage and never free mapped weights. */
static nya_llm_tensor nya_llm_expert_slice(const nya_llm_tensor *bank, uint32_t expert, uint32_t count)
{
    nya_llm_tensor view = *bank;
    view.data_size = bank->data_size / count;
    view.data = bank->data + (size_t)expert * view.data_size;
    return view;
}

static int nya_llm_experts(const nya_llm_context *c, const nya_llm_layer *w, nya_llm_run_state *s)
{
    size_t d = c->embedding_length, n = c->expert_count, k = c->experts_used, h = w->expert_hidden_length;
    double mass = 0.0;
    /* The router sees the post-attention residual, not the MLP-normalized
       vector. Learned input scale precedes routing; expert scales follow the
       top-k renormalization and must not affect which experts are selected. */
    nya_llm_rms_norm(s->normalized, s->activation, w->router_scale, d, c->norm_epsilon);
    for (size_t j = 0; j < d; ++j) s->normalized[j] /= sqrtf((float)d);
    nya_llm_matvec(c->compute, s->router_probabilities, w->router, s->normalized, d, n);
    if (nya_llm_softmax(s->router_probabilities, n) != 0) return -1;
    for (size_t j = 0, used = 0; j < n; ++j) {
        size_t slot = used;
        if (slot < k) ++used;
        while (slot > 0 && s->router_probabilities[j] > s->router_probabilities[s->expert_indices[slot - 1]]) {
            if (slot < k) s->expert_indices[slot] = s->expert_indices[slot - 1];
            --slot;
        }
        if (slot < k) s->expert_indices[slot] = (uint32_t)j;
    }
    for (size_t j = 0; j < k; ++j) mass += s->router_probabilities[s->expert_indices[j]];
    if (!(mass > 0.0)) return -1;
    nya_llm_rms_norm(s->expert_input, s->activation, w->expert_pre_norm, d, c->norm_epsilon);
    memset(s->expert_output, 0, d * sizeof(float));
    for (size_t j = 0; j < k; ++j) {
        uint32_t id = s->expert_indices[j];
        nya_llm_tensor gate, up, down = nya_llm_expert_slice(w->expert_down, id, c->expert_count);
        float scale = (float)((double)s->router_probabilities[id] / mass) * nya_llm_tensor_value(w->expert_scale, id);
        if (w->expert_gate_up != NULL) {
            gate = nya_llm_expert_slice(w->expert_gate_up, id, c->expert_count);
            up = gate;
            up.data += gate.data_size / 2;
        } else {
            gate = nya_llm_expert_slice(w->expert_gate, id, c->expert_count);
            up = nya_llm_expert_slice(w->expert_up, id, c->expert_count);
        }
        nya_llm_matvec(c->compute, s->feed_forward_gate, &gate, s->expert_input, d, h);
        nya_llm_matvec(c->compute, s->feed_forward_up, &up, s->expert_input, d, h);
        nya_llm_gated_activation(s->feed_forward_gate, s->feed_forward_up, h, 1);
        nya_llm_matvec(c->compute, s->normalized, &down, s->feed_forward_gate, h, d);
        for (size_t v = 0; v < d; ++v) s->expert_output[v] += scale * s->normalized[v];
    }
    nya_llm_rms_norm(s->expert_output, s->expert_output, w->expert_post_norm, d, c->norm_epsilon);
    return 0;
}

/* One causal token step. Gemma's layer-local dimensions are independent of the
   residual width. Shared layers read an earlier layer's cache slice, and local
   attention excludes old positions before the softmax (never after it). */
static int nya_llm_forward_impl(const nya_llm_context *c, nya_llm_run_state *s,
    uint32_t token, size_t position, int compute_logits,
    const nya_llm_context *target, const nya_llm_run_state *target_state, size_t cache_length)
{
    size_t d = c->embedding_length, ple = c->per_layer_embedding_length;
    if (token >= c->tokenizer.vocabulary_size || position >= s->sequence_capacity) return -1;
    for (size_t j = 0; j < s->rope_stride * 2; ++j) {
        float angle = (float)position * s->rope_frequencies[j];
        s->rope_cosines[j] = cosf(angle);
        s->rope_sines[j] = sinf(angle);
    }
    if (c->is_assistant) {
        if (target == NULL || target_state == NULL || cache_length == 0 || cache_length > target_state->sequence_capacity) return -1;
        /* MTP input concatenates the TARGET's scaled token embedding and the
           previous target/recurrent hidden state, then projects to draft width.
           The assistant's own embedding matrix is used as its vocabulary head. */
        size_t backbone = c->backbone_length;
        nya_llm_copy_row(s->mtp_input, target->token_embedding, token, backbone);
        for (size_t j = 0; j < backbone; ++j) s->mtp_input[j] *= sqrtf((float)backbone);
        memcpy(s->mtp_input + backbone, s->mtp_hidden, backbone * sizeof(float));
        nya_llm_matvec(c->compute, s->activation, c->nextn_pre, s->mtp_input, backbone * 2, d);
    } else if (s->embedded_input != NULL) {
        memcpy(s->activation, s->embedded_input, d * sizeof(float));
    } else {
        nya_llm_copy_row(s->activation, c->token_embedding, token, d);
        if (c->is_gemma) for (size_t j = 0; j < d; ++j) s->activation[j] *= sqrtf((float)d);
    }
    if (ple != 0) {
        size_t packed = ple * c->block_count;
        nya_llm_matvec(c->compute, s->ple_inputs, c->ple_projection, s->activation, d, packed);
        for (size_t j = 0; j < packed; ++j) s->ple_inputs[j] /= sqrtf((float)d);
        for (size_t layer = 0; layer < c->block_count; ++layer) {
            float *v = s->ple_inputs + layer * ple;
            nya_llm_rms_norm(v, v, c->ple_norm, ple, c->norm_epsilon);
            for (size_t j = 0; j < ple; ++j) {
                float identity = nya_llm_tensor_value(c->ple_embedding, (size_t)token * packed + layer * ple + j);
                v[j] = (v[j] + identity * sqrtf((float)ple)) * 0.7071067811865475f;
            }
        }
    }
    if (s->layer_input != NULL) memcpy(s->activation, s->layer_input, d * sizeof(float));
    size_t layer_end = s->layer_end == 0 ? c->block_count : s->layer_end;
    for (size_t layer = s->layer_begin; layer < layer_end; ++layer) {
        const nya_llm_layer *w = &c->layers[layer];
        size_t hd = w->head_dimension, heads = w->head_count, kv_heads = w->kv_head_count;
        size_t qdim = heads * hd, kvdim = kv_heads * hd, h = w->hidden_length;
        size_t offset = w->cache_offset * s->sequence_capacity;
        const float *read_keys = s->key_cache, *read_values = s->value_cache;
        size_t attention_end = position;
        if (w->sliding_window != 0 && s->attention_extent > position) attention_end = s->attention_extent;
        if (c->is_assistant) {
            const nya_llm_layer *source = &target->layers[s->mtp_sources[layer]];
            offset = source->cache_offset * target_state->sequence_capacity;
            read_keys = target_state->key_cache;
            read_values = target_state->value_cache;
            attention_end = cache_length - 1;
        }
        size_t rope_offset = w->sliding_window != 0 ? s->rope_stride : 0;
        const float *co = s->rope_cosines + rope_offset, *si = s->rope_sines + rope_offset;
        /* Do not even form out-of-bounds pointers into the assistant's empty
           KV allocation: pointer arithmetic beyond an object is itself UB. */
        float *key = c->is_assistant ? NULL : s->key_cache + offset + position * kvdim;
        float *value = c->is_assistant ? NULL : s->value_cache + offset + position * kvdim;
        /* Gemma's vision overlay extends the causal upper boundary to the end
           of this image; the local window supplies only the lower boundary.
           MTP instead reads the final window of its fixed borrowed prefix. */
        size_t window_anchor = c->is_assistant ? attention_end : position;
        size_t first = w->sliding_window != 0 && window_anchor >= w->sliding_window ? window_anchor - w->sliding_window + 1 : 0;
        if (first > attention_end) return -1;
        size_t length = attention_end - first + 1;
        nya_llm_rms_norm(s->normalized, s->activation, w->attention_norm, d, c->norm_epsilon);
        nya_llm_matvec(c->compute, s->query, w->query, s->normalized, d, qdim);
        if (c->is_gemma) nya_llm_head_norm(s->query, heads, hd, w->query_norm, c->norm_epsilon);
        if (w->kv_source == layer) {
            nya_llm_matvec(c->compute, key, w->key, s->normalized, d, kvdim);
            if (w->value != NULL) nya_llm_matvec(c->compute, value, w->value, s->normalized, d, kvdim);
            else memcpy(value, key, kvdim * sizeof(float));
            if (c->is_gemma) {
                nya_llm_head_norm(key, kv_heads, hd, w->key_norm, c->norm_epsilon);
                nya_llm_head_norm(value, kv_heads, hd, NULL, c->norm_epsilon);
                nya_llm_gemma_rope(key, kv_heads, hd, co, si);
            } else nya_llm_rope(key, kv_heads, hd, co, si);
        }
        if (s->prepare_kv_only) continue;
        if (c->is_gemma) nya_llm_gemma_rope(s->query, heads, hd, co, si);
        else nya_llm_rope(s->query, heads, hd, co, si);
        for (size_t head = 0; head < heads; ++head) {
            const float *query = s->query + head * hd;
            size_t kv_head = head / (heads / kv_heads);
            float *out = s->normalized + head * hd;
            float attention_scale = c->is_gemma ? 1.0f : 1.0f / sqrtf((float)hd);
            for (size_t t = first; t <= attention_end; ++t) {
                const float *k = read_keys + offset + t * kvdim + kv_head * hd;
                float score = 0.0f;
                for (size_t j = 0; j < hd; ++j) score += query[j] * k[j];
                s->attention[t - first] = score * attention_scale;
            }
            if (nya_llm_softmax(s->attention, length) != 0) return -1;
            memset(out, 0, hd * sizeof(float));
            for (size_t t = first; t <= attention_end; ++t) {
                const float *v = read_values + offset + t * kvdim + kv_head * hd;
                float score = s->attention[t - first];
                for (size_t j = 0; j < hd; ++j) out[j] += score * v[j];
            }
        }
        nya_llm_matvec(c->compute, s->residual, w->attention_output, s->normalized, qdim, d);
        if (c->is_gemma) nya_llm_rms_norm(s->residual, s->residual, w->attention_post_norm, d, c->norm_epsilon);
        for (size_t j = 0; j < d; ++j) s->activation[j] += s->residual[j];
        nya_llm_rms_norm(s->normalized, s->activation, w->feed_forward_norm, d, c->norm_epsilon);
        nya_llm_matvec(c->compute, s->feed_forward_gate, w->feed_forward_gate, s->normalized, d, h);
        nya_llm_matvec(c->compute, s->feed_forward_up, w->feed_forward_up, s->normalized, d, h);
        nya_llm_gated_activation(s->feed_forward_gate, s->feed_forward_up, h, c->is_gemma);
        nya_llm_matvec(c->compute, s->residual, w->feed_forward_down, s->feed_forward_gate, h, d);
        if (c->expert_count != 0) {
            nya_llm_rms_norm(s->residual, s->residual, w->shared_post_norm, d, c->norm_epsilon);
            if (nya_llm_experts(c, w, s) != 0) return -1;
            for (size_t j = 0; j < d; ++j) s->residual[j] += s->expert_output[j];
        }
        if (c->is_gemma) nya_llm_rms_norm(s->residual, s->residual, w->ffn_post_norm, d, c->norm_epsilon);
        for (size_t j = 0; j < d; ++j) s->activation[j] += s->residual[j];
        if (ple != 0) {
            nya_llm_matvec(c->compute, s->ple_gate, w->ple_gate, s->activation, d, ple);
            for (size_t j = 0; j < ple; ++j) s->ple_gate[j] = nya_llm_gelu(s->ple_gate[j]) * s->ple_inputs[layer * ple + j];
            nya_llm_matvec(c->compute, s->residual, w->ple_projection, s->ple_gate, ple, d);
            nya_llm_rms_norm(s->residual, s->residual, w->ple_norm, d, c->norm_epsilon);
            for (size_t j = 0; j < d; ++j) s->activation[j] += s->residual[j];
        }
        if (w->output_scale != NULL) {
            float scale = nya_llm_tensor_value(w->output_scale, 0);
            for (size_t j = 0; j < d; ++j) s->activation[j] *= scale;
        }
    }
    /* Prompt positions only need caches. Keep the full vocabulary projection
       out of prefill until a sampling/verification position requests it. */
    if (!compute_logits) return 0;
    nya_llm_rms_norm(s->normalized, s->activation, c->output_norm, d, c->norm_epsilon);
    nya_llm_matvec(c->compute, s->logits, c->output, s->normalized, d, c->tokenizer.vocabulary_size);
    if (c->is_assistant) nya_llm_matvec(c->compute, s->mtp_hidden, c->nextn_post, s->normalized, d, c->backbone_length);
    if (c->final_logit_softcap > 0.0f) {
        float cap = c->final_logit_softcap;
        for (size_t j = 0; j < c->tokenizer.vocabulary_size; ++j) s->logits[j] = cap * tanhf(s->logits[j] / cap);
    }
    /* Reject non-finite model results before the checked suppression list is
       allowed to introduce negative infinity for intentionally masked tokens. */
    for (size_t j = 0; j < c->tokenizer.vocabulary_size; ++j) if (!isfinite(s->logits[j])) return -1;
    for (size_t j = 0; j < c->tokenizer.suppressed_count; ++j) s->logits[c->tokenizer.suppressed_tokens[j]] = -INFINITY;
    return 0;
}

static int nya_llm_forward(const nya_llm_context *c, nya_llm_run_state *s,
    uint32_t token, size_t position, int compute_logits)
{
    return nya_llm_forward_impl(c, s, token, position, compute_logits, NULL, NULL, 0);
}

/* Multimodal prefill traverses layers before positions. Every source layer
   first prepares the complete prompt KV slice, allowing an image token to read
   later tokens of that SAME image in local attention. Full attention remains
   causal. Two activation matrices suffice: immutable input embeddings for PLE,
   and the current residual for each position. Cache slices retain their normal
   decode layout, so subsequent generation uses the ordinary one-token kernel. */
static int nya_llm_prefill(const nya_llm_context *c, nya_llm_run_state *s,
    const uint32_t *tokens, size_t count, const nya_generation_request *r,
    char *error, size_t capacity)
{
    float *initial = NULL, *hidden = NULL;
    size_t *image_end = NULL;
    unsigned char *soft = NULL;
    size_t elements, bytes, previous_end = 0, d = c->embedding_length;
    int result = -1;
    const char *failure = "invalid multimodal prompt spans";
    if (r->soft_token_spans == 0) {
        for (size_t i = 0; i < count; ++i) if (nya_llm_forward(c, s, tokens[i], i, i + 1 == count) != 0) return -1;
        return 0;
    }
    if (!c->is_gemma || c->is_assistant || r->soft_tokens == NULL || r->soft_token_spans > count) goto cleanup;
    for (size_t i = 0; i < r->soft_token_spans; ++i) {
        const nya_generation_soft_tokens *span = &r->soft_tokens[i];
        if (span->count == 0 || span->position < previous_end || span->position >= count ||
            span->count > count - span->position || span->embedding_length != d || span->embeddings == NULL ||
            span->count > SIZE_MAX / d || span->embedding_count != span->count * d ||
            (span->vision != 0 && span->vision != 1)) goto cleanup;
        for (size_t j = 0; j < span->embedding_count; ++j) if (!isfinite(span->embeddings[j])) goto cleanup;
        previous_end = span->position + span->count;
    }
    failure = "multimodal prefill allocation failed";
    if (nya_llm_cpu_product(count, d, &elements) != 0 || nya_llm_cpu_product(elements, sizeof(float), &bytes) != 0 ||
        count > SIZE_MAX / sizeof(size_t)) goto cleanup;
    initial = (float *)malloc(bytes); hidden = (float *)malloc(bytes);
    image_end = (size_t *)calloc(count, sizeof(size_t)); soft = (unsigned char *)calloc(count, 1);
    if (initial == NULL || hidden == NULL || image_end == NULL || soft == NULL) goto cleanup;
    for (size_t i = 0; i < count; ++i) {
        nya_llm_copy_row(initial + i * d, c->token_embedding, tokens[i], d);
        for (size_t j = 0; j < d; ++j) initial[i * d + j] *= sqrtf((float)d);
    }
    for (size_t i = 0; i < r->soft_token_spans; ++i) {
        const nya_generation_soft_tokens *span = &r->soft_tokens[i];
        memcpy(initial + span->position * d, span->embeddings, span->embedding_count * sizeof(float));
        for (size_t j = span->position; j < span->position + span->count; ++j) {
            soft[j] = 1;
            if (span->vision) image_end[j] = span->position + span->count - 1;
        }
    }
    memcpy(hidden, initial, bytes);
    failure = "multimodal transformer produced an invalid value";
    for (size_t layer = 0; layer < c->block_count; ++layer) {
        s->layer_begin = layer; s->layer_end = layer + 1;
        for (int phase = 0; phase < 2; ++phase) {
            if (phase == 0 && c->layers[layer].kv_source != layer) continue;
            s->prepare_kv_only = phase == 0;
            for (size_t i = 0; i < count; ++i) {
                s->embedded_input = initial + i * d; s->layer_input = hidden + i * d;
                s->attention_extent = image_end[i];
                /* Published Gemma PLE looks up PAD (ID 0) at soft positions. */
                if (nya_llm_forward(c, s, soft[i] ? 0 : tokens[i], i,
                    phase == 1 && layer + 1 == c->block_count && i + 1 == count) != 0) goto cleanup;
                if (phase == 1) memcpy(hidden + i * d, s->activation, d * sizeof(float));
            }
        }
    }
    result = 0;
cleanup:
    s->embedded_input = NULL; s->layer_input = NULL; s->layer_begin = 0; s->layer_end = 0;
    s->attention_extent = 0; s->prepare_kv_only = 0;
    free(initial); free(hidden); free(image_end); free(soft);
    if (result != 0) nya_llm_cpu_error(error, capacity, "%s", failure);
    return result;
}

/* Produce one deterministic xorshift64* word. */
static uint32_t nya_llm_random_u32(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * UINT64_C(0x2545F4914F6CDD1D)) >> 32);
}

/* Convert one random word into a float in the half-open unit interval. */
static float nya_llm_random_float(uint64_t *state)
{
    return (float)(nya_llm_random_u32(state) >> 8) / 16777216.0f;
}

/* Sort candidates in descending probability with token ID as a stable tie break. */
static int nya_llm_probability_compare(const void *left, const void *right)
{
    const nya_llm_probability *left_probability;
    const nya_llm_probability *right_probability;

    left_probability = (const nya_llm_probability *)left;
    right_probability = (const nya_llm_probability *)right;
    if (left_probability->probability > right_probability->probability) return -1;
    if (left_probability->probability < right_probability->probability) return 1;
    if (left_probability->token < right_probability->token) return -1;
    if (left_probability->token > right_probability->token) return 1;
    return 0;
}

/* Keep the least desirable retained candidate at the heap root. The same
   probability/token ordering is used by the heap and the final sort, making
   equal-probability top-k selection deterministic across C library versions. */
static void nya_llm_candidate_sift(nya_llm_probability *items, size_t count, size_t parent)
{
    for (;;) {
        size_t child = parent * 2 + 1;
        nya_llm_probability temporary;
        if (child >= count) return;
        if (child + 1 < count && nya_llm_probability_compare(&items[child + 1], &items[child]) > 0) child += 1;
        if (nya_llm_probability_compare(&items[parent], &items[child]) >= 0) return;
        temporary = items[parent];
        items[parent] = items[child];
        items[child] = temporary;
        parent = child;
    }
}

/* Build the actual sampling distribution after temperature, top-k and top-p.
   Selection on raw logits is equivalent to selection after softmax. It avoids
   evaluating exp for every vocabulary entry when only a small top-k is kept.
   Dense probabilities are needed by speculative rejection/residual sampling. */
static int nya_llm_distribution(float *logits, size_t vocabulary, float temperature,
    float top_p, size_t top_k, nya_llm_probability *candidates, size_t *count)
{
    size_t kept = top_k == 0 || top_k > vocabulary ? vocabulary : top_k;
    double mass = 0.0, cumulative = 0.0;
    float maximum;
    if (vocabulary == 0 || candidates == NULL || !(temperature > 0.0f)) return -1;
    for (size_t j = 0; j < vocabulary; ++j) if (isnan(logits[j]) || logits[j] == INFINITY) return -1;
    for (size_t j = 0; j < kept; ++j) {
        candidates[j].probability = logits[j];
        candidates[j].token = (uint32_t)j;
    }
    if (kept < vocabulary) {
        for (size_t j = kept / 2; j > 0; --j) nya_llm_candidate_sift(candidates, kept, j - 1);
        for (size_t j = kept; j < vocabulary; ++j) {
            nya_llm_probability candidate = {logits[j], (uint32_t)j};
            if (nya_llm_probability_compare(&candidate, &candidates[0]) < 0) {
                candidates[0] = candidate;
                nya_llm_candidate_sift(candidates, kept, 0);
            }
        }
    }
    qsort(candidates, kept, sizeof(*candidates), nya_llm_probability_compare);
    maximum = candidates[0].probability;
    if (!isfinite(maximum)) return -1;
    for (size_t j = 0; j < kept; ++j) {
        candidates[j].probability = expf((candidates[j].probability - maximum) / temperature);
        mass += candidates[j].probability;
    }
    for (size_t j = 0; j < kept; ++j) {
        cumulative += candidates[j].probability;
        if (cumulative >= (double)top_p * mass) { kept = j + 1; break; }
    }
    if (!(cumulative > 0.0) || !isfinite(cumulative)) return -1;
    memset(logits, 0, vocabulary * sizeof(float));
    for (size_t j = 0; j < kept; ++j) {
        candidates[j].probability = (float)((double)candidates[j].probability / cumulative);
        logits[candidates[j].token] = candidates[j].probability;
    }
    *count = kept;
    return 0;
}

static int nya_llm_sample(float *logits, size_t vocabulary, float temperature, float top_p,
    size_t top_k, uint64_t *rng, nya_llm_probability *candidates, uint32_t *token)
{
    size_t count;
    double cumulative = 0.0, target;
    if (vocabulary == 0) return -1;
    if (temperature == 0.0f) {
        size_t best = 0;
        for (size_t j = 0; j < vocabulary; ++j) {
            if (isnan(logits[j]) || logits[j] == INFINITY) return -1;
            if (logits[j] > logits[best]) best = j;
        }
        if (!isfinite(logits[best])) return -1;
        *token = (uint32_t)best;
        return 0;
    }
    if (nya_llm_distribution(logits, vocabulary, temperature, top_p, top_k, candidates, &count) != 0) return -1;
    target = nya_llm_random_float(rng);
    for (size_t j = 0; j < count; ++j) {
        cumulative += candidates[j].probability;
        if (target < cumulative) { *token = candidates[j].token; return 0; }
    }
    *token = candidates[count - 1].token;
    return 0;
}

/* Draw from a dense nonnegative vector. Accumulate in double and fall back to
   the last positive entry, never a zero-mass tail, after rounding at the CDF. */
static int nya_llm_draw(const float *probabilities, size_t n, uint64_t *rng, uint32_t *token)
{
    double mass = 0.0, cdf = 0.0, target;
    size_t last = 0;
    for (size_t j = 0; j < n; ++j) {
        if (probabilities[j] < 0.0f || !isfinite(probabilities[j])) return -1;
        mass += probabilities[j];
        if (probabilities[j] > 0.0f) last = j;
    }
    if (!(mass > 0.0)) return -1;
    target = (double)nya_llm_random_float(rng) * mass;
    for (size_t j = 0; j < n; ++j) {
        cdf += probabilities[j];
        if (target < cdf) { *token = (uint32_t)j; return 0; }
    }
    *token = (uint32_t)last;
    return 0;
}

/* Run prompt ingestion followed by bounded autoregressive generation. */
int nya_llm_generate(
    const nya_llm_context *context,
    const nya_generation_request *request,
    nya_generation_response *response,
    char *error,
    size_t error_capacity
)
{
    uint32_t *prompt_tokens;
    size_t prompt_count;
    size_t available_tokens;
    size_t generation_limit;
    size_t sequence_capacity;
    nya_llm_run_state state;
    nya_llm_probability *candidates;
    uint64_t random_state;
    uint32_t token;
    size_t position;
    size_t generated;
    size_t output_length;
    int result;
    int prefilled = 0;

    if (response != NULL) memset(response, 0, sizeof(*response));
    if (context == NULL || request == NULL || response == NULL || request->prompt == NULL || context->is_assistant ||
        request->max_tokens == 0 || request->max_output_bytes == 0 ||
        !isfinite(request->temperature) || !isfinite(request->top_p) ||
        request->temperature < 0.0f || request->temperature > 5.0f ||
        request->top_p <= 0.0f || request->top_p > 1.0f) {
        nya_llm_cpu_error(error, error_capacity, "invalid generation request");
        return -1;
    }
    memset(response, 0, sizeof(*response));
    prompt_tokens = NULL;
    prompt_count = 0;
    memset(&state, 0, sizeof(state));
    candidates = NULL;
    result = -1;

    if (nya_llm_tokenize(
            context,
            request->prompt,
            &prompt_tokens,
            &prompt_count,
            error,
            error_capacity
        ) != 0) goto cleanup;
    if (prompt_count >= context->context_length) {
        nya_llm_cpu_error(error, error_capacity, "the tokenized prompt fills or exceeds the model context");
        goto cleanup;
    }

    available_tokens = context->context_length - prompt_count;
    generation_limit = request->max_tokens < available_tokens ? request->max_tokens : available_tokens;
    if (generation_limit > SIZE_MAX - prompt_count) {
        nya_llm_cpu_error(error, error_capacity, "generation sequence length overflowed");
        goto cleanup;
    }
    sequence_capacity = prompt_count + generation_limit;
    if (nya_llm_state_create(context, sequence_capacity, &state, error, error_capacity) != 0) goto cleanup;

    /* Greedy decoding uses no probability candidates and needs no allocation. */
    if (request->temperature > 0.0f) {
        size_t candidate_count = request->top_k == 0 || request->top_k > context->tokenizer.vocabulary_size ?
            context->tokenizer.vocabulary_size : request->top_k;
        if (candidate_count > SIZE_MAX / sizeof(*candidates)) {
            nya_llm_cpu_error(error, error_capacity, "sampling allocation size overflowed");
            goto cleanup;
        }
        candidates = (nya_llm_probability *)malloc(
            candidate_count * sizeof(*candidates)
        );
    }
    if ((request->temperature > 0.0f && candidates == NULL) || request->max_output_bytes == SIZE_MAX) {
        nya_llm_cpu_error(error, error_capacity, "sampling or output allocation failed");
        goto cleanup;
    }
    response->text = (char *)malloc(request->max_output_bytes + 1);
    if (response->text == NULL) {
        nya_llm_cpu_error(error, error_capacity, "generation output allocation failed");
        goto cleanup;
    }
    response->text[0] = '\0';
    response->prompt_tokens = prompt_count;
    response->seed = request->seed;
    response->stop_reason = generation_limit < request->max_tokens ?
        NYA_GENERATION_STOP_CONTEXT : NYA_GENERATION_STOP_LENGTH;

    random_state = request->seed == 0 ? UINT64_C(0x9E3779B97F4A7C15) : request->seed;
    token = prompt_tokens[0];
    position = 0;
    generated = 0;
    output_length = 0;

    if (request->soft_token_spans != 0) {
        if (nya_llm_prefill(context, &state, prompt_tokens, prompt_count, request, error, error_capacity) != 0) goto cleanup;
        position = prompt_count - 1; token = prompt_tokens[position];
        response->target_steps = prompt_count; prefilled = 1;
    }

    while (generated < generation_limit) {
        uint32_t next;

        if (!prefilled && nya_llm_forward(context, &state, token, position, position + 1 >= prompt_count) != 0) {
            nya_llm_cpu_error(error, error_capacity, "transformer forward pass produced an invalid value");
            goto cleanup;
        }
        if (!prefilled) response->target_steps += 1;
        prefilled = 0;

        if (position + 1 < prompt_count) {
            next = prompt_tokens[position + 1];
        } else {
            if (nya_llm_sample(
                    state.logits,
                    context->tokenizer.vocabulary_size,
                    request->temperature,
                    request->top_p,
                    request->top_k,
                    &random_state,
                    candidates,
                    &next
                ) != 0) {
                nya_llm_cpu_error(error, error_capacity, "next-token sampling failed");
                goto cleanup;
            }
            generated += 1;
            if (context->tokenizer.stop_tokens[next]) {
                response->stop_reason = NYA_GENERATION_STOP_EOS;
                break;
            }
            if (nya_llm_decode_token(
                    context,
                    token,
                    next,
                    response->text,
                    request->max_output_bytes + 1,
                    &output_length
                ) != 0) {
                nya_llm_cpu_error(error, error_capacity, "generated text exceeded max_output_bytes");
                goto cleanup;
            }
        }

        token = next;
        position += 1;
    }

    /* Byte-fallback pieces can split a code point across tokens. Validate only
       after sampling stops so complete sequences are accepted, but an EOS or
       length boundary can never return malformed text to the JSON transport. */
    if (!nya_llm_text_is_utf8(response->text, output_length)) {
        nya_llm_cpu_error(error, error_capacity, "generated token bytes do not form valid UTF-8");
        goto cleanup;
    }
    response->text_length = output_length;
    response->generated_tokens = generated;
    result = 0;

cleanup:
    if (result != 0) {
        free(response->text);
        memset(response, 0, sizeof(*response));
    }
    free(candidates);
    nya_llm_state_free(&state);
    free(prompt_tokens);
    return result;
}

/* Matching token text alone is insufficient: the same text can have different
   IDs, pair ranks, normalization, or stop semantics in two model packages. */
static int nya_llm_tokenizers_match(const nya_llm_tokenizer *a, const nya_llm_tokenizer *b, int assistant)
{
    size_t n = a->vocabulary_size;
    if (n != b->vocabulary_size || a->unknown_token != b->unknown_token ||
        a->beginning_token != b->beginning_token || a->end_token != b->end_token ||
        a->add_beginning_token != b->add_beginning_token || a->add_end_token != b->add_end_token ||
        a->add_space_prefix != b->add_space_prefix || a->uses_sentencepiece_space != b->uses_sentencepiece_space ||
        a->uses_merge_ranks != b->uses_merge_ranks || a->merge_count != b->merge_count) return 0;
    for (size_t j = 0; j < n; ++j) {
        /* Some official assistant exports label added pieces (e.g. <|video|>)
           NORMAL while the target labels the identical ID/spelling CONTROL.
           MTP never tokenizes or decodes independently: it consumes the target's
           IDs and hidden states. This classification-only discrepancy is safe;
           byte/unknown/user-defined changes, ID text, ranks and stops stay exact. */
        int same_type = a->types[j] == b->types[j] || (assistant && a->uses_merge_ranks &&
            ((a->types[j] == 1 && b->types[j] == 3) || (a->types[j] == 3 && b->types[j] == 1)));
        if (!same_type || (!a->uses_merge_ranks && a->scores[j] != b->scores[j]) ||
            a->stop_tokens[j] != b->stop_tokens[j] || strcmp(a->pieces[j], b->pieces[j]) != 0) return 0;
    }
    for (size_t j = 0; j < a->merge_count; ++j) {
        if (a->merges[j].rank != b->merges[j].rank || a->merges[j].left_length != b->merges[j].left_length ||
            strcmp(a->merges[j].text, b->merges[j].text) != 0) return 0;
    }
    return 1;
}

/* Emit only a target-verified token. In particular, draft EOS and control tokens
   never terminate a response until the target has accepted them. */
static int nya_llm_emit(const nya_llm_context *c, const nya_generation_request *r,
    nya_generation_response *out, uint32_t previous, uint32_t token)
{
    ++out->generated_tokens;
    if (c->tokenizer.stop_tokens[token]) { out->stop_reason = NYA_GENERATION_STOP_EOS; return 1; }
    return nya_llm_decode_token(c, previous, token, out->text, r->max_output_bytes + 1, &out->text_length) == 0 ? 0 : -1;
}

/* Exact speculative sampling (p = target, q = draft): accept x~q with
   min(1,p(x)/q(x)); after rejection draw from normalized max(p-q,0). Applying
   top-k/top-p BEFORE this rule preserves the requested target distribution.
   A rejected suffix is rolled back by the logical position; old KV bytes above
   it are never read and get overwritten by the corrected continuation.

   Verification currently uses scalar token steps. This provides correctness and
   model-pair interoperability; throughput gains require batched verification
   and are not implied by the number of accepted proposals. */
int nya_llm_generate_draft(const nya_llm_context *c, const nya_llm_context *draft,
    const nya_generation_request *r, nya_generation_response *out, char *error, size_t error_capacity)
{
    nya_llm_run_state target_state = {0}, draft_state = {0};
    nya_llm_probability *candidates = NULL;
    uint32_t *prompt = NULL, proposals[32], previous;
    size_t prompt_count = 0, position, limit, window, vocabulary, count, bytes;
    float *q_bank = NULL;
    uint64_t rng, draft_rng;
    int result = -1, stopped = 0;
    if (draft == NULL) {
        if (r != NULL && r->speculative_tokens != 0) {
            if (out != NULL) memset(out, 0, sizeof(*out));
            nya_llm_cpu_error(error, error_capacity, "speculative_tokens requires a draft model");
            return -1;
        }
        return nya_llm_generate(c, r, out, error, error_capacity);
    }
    if (out != NULL) memset(out, 0, sizeof(*out));
    if (c == NULL || r == NULL || out == NULL || r->prompt == NULL || r->max_tokens == 0 ||
        r->max_output_bytes == 0 || r->max_output_bytes == SIZE_MAX || r->speculative_tokens > 32 ||
        !isfinite(r->temperature) || r->temperature < 0.0f || r->temperature > 5.0f ||
        !isfinite(r->top_p) || r->top_p <= 0.0f || r->top_p > 1.0f) {
        nya_llm_cpu_error(error, error_capacity, "invalid speculative generation request"); return -1;
    }
    if (!nya_llm_tokenizers_match(&c->tokenizer, &draft->tokenizer, draft->is_assistant)) {
        nya_llm_cpu_error(error, error_capacity, "target and draft tokenizers are incompatible"); return -1;
    }
    if (c->is_assistant || (draft->is_assistant && (!c->is_gemma || draft->backbone_length != c->embedding_length))) {
        nya_llm_cpu_error(error, error_capacity, "MTP assistant requires a Gemma 4 target with the matching hidden width"); return -1;
    }
    if (nya_llm_tokenize(c, r->prompt, &prompt, &prompt_count, error, error_capacity) != 0) goto cleanup;
    count = c->context_length < draft->context_length ? c->context_length : draft->context_length;
    if (prompt_count >= count) { nya_llm_cpu_error(error, error_capacity, "prompt fills target or draft context"); goto cleanup; }
    limit = r->max_tokens < count - prompt_count ? r->max_tokens : count - prompt_count;
    if (nya_llm_state_create(c, prompt_count + limit, &target_state, error, error_capacity) != 0 ||
        nya_llm_state_create(draft, prompt_count + limit, &draft_state, error, error_capacity) != 0) goto cleanup;
    if (draft->is_assistant) {
        /* Bind the last non-sharing target layer of each attention kind. This
           matches the full/local cache dictionaries in the published assistant.
           Shape agreement is necessary even when vocabulary IDs agree. */
        for (size_t layer = 0; layer < draft->block_count; ++layer) {
            const nya_llm_layer *w = &draft->layers[layer];
            uint32_t source = UINT32_MAX;
            for (uint32_t j = 0; j < c->block_count; ++j)
                if (c->layers[j].kv_source == j && (c->layers[j].sliding_window != 0) == (w->sliding_window != 0)) source = j;
            if (source == UINT32_MAX || c->layers[source].head_dimension != w->head_dimension ||
                c->layers[source].kv_head_count != w->kv_head_count) {
                nya_llm_cpu_error(error, error_capacity, "MTP assistant KV layout is incompatible with target"); goto cleanup;
            }
            draft_state.mtp_sources[layer] = source;
        }
    }
    window = r->speculative_tokens == 0 ? 4 : r->speculative_tokens;
    vocabulary = c->tokenizer.vocabulary_size;
    if (r->temperature > 0.0f) {
        count = r->top_k == 0 || r->top_k > vocabulary ? vocabulary : r->top_k;
        if (nya_llm_cpu_product(count, sizeof(*candidates), &bytes) != 0) goto cleanup;
        candidates = (nya_llm_probability *)malloc(bytes);
        if (nya_llm_cpu_product(window, vocabulary, &count) != 0 ||
            nya_llm_cpu_product(count, sizeof(float), &bytes) != 0) goto cleanup;
        q_bank = (float *)malloc(bytes);
        if (candidates == NULL || q_bank == NULL) goto cleanup;
    }
    out->text = (char *)malloc(r->max_output_bytes + 1);
    if (out->text == NULL) goto cleanup;
    out->text[0] = '\0'; out->prompt_tokens = prompt_count; out->seed = r->seed;
    out->stop_reason = limit < r->max_tokens ? NYA_GENERATION_STOP_CONTEXT : NYA_GENERATION_STOP_LENGTH;
    rng = r->seed == 0 ? UINT64_C(0x9E3779B97F4A7C15) : r->seed;
    draft_rng = rng ^ UINT64_C(0xD1B54A32D192ED03);
    if (draft_rng == 0) draft_rng = 1;
    if (nya_llm_prefill(c, &target_state, prompt, prompt_count, r, error, error_capacity) != 0 ||
        (!draft->is_assistant && nya_llm_prefill(draft, &draft_state, prompt, prompt_count, r, error, error_capacity) != 0)) goto cleanup;
    position = prompt_count; out->target_steps = prompt_count;
    previous = prompt[prompt_count - 1];
    while (!stopped && out->generated_tokens < limit) {
        size_t proposed = 0, round_start, maximum, mtp_cache_length = position;
        int rejected = 0;
        if (draft->is_assistant) {
            uint32_t base_token;
            /* MTP predicts after a token already sampled by the target. Freeze
               the preceding hidden state/KV prefix before advancing that token;
               later draft steps use recurrent projected states and this same
               cache prefix. This differs from an ordinary standalone draft. */
            if (nya_llm_sample(target_state.logits, vocabulary, r->temperature, r->top_p,
                r->top_k, &rng, candidates, &base_token) != 0) goto invalid;
            stopped = nya_llm_emit(c, r, out, previous, base_token);
            if (stopped < 0) goto invalid;
            if (stopped || out->generated_tokens == limit) break;
            memcpy(draft_state.mtp_hidden, target_state.normalized, c->embedding_length * sizeof(float));
            if (nya_llm_forward_impl(draft, &draft_state, base_token, position, 1, c, &target_state, mtp_cache_length) != 0 ||
                nya_llm_forward(c, &target_state, base_token, position, 1) != 0) goto invalid;
            ++out->target_steps;
            previous = base_token; ++position;
        }
        round_start = position;
        maximum = window < limit - out->generated_tokens ? window : limit - out->generated_tokens;
        /* q for each speculative position belongs to that exact draft prefix.
           Preserve it before the draft's next forward overwrites its logits. */
        for (; proposed < maximum; ++proposed) {
            uint32_t token;
            if (nya_llm_sample(draft_state.logits, vocabulary, r->temperature, r->top_p, r->top_k,
                &draft_rng, candidates, &token) != 0) goto invalid;
            proposals[proposed] = token;
            ++out->draft_tokens;
            if (q_bank != NULL) memcpy(q_bank + proposed * vocabulary, draft_state.logits, vocabulary * sizeof(float));
            if (draft->tokenizer.stop_tokens[token]) { ++proposed; break; }
            if (draft->is_assistant) {
                if (nya_llm_forward_impl(draft, &draft_state, token, round_start + proposed, 1,
                    c, &target_state, mtp_cache_length) != 0) goto invalid;
            } else if (nya_llm_forward(draft, &draft_state, token, round_start + proposed, 1) != 0) goto invalid;
        }
        for (size_t j = 0; j < proposed && !stopped; ++j) {
            uint32_t token = proposals[j];
            int accepted;
            if (r->temperature == 0.0f) {
                uint32_t best;
                if (nya_llm_sample(target_state.logits, vocabulary, 0.0f, r->top_p, r->top_k,
                    &rng, NULL, &best) != 0) goto invalid;
                accepted = token == best;
                token = best;
            } else {
                float *q = q_bank + j * vocabulary;
                if (nya_llm_distribution(target_state.logits, vocabulary, r->temperature,
                    r->top_p, r->top_k, candidates, &count) != 0) goto invalid;
                if (!(q[token] > 0.0f)) goto invalid;
                accepted = (double)nya_llm_random_float(&rng) * q[token] < target_state.logits[token];
                if (!accepted) {
                    for (size_t v = 0; v < vocabulary; ++v)
                        target_state.logits[v] = fmaxf(target_state.logits[v] - q[v], 0.0f);
                    if (nya_llm_draw(target_state.logits, vocabulary, &rng, &token) != 0) goto invalid;
                }
            }
            if (accepted) ++out->accepted_draft_tokens;
            stopped = nya_llm_emit(c, r, out, previous, token);
            if (stopped < 0) goto invalid;
            if (!stopped && out->generated_tokens < limit) {
                if (nya_llm_forward(c, &target_state, token, position, 1) != 0) goto invalid;
                ++out->target_steps;
                if (!accepted && !draft->is_assistant && nya_llm_forward(draft, &draft_state, token, position, 1) != 0) goto invalid;
            }
            previous = token; ++position;
            if (!accepted) { rejected = 1; break; }
        }
        /* If every proposal was accepted, p at the next position is already
           available. One bonus token guarantees progress beyond the draft. */
        if (!draft->is_assistant && !rejected && !stopped && out->generated_tokens < limit) {
            uint32_t token;
            if (nya_llm_sample(target_state.logits, vocabulary, r->temperature, r->top_p,
                r->top_k, &rng, candidates, &token) != 0) goto invalid;
            stopped = nya_llm_emit(c, r, out, previous, token);
            if (stopped < 0) goto invalid;
            if (!stopped && out->generated_tokens < limit) {
                if (nya_llm_forward(c, &target_state, token, position, 1) != 0 ||
                    nya_llm_forward(draft, &draft_state, token, position, 1) != 0) goto invalid;
                ++out->target_steps;
            }
            previous = token; ++position;
        }
    }
    if (!nya_llm_text_is_utf8(out->text, out->text_length)) goto invalid;
    result = 0;
    goto cleanup;
invalid:
    nya_llm_cpu_error(error, error_capacity, "speculative forward, sampling, or bounded UTF-8 output failed");
cleanup:
    free(prompt); free(candidates); free(q_bank);
    nya_llm_state_free(&target_state); nya_llm_state_free(&draft_state);
    if (result != 0) {
        free(out->text); memset(out, 0, sizeof(*out));
        if (error != NULL && error_capacity > 0 && error[0] == '\0')
            nya_llm_cpu_error(error, error_capacity, "speculative allocation failed");
    }
    return result;
}
