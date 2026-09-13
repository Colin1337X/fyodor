#include "llm_internal.h"
#include <string.h>

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

/* Round to IEEE binary16, then expand for reference arithmetic. Integer
   rounding implements ties-to-even independently of the host FP rounding mode;
   memcpy avoids aliasing UB. Subnormals, signed zero and overflow are retained.
   Used by cache diagnostics to study rounding boundaries; normal inference
   retains F32 KV. This is never an autograd operation. */
float nya_llm_round_f16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U, magnitude = bits & 0x7fffffffU;
    uint32_t result;
    if (magnitude >= 0x7f800000U) result = magnitude == 0x7f800000U ? 0x7c00U : 0x7e00U;
    else if (magnitude >= 0x477ff000U) result = 0x7c00U;
    else if (magnitude < 0x33000000U) result = 0;
    else {
        uint32_t exponent = magnitude >> 23;
        uint32_t shift = exponent < 113U ? 126U-exponent : 13U;
        uint32_t mantissa = (magnitude & 0x7fffffU) | (exponent < 113U ? 0x800000U : 0U);
        uint32_t rounded = mantissa >> shift;
        uint32_t remainder = mantissa & ((1U << shift)-1U), midpoint = 1U << (shift-1U);
        if (remainder > midpoint || (remainder == midpoint && (rounded & 1U))) ++rounded;
        result = exponent < 113U ? rounded : ((exponent-112U) << 10) + rounded;
    }
    uint16_t half = (uint16_t)(sign | result);
    unsigned char bytes[2] = {(unsigned char)half, (unsigned char)(half >> 8)};
    return nya_llm_half_to_float(bytes);
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
