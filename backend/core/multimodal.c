#include "multimodal.h"
#include "llm_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Input tensors can come directly from a network byte buffer. Never cast that
   buffer to float*: neither alignment nor its effective C type is guaranteed. */
static float nya_mm_read(const void *data, size_t index)
{
    float value;
    memcpy(&value, (const unsigned char *)data + index * sizeof(float), sizeof(value));
    return value;
}

/* LayerNorm subtracts the mean; the final projector RMSNorm does not. Double
   accumulation avoids overflow for large finite F32 inputs. Affine parameters
   are applied after normalization, matching the published Unified graph. */
static void nya_mm_norm(float *v, size_t count, const nya_llm_tensor *weight,
    const nya_llm_tensor *bias, float epsilon, int center)
{
    double mean = 0.0, variance = 0.0;
    if (center) {
        for (size_t i = 0; i < count; ++i) mean += v[i];
        mean /= (double)count;
    }
    for (size_t i = 0; i < count; ++i) { double delta = (double)v[i] - mean; variance += delta * delta; }
    double inverse = 1.0 / sqrt(variance / (double)count + epsilon);
    for (size_t i = 0; i < count; ++i) {
        double value = ((double)v[i] - mean) * inverse;
        if (weight != NULL) value *= nya_llm_tensor_value(weight, i);
        if (bias != NULL) value += nya_llm_tensor_value(bias, i);
        v[i] = (float)value;
    }
}

int nya_multimodal_attach(nya_model *model, char *error, size_t capacity)
{
    nya_llm_context *context = NULL;
    if (model->execution_context != NULL) return model->inference_supported ? 0 : -1;
    if (nya_llm_load_projector(model->path, model->file_size, &context, error, capacity) != 0) return -1;
    model->execution_context = context;
    model->inference_supported = 1;
    return 0;
}

void nya_multimodal_detach(nya_model *model)
{
    nya_llm_free((nya_llm_context *)model->execution_context);
    model->execution_context = NULL;
    model->inference_supported = 0;
}

int nya_multimodal_run(nya_model *model, const nya_execution_request *r,
    nya_execution_response *out, char *error, size_t capacity)
{
    const char *failure = "invalid projector request";
    nya_llm_context *c;
    const nya_llm_projector *p;
    float *scratch = NULL, *hidden = NULL, *output = NULL;
    size_t elements = 1, tokens = 0, input_width = 0, width = 0, height = 0, columns = 0, output_count;
    int audio = 0, packed = 0;
    if (out != NULL) memset(out, 0, sizeof(*out));
    if (model == NULL || r == NULL || out == NULL || !model->inference_supported ||
        model->execution_context == NULL || r->input_data == NULL || r->input_shape == NULL ||
        r->input_name == NULL || r->output_name == NULL || r->input_type != NYA_TENSOR_FLOAT32 ||
        r->input_rank == 0 || r->input_rank > 3 || strcmp(r->output_name, "embeddings") != 0) goto fail;
    c = (nya_llm_context *)model->execution_context;
    p = &c->projector;
    for (size_t i = 0; i < r->input_rank; ++i) {
        if (r->input_shape[i] <= 0 || (uint64_t)r->input_shape[i] > SIZE_MAX / elements) goto fail;
        elements *= (size_t)r->input_shape[i];
    }
    if (elements > SIZE_MAX / sizeof(float) || elements * sizeof(float) != r->input_data_size) goto fail;
    if (strcmp(r->input_name, "audio_features") == 0) {
        audio = 1;
        if (p->audio == NULL || r->input_rank != 2 || (uint64_t)r->input_shape[1] != p->audio_width) goto fail;
        tokens = (size_t)r->input_shape[0]; input_width = p->audio_width;
    } else if (strcmp(r->input_name, "image_rgb") == 0) {
        if (p->vision == NULL || r->input_rank != 3 || r->input_shape[2] != 3) goto fail;
        height = (size_t)r->input_shape[0]; width = (size_t)r->input_shape[1];
        if (height % p->patch_size != 0 || width % p->patch_size != 0 ||
            height / p->patch_size > p->position_count || width / p->patch_size > p->position_count) goto fail;
        columns = width / p->patch_size;
        tokens = columns * (height / p->patch_size);
        input_width = (size_t)p->patch_size * p->patch_size * 3;
    } else if (strcmp(r->input_name, "image_patches") == 0) {
        /* Each row contains x,y followed by one channel-first raw pixel patch.
           Coordinates are exact F32 integers; padding rows must be removed by
           the caller, so every returned row is a real language soft token. */
        packed = 1;
        input_width = (size_t)p->patch_size * p->patch_size * 3;
        if (p->vision == NULL || r->input_rank != 2 || (uint64_t)r->input_shape[1] != input_width + 2) goto fail;
        tokens = (size_t)r->input_shape[0];
    } else goto fail;
    failure = "projector output exceeds the requested byte limit";
    if (tokens == 0 || tokens > SIZE_MAX / p->output_width) goto fail;
    output_count = tokens * p->output_width;
    if (output_count > SIZE_MAX / sizeof(float) || output_count * sizeof(float) > r->max_output_bytes) goto fail;
    failure = "projector input contains a non-finite value or pixels outside [0,1]";
    for (size_t i = 0; i < elements; ++i) {
        float value = nya_mm_read(r->input_data, i);
        int coordinate = packed && i % (input_width + 2) < 2;
        if (!isfinite(value) || (!audio && !coordinate && (value < 0.0f || value > 1.0f))) goto fail;
    }
    failure = "projector scratch allocation failed";
    scratch = (float *)malloc(input_width * sizeof(float));
    hidden = (float *)malloc((audio ? 1 : p->vision_width) * sizeof(float));
    output = (float *)malloc(output_count * sizeof(float));
    if (scratch == NULL || hidden == NULL || output == NULL) goto fail;
    for (size_t t = 0; t < tokens; ++t) {
        size_t x = 0, y = 0;
        if (audio) {
            for (size_t j = 0; j < input_width; ++j) scratch[j] = nya_mm_read(r->input_data, t * input_width + j);
            nya_mm_norm(scratch, input_width, NULL, NULL, p->audio_epsilon, 0);
            nya_llm_matvec(c->compute, output + t * p->output_width, p->audio, scratch, input_width, p->output_width);
        } else {
            if (packed) {
                float xf = nya_mm_read(r->input_data, t * (input_width + 2));
                float yf = nya_mm_read(r->input_data, t * (input_width + 2) + 1);
                failure = "image patch position is outside the positional table";
                /* Validate before float-to-integer conversion, which would be
                   undefined for out-of-range or non-finite values in C. */
                if (xf < 0 || yf < 0 || xf >= (float)p->position_count || yf >= (float)p->position_count ||
                    floorf(xf) != xf || floorf(yf) != yf) goto fail;
                x = (size_t)xf; y = (size_t)yf;
                for (size_t j = 0; j < input_width; ++j) scratch[j] = nya_mm_read(r->input_data, t * (input_width + 2) + 2 + j);
            } else {
                x = t % columns; y = t / columns;
                size_t side = p->patch_size;
                /* Public RGB is HWC; GGUF patch weights require CHW. No resize
                   is implicit: supplied dimensions must be whole model patches. */
                for (size_t channel = 0; channel < 3; ++channel)
                    for (size_t row = 0; row < side; ++row) for (size_t col = 0; col < side; ++col)
                        scratch[(channel * side + row) * side + col] = nya_mm_read(r->input_data,
                            ((y * side + row) * width + x * side + col) * 3 + channel);
            }
            nya_mm_norm(scratch, input_width, p->norm_weight[0], p->norm_bias[0], 1e-5f, 1);
            nya_llm_matvec(c->compute, hidden, p->patch, scratch, input_width, p->vision_width);
            for (size_t j = 0; j < p->vision_width; ++j) hidden[j] += nya_llm_tensor_value(p->patch_bias, j);
            nya_mm_norm(hidden, p->vision_width, p->norm_weight[1], p->norm_bias[1], 1e-5f, 1);
            for (size_t j = 0; j < p->vision_width; ++j) {
                hidden[j] += nya_llm_tensor_value(p->position, x * p->vision_width + j);
                hidden[j] += nya_llm_tensor_value(p->position, ((size_t)p->position_count + y) * p->vision_width + j);
            }
            nya_mm_norm(hidden, p->vision_width, p->norm_weight[2], p->norm_bias[2], 1e-5f, 1);
            nya_mm_norm(hidden, p->vision_width, NULL, NULL, p->vision_epsilon, 0);
            nya_llm_matvec(c->compute, output + t * p->output_width, p->vision, hidden, p->vision_width, p->output_width);
        }
    }
    failure = "projector produced a non-finite embedding";
    for (size_t i = 0; i < output_count; ++i) if (!isfinite(output[i])) goto fail;
    out->output_type = NYA_TENSOR_FLOAT32; out->output_rank = 2;
    out->output_shape[0] = (int64_t)tokens; out->output_shape[1] = p->output_width;
    out->output_data = output; out->output_data_size = output_count * sizeof(float);
    free(scratch); free(hidden);
    return 0;
fail:
    free(scratch); free(hidden); free(output);
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", failure);
    return -1;
}
