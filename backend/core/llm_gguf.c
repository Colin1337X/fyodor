#include "llm_internal.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
/* Windows supplies UTF-8 path conversion and read-only file mappings. */
#include <windows.h>
#else
/* POSIX supplies read-only file mappings through open, fstat, and mmap. */
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Model parsing is bounded independently from the broader inspection layer. */
#define NYA_LLM_METADATA_LIMIT 1000000ULL
#define NYA_LLM_TENSOR_LIMIT 100000ULL
#define NYA_LLM_VOCABULARY_LIMIT 1000000ULL
#define NYA_LLM_STRING_LIMIT 1048576ULL
#define NYA_LLM_ARRAY_LIMIT 100000000ULL
#define NYA_LLM_NESTING_LIMIT 8U
#define NYA_LLM_DIMENSION_LIMIT 65536U
#define NYA_LLM_LAYER_LIMIT 256U
#define NYA_LLM_CONTEXT_LIMIT 1048576U

/* GGUF metadata value identifiers are fixed by the published format. */
enum nya_llm_gguf_value_type {
    NYA_LLM_GGUF_UINT8 = 0,
    NYA_LLM_GGUF_INT8 = 1,
    NYA_LLM_GGUF_UINT16 = 2,
    NYA_LLM_GGUF_INT16 = 3,
    NYA_LLM_GGUF_UINT32 = 4,
    NYA_LLM_GGUF_INT32 = 5,
    NYA_LLM_GGUF_FLOAT32 = 6,
    NYA_LLM_GGUF_BOOL = 7,
    NYA_LLM_GGUF_STRING = 8,
    NYA_LLM_GGUF_ARRAY = 9,
    NYA_LLM_GGUF_UINT64 = 10,
    NYA_LLM_GGUF_INT64 = 11,
    NYA_LLM_GGUF_FLOAT64 = 12
};

/* A memory reader advances only after every requested byte has been checked. */
typedef struct nya_llm_reader {
    const unsigned char *data;
    size_t size;
    size_t position;
} nya_llm_reader;

/* Temporary text settings are validated before tensor binding begins. */
typedef struct nya_llm_metadata_state {
    char architecture[32];
    char tokenizer_model[32];
    char tokenizer_pretokenizer[64];
    char rope_scaling_type[32];
    uint32_t alignment;
    uint32_t vocabulary_size;
    uint32_t expert_count;
} nya_llm_metadata_state;

/* Store one detailed loader error without exceeding caller-owned storage. */
static void nya_llm_error(char *error, size_t capacity, const char *format, ...)
{
    va_list arguments;

    if (error == NULL || capacity == 0) {
        return;
    }

    va_start(arguments, format);
    vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
    error[capacity - 1] = '\0';
}

/* Checked multiplication prevents model-controlled allocation overflow. */
static int nya_llm_size_product(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left) {
        return -1;
    }

    *result = left * right;
    return 0;
}

/* Read exact bytes from the already mapped model. */
static int nya_llm_read(nya_llm_reader *reader, void *output, size_t length)
{
    if (reader == NULL || reader->position > reader->size || length > reader->size - reader->position) {
        return -1;
    }

    if (length > 0 && output != NULL) {
        memcpy(output, reader->data + reader->position, length);
    }
    reader->position += length;
    return 0;
}

/* Decode unsigned little-endian integers without alignment assumptions. */
static int nya_llm_u8(nya_llm_reader *reader, uint8_t *value)
{
    return nya_llm_read(reader, value, 1);
}

static int nya_llm_u16(nya_llm_reader *reader, uint16_t *value)
{
    unsigned char bytes[2];

    if (nya_llm_read(reader, bytes, sizeof(bytes)) != 0) return -1;
    *value = (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
    return 0;
}

static int nya_llm_u32(nya_llm_reader *reader, uint32_t *value)
{
    unsigned char bytes[4];

    if (nya_llm_read(reader, bytes, sizeof(bytes)) != 0) return -1;
    *value = (uint32_t)bytes[0] |
             ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) |
             ((uint32_t)bytes[3] << 24);
    return 0;
}

static int nya_llm_u64(nya_llm_reader *reader, uint64_t *value)
{
    unsigned char bytes[8];
    size_t index;
    uint64_t result;

    if (nya_llm_read(reader, bytes, sizeof(bytes)) != 0) return -1;
    result = 0;
    for (index = 0; index < sizeof(bytes); ++index) {
        result |= (uint64_t)bytes[index] << (index * 8);
    }
    *value = result;
    return 0;
}

/* Decode IEEE floating-point metadata through its integer representation. */
static int nya_llm_f32(nya_llm_reader *reader, float *value)
{
    uint32_t bits;

    if (sizeof(float) != 4 || nya_llm_u32(reader, &bits) != 0) return -1;
    memcpy(value, &bits, sizeof(bits));
    return isfinite(*value) ? 0 : -1;
}

static int nya_llm_f64(nya_llm_reader *reader, double *value)
{
    uint64_t bits;

    if (sizeof(double) != 8 || nya_llm_u64(reader, &bits) != 0) return -1;
    memcpy(value, &bits, sizeof(bits));
    return isfinite(*value) ? 0 : -1;
}

/* Allocate one bounded GGUF string and reject embedded terminators. */
static int nya_llm_string(nya_llm_reader *reader, char **output, size_t *output_length)
{
    uint64_t length;
    char *text;

    if (nya_llm_u64(reader, &length) != 0 || length > NYA_LLM_STRING_LIMIT ||
        length > (uint64_t)SIZE_MAX - 1 || length > reader->size - reader->position) {
        return -1;
    }

    text = (char *)malloc((size_t)length + 1);
    if (text == NULL) return -1;
    if (nya_llm_read(reader, text, (size_t)length) != 0 ||
        memchr(text, '\0', (size_t)length) != NULL) {
        free(text);
        return -1;
    }
    text[length] = '\0';
    *output = text;
    if (output_length != NULL) *output_length = (size_t)length;
    return 0;
}

/* Copy a parsed string into one fixed metadata field. */
static int nya_llm_fixed_string(nya_llm_reader *reader, char *output, size_t capacity)
{
    char *text;
    size_t length;

    text = NULL;
    if (nya_llm_string(reader, &text, &length) != 0 || length >= capacity) {
        free(text);
        return -1;
    }
    memcpy(output, text, length + 1);
    free(text);
    return 0;
}

/* Return the fixed width of one primitive GGUF metadata type. */
static size_t nya_llm_primitive_size(uint32_t type)
{
    switch (type) {
        case NYA_LLM_GGUF_UINT8:
        case NYA_LLM_GGUF_INT8:
        case NYA_LLM_GGUF_BOOL:
            return 1;
        case NYA_LLM_GGUF_UINT16:
        case NYA_LLM_GGUF_INT16:
            return 2;
        case NYA_LLM_GGUF_UINT32:
        case NYA_LLM_GGUF_INT32:
        case NYA_LLM_GGUF_FLOAT32:
            return 4;
        case NYA_LLM_GGUF_UINT64:
        case NYA_LLM_GGUF_INT64:
        case NYA_LLM_GGUF_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

/* Skip one unknown metadata value while bounding arrays and recursion. */
static int nya_llm_skip_value(nya_llm_reader *reader, uint32_t type, unsigned int depth)
{
    size_t primitive_size;

    primitive_size = nya_llm_primitive_size(type);
    if (primitive_size != 0) return nya_llm_read(reader, NULL, primitive_size);

    if (type == NYA_LLM_GGUF_STRING) {
        uint64_t length;

        if (nya_llm_u64(reader, &length) != 0 || length > reader->size - reader->position) return -1;
        return nya_llm_read(reader, NULL, (size_t)length);
    }

    if (type == NYA_LLM_GGUF_ARRAY) {
        uint32_t element_type;
        uint64_t element_count;
        size_t element_size;
        uint64_t index;

        if (depth >= NYA_LLM_NESTING_LIMIT ||
            nya_llm_u32(reader, &element_type) != 0 ||
            nya_llm_u64(reader, &element_count) != 0 ||
            element_count > NYA_LLM_ARRAY_LIMIT) return -1;

        element_size = nya_llm_primitive_size(element_type);
        if (element_size != 0) {
            if (element_count > (uint64_t)SIZE_MAX / element_size) return -1;
            return nya_llm_read(reader, NULL, (size_t)element_count * element_size);
        }

        if (element_type != NYA_LLM_GGUF_STRING && element_type != NYA_LLM_GGUF_ARRAY) return -1;
        for (index = 0; index < element_count; ++index) {
            if (nya_llm_skip_value(reader, element_type, depth + 1) != 0) return -1;
        }
        return 0;
    }

    return -1;
}

/* Read a non-negative integer from any compatible scalar metadata type. */
static int nya_llm_unsigned_value(nya_llm_reader *reader, uint32_t type, uint64_t *value)
{
    uint8_t value8;
    uint16_t value16;
    uint32_t value32;

    switch (type) {
        case NYA_LLM_GGUF_UINT8:
            if (nya_llm_u8(reader, &value8) != 0) return -1;
            *value = value8;
            return 0;
        case NYA_LLM_GGUF_UINT16:
            if (nya_llm_u16(reader, &value16) != 0) return -1;
            *value = value16;
            return 0;
        case NYA_LLM_GGUF_UINT32:
            if (nya_llm_u32(reader, &value32) != 0) return -1;
            *value = value32;
            return 0;
        case NYA_LLM_GGUF_UINT64:
            return nya_llm_u64(reader, value);
        case NYA_LLM_GGUF_INT8:
            if (nya_llm_u8(reader, &value8) != 0 || (int8_t)value8 < 0) return -1;
            *value = (uint64_t)(int8_t)value8;
            return 0;
        case NYA_LLM_GGUF_INT16:
            if (nya_llm_u16(reader, &value16) != 0 || (int16_t)value16 < 0) return -1;
            *value = (uint64_t)(int16_t)value16;
            return 0;
        case NYA_LLM_GGUF_INT32:
            if (nya_llm_u32(reader, &value32) != 0 || (int32_t)value32 < 0) return -1;
            *value = (uint64_t)(int32_t)value32;
            return 0;
        default:
            return -1;
    }
}

/* Read one finite floating-point metadata value. */
static int nya_llm_float_value(nya_llm_reader *reader, uint32_t type, float *value)
{
    double value64;

    if (type == NYA_LLM_GGUF_FLOAT32) return nya_llm_f32(reader, value);
    if (type != NYA_LLM_GGUF_FLOAT64 || nya_llm_f64(reader, &value64) != 0 ||
        value64 < -FLT_MAX || value64 > FLT_MAX) return -1;
    *value = (float)value64;
    return 0;
}

/* Read one exact GGUF boolean byte. */
static int nya_llm_boolean_value(nya_llm_reader *reader, uint32_t type, int *value)
{
    uint8_t byte;

    if (type != NYA_LLM_GGUF_BOOL || nya_llm_u8(reader, &byte) != 0 || byte > 1) return -1;
    *value = byte != 0;
    return 0;
}

/* Parse the tokenizer's complete string vocabulary array. */
static int nya_llm_token_pieces(nya_llm_reader *reader, uint32_t type, nya_llm_tokenizer *tokenizer)
{
    uint32_t element_type;
    uint64_t count;
    size_t allocation_size;
    uint64_t index;

    if (type != NYA_LLM_GGUF_ARRAY || tokenizer->pieces != NULL ||
        nya_llm_u32(reader, &element_type) != 0 || element_type != NYA_LLM_GGUF_STRING ||
        nya_llm_u64(reader, &count) != 0 || count == 0 || count > NYA_LLM_VOCABULARY_LIMIT ||
        (tokenizer->vocabulary_size != 0 && tokenizer->vocabulary_size != count) ||
        nya_llm_size_product((size_t)count, sizeof(*tokenizer->pieces), &allocation_size) != 0) return -1;

    /* Establish ownership before allocating: cleanup must iterate the number of
       pointer slots actually allocated, even if a later allocation/read fails.
       In particular, a scores array preceding a shorter pieces array must not
       make cleanup walk beyond the pieces allocation. */
    tokenizer->vocabulary_size = (uint32_t)count;
    tokenizer->pieces = (char **)calloc(1, allocation_size);
    tokenizer->piece_lengths = (size_t *)calloc((size_t)count, sizeof(*tokenizer->piece_lengths));
    if (tokenizer->pieces == NULL || tokenizer->piece_lengths == NULL) return -1;

    for (index = 0; index < count; ++index) {
        size_t length;

        if (nya_llm_string(reader, &tokenizer->pieces[index], &length) != 0) return -1;
        tokenizer->piece_lengths[index] = length;
        if (length > tokenizer->maximum_piece_length) tokenizer->maximum_piece_length = length;
    }
    return 0;
}

/* Parse one float score for every vocabulary piece. */
static int nya_llm_token_scores(nya_llm_reader *reader, uint32_t type, nya_llm_tokenizer *tokenizer)
{
    uint32_t element_type;
    uint64_t count;
    uint64_t index;

    if (type != NYA_LLM_GGUF_ARRAY || tokenizer->scores != NULL ||
        nya_llm_u32(reader, &element_type) != 0 || element_type != NYA_LLM_GGUF_FLOAT32 ||
        nya_llm_u64(reader, &count) != 0 || count == 0 || count > NYA_LLM_VOCABULARY_LIMIT) return -1;

    tokenizer->scores = (float *)malloc((size_t)count * sizeof(*tokenizer->scores));
    if (tokenizer->scores == NULL) return -1;
    for (index = 0; index < count; ++index) {
        if (nya_llm_f32(reader, &tokenizer->scores[index]) != 0) return -1;
    }
    if (tokenizer->vocabulary_size != 0 && tokenizer->vocabulary_size != count) return -1;
    if (tokenizer->vocabulary_size == 0) tokenizer->vocabulary_size = (uint32_t)count;
    return 0;
}

/* Parse one signed token class for every vocabulary piece. */
static int nya_llm_token_types(nya_llm_reader *reader, uint32_t type, nya_llm_tokenizer *tokenizer)
{
    uint32_t element_type;
    uint64_t count;
    uint64_t index;

    if (type != NYA_LLM_GGUF_ARRAY || tokenizer->types != NULL ||
        nya_llm_u32(reader, &element_type) != 0 || element_type != NYA_LLM_GGUF_INT32 ||
        nya_llm_u64(reader, &count) != 0 || count == 0 || count > NYA_LLM_VOCABULARY_LIMIT) return -1;

    tokenizer->types = (int32_t *)malloc((size_t)count * sizeof(*tokenizer->types));
    if (tokenizer->types == NULL) return -1;
    for (index = 0; index < count; ++index) {
        uint32_t bits;

        if (nya_llm_u32(reader, &bits) != 0) return -1;
        tokenizer->types[index] = (int32_t)bits;
    }
    if (tokenizer->vocabulary_size != 0 && tokenizer->vocabulary_size != count) return -1;
    if (tokenizer->vocabulary_size == 0) tokenizer->vocabulary_size = (uint32_t)count;
    return 0;
}

/* Assign a bounded integer metadata field after decoding its source type. */
static int nya_llm_assign_u32(
    nya_llm_reader *reader,
    uint32_t type,
    uint32_t maximum,
    uint32_t *destination
)
{
    uint64_t value;

    if (nya_llm_unsigned_value(reader, type, &value) != 0 || value > maximum) return -1;
    *destination = (uint32_t)value;
    return 0;
}

/* Read scalar-or-layer-array metadata without depending on the order in which
   block_count appears. Validation later requires either one value or N layers. */
static int nya_llm_layer_values(nya_llm_reader *reader, uint32_t type,
    uint32_t *values, uint32_t *count, uint32_t maximum)
{
    uint64_t length = 1;
    uint32_t item_type = type;
    size_t index;
    if (*count != 0) return -1;
    if (type == NYA_LLM_GGUF_ARRAY &&
        (nya_llm_u32(reader, &item_type) != 0 || nya_llm_u64(reader, &length) != 0)) return -1;
    if (length == 0 || length > NYA_LLM_LAYER_LIMIT) return -1;
    for (index = 0; index < (size_t)length; ++index) {
        if (item_type == NYA_LLM_GGUF_BOOL) {
            int value;
            if (nya_llm_boolean_value(reader, item_type, &value) != 0) return -1;
            values[index] = (uint32_t)value;
        } else if (nya_llm_assign_u32(reader, item_type, maximum, &values[index]) != 0) return -1;
        if (values[index] > maximum) return -1;
    }
    *count = (uint32_t)length;
    return 0;
}

static int nya_llm_rule_compare(const void *a, const void *b)
{
    const nya_llm_bpe_rule *left = (const nya_llm_bpe_rule *)a;
    const nya_llm_bpe_rule *right = (const nya_llm_bpe_rule *)b;
    int order = strcmp(left->text, right->text);
    if (order != 0) return order;
    return left->left_length < right->left_length ? -1 : left->left_length > right->left_length;
}

/* Gemma 4's BPE uses explicit pair ranks. SentencePiece scores alone do not
   reproduce this tokenizer. Preserve the pair boundary while removing the
   metadata separator, then sort for logarithmic lookup during merging. */
static int nya_llm_read_merges(nya_llm_reader *reader, uint32_t type, nya_llm_tokenizer *tok)
{
    uint32_t item_type;
    uint64_t count;
    size_t index;
    if (tok->merges != NULL || type != NYA_LLM_GGUF_ARRAY ||
        nya_llm_u32(reader, &item_type) != 0 || item_type != NYA_LLM_GGUF_STRING ||
        nya_llm_u64(reader, &count) != 0 || count > NYA_LLM_VOCABULARY_LIMIT) return -1;
    tok->merges = (nya_llm_bpe_rule *)calloc((size_t)count + 1, sizeof(*tok->merges));
    if (tok->merges == NULL) return -1;
    for (index = 0; index < (size_t)count; ++index) {
        char *separator;
        size_t length;
        nya_llm_bpe_rule *rule = &tok->merges[index];
        tok->merge_count = index + 1; /* Own partial strings even on parse failure. */
        if (nya_llm_string(reader, &rule->text, &length) != 0 || length < 3) return -1;
        separator = strchr(rule->text + 1, ' ');
        if (separator == NULL || separator[1] == '\0') return -1;
        rule->left_length = (size_t)(separator - rule->text);
        memmove(separator, separator + 1, length - rule->left_length);
        rule->rank = (uint32_t)index;
    }
    qsort(tok->merges, tok->merge_count, sizeof(*tok->merges), nya_llm_rule_compare);
    for (index = 1; index < tok->merge_count; ++index)
        if (nya_llm_rule_compare(&tok->merges[index - 1], &tok->merges[index]) == 0) return -1;
    return 0;
}

/* Gemma metadata is dispatched by an exact prefix, including before the
   general.architecture entry. Unknown graph parameters fail closed. */
static int nya_llm_gemma_metadata(nya_llm_reader *r, uint32_t t,
    const char *key, nya_llm_context *c)
{
#define U32(k, field, max) if (strcmp(key, k) == 0) return nya_llm_assign_u32(r, t, max, &c->field)
#define F32(k, field) if (strcmp(key, k) == 0) return nya_llm_float_value(r, t, &c->field)
#define ARR(k, field, count, max) if (strcmp(key, k) == 0) return nya_llm_layer_values(r, t, c->field, &c->count, max)
    U32("context_length", context_length, NYA_LLM_CONTEXT_LIMIT);
    U32("embedding_length", embedding_length, NYA_LLM_DIMENSION_LIMIT);
    U32("embedding_length_out", backbone_length, NYA_LLM_DIMENSION_LIMIT);
    U32("nextn_predict_layers", nextn_layers, NYA_LLM_LAYER_LIMIT);
    U32("block_count", block_count, NYA_LLM_LAYER_LIMIT);
    U32("embedding_length_per_layer_input", per_layer_embedding_length, NYA_LLM_DIMENSION_LIMIT);
    U32("attention.key_length", head_dimension, NYA_LLM_DIMENSION_LIMIT);
    U32("attention.value_length", value_dimension, NYA_LLM_DIMENSION_LIMIT);
    U32("attention.key_length_swa", sliding_head_dimension, NYA_LLM_DIMENSION_LIMIT);
    U32("attention.value_length_swa", sliding_value_dimension, NYA_LLM_DIMENSION_LIMIT);
    U32("attention.sliding_window", sliding_window, NYA_LLM_CONTEXT_LIMIT);
    U32("attention.shared_kv_layers", shared_kv_layers, NYA_LLM_LAYER_LIMIT);
    U32("rope.dimension_count", rope_dimension_count, NYA_LLM_DIMENSION_LIMIT);
    U32("rope.dimension_count_swa", sliding_rope_dimension, NYA_LLM_DIMENSION_LIMIT);
    U32("expert_count", expert_count, 1024);
    U32("expert_used_count", experts_used, 1024);
    F32("attention.layer_norm_rms_epsilon", norm_epsilon);
    F32("rope.freq_base", rope_frequency_base);
    F32("rope.freq_base_swa", sliding_rope_base);
    F32("final_logit_softcapping", final_logit_softcap);
    ARR("attention.head_count", layer_heads, heads_count, NYA_LLM_DIMENSION_LIMIT);
    ARR("attention.head_count_kv", layer_kv_heads, kv_heads_count, NYA_LLM_DIMENSION_LIMIT);
    ARR("feed_forward_length", layer_hidden, hidden_count, NYA_LLM_DIMENSION_LIMIT * 4U);
    ARR("expert_feed_forward_length", layer_expert_hidden, expert_hidden_count, NYA_LLM_DIMENSION_LIMIT * 4U);
    ARR("attention.sliding_window_pattern", layer_sliding, sliding_count, NYA_LLM_LAYER_LIMIT);
#undef ARR
#undef F32
#undef U32
    return -1;
}

/* Parse one metadata entry needed by a native generation provider. */
static int nya_llm_metadata(
    nya_llm_reader *reader,
    nya_llm_context *context,
    nya_llm_metadata_state *state
)
{
    size_t entry_begin = reader->position;
    char *key;
    uint32_t type;
    int result;

    key = NULL;
    if (nya_llm_string(reader, &key, NULL) != 0 || nya_llm_u32(reader, &type) != 0) {
        free(key);
        return -1;
    }

    result = 0;
    if (strcmp(key, "general.architecture") == 0) {
        result = type == NYA_LLM_GGUF_STRING ?
            nya_llm_fixed_string(reader, state->architecture, sizeof(state->architecture)) : -1;
    } else if (strncmp(key, "gemma4.", 7) == 0) {
        result = nya_llm_gemma_metadata(reader, type, key + 7, context);
    } else if (strncmp(key, "gemma4-assistant.", 17) == 0) {
        result = nya_llm_gemma_metadata(reader, type, key + 17, context);
    } else if (strcmp(key, "clip.vision.projector_type") == 0 || strcmp(key, "clip.audio.projector_type") == 0) {
        char *destination = key[5] == 'v' ? context->projector.vision_type : context->projector.audio_type;
        result = type == NYA_LLM_GGUF_STRING ? nya_llm_fixed_string(reader, destination, 32) : -1;
    } else if (strcmp(key, "clip.vision.embedding_length") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &context->projector.vision_width);
    } else if (strcmp(key, "clip.audio.embedding_length") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &context->projector.audio_width);
    } else if (strcmp(key, "clip.vision.projection_dim") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &context->projector.output_width);
    } else if (strcmp(key, "clip.audio.projection_dim") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &context->projector.audio_output_width);
    } else if (strcmp(key, "clip.vision.block_count") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_LAYER_LIMIT, &context->projector.vision_blocks);
    } else if (strcmp(key, "clip.audio.block_count") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_LAYER_LIMIT, &context->projector.audio_blocks);
    } else if (strcmp(key, "clip.vision.attention.layer_norm_epsilon") == 0) {
        result = nya_llm_float_value(reader, type, &context->projector.vision_epsilon);
    } else if (strcmp(key, "clip.audio.attention.layer_norm_epsilon") == 0) {
        result = nya_llm_float_value(reader, type, &context->projector.audio_epsilon);
    } else if (strcmp(key, "tokenizer.ggml.merges") == 0) {
        result = nya_llm_read_merges(reader, type, &context->tokenizer);
    } else if (strcmp(key, "tokenizer.ggml.suppress_tokens") == 0) {
        uint64_t length;
        uint32_t item_type;
        nya_llm_tokenizer *tok = &context->tokenizer;
        if (tok->suppressed_tokens != NULL || type != NYA_LLM_GGUF_ARRAY ||
            nya_llm_u32(reader, &item_type) != 0 || nya_llm_u64(reader, &length) != 0 || length > NYA_LLM_VOCABULARY_LIMIT) result = -1;
        else {
            tok->suppressed_tokens = (uint32_t *)calloc((size_t)length + 1, sizeof(uint32_t));
            if (tok->suppressed_tokens == NULL) result = -1;
            else for (size_t j = 0; j < (size_t)length; ++j) {
                if (nya_llm_assign_u32(reader, item_type, (uint32_t)NYA_LLM_VOCABULARY_LIMIT, &tok->suppressed_tokens[j]) != 0) { result = -1; break; }
                tok->suppressed_count = j + 1;
            }
        }
    } else if (strcmp(key, "general.alignment") == 0) {
        result = nya_llm_assign_u32(reader, type, 1048576U, &state->alignment);
    } else if (strcmp(key, "llama.context_length") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_CONTEXT_LIMIT, &context->context_length);
    } else if (strcmp(key, "llama.embedding_length") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &context->embedding_length);
    } else if (strcmp(key, "llama.feed_forward_length") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT * 4U, &context->feed_forward_length);
    } else if (strcmp(key, "llama.block_count") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_LAYER_LIMIT, &context->block_count);
    } else if (strcmp(key, "llama.attention.head_count") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &context->head_count);
    } else if (strcmp(key, "llama.attention.head_count_kv") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &context->key_value_head_count);
    } else if (strcmp(key, "llama.rope.dimension_count") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &context->rope_dimension_count);
    } else if (strcmp(key, "llama.attention.layer_norm_rms_epsilon") == 0) {
        result = nya_llm_float_value(reader, type, &context->norm_epsilon);
    } else if (strcmp(key, "llama.rope.freq_base") == 0) {
        result = nya_llm_float_value(reader, type, &context->rope_frequency_base);
    } else if (strcmp(key, "llama.rope.scaling.type") == 0) {
        result = type == NYA_LLM_GGUF_STRING ?
            nya_llm_fixed_string(reader, state->rope_scaling_type, sizeof(state->rope_scaling_type)) : -1;
    } else if (strcmp(key, "llama.expert_count") == 0) {
        result = nya_llm_assign_u32(reader, type, NYA_LLM_DIMENSION_LIMIT, &state->expert_count);
    } else if (strcmp(key, "llama.vocab_size") == 0) {
        result = nya_llm_assign_u32(reader, type, (uint32_t)NYA_LLM_VOCABULARY_LIMIT, &state->vocabulary_size);
    } else if (strcmp(key, "tokenizer.ggml.model") == 0) {
        result = type == NYA_LLM_GGUF_STRING ?
            nya_llm_fixed_string(reader, state->tokenizer_model, sizeof(state->tokenizer_model)) : -1;
    } else if (strcmp(key, "tokenizer.ggml.pre") == 0) {
        result = type == NYA_LLM_GGUF_STRING ?
            nya_llm_fixed_string(reader, state->tokenizer_pretokenizer, sizeof(state->tokenizer_pretokenizer)) : -1;
    } else if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
        result = nya_llm_token_pieces(reader, type, &context->tokenizer);
    } else if (strcmp(key, "tokenizer.ggml.scores") == 0) {
        result = nya_llm_token_scores(reader, type, &context->tokenizer);
    } else if (strcmp(key, "tokenizer.ggml.token_type") == 0) {
        result = nya_llm_token_types(reader, type, &context->tokenizer);
    } else if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0) {
        result = nya_llm_assign_u32(reader, type, UINT32_MAX, &context->tokenizer.unknown_token);
    } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
        result = nya_llm_assign_u32(reader, type, UINT32_MAX, &context->tokenizer.beginning_token);
    } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
        result = nya_llm_assign_u32(reader, type, UINT32_MAX, &context->tokenizer.end_token);
    } else if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0) {
        result = nya_llm_boolean_value(reader, type, &context->tokenizer.add_beginning_token);
    } else if (strcmp(key, "tokenizer.ggml.add_eos_token") == 0) {
        result = nya_llm_boolean_value(reader, type, &context->tokenizer.add_end_token);
    } else if (strcmp(key, "tokenizer.ggml.add_space_prefix") == 0) {
        result = nya_llm_boolean_value(reader, type, &context->tokenizer.add_space_prefix);
    } else if (strcmp(key, "tokenizer.ggml.remove_extra_whitespaces") == 0) {
        int remove_whitespace = 0;
        result = nya_llm_boolean_value(reader, type, &remove_whitespace);
        if (remove_whitespace) result = -1;
    } else if (strncmp(key, "llama.", 6) == 0) {
        /* Unknown architecture parameters may alter the graph without changing
           tensor shapes (for example attention scaling). Never silently claim
           support for those variants or for unimplemented text normalization. */
        result = -1;
    } else {
        result = nya_llm_skip_value(reader, type, 0);
    }

    if (result == 0 && strcmp(key,"general.file_type") == 0) {
        if (context->file_type_begin != 0) result = -1;
        context->file_type_begin = entry_begin; context->file_type_end = reader->position;
    }
    free(key);
    return result;
}

/* Return the byte size of one supported tensor after checked element counting. */
static int nya_llm_tensor_size(nya_llm_tensor *tensor)
{
    uint64_t element_count;
    uint32_t dimension;
    uint64_t block_count;
    uint64_t byte_count;

    element_count = 1;
    for (dimension = 0; dimension < tensor->dimension_count; ++dimension) {
        if (tensor->dimensions[dimension] == 0 ||
            element_count > UINT64_MAX / tensor->dimensions[dimension]) return -1;
        element_count *= tensor->dimensions[dimension];
    }

    switch (tensor->type) {
        case NYA_LLM_TENSOR_F32:
            if (element_count > UINT64_MAX / 4) return -1;
            byte_count = element_count * 4;
            break;
        case NYA_LLM_TENSOR_BF16:
        case NYA_LLM_TENSOR_F16:
            if (element_count > UINT64_MAX / 2) return -1;
            byte_count = element_count * 2;
            break;
        case NYA_LLM_TENSOR_Q4_0:
            if (element_count % 32 != 0) return -1;
            block_count = element_count / 32;
            if (block_count > UINT64_MAX / 18) return -1;
            byte_count = block_count * 18;
            break;
        case NYA_LLM_TENSOR_Q8_0:
            if (element_count % 32 != 0) return -1;
            block_count = element_count / 32;
            if (block_count > UINT64_MAX / 34) return -1;
            byte_count = block_count * 34;
            break;
        case NYA_LLM_TENSOR_Q4_K:
        case NYA_LLM_TENSOR_Q6_K:
            if (element_count % 256 != 0) return -1;
            block_count = element_count / 256;
            if (block_count > UINT64_MAX / 210) return -1;
            byte_count = block_count * (tensor->type == NYA_LLM_TENSOR_Q4_K ? 144U : 210U);
            break;
        default:
            return 1;
    }

    if (byte_count > SIZE_MAX) return -1;
    tensor->data_size = (size_t)byte_count;
    return 0;
}

/* Parse every tensor directory entry before locating the aligned data section. */
static int nya_llm_tensor_directory(
    nya_llm_reader *reader,
    nya_llm_context *context,
    uint64_t tensor_count,
    uint32_t alignment
)
{
    uint64_t index;
    size_t data_start;
    size_t padding;

    context->tensors = (nya_llm_tensor *)calloc((size_t)tensor_count, sizeof(*context->tensors));
    if (context->tensors == NULL && tensor_count != 0) return -1;
    context->tensor_count = (size_t)tensor_count;

    for (index = 0; index < tensor_count; ++index) {
        nya_llm_tensor *tensor;
        char *name;
        size_t name_length;
        uint32_t dimension;

        tensor = &context->tensors[index];
        name = NULL;
        if (nya_llm_string(reader, &name, &name_length) != 0 || name_length >= sizeof(tensor->name) ||
            nya_llm_u32(reader, &tensor->dimension_count) != 0 ||
            tensor->dimension_count == 0 || tensor->dimension_count > 4) {
            free(name);
            return -1;
        }
        memcpy(tensor->name, name, name_length + 1);
        free(name);

        for (dimension = 0; dimension < tensor->dimension_count; ++dimension) {
            if (nya_llm_u64(reader, &tensor->dimensions[dimension]) != 0 ||
                tensor->dimensions[dimension] == 0) return -1;
        }
        if (nya_llm_u32(reader, &tensor->type) != 0 ||
            nya_llm_u64(reader, &tensor->relative_offset) != 0 ||
            tensor->relative_offset % alignment != 0) return -1;
    }

    /* GGUF requires an alignment that is a multiple of eight, not necessarily
       a power of two. Modulo padding also handles values such as 24 correctly. */
    padding = ((size_t)alignment - reader->position % alignment) % alignment;
    if (reader->position > SIZE_MAX - padding) return -1;
    data_start = reader->position + padding;
    if (data_start > reader->size) return -1;

    for (index = 0; index < tensor_count; ++index) {
        nya_llm_tensor *tensor;
        int size_result;

        tensor = &context->tensors[index];
        if (tensor->relative_offset > reader->size - data_start) return -1;
        size_result = nya_llm_tensor_size(tensor);
        if (size_result < 0) return -1;
        if (size_result == 0) {
            if (tensor->data_size > reader->size - data_start - (size_t)tensor->relative_offset) return -1;
            tensor->data = reader->data + data_start + (size_t)tensor->relative_offset;
        }
    }
    return 0;
}

/* Locate one tensor by its exact canonical GGUF name. */
static const nya_llm_tensor *nya_llm_find_tensor(const nya_llm_context *context, const char *name)
{
    size_t index;

    for (index = 0; index < context->tensor_count; ++index) {
        if (strcmp(context->tensors[index].name, name) == 0) return &context->tensors[index];
    }
    return NULL;
}

/* Require one supported vector with an exact element count. */
static const nya_llm_tensor *nya_llm_bind_vector(
    const nya_llm_context *context,
    const char *name,
    uint32_t length
)
{
    const nya_llm_tensor *tensor;

    tensor = nya_llm_find_tensor(context, name);
    if (tensor == NULL || tensor->data == NULL || tensor->dimension_count != 1 ||
        tensor->dimensions[0] != length) return NULL;
    context->tensors[tensor - context->tensors].bound = 1;
    return tensor;
}

/* Require one supported matrix stored as GGML columns by rows. */
static const nya_llm_tensor *nya_llm_bind_matrix(
    const nya_llm_context *context,
    const char *name,
    uint32_t columns,
    uint32_t rows
)
{
    const nya_llm_tensor *tensor;

    tensor = nya_llm_find_tensor(context, name);
    if (tensor == NULL || tensor->data == NULL || tensor->dimension_count != 2 ||
        tensor->dimensions[0] != columns || tensor->dimensions[1] != rows) return NULL;
    if ((tensor->type == NYA_LLM_TENSOR_Q4_0 || tensor->type == NYA_LLM_TENSOR_Q8_0) &&
        columns % 32 != 0) return NULL;
    if ((tensor->type == NYA_LLM_TENSOR_Q4_K || tensor->type == NYA_LLM_TENSOR_Q6_K) && columns % 256 != 0) return NULL;
    context->tensors[tensor - context->tensors].bound = 1;
    return tensor;
}

/* Expert matrices remain mapped. A request evaluates only the selected expert
   slices, avoiding a full dequantization/copy of the MoE parameter bank. */
static const nya_llm_tensor *nya_llm_bind_experts(const nya_llm_context *c,
    const char *name, uint32_t columns, uint32_t rows)
{
    const nya_llm_tensor *t = nya_llm_find_tensor(c, name);
    if (t == NULL || t->data == NULL || t->dimension_count != 3 ||
        t->dimensions[0] != columns || t->dimensions[1] != rows ||
        t->dimensions[2] != c->expert_count) return NULL;
    if ((t->type == NYA_LLM_TENSOR_Q4_0 || t->type == NYA_LLM_TENSOR_Q8_0) && columns % 32 != 0) return NULL;
    if ((t->type == NYA_LLM_TENSOR_Q4_K || t->type == NYA_LLM_TENSOR_Q6_K) && columns % 256 != 0) return NULL;
    c->tensors[t - c->tensors].bound = 1;
    return t;
}

static uint32_t nya_llm_layer_setting(const uint32_t *values, uint32_t count, uint32_t layer)
{
    return values[count == 1 ? 0 : layer];
}

/* Bind the exact Gemma 4 text graph. No choice depends on a marketing size:
   dense, MoE, PLE, and KV sharing are determined by checked metadata/tensors. */
static int nya_llm_bind_gemma(nya_llm_context *c, char *error, size_t error_capacity)
{
    uint32_t i, last_local = UINT32_MAX, last_global = UINT32_MAX;
    uint32_t d = c->embedding_length, p = c->per_layer_embedding_length;
    uint32_t counts[] = {c->heads_count, c->kv_heads_count, c->hidden_count,
        c->expert_hidden_count, c->sliding_count};
    if (d == 0 || c->block_count == 0 || c->context_length == 0 ||
        c->heads_count == 0 || c->kv_heads_count == 0 || c->hidden_count == 0 || c->sliding_count == 0 ||
        c->head_dimension == 0 || c->head_dimension % 2 != 0 ||
        c->sliding_head_dimension == 0 || c->sliding_head_dimension % 2 != 0 ||
        c->head_dimension != c->value_dimension || c->sliding_head_dimension != c->sliding_value_dimension ||
        c->rope_dimension_count != c->head_dimension || c->sliding_rope_dimension != c->sliding_head_dimension ||
        ((!c->is_assistant && c->shared_kv_layers >= c->block_count) ||
          (c->is_assistant && c->shared_kv_layers != c->block_count)) || c->sliding_window == 0 ||
        !(c->norm_epsilon > 0.0f) || !(c->rope_frequency_base > 0.0f) ||
        !(c->sliding_rope_base > 0.0f) || c->final_logit_softcap < 0.0f ||
        (c->expert_count != 0 && (c->experts_used == 0 || c->experts_used > c->expert_count ||
                                  c->expert_hidden_count == 0))) goto invalid;
    for (i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i)
        if (counts[i] != 0 && counts[i] != 1 && counts[i] != c->block_count) goto invalid;
    c->token_embedding = nya_llm_bind_matrix(c, "token_embd.weight", d, c->tokenizer.vocabulary_size);
    c->output_norm = nya_llm_bind_vector(c, "output_norm.weight", d);
    c->output = nya_llm_bind_matrix(c, "output.weight", d, c->tokenizer.vocabulary_size);
    if (c->output == NULL && nya_llm_find_tensor(c, "output.weight") == NULL) c->output = c->token_embedding;
    if (c->token_embedding == NULL || c->output_norm == NULL || c->output == NULL) goto invalid;
    if (c->is_assistant) {
        if (c->backbone_length == 0 || c->nextn_layers != c->block_count || p != 0 ||
            c->expert_count != 0 || c->final_logit_softcap != 0.0f) goto invalid;
        c->nextn_pre = nya_llm_bind_matrix(c, "nextn.pre_projection.weight", c->backbone_length * 2, d);
        c->nextn_post = nya_llm_bind_matrix(c, "nextn.post_projection.weight", d, c->backbone_length);
        if (c->nextn_pre == NULL || c->nextn_post == NULL) goto invalid;
    }
    if (p != 0) {
        /* Both factors are bounded to 65536/256, so the packed dimension fits
           uint32_t; the generic tensor reader still checks its full byte size. */
        c->ple_embedding = nya_llm_bind_matrix(c, "per_layer_token_embd.weight", p * c->block_count, c->tokenizer.vocabulary_size);
        c->ple_projection = nya_llm_bind_matrix(c, "per_layer_model_proj.weight", d, p * c->block_count);
        c->ple_norm = nya_llm_bind_vector(c, "per_layer_proj_norm.weight", p);
        if (c->ple_embedding == NULL || c->ple_projection == NULL || c->ple_norm == NULL) goto invalid;
    }
    c->layers = (nya_llm_layer *)calloc(c->block_count, sizeof(*c->layers));
    if (c->layers == NULL) goto invalid;
    for (i = 0; i < c->block_count; ++i) {
        nya_llm_layer *w = &c->layers[i];
        char name[128];
        uint32_t q, kv, h, local = nya_llm_layer_setting(c->layer_sliding, c->sliding_count, i);
        if (c->sliding_count == 1 && local > 1) local = (i + 1) % local != 0;
        if (local > 1) goto invalid;
        w->sliding_window = local ? c->sliding_window : 0;
        w->head_dimension = local ? c->sliding_head_dimension : c->head_dimension;
        w->head_count = nya_llm_layer_setting(c->layer_heads, c->heads_count, i);
        w->kv_head_count = nya_llm_layer_setting(c->layer_kv_heads, c->kv_heads_count, i);
        w->hidden_length = nya_llm_layer_setting(c->layer_hidden, c->hidden_count, i);
        w->expert_hidden_length = c->expert_hidden_count == 0 ? 0 :
            nya_llm_layer_setting(c->layer_expert_hidden, c->expert_hidden_count, i);
        if (w->head_count == 0 || w->kv_head_count == 0 || w->hidden_length == 0 ||
            w->head_count % w->kv_head_count != 0 ||
            w->head_count > NYA_LLM_DIMENSION_LIMIT / w->head_dimension) goto invalid;
        q = w->head_count * w->head_dimension;
        kv = w->kv_head_count * w->head_dimension;
        h = w->hidden_length;
        if (q > c->maximum_query_length) c->maximum_query_length = q;
        if (h > c->maximum_hidden_length) c->maximum_hidden_length = h;
        if (w->expert_hidden_length > c->maximum_hidden_length) c->maximum_hidden_length = w->expert_hidden_length;
        w->kv_source = i;
        if (c->is_assistant) {
            /* The target-dependent cache slice is resolved when pairing the
               models, not borrowed from an assistant layer that has no KV. */
            w->kv_source = UINT32_MAX;
        } else if (i >= c->block_count - c->shared_kv_layers) {
            w->kv_source = local ? last_local : last_global;
            if (w->kv_source == UINT32_MAX || c->layers[w->kv_source].head_dimension != w->head_dimension ||
                c->layers[w->kv_source].kv_head_count != w->kv_head_count) goto invalid;
            w->cache_offset = c->layers[w->kv_source].cache_offset;
        } else {
            if (local) last_local = i; else last_global = i;
            w->cache_offset = c->cache_width;
            if (kv > SIZE_MAX - c->cache_width) goto invalid;
            c->cache_width += kv;
        }
#define V(field, suffix, length) do { snprintf(name, sizeof(name), "blk.%u." suffix, i); \
    w->field = nya_llm_bind_vector(c, name, length); } while (0)
#define M(field, suffix, columns, rows) do { snprintf(name, sizeof(name), "blk.%u." suffix, i); \
    w->field = nya_llm_bind_matrix(c, name, columns, rows); } while (0)
#define E(field, suffix, columns, rows) do { snprintf(name, sizeof(name), "blk.%u." suffix, i); \
    w->field = nya_llm_bind_experts(c, name, columns, rows); } while (0)
        V(attention_norm, "attn_norm.weight", d);
        M(query, "attn_q.weight", d, q);
        V(query_norm, "attn_q_norm.weight", w->head_dimension);
        M(attention_output, "attn_output.weight", q, d);
        V(attention_post_norm, "post_attention_norm.weight", d);
        if (!c->is_assistant) {
            M(key, "attn_k.weight", d, kv);
            M(value, "attn_v.weight", d, kv);
            V(key_norm, "attn_k_norm.weight", w->head_dimension);
            /* K==V is permitted for global attention only. Invalid existing
               tensors remain unbound and are rejected by the final scan. */
            if (w->kv_source == i && (w->key == NULL || w->key_norm == NULL || (local && w->value == NULL))) goto invalid;
            /* Some exporters retain trained K/V weights on shared layers.
               Validate their shapes when present; execution still reads the
               metadata-selected source cache and does not run these matrices. */
        }
        V(feed_forward_norm, "ffn_norm.weight", d);
        M(feed_forward_gate, "ffn_gate.weight", d, h);
        M(feed_forward_up, "ffn_up.weight", d, h);
        M(feed_forward_down, "ffn_down.weight", h, d);
        V(ffn_post_norm, "post_ffw_norm.weight", d);
        V(output_scale, "layer_output_scale.weight", 1);
        if (w->attention_norm == NULL || w->query == NULL || w->query_norm == NULL ||
            w->attention_output == NULL || w->attention_post_norm == NULL || w->feed_forward_norm == NULL ||
            w->feed_forward_gate == NULL || w->feed_forward_up == NULL ||
            w->feed_forward_down == NULL || w->ffn_post_norm == NULL) goto invalid;
        if (!local) {
            w->rope_factors = nya_llm_bind_vector(c, "rope_freqs.weight", w->head_dimension / 2);
            if (w->rope_factors == NULL) goto invalid;
        }
        if (c->expert_count != 0) {
            uint32_t eh = w->expert_hidden_length;
            if (eh == 0) goto invalid;
            M(router, "ffn_gate_inp.weight", d, c->expert_count);
            V(router_scale, "ffn_gate_inp.scale", d);
            V(expert_scale, "ffn_down_exps.scale", c->expert_count);
            V(expert_pre_norm, "pre_ffw_norm_2.weight", d);
            V(shared_post_norm, "post_ffw_norm_1.weight", d);
            V(expert_post_norm, "post_ffw_norm_2.weight", d);
            E(expert_gate_up, "ffn_gate_up_exps.weight", d, eh * 2);
            if (w->expert_gate_up == NULL) {
                E(expert_gate, "ffn_gate_exps.weight", d, eh);
                E(expert_up, "ffn_up_exps.weight", d, eh);
                if (w->expert_gate == NULL || w->expert_up == NULL) goto invalid;
            }
            E(expert_down, "ffn_down_exps.weight", eh, d);
            if (w->router == NULL || w->router_scale == NULL || w->expert_scale == NULL ||
                w->expert_pre_norm == NULL || w->shared_post_norm == NULL ||
                w->expert_post_norm == NULL || w->expert_down == NULL) goto invalid;
        }
        if (p != 0) {
            M(ple_gate, "inp_gate.weight", d, p);
            M(ple_projection, "proj.weight", p, d);
            V(ple_norm, "post_norm.weight", d);
            if (w->ple_gate == NULL || w->ple_projection == NULL || w->ple_norm == NULL) goto invalid;
        }
#undef E
#undef M
#undef V
    }
    if (c->layers[c->block_count - 1].sliding_window != 0) goto invalid;
    for (i = 0; i < c->tensor_count; ++i) {
        if (!c->tensors[i].bound) {
            nya_llm_error(error, error_capacity, "Gemma tensor '%s' has an unsupported shape, role or quantization", c->tensors[i].name);
            return -1;
        }
    }
    return 0;
invalid:
    nya_llm_error(error, error_capacity, "Gemma 4 text dimensions, layer arrays or required tensors are inconsistent");
    return -1;
}

/* Bind the complete dense LLaMA tensor set and reject partial architectures. */
static int nya_llm_bind_weights(nya_llm_context *context, char *error, size_t error_capacity)
{
    uint32_t key_value_dimension;
    uint32_t layer;

    key_value_dimension = context->embedding_length / context->head_count * context->key_value_head_count;
    context->token_embedding = nya_llm_bind_matrix(
        context,
        "token_embd.weight",
        context->embedding_length,
        context->tokenizer.vocabulary_size
    );
    context->output_norm = nya_llm_bind_vector(context, "output_norm.weight", context->embedding_length);
    context->output = nya_llm_bind_matrix(
        context,
        "output.weight",
        context->embedding_length,
        context->tokenizer.vocabulary_size
    );
    /* GGUF permits tied embeddings by omitting output.weight. An explicitly
       present but invalid output tensor is a model error, not tied weights. */
    if (context->output == NULL && nya_llm_find_tensor(context, "output.weight") == NULL) {
        context->output = context->token_embedding;
    }
    if (context->token_embedding == NULL || context->output_norm == NULL || context->output == NULL) {
        nya_llm_error(error, error_capacity, "required embedding or output tensors are missing or use an unsupported quantization");
        return -1;
    }

    context->layers = (nya_llm_layer *)calloc(context->block_count, sizeof(*context->layers));
    if (context->layers == NULL) {
        nya_llm_error(error, error_capacity, "transformer layer allocation failed");
        return -1;
    }

    for (layer = 0; layer < context->block_count; ++layer) {
        nya_llm_layer *weights;
        char name[128];

        weights = &context->layers[layer];
        weights->head_count = context->head_count;
        weights->kv_head_count = context->key_value_head_count;
        weights->head_dimension = context->embedding_length / context->head_count;
        weights->hidden_length = context->feed_forward_length;
        weights->kv_source = layer;
        weights->cache_offset = (size_t)layer * key_value_dimension;
#define NYA_BIND_VECTOR(field, format, length) \
        do { \
            snprintf(name, sizeof(name), format, layer); \
            weights->field = nya_llm_bind_vector(context, name, length); \
        } while (0)
#define NYA_BIND_MATRIX(field, format, columns, rows) \
        do { \
            snprintf(name, sizeof(name), format, layer); \
            weights->field = nya_llm_bind_matrix(context, name, columns, rows); \
        } while (0)

        NYA_BIND_VECTOR(attention_norm, "blk.%u.attn_norm.weight", context->embedding_length);
        NYA_BIND_MATRIX(query, "blk.%u.attn_q.weight", context->embedding_length, context->embedding_length);
        NYA_BIND_MATRIX(key, "blk.%u.attn_k.weight", context->embedding_length, key_value_dimension);
        NYA_BIND_MATRIX(value, "blk.%u.attn_v.weight", context->embedding_length, key_value_dimension);
        NYA_BIND_MATRIX(attention_output, "blk.%u.attn_output.weight", context->embedding_length, context->embedding_length);
        NYA_BIND_VECTOR(feed_forward_norm, "blk.%u.ffn_norm.weight", context->embedding_length);
        NYA_BIND_MATRIX(feed_forward_gate, "blk.%u.ffn_gate.weight", context->embedding_length, context->feed_forward_length);
        NYA_BIND_MATRIX(feed_forward_down, "blk.%u.ffn_down.weight", context->feed_forward_length, context->embedding_length);
        NYA_BIND_MATRIX(feed_forward_up, "blk.%u.ffn_up.weight", context->embedding_length, context->feed_forward_length);

#undef NYA_BIND_MATRIX
#undef NYA_BIND_VECTOR

        if (weights->attention_norm == NULL || weights->query == NULL || weights->key == NULL ||
            weights->value == NULL || weights->attention_output == NULL ||
            weights->feed_forward_norm == NULL || weights->feed_forward_gate == NULL ||
            weights->feed_forward_down == NULL || weights->feed_forward_up == NULL) {
            nya_llm_error(error, error_capacity, "layer %u is missing a required tensor or uses an unsupported quantization", layer);
            return -1;
        }
    }

    /* Every tensor must participate in the implemented graph. Silently ignoring
       bias, learned RoPE factors, or an architectural extension can produce
       plausible text from the wrong computation, which is worse than a clear
       inspect-only result. This also rejects duplicate tensor directory names. */
    {
        size_t index;
        for (index = 0; index < context->tensor_count; ++index) {
            const nya_llm_tensor *tensor = &context->tensors[index];
            int used = tensor == context->token_embedding || tensor == context->output_norm ||
                       tensor == context->output;
            for (layer = 0; !used && layer < context->block_count; ++layer) {
                const nya_llm_layer *weights = &context->layers[layer];
                used = tensor == weights->attention_norm || tensor == weights->query ||
                       tensor == weights->key || tensor == weights->value ||
                       tensor == weights->attention_output || tensor == weights->feed_forward_norm ||
                       tensor == weights->feed_forward_gate || tensor == weights->feed_forward_up ||
                       tensor == weights->feed_forward_down;
            }
            if (!used) {
                nya_llm_error(error, error_capacity, "tensor '%s' is not supported by the dense LLaMA graph", tensor->name);
                return -1;
            }
        }
    }
    context->cache_width = (size_t)context->block_count * key_value_dimension;
    context->maximum_query_length = context->embedding_length;
    context->maximum_hidden_length = context->feed_forward_length;
    return 0;
}

/* Compare two sorted vocabulary entries by their complete token text. */
static int nya_llm_token_compare(const void *left, const void *right)
{
    const nya_llm_token_index *left_token;
    const nya_llm_token_index *right_token;

    left_token = (const nya_llm_token_index *)left;
    right_token = (const nya_llm_token_index *)right;
    return strcmp(left_token->piece, right_token->piece);
}

/* Parse the canonical six-byte byte-fallback token spelling. */
static int nya_llm_byte_piece(const char *piece, size_t length, unsigned int *byte)
{
    unsigned int high;
    unsigned int low;

    if (length != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') return -1;
    high = piece[3] >= '0' && piece[3] <= '9' ? (unsigned int)(piece[3] - '0') :
           piece[3] >= 'A' && piece[3] <= 'F' ? (unsigned int)(piece[3] - 'A' + 10) :
           piece[3] >= 'a' && piece[3] <= 'f' ? (unsigned int)(piece[3] - 'a' + 10) : 16U;
    low = piece[4] >= '0' && piece[4] <= '9' ? (unsigned int)(piece[4] - '0') :
          piece[4] >= 'A' && piece[4] <= 'F' ? (unsigned int)(piece[4] - 'A' + 10) :
          piece[4] >= 'a' && piece[4] <= 'f' ? (unsigned int)(piece[4] - 'a' + 10) : 16U;
    if (high > 15 || low > 15) return -1;
    *byte = high * 16 + low;
    return 0;
}

/* Finalize token defaults, lookup order, and byte-fallback mappings. */
static int nya_llm_finalize_tokenizer(
    nya_llm_context *context,
    const nya_llm_metadata_state *state,
    char *error,
    size_t error_capacity
)
{
    nya_llm_tokenizer *tokenizer;
    uint32_t token;

    tokenizer = &context->tokenizer;
    if (tokenizer->pieces == NULL || tokenizer->vocabulary_size == 0 ||
        (state->vocabulary_size != 0 && state->vocabulary_size != tokenizer->vocabulary_size)) {
        nya_llm_error(error, error_capacity, "the GGUF tokenizer vocabulary is missing or inconsistent");
        return -1;
    }
    /* SentencePiece merge order is model data. Fabricating zero scores changes
       prompt tokenization and therefore changes the entire generated sequence. */
    if (tokenizer->scores == NULL) {
        nya_llm_error(error, error_capacity, "SentencePiece generation requires tokenizer.ggml.scores");
        return -1;
    }
    if (tokenizer->types == NULL) {
        tokenizer->types = (int32_t *)malloc(tokenizer->vocabulary_size * sizeof(*tokenizer->types));
        if (tokenizer->types != NULL) {
            for (token = 0; token < tokenizer->vocabulary_size; ++token) tokenizer->types[token] = 1;
        }
    }
    tokenizer->sorted = (nya_llm_token_index *)malloc(tokenizer->vocabulary_size * sizeof(*tokenizer->sorted));
    if (tokenizer->scores == NULL || tokenizer->types == NULL || tokenizer->sorted == NULL) {
        nya_llm_error(error, error_capacity, "tokenizer allocation failed");
        return -1;
    }

    for (token = 0; token < 256; ++token) tokenizer->byte_tokens[token] = -1;
    for (token = 0; token < tokenizer->vocabulary_size; ++token) {
        unsigned int byte;

        /* User-defined pieces require atomic special-token matching before BPE,
           which this deliberately narrow plain-text tokenizer does not offer. */
        if (tokenizer->types[token] < 1 || tokenizer->types[token] > 6 ||
            (tokenizer->types[token] == 4 && !tokenizer->uses_merge_ranks)) {
            nya_llm_error(error, error_capacity, "tokenizer contains an unsupported token class");
            return -1;
        }
        tokenizer->sorted[token].piece = tokenizer->pieces[token];
        tokenizer->sorted[token].token = token;
        if (nya_llm_byte_piece(tokenizer->pieces[token], tokenizer->piece_lengths[token], &byte) == 0) {
            tokenizer->byte_tokens[byte] = (int32_t)token;
        }
    }
    qsort(tokenizer->sorted, tokenizer->vocabulary_size, sizeof(*tokenizer->sorted), nya_llm_token_compare);

    for (token = 1; token < tokenizer->vocabulary_size; ++token) {
        if (strcmp(tokenizer->sorted[token - 1].piece, tokenizer->sorted[token].piece) == 0) {
            nya_llm_error(error, error_capacity, "duplicate tokenizer pieces make token lookup ambiguous");
            return -1;
        }
    }

    tokenizer->uses_sentencepiece_space = 0;
    for (token = 0; token < tokenizer->vocabulary_size; ++token) {
        /* The standalone marker need not exist: it may occur only inside longer
           pieces whose prefix can still be discovered by text-symbol merges. */
        if (strstr(tokenizer->pieces[token], "\xE2\x96\x81") != NULL) {
            tokenizer->uses_sentencepiece_space = 1;
            break;
        }
    }

    if (tokenizer->unknown_token >= tokenizer->vocabulary_size) tokenizer->unknown_token = 0;
    if (tokenizer->beginning_token >= tokenizer->vocabulary_size ||
        tokenizer->end_token >= tokenizer->vocabulary_size) {
        nya_llm_error(error, error_capacity, "tokenizer BOS or EOS identifier is outside the vocabulary");
        return -1;
    }
    tokenizer->stop_tokens = (unsigned char *)calloc(tokenizer->vocabulary_size, 1);
    if (tokenizer->stop_tokens == NULL) return -1;
    tokenizer->stop_tokens[tokenizer->end_token] = 1;
    if (tokenizer->suppressed_count >= tokenizer->vocabulary_size) return -1;
    for (size_t j = 0; j < tokenizer->suppressed_count; ++j)
        if (tokenizer->suppressed_tokens[j] >= tokenizer->vocabulary_size) return -1;
    if (tokenizer->uses_merge_ranks) {
        if (tokenizer->merges == NULL) {
            nya_llm_error(error, error_capacity, "Gemma 4 requires tokenizer.ggml.merges");
            return -1;
        }
        tokenizer->special_tokens = (uint32_t *)malloc(tokenizer->vocabulary_size * sizeof(uint32_t));
        if (tokenizer->special_tokens == NULL) return -1;
        for (token = 0; token < tokenizer->vocabulary_size; ++token) {
            if ((tokenizer->types[token] == 3 || tokenizer->types[token] == 4) &&
                tokenizer->piece_lengths[token] > 0)
                tokenizer->special_tokens[tokenizer->special_count++] = token;
            if (strcmp(tokenizer->pieces[token], "<turn|>") == 0 ||
                strcmp(tokenizer->pieces[token], "<|tool_response>") == 0)
                tokenizer->stop_tokens[token] = 1;
        }
    }
    return 0;
}

/* Map one model file read-only without copying its tensor payload. */
static int nya_llm_mapping_open(
    const char *path,
    uint64_t expected_size,
    nya_llm_mapping *mapping,
    char *error,
    size_t error_capacity
)
{
    memset(mapping, 0, sizeof(*mapping));
    if (expected_size == 0 || expected_size > SIZE_MAX) {
        nya_llm_error(error, error_capacity, "the model is empty or too large for this process");
        return -1;
    }

#ifdef _WIN32
    {
        int wide_length;
        wchar_t *wide_path;
        HANDLE file;
        LARGE_INTEGER file_size;
        HANDLE file_mapping;
        void *view;

        wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
        if (wide_length <= 0) {
            nya_llm_error(error, error_capacity, "the GGUF path is not valid UTF-8");
            return -1;
        }
        wide_path = (wchar_t *)malloc((size_t)wide_length * sizeof(*wide_path));
        if (wide_path == NULL || MultiByteToWideChar(
                CP_UTF8,
                MB_ERR_INVALID_CHARS,
                path,
                -1,
                wide_path,
                wide_length
            ) != wide_length) {
            free(wide_path);
            nya_llm_error(error, error_capacity, "the GGUF path could not be converted");
            return -1;
        }

        file = CreateFileW(wide_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        free(wide_path);
        if (file == INVALID_HANDLE_VALUE || !GetFileSizeEx(file, &file_size) ||
            file_size.QuadPart < 0 || (uint64_t)file_size.QuadPart != expected_size) {
            if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
            nya_llm_error(error, error_capacity, "the GGUF file changed before it could be mapped");
            return -1;
        }

        file_mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
        if (file_mapping == NULL) {
            CloseHandle(file);
            nya_llm_error(error, error_capacity, "the GGUF file mapping could not be created");
            return -1;
        }
        view = MapViewOfFile(file_mapping, FILE_MAP_READ, 0, 0, 0);
        if (view == NULL) {
            CloseHandle(file_mapping);
            CloseHandle(file);
            nya_llm_error(error, error_capacity, "the GGUF file could not be memory mapped");
            return -1;
        }

        mapping->data = (unsigned char *)view;
        mapping->size = (size_t)expected_size;
        mapping->file_handle = (uintptr_t)file;
        mapping->mapping_handle = (uintptr_t)file_mapping;
    }
#else
    {
        int file;
        struct stat information;
        void *view;

        file = open(path, O_RDONLY);
        if (file < 0 || fstat(file, &information) != 0 || information.st_size < 0 ||
            (uint64_t)information.st_size != expected_size) {
            if (file >= 0) close(file);
            nya_llm_error(error, error_capacity, "the GGUF file changed before it could be mapped");
            return -1;
        }
        view = mmap(NULL, (size_t)expected_size, PROT_READ, MAP_PRIVATE, file, 0);
        close(file);
        if (view == MAP_FAILED) {
            nya_llm_error(error, error_capacity, "the GGUF file could not be memory mapped");
            return -1;
        }
        mapping->data = (unsigned char *)view;
        mapping->size = (size_t)expected_size;
    }
#endif
    return 0;
}

/* Release one platform mapping and its owned handles. */
static void nya_llm_mapping_close(nya_llm_mapping *mapping)
{
    if (mapping == NULL || mapping->data == NULL) return;
#ifdef _WIN32
    UnmapViewOfFile(mapping->data);
    if (mapping->mapping_handle != 0) CloseHandle((HANDLE)mapping->mapping_handle);
    if (mapping->file_handle != 0) CloseHandle((HANDLE)mapping->file_handle);
#else
    munmap(mapping->data, mapping->size);
#endif
    memset(mapping, 0, sizeof(*mapping));
}

/* Bind the Unified projector graph exactly. In GGUF the patch columns and its
   first LayerNorm are already permuted to channel-first order by conversion.
   Derive the effective patch size from those columns: the clip.patch_size
   metadata describes teacher patches (16), while these weights consume 48. */
static int nya_llm_bind_projector(nya_llm_context *c)
{
    nya_llm_projector *p = &c->projector;
    if (p->vision_type[0] == '\0' && p->audio_type[0] == '\0') return -1;
    if (p->vision_blocks != 0 || p->audio_blocks != 0) return -1;
    if (p->vision_type[0] != '\0') {
        if (strcmp(p->vision_type, "gemma4uv") != 0 || p->vision_width == 0 || p->output_width == 0 ||
            !isfinite(p->vision_epsilon) || p->vision_epsilon <= 0.0f) return -1;
        const nya_llm_tensor *patch = nya_llm_find_tensor(c, "v.patch_embd.weight");
        if (patch == NULL || patch->dimension_count != 2 || patch->dimensions[0] > 3U * 256U * 256U) return -1;
        uint32_t columns = (uint32_t)patch->dimensions[0];
        for (uint32_t side = 1; side <= 256; ++side) if (side * side * 3 == columns) p->patch_size = side;
        if (p->patch_size == 0) return -1;
        p->patch = nya_llm_bind_matrix(c, "v.patch_embd.weight", columns, p->vision_width);
        p->patch_bias = nya_llm_bind_vector(c, "v.patch_embd.bias", p->vision_width);
        p->vision = nya_llm_bind_matrix(c, "mm.input_projection.weight", p->vision_width, p->output_width);
        if (p->patch == NULL || p->patch_bias == NULL || p->vision == NULL) return -1;
        for (size_t i = 0; i < 3; ++i) {
            char name[64];
            uint32_t width = i == 0 ? columns : p->vision_width;
            snprintf(name, sizeof(name), "v.patch_norm.%zu.weight", i + 1);
            p->norm_weight[i] = nya_llm_bind_vector(c, name, width);
            snprintf(name, sizeof(name), "v.patch_norm.%zu.bias", i + 1);
            p->norm_bias[i] = nya_llm_bind_vector(c, name, width);
            if (p->norm_weight[i] == NULL || p->norm_bias[i] == NULL) return -1;
        }
        p->position = nya_llm_find_tensor(c, "v.position_embd.weight");
        if (p->position == NULL || p->position->data == NULL || p->position->dimension_count != 3 ||
            p->position->dimensions[0] != p->vision_width || p->position->dimensions[2] != 2 ||
            p->position->dimensions[1] > 65536 ||
            (p->position->type != NYA_LLM_TENSOR_F32 && p->position->type != NYA_LLM_TENSOR_F16 &&
             p->position->type != NYA_LLM_TENSOR_BF16)) return -1;
        p->position_count = (uint32_t)p->position->dimensions[1];
        c->tensors[p->position - c->tensors].bound = 1;
    }
    if (p->audio_type[0] != '\0') {
        if (strcmp(p->audio_type, "gemma4ua") != 0 || p->audio_width == 0 || p->audio_output_width == 0 ||
            !isfinite(p->audio_epsilon) || p->audio_epsilon <= 0.0f) return -1;
        if (p->output_width != 0 && p->output_width != p->audio_output_width) return -1;
        p->output_width = p->audio_output_width;
        p->audio = nya_llm_bind_matrix(c, "mm.a.input_projection.weight", p->audio_width, p->output_width);
        if (p->audio == NULL) return -1;
    }
    for (size_t i = 0; i < c->tensor_count; ++i) if (!c->tensors[i].bound) return -1;
    return 0;
}

/* Both native providers share the checked GGUF reader and mapping lifetime. */
static int nya_llm_load_impl(
    const char *path,
    uint64_t expected_size,
    nya_llm_context **output,
    char *error,
    size_t error_capacity,
    int projector
)
{
    nya_llm_context *context;
    nya_llm_metadata_state state;
    nya_llm_reader reader;
    unsigned char magic[4];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t index;
    uint32_t head_dimension;

    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (output == NULL || path == NULL) return -1;
    *output = NULL;
    context = (nya_llm_context *)calloc(1, sizeof(*context));
    if (context == NULL) {
        nya_llm_error(error, error_capacity, "LLM context allocation failed");
        return -1;
    }

    context->norm_epsilon = 0.00001f;
    context->rope_frequency_base = 10000.0f;
    context->sliding_rope_base = 10000.0f;
    context->projector.vision_epsilon = 0.000001f;
    context->projector.audio_epsilon = 0.000001f;
    context->tokenizer.unknown_token = UINT32_MAX;
    context->tokenizer.beginning_token = UINT32_MAX;
    context->tokenizer.end_token = UINT32_MAX;
    context->tokenizer.add_beginning_token = 1;
    context->tokenizer.add_end_token = 0;
    context->tokenizer.add_space_prefix = 1;
    memset(&state, 0, sizeof(state));
    state.alignment = 32;

    if (nya_llm_mapping_open(path, expected_size, &context->mapping, error, error_capacity) != 0) {
        nya_llm_free(context);
        return -1;
    }
    reader.data = context->mapping.data;
    reader.size = context->mapping.size;
    reader.position = 0;

    if (nya_llm_read(&reader, magic, sizeof(magic)) != 0 || memcmp(magic, "GGUF", 4) != 0 ||
        nya_llm_u32(&reader, &version) != 0 || (version != 2 && version != 3) ||
        nya_llm_u64(&reader, &tensor_count) != 0 || tensor_count == 0 || tensor_count > NYA_LLM_TENSOR_LIMIT ||
        nya_llm_u64(&reader, &metadata_count) != 0 || metadata_count > NYA_LLM_METADATA_LIMIT) {
        nya_llm_error(error, error_capacity, "the GGUF header is not a supported generation model");
        nya_llm_free(context);
        return -1;
    }

    for (index = 0; index < metadata_count; ++index) {
        if (nya_llm_metadata(&reader, context, &state) != 0) {
            nya_llm_error(error, error_capacity, "GGUF generation metadata is malformed or unsupported");
            nya_llm_free(context);
            return -1;
        }
    }

    context->metadata_end = reader.position;
    context->gguf_metadata_count = metadata_count;
    context->tensor_alignment = state.alignment;
    if (projector) {
        if (strcmp(state.architecture, "clip") != 0 || state.alignment < 8 || state.alignment % 8 != 0 ||
            nya_llm_tensor_directory(&reader, context, tensor_count, state.alignment) != 0 ||
            nya_llm_bind_projector(context) != 0) {
            nya_llm_error(error, error_capacity, "unsupported or malformed Gemma 4 Unified projector tensors/metadata");
            nya_llm_free(context); return -1;
        }
        context->compute = nya_compute_create();
        *output = context;
        return 0;
    }
    context->is_assistant = strcmp(state.architecture, "gemma4-assistant") == 0;
    context->is_gemma = context->is_assistant || strcmp(state.architecture, "gemma4") == 0;
    context->tokenizer.uses_merge_ranks = strcmp(state.tokenizer_model, "gemma4") == 0;
    if ((!context->is_gemma && strcmp(state.architecture, "llama") != 0) ||
        (context->is_gemma ? !context->tokenizer.uses_merge_ranks : strcmp(state.tokenizer_model, "llama") != 0) ||
        (state.rope_scaling_type[0] != '\0' && strcmp(state.rope_scaling_type, "none") != 0) ||
        state.expert_count != 0 || state.alignment < 8 || state.alignment % 8 != 0) {
        nya_llm_error(error, error_capacity, "generation requires a supported LLaMA or Gemma 4 architecture/tokenizer pair");
        nya_llm_free(context);
        return -1;
    }
    if (!context->is_gemma) {
    if (context->embedding_length == 0 || context->feed_forward_length == 0 ||
        context->block_count == 0 || context->head_count == 0 || context->context_length == 0 ||
        context->embedding_length % context->head_count != 0) {
        nya_llm_error(error, error_capacity, "required LLaMA dimensions are missing or inconsistent");
        nya_llm_free(context);
        return -1;
    }
    if (context->key_value_head_count == 0) context->key_value_head_count = context->head_count;
    head_dimension = context->embedding_length / context->head_count;
    if (context->head_count % context->key_value_head_count != 0) {
        nya_llm_error(error, error_capacity, "the key/value head count does not divide the attention head count");
        nya_llm_free(context);
        return -1;
    }
    if (context->rope_dimension_count == 0) context->rope_dimension_count = head_dimension;
    if (context->rope_dimension_count != head_dimension || context->rope_dimension_count % 2 != 0 ||
        context->norm_epsilon <= 0.0f || context->rope_frequency_base <= 0.0f) {
        nya_llm_error(error, error_capacity, "the model uses unsupported head or RoPE dimensions");
        nya_llm_free(context);
        return -1;
    }

    }

    if (nya_llm_finalize_tokenizer(context, &state, error, error_capacity) != 0 ||
        nya_llm_tensor_directory(&reader, context, tensor_count, state.alignment) != 0) {
        if (error != NULL && error_capacity > 0 && error[0] == '\0') nya_llm_error(error, error_capacity, "GGUF tokenizer or tensor directory is invalid");
        nya_llm_free(context);
        return -1;
    }
    if ((context->is_gemma ? nya_llm_bind_gemma(context, error, error_capacity) :
         nya_llm_bind_weights(context, error, error_capacity)) != 0) {
        nya_llm_free(context);
        return -1;
    }

    /* Accelerators are optional: an unavailable driver/device must not prevent
       a valid model from loading or the CPU execution path from being used. */
    context->compute = nya_compute_create();
    *output = context;
    return 0;
}

int nya_llm_load(const char *path, uint64_t expected_size, nya_llm_context **output,
    char *error, size_t error_capacity)
{
    return nya_llm_load_impl(path, expected_size, output, error, error_capacity, 0);
}

int nya_llm_load_projector(const char *path, uint64_t expected_size, nya_llm_context **output,
    char *error, size_t error_capacity)
{
    return nya_llm_load_impl(path, expected_size, output, error, error_capacity, 1);
}

/* Release tokenizer strings, tensor bindings, and the mapped model. */
void nya_llm_free(nya_llm_context *context)
{
    uint32_t token;

    if (context == NULL) return;
    /* Release cached device weights before unmapping the source tensor bytes. */
    nya_compute_free(context->compute);
    if (context->tokenizer.pieces != NULL) {
        for (token = 0; token < context->tokenizer.vocabulary_size; ++token) {
            free(context->tokenizer.pieces[token]);
        }
    }
    free(context->tokenizer.pieces);
    free(context->tokenizer.piece_lengths);
    free(context->tokenizer.scores);
    free(context->tokenizer.types);
    free(context->tokenizer.sorted);
    for (size_t i = 0; i < context->tokenizer.merge_count; ++i) free(context->tokenizer.merges[i].text);
    free(context->tokenizer.merges);
    free(context->tokenizer.stop_tokens);
    free(context->tokenizer.suppressed_tokens);
    free(context->tokenizer.special_tokens);
    free(context->layers);
    free(context->tensors);
    nya_llm_mapping_close(&context->mapping);
    free(context);
}

/* Append one token while preserving the allocation bound derived from prompt bytes. */
static int nya_llm_push_token(uint32_t *tokens, size_t capacity, size_t *count, uint32_t token)
{
    if (*count >= capacity) return -1;
    tokens[*count] = token;
    *count += 1;
    return 0;
}

/* Encode one byte through a dedicated byte token or the model's unknown token. */
static int nya_llm_push_byte(
    const nya_llm_tokenizer *tokenizer,
    uint32_t *tokens,
    size_t capacity,
    size_t *count,
    unsigned char byte
)
{
    int32_t token;

    token = tokenizer->byte_tokens[byte];
    return nya_llm_push_token(
        tokens,
        capacity,
        count,
        token < 0 ? tokenizer->unknown_token : (uint32_t)token
    );
}

/* Return the bounded byte length of one valid UTF-8 code point. */
static size_t nya_llm_utf8_length(const unsigned char *text, size_t remaining)
{
    size_t length;
    size_t index;

    if (remaining == 0) return 0;
    if (text[0] < 0x80) return 1;
    if (text[0] >= 0xC2 && text[0] <= 0xDF) length = 2;
    else if (text[0] >= 0xE0 && text[0] <= 0xEF) length = 3;
    else if (text[0] >= 0xF0 && text[0] <= 0xF4) length = 4;
    else return 0;
    if (length > remaining) return 0;
    for (index = 1; index < length; ++index) {
        if ((text[index] & 0xC0U) != 0x80U) return 0;
    }
    if ((length == 3 && text[0] == 0xE0 && text[1] < 0xA0) ||
        (length == 3 && text[0] == 0xED && text[1] >= 0xA0) ||
        (length == 4 && text[0] == 0xF0 && text[1] < 0x90) ||
        (length == 4 && text[0] == 0xF4 && text[1] >= 0x90)) return 0;
    return length;
}

int nya_llm_text_is_utf8(const char *text, size_t length)
{
    size_t position = 0;
    if (text == NULL) return 0;
    while (position < length) {
        size_t width = nya_llm_utf8_length((const unsigned char *)text + position, length - position);
        if (width == 0) return 0;
        position += width;
    }
    return 1;
}

/* A symbol is a contiguous span in normalized prompt text. Neighbor indices
   form a linked list; merging never copies the text or shifts the token array. */
typedef struct nya_llm_symbol {
    size_t start;
    size_t length;
    size_t previous;
    size_t next;
    int32_t fixed_token;
} nya_llm_symbol;

typedef struct nya_llm_merge {
    size_t left;
    size_t right;
    size_t length;
    float score;
} nya_llm_merge;

/* Binary lookup over a byte span avoids allocating/terminating a temporary
   string for every candidate pair. Only normal vocabulary pieces participate;
   BOS/EOS and byte-token spellings are never parsed as literal prompt tokens. */
static int32_t nya_llm_span_lookup(const nya_llm_tokenizer *tokenizer, const char *text, size_t length)
{
    size_t low = 0, high = tokenizer->vocabulary_size;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        uint32_t token = tokenizer->sorted[middle].token;
        size_t piece_length = tokenizer->piece_lengths[token];
        size_t common = length < piece_length ? length : piece_length;
        int order = memcmp(text, tokenizer->sorted[middle].piece, common);
        if (order == 0) order = length < piece_length ? -1 : length > piece_length ? 1 : 0;
        if (order < 0) high = middle;
        else if (order > 0) low = middle + 1;
        else return tokenizer->types[token] == 1 ? (int32_t)token : -1;
    }
    return -1;
}

/* Equal-score SentencePiece merges choose the leftmost pair. Original symbol
   indices retain that text order even after intermediate symbols disappear. */
static int nya_llm_merge_before(const nya_llm_merge *left, const nya_llm_merge *right)
{
    return left->score > right->score || (left->score == right->score && left->left < right->left);
}

static int32_t nya_llm_pair_rank(const nya_llm_tokenizer *tok, const char *text, size_t length, size_t left_length)
{
    size_t low = 0, high = tok->merge_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        const nya_llm_bpe_rule *rule = &tok->merges[middle];
        size_t rule_length = strlen(rule->text), common = length < rule_length ? length : rule_length;
        int order = memcmp(text, rule->text, common);
        if (order == 0) order = length < rule_length ? -1 : length > rule_length;
        if (order == 0) order = left_length < rule->left_length ? -1 : left_length > rule->left_length;
        if (order < 0) high = middle;
        else if (order > 0) low = middle + 1;
        else return (int32_t)rule->rank;
    }
    return -1;
}

/* Insert one currently adjacent candidate into a max-heap. At most N-1 initial
   candidates and two candidates per successful merge can ever be enqueued. */
static int nya_llm_queue_merge(
    const nya_llm_tokenizer *tokenizer, const char *text, const nya_llm_symbol *symbols,
    size_t left, nya_llm_merge *heap, size_t capacity, size_t *count
)
{
    size_t right, slot;
    int32_t token;
    nya_llm_merge candidate;
    if (left == SIZE_MAX || symbols[left].length == 0) return 0;
    right = symbols[left].next;
    if (right == SIZE_MAX || symbols[right].length == 0) return 0;
    if (symbols[left].fixed_token >= 0 || symbols[right].fixed_token >= 0) return 0;
    candidate.length = symbols[left].length + symbols[right].length;
    if (candidate.length > tokenizer->maximum_piece_length) return 0;
    token = nya_llm_span_lookup(tokenizer, text + symbols[left].start, candidate.length);
    if (token < 0) return 0;
    if (*count == capacity) return -1;
    candidate.left = left;
    candidate.right = right;
    candidate.score = tokenizer->scores[token];
    if (tokenizer->uses_merge_ranks) {
        int32_t rank;
        /* Newline runs are pre-tokenized separately by Gemma's published
           tokenizer. No regular pair may merge across that boundary. */
        if (memchr(text + symbols[left].start, '\n', candidate.length) != NULL) return 0;
        rank = nya_llm_pair_rank(tokenizer, text + symbols[left].start, candidate.length, symbols[left].length);
        if (rank < 0) return 0;
        candidate.score = -(float)rank;
    }
    slot = (*count)++;
    while (slot > 0) {
        size_t parent = (slot - 1) / 2;
        if (!nya_llm_merge_before(&candidate, &heap[parent])) break;
        heap[slot] = heap[parent];
        slot = parent;
    }
    heap[slot] = candidate;
    return 0;
}

static nya_llm_merge nya_llm_pop_merge(nya_llm_merge *heap, size_t *count)
{
    nya_llm_merge result = heap[0];
    nya_llm_merge tail = heap[--(*count)];
    size_t parent = 0;
    while (parent < *count / 2) {
        size_t child = parent * 2 + 1;
        if (child + 1 < *count && nya_llm_merge_before(&heap[child + 1], &heap[child])) child += 1;
        if (!nya_llm_merge_before(&heap[child], &tail)) break;
        heap[parent] = heap[child];
        parent = child;
    }
    if (*count != 0) heap[parent] = tail;
    return result;
}

/* Encode UTF-8 prompt text, then apply score-ordered SentencePiece BPE. Byte
   fallback must happen AFTER merges: an absent individual Unicode character can
   still be part of a valid multi-character vocabulary piece. The heap avoids
   rescanning every prompt pair and moving token arrays after every merge. */
int nya_llm_tokenize(
    const nya_llm_context *context,
    const char *text,
    uint32_t **output_tokens,
    size_t *output_count,
    char *error,
    size_t error_capacity
)
{
    const nya_llm_tokenizer *tokenizer;
    size_t text_length, normalized_length, capacity, allocation_size;
    size_t count = 0, position, symbol_count = 0, heap_count = 0, heap_capacity;
    uint32_t *tokens = NULL;
    char *normalized = NULL;
    nya_llm_symbol *symbols = NULL;
    nya_llm_merge *heap = NULL;

    if (context == NULL || text == NULL || output_tokens == NULL || output_count == NULL) return -1;
    *output_tokens = NULL;
    *output_count = 0;
    tokenizer = &context->tokenizer;
    text_length = strlen(text);
    if (!nya_llm_text_is_utf8(text, text_length)) {
        nya_llm_error(error, error_capacity, "the generation prompt is not valid UTF-8");
        return -1;
    }
    /* Each ASCII space can expand to the three-byte SentencePiece marker. One
       extra prefix symbol plus BOS/EOS bounds every scratch allocation below. */
    if (text_length > (SIZE_MAX - 6) / 3) goto token_failure;
    capacity = text_length * 3 + 6;
    normalized = (char *)malloc(capacity);
    if (nya_llm_size_product(capacity, sizeof(*tokens), &allocation_size) != 0) goto token_failure;
    tokens = (uint32_t *)malloc(allocation_size);
    if (nya_llm_size_product(text_length + 1, sizeof(*symbols), &allocation_size) != 0) goto token_failure;
    symbols = (nya_llm_symbol *)malloc(allocation_size);
    if (normalized == NULL || tokens == NULL || symbols == NULL) goto token_failure;
    normalized_length = 0;
    for (position = 0; position < text_length + (text_length > 0 && tokenizer->add_space_prefix ? 1U : 0U); ++position) {
        char byte = text_length > 0 && tokenizer->add_space_prefix ?
            (position == 0 ? ' ' : text[position - 1]) : text[position];
        if (byte == ' ' && tokenizer->uses_sentencepiece_space) {
            memcpy(normalized + normalized_length, "\xE2\x96\x81", 3);
            normalized_length += 3;
        } else normalized[normalized_length++] = byte;
    }
    normalized[normalized_length] = '\0';
    for (position = 0; position < normalized_length;) {
        size_t length = nya_llm_utf8_length((const unsigned char *)normalized + position, normalized_length - position);
        int32_t fixed_token = -1;
        if (length == 0) length = 1;
        if (tokenizer->uses_merge_ranks) {
            /* Added control/user tokens are atomic. Longest-match wins before
               regular BPE, making chat markers usable without merging their
               spelling into adjacent text. A newline run is likewise atomic
               when its complete spelling has a vocabulary entry. */
            for (size_t s = 0; s < tokenizer->special_count; ++s) {
                uint32_t id = tokenizer->special_tokens[s];
                size_t n = tokenizer->piece_lengths[id];
                if (n >= length && n <= normalized_length - position &&
                    memcmp(normalized + position, tokenizer->pieces[id], n) == 0) {
                    length = n;
                    fixed_token = (int32_t)id;
                }
            }
            if (normalized[position] == '\n' && fixed_token < 0) {
                size_t n = 1;
                while (n < normalized_length - position && normalized[position + n] == '\n') ++n;
                fixed_token = nya_llm_span_lookup(tokenizer, normalized + position, n);
                if (fixed_token >= 0) length = n;
            }
        }
        symbols[symbol_count].start = position;
        symbols[symbol_count].length = length;
        symbols[symbol_count].fixed_token = fixed_token;
        symbols[symbol_count].previous = symbol_count == 0 ? SIZE_MAX : symbol_count - 1;
        symbols[symbol_count].next = symbol_count + 1;
        symbol_count += 1;
        position += length;
    }
    if (symbol_count > 0) symbols[symbol_count - 1].next = SIZE_MAX;
    if (nya_llm_size_product(symbol_count + 1, 3, &heap_capacity) != 0 ||
        nya_llm_size_product(heap_capacity, sizeof(*heap), &allocation_size) != 0) goto token_failure;
    heap = (nya_llm_merge *)malloc(allocation_size);
    if (heap == NULL) goto token_failure;
    for (position = 0; position < symbol_count; ++position) {
        if (nya_llm_queue_merge(tokenizer, normalized, symbols, position, heap, heap_capacity, &heap_count) != 0) goto token_failure;
    }
    while (heap_count != 0) {
        nya_llm_merge candidate = nya_llm_pop_merge(heap, &heap_count);
        nya_llm_symbol *left = &symbols[candidate.left];
        nya_llm_symbol *right = &symbols[candidate.right];
        /* Neighboring merges leave stale heap entries. Validate both adjacency
           and span length before using a candidate; stale entries are harmless. */
        if (left->length == 0 || right->length == 0 || left->next != candidate.right ||
            left->length + right->length != candidate.length) continue;
        left->length = candidate.length;
        left->next = right->next;
        right->length = 0;
        if (left->next != SIZE_MAX) symbols[left->next].previous = candidate.left;
        if (nya_llm_queue_merge(tokenizer, normalized, symbols, left->previous, heap, heap_capacity, &heap_count) != 0 ||
            nya_llm_queue_merge(tokenizer, normalized, symbols, candidate.left, heap, heap_capacity, &heap_count) != 0) goto token_failure;
    }
    if (tokenizer->add_beginning_token &&
        nya_llm_push_token(tokens, capacity, &count, tokenizer->beginning_token) != 0) goto token_failure;
    for (position = symbol_count == 0 ? SIZE_MAX : 0; position != SIZE_MAX; position = symbols[position].next) {
        const nya_llm_symbol *symbol = &symbols[position];
        int32_t token = symbol->fixed_token >= 0 ? symbol->fixed_token :
            nya_llm_span_lookup(tokenizer, normalized + symbol->start, symbol->length);
        if (token >= 0) {
            if (nya_llm_push_token(tokens, capacity, &count, (uint32_t)token) != 0) goto token_failure;
        } else {
            size_t byte;
            for (byte = 0; byte < symbol->length; ++byte) {
                if (nya_llm_push_byte(tokenizer, tokens, capacity, &count,
                    (unsigned char)normalized[symbol->start + byte]) != 0) goto token_failure;
            }
        }
    }
    /* Empty prompts still need one position from which generation can sample. */
    if (count == 0 && nya_llm_push_token(tokens, capacity, &count, tokenizer->beginning_token) != 0) goto token_failure;
    /* EOS belongs after all text merges. Appending it earlier would allow the
       special token's spelling to participate in a regular vocabulary merge. */
    if (tokenizer->add_end_token &&
        nya_llm_push_token(tokens, capacity, &count, tokenizer->end_token) != 0) goto token_failure;
    free(normalized);
    free(symbols);
    free(heap);
    *output_tokens = tokens;
    *output_count = count;
    return 0;

token_failure:
    free(normalized);
    free(symbols);
    free(heap);
    free(tokens);
    nya_llm_error(error, error_capacity, "prompt tokenization failed within its bounds");
    return -1;
}

/* Append exact bytes while reserving one terminator byte. */
static int nya_llm_append_bytes(
    char *output,
    size_t capacity,
    size_t *length,
    const char *bytes,
    size_t byte_count
)
{
    if (*length >= capacity || byte_count > capacity - *length - 1) return -1;
    memcpy(output + *length, bytes, byte_count);
    *length += byte_count;
    output[*length] = '\0';
    return 0;
}

/* Decode one token, replacing SentencePiece's visible space marker. */
int nya_llm_decode_token(
    const nya_llm_context *context,
    uint32_t previous_token,
    uint32_t token,
    char *output,
    size_t output_capacity,
    size_t *output_length
)
{
    const nya_llm_tokenizer *tokenizer;
    const char *piece;
    size_t piece_length;
    unsigned int byte;
    size_t position;

    if (context == NULL || output == NULL || output_length == NULL || output_capacity == 0 ||
        token >= context->tokenizer.vocabulary_size) return -1;
    tokenizer = &context->tokenizer;
    if (token == tokenizer->beginning_token || token == tokenizer->end_token || tokenizer->types[token] == 3) return 0;
    piece = tokenizer->pieces[token];
    piece_length = tokenizer->piece_lengths[token];

    if (nya_llm_byte_piece(piece, piece_length, &byte) == 0) {
        if (byte == 0) return nya_llm_append_bytes(output, output_capacity, output_length, "\xEF\xBF\xBD", 3);
        {
            char value;

            value = (char)byte;
            return nya_llm_append_bytes(output, output_capacity, output_length, &value, 1);
        }
    }

    position = 0;
    while (position < piece_length) {
        if (tokenizer->uses_sentencepiece_space && position + 3 <= piece_length &&
            memcmp(piece + position, "\xE2\x96\x81", 3) == 0) {
            if (!(tokenizer->add_space_prefix && previous_token == tokenizer->beginning_token && *output_length == 0 && position == 0) &&
                nya_llm_append_bytes(output, output_capacity, output_length, " ", 1) != 0) return -1;
            position += 3;
        } else {
            if (nya_llm_append_bytes(output, output_capacity, output_length, piece + position, 1) != 0) return -1;
            position += 1;
        }
    }
    return 0;
}
