#include "format_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Official GGUF keys cannot exceed 65535 bytes. */
#define NYA_GGUF_KEY_LIMIT 65535

/* These limits prevent valid-looking files from consuming unreasonable CPU. */
#define NYA_GGUF_METADATA_LIMIT 1000000ULL
#define NYA_GGUF_TENSOR_LIMIT 1000000ULL
#define NYA_GGUF_ARRAY_LIMIT 100000000ULL
#define NYA_GGUF_DEPTH_LIMIT 8U

/* GGUF metadata value identifiers from the published format. */
enum nya_gguf_value_type {
    NYA_GGUF_UINT8 = 0,
    NYA_GGUF_INT8 = 1,
    NYA_GGUF_UINT16 = 2,
    NYA_GGUF_INT16 = 3,
    NYA_GGUF_UINT32 = 4,
    NYA_GGUF_INT32 = 5,
    NYA_GGUF_FLOAT32 = 6,
    NYA_GGUF_BOOL = 7,
    NYA_GGUF_STRING = 8,
    NYA_GGUF_ARRAY = 9,
    NYA_GGUF_UINT64 = 10,
    NYA_GGUF_INT64 = 11,
    NYA_GGUF_FLOAT64 = 12
};

/* Skip or capture one length-prefixed GGUF string. */
static int nya_gguf_string(nya_reader *reader, char *output, size_t capacity)
{
    uint64_t length;

    if (nya_reader_u64_le(reader, &length) != 0) {
        return -1;
    }

    if (output == NULL || length >= capacity) {
        if (output != NULL && capacity != 0) output[0] = '\0';
        return nya_reader_skip(reader, length);
    }

    if (nya_reader_read(reader, output, (size_t)length) != 0) {
        return -1;
    }
    /* Captured metadata is exposed as a C string, so embedded terminators would
     * make the public architecture disagree with the actual GGUF value. */
    if (memchr(output, 0, (size_t)length) != NULL) return -1;
    output[length] = '\0';
    return 0;
}

/* Return the fixed byte width for a primitive metadata type. */
static uint64_t nya_gguf_primitive_size(uint32_t type)
{
    switch (type) {
        case NYA_GGUF_UINT8:
        case NYA_GGUF_INT8:
        case NYA_GGUF_BOOL:
            return 1;
        case NYA_GGUF_UINT16:
        case NYA_GGUF_INT16:
            return 2;
        case NYA_GGUF_UINT32:
        case NYA_GGUF_INT32:
        case NYA_GGUF_FLOAT32:
            return 4;
        case NYA_GGUF_UINT64:
        case NYA_GGUF_INT64:
        case NYA_GGUF_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

/* Skip one metadata value while bounding recursive nested arrays. */
static int nya_gguf_skip_value(nya_reader *reader, uint32_t type, unsigned int depth)
{
    uint64_t primitive_size;

    primitive_size = nya_gguf_primitive_size(type);
    if (primitive_size != 0) {
        return nya_reader_skip(reader, primitive_size);
    }

    if (type == NYA_GGUF_STRING) {
        return nya_gguf_string(reader, NULL, 0);
    }

    if (type == NYA_GGUF_ARRAY) {
        uint32_t element_type;
        uint64_t element_count;
        uint64_t element_size;
        uint64_t index;

        if (depth >= NYA_GGUF_DEPTH_LIMIT ||
            nya_reader_u32_le(reader, &element_type) != 0 ||
            nya_reader_u64_le(reader, &element_count) != 0 ||
            element_count > NYA_GGUF_ARRAY_LIMIT) {
            return -1;
        }

        element_size = nya_gguf_primitive_size(element_type);
        if (element_size != 0) {
            if (element_count > UINT64_MAX / element_size) {
                return -1;
            }
            return nya_reader_skip(reader, element_count * element_size);
        }

        if (element_type != NYA_GGUF_STRING && element_type != NYA_GGUF_ARRAY) {
            return -1;
        }

        for (index = 0; index < element_count; ++index) {
            if (nya_gguf_skip_value(reader, element_type, depth + 1) != 0) {
                return -1;
            }
        }
        return 0;
    }

    return -1;
}

/* Read one metadata item and capture the two fields needed by the common API. */
static int nya_gguf_metadata(
    nya_reader *reader,
    nya_format_info *information,
    uint32_t *alignment,
    unsigned int *seen_fields
)
{
    uint64_t key_length;
    char key[128];
    uint32_t value_type;

    if (nya_reader_u64_le(reader, &key_length) != 0 || key_length == 0 || key_length > NYA_GGUF_KEY_LIMIT) {
        return -1;
    }

    if (key_length < sizeof(key)) {
        if (nya_reader_read(reader, key, (size_t)key_length) != 0) {
            return -1;
        }
        if (memchr(key, 0, (size_t)key_length) != NULL) return -1;
        key[key_length] = '\0';
    } else {
        key[0] = '\0';
        if (nya_reader_skip(reader, key_length) != 0) {
            return -1;
        }
    }

    if (nya_reader_u32_le(reader, &value_type) != 0) {
        return -1;
    }

    if (strcmp(key, "general.architecture") == 0) {
        if (value_type != NYA_GGUF_STRING || (*seen_fields & 1U) != 0) return -1;
        *seen_fields |= 1U;
        return nya_gguf_string(reader, information->architecture, sizeof(information->architecture));
    }

    if (strcmp(key, "general.alignment") == 0) {
        if (value_type != NYA_GGUF_UINT32 || (*seen_fields & 2U) != 0) return -1;
        *seen_fields |= 2U;
        return nya_reader_u32_le(reader, alignment);
    }

    return nya_gguf_skip_value(reader, value_type, 0);
}

/* File block sizes, not host C struct sizes: each entry is {elements, bytes}.
 * These layout facts follow ggml's ggml_type enum and ggml-common.h definitions.
 * Zero entries are retired encodings; future encodings are unsupported until
 * their layout is known, because accepting them would skip payload validation. */
static const uint16_t nya_gguf_blocks[][2] = {
    {1, 4}, {1, 2}, {32, 18}, {32, 20}, {0, 0}, {0, 0}, /* F32/F16/Q4 */
    {32, 22}, {32, 24}, {32, 34}, {32, 36},              /* Q5/Q8 */
    {256, 84}, {256, 110}, {256, 144}, {256, 176}, {256, 210}, {256, 292},
    {256, 66}, {256, 74}, {256, 98}, {256, 50}, {32, 18}, /* IQ variants */
    {256, 110}, {256, 82}, {256, 136},
    {1, 1}, {1, 2}, {1, 4}, {1, 8}, {1, 8}, {256, 56}, {1, 2},
    {0, 0}, {0, 0}, {0, 0}, {256, 54}, {256, 66},       /* TQ1/TQ2 */
    {0, 0}, {0, 0}, {0, 0}, {32, 17}, {64, 36}, {128, 18}, {64, 18}
};

/* Retain only the small directory, never the tensor payload. Sorting twice
 * makes duplicate-name and overlapping-range validation O(n log n). */
typedef struct nya_gguf_tensor_range {
    char name[65];
    uint64_t begin;
    uint64_t end;
} nya_gguf_tensor_range;

static int nya_gguf_name_compare(const void *left, const void *right)
{
    return strcmp(((const nya_gguf_tensor_range *)left)->name,
                  ((const nya_gguf_tensor_range *)right)->name);
}

static int nya_gguf_range_compare(const void *left_pointer, const void *right_pointer)
{
    const nya_gguf_tensor_range *left = (const nya_gguf_tensor_range *)left_pointer;
    const nya_gguf_tensor_range *right = (const nya_gguf_tensor_range *)right_pointer;
    return left->begin < right->begin ? -1 : (left->begin > right->begin ? 1 : 0);
}

/* Validate the GGUF header, metadata, tensor directory, and data boundary. */
nya_format_result nya_gguf_inspect(nya_reader *reader, nya_format_info *information)
{
    unsigned char magic[4];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t index;
    uint32_t alignment;
    uint64_t data_offset;
    uint64_t padding;
    unsigned int seen_fields = 0;
    nya_gguf_tensor_range *ranges = NULL;
    nya_format_result result = NYA_FORMAT_INVALID;

    if (nya_reader_read(reader, magic, sizeof(magic)) != 0 || memcmp(magic, "GGUF", 4) != 0 ||
        nya_reader_u32_le(reader, &version) != 0 || (version != 2 && version != 3) ||
        nya_reader_u64_le(reader, &tensor_count) != 0 ||
        nya_reader_u64_le(reader, &metadata_count) != 0) {
        return NYA_FORMAT_INVALID;
    }
    if (tensor_count > NYA_GGUF_TENSOR_LIMIT || metadata_count > NYA_GGUF_METADATA_LIMIT ||
        tensor_count > SIZE_MAX / sizeof(*ranges)) return NYA_FORMAT_RESOURCE_LIMIT;

    information->format = NYA_FORMAT_GGUF;
    information->format_version = version;
    information->tensor_count = tensor_count;
    information->metadata_count = metadata_count;
    alignment = 32;

    for (index = 0; index < metadata_count; ++index) {
        if (nya_gguf_metadata(reader, information, &alignment, &seen_fields) != 0) {
            return NYA_FORMAT_INVALID;
        }
    }

    if (alignment < 8 || alignment % 8 != 0) {
        return NYA_FORMAT_INVALID;
    }

    /* A directory entry needs at least 33 bytes. Check that inexpensive bound
     * before allocating from an attacker-controlled count in a tiny file. */
    if (tensor_count > (reader->size - reader->position) / 33) return NYA_FORMAT_INVALID;
    if (tensor_count != 0) {
        ranges = (nya_gguf_tensor_range *)calloc((size_t)tensor_count, sizeof(*ranges));
        if (ranges == NULL) return NYA_FORMAT_RESOURCE_LIMIT;
    }
    for (index = 0; index < tensor_count; ++index) {
        uint32_t dimensions;
        uint32_t type;
        uint64_t tensor_offset;
        uint32_t dimension;
        uint64_t name_length;
        uint64_t element_count = 1;
        uint64_t first_extent = 0;
        uint64_t block_count;
        uint64_t byte_count;
        uint64_t block_size;
        uint64_t block_bytes;

        if (nya_reader_u64_le(reader, &name_length) != 0 || name_length == 0 || name_length > 64 ||
            nya_reader_read(reader, ranges[index].name, (size_t)name_length) != 0 ||
            memchr(ranges[index].name, 0, (size_t)name_length) != NULL ||
            nya_reader_u32_le(reader, &dimensions) != 0 || dimensions == 0 || dimensions > 4) goto cleanup;

        for (dimension = 0; dimension < dimensions; ++dimension) {
            uint64_t extent;

            if (nya_reader_u64_le(reader, &extent) != 0 || extent == 0 || extent > INT64_MAX ||
                element_count > UINT64_MAX / extent) goto cleanup;
            if (dimension == 0) first_extent = extent;
            element_count *= extent;
        }

        if (nya_reader_u32_le(reader, &type) != 0 ||
            nya_reader_u64_le(reader, &tensor_offset) != 0 || tensor_offset % alignment != 0) goto cleanup;
        if (type >= sizeof(nya_gguf_blocks) / sizeof(nya_gguf_blocks[0]) || nya_gguf_blocks[type][0] == 0) {
            result = NYA_FORMAT_UNSUPPORTED;
            goto cleanup;
        }
        block_size = nya_gguf_blocks[type][0];
        block_bytes = nya_gguf_blocks[type][1];
        /* Quantization blocks cannot straddle rows. Dividing only the overall
         * element count would accept malformed shapes such as Q4_0 [16, 2]. */
        if (first_extent % block_size != 0) goto cleanup;
        block_count = element_count / block_size;
        if (block_count > UINT64_MAX / block_bytes) goto cleanup;
        byte_count = block_count * block_bytes;
        if (tensor_offset > UINT64_MAX - byte_count) goto cleanup;
        ranges[index].begin = tensor_offset;
        ranges[index].end = tensor_offset + byte_count;
    }

    /* GGUF allows any multiple of eight, so bit masks are insufficient for
     * alignments such as 24. Check before adding the computed padding. */
    padding = (alignment - reader->position % alignment) % alignment;
    if (reader->position > UINT64_MAX - padding) goto cleanup;
    data_offset = reader->position + padding;
    if (data_offset > reader->size) goto cleanup;
    if (tensor_count > 1) {
        qsort(ranges, (size_t)tensor_count, sizeof(*ranges), nya_gguf_name_compare);
        for (index = 1; index < tensor_count; ++index) {
            if (strcmp(ranges[index - 1].name, ranges[index].name) == 0) goto cleanup;
        }
        qsort(ranges, (size_t)tensor_count, sizeof(*ranges), nya_gguf_range_compare);
    }
    for (index = 0; index < tensor_count; ++index) {
        if (ranges[index].end > reader->size - data_offset ||
            (index != 0 && ranges[index].begin < ranges[index - 1].end)) goto cleanup;
    }

    information->data_offset = data_offset;
    result = NYA_FORMAT_OK;
cleanup:
    free(ranges);
    return result;
}
