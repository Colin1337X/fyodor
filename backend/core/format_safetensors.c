#include "format_internal.h"

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The reference SafeTensors implementation enforces a 100,000,000-byte header. */
#define NYA_SAFETENSORS_HEADER_LIMIT 100000000ULL

/* A bounded tensor count prevents metadata-only CPU and allocation abuse. */
#define NYA_SAFETENSORS_TENSOR_LIMIT 1000000ULL

/* Nested arbitrary metadata still receives a conventional parser depth bound. */
#define NYA_JSON_DEPTH_LIMIT 64U

/* The JSON parser owns no memory and advances through one validated header buffer. */
typedef struct nya_json_parser {
    unsigned char *data;
    size_t length;
    size_t position;
    unsigned int depth;
} nya_json_parser;

/* Decoded strings borrow space inside the header, including names containing NUL.
 * Explicit lengths preserve JSON's string identity without C-string truncation. */
typedef struct nya_json_string {
    const unsigned char *data;
    size_t length;
} nya_json_string;

/* One interval describes one tensor slice within the SafeTensors data buffer. */
typedef struct nya_tensor_interval {
    uint64_t begin;
    uint64_t end;
    nya_json_string name;
} nya_tensor_interval;

/* A growable bounded list is used only while validating non-overlapping offsets. */
typedef struct nya_interval_list {
    nya_tensor_interval *items;
    size_t count;
    size_t capacity;
} nya_interval_list;

/* Reject malformed UTF-8 before interpreting the header as JSON. */
static int nya_utf8_valid(const unsigned char *data, size_t length)
{
    size_t index;

    index = 0;
    while (index < length) {
        unsigned char first;
        unsigned int continuation_count;
        uint32_t codepoint;
        uint32_t minimum;
        unsigned int offset;

        first = data[index];
        if (first < 0x80) {
            index += 1;
            continue;
        }

        if ((first & 0xe0U) == 0xc0U) {
            continuation_count = 1;
            codepoint = first & 0x1fU;
            minimum = 0x80;
        } else if ((first & 0xf0U) == 0xe0U) {
            continuation_count = 2;
            codepoint = first & 0x0fU;
            minimum = 0x800;
        } else if ((first & 0xf8U) == 0xf0U) {
            continuation_count = 3;
            codepoint = first & 0x07U;
            minimum = 0x10000;
        } else {
            return 0;
        }

        if (index + continuation_count >= length) {
            return 0;
        }

        for (offset = 1; offset <= continuation_count; ++offset) {
            unsigned char next;

            next = data[index + offset];
            if ((next & 0xc0U) != 0x80U) {
                return 0;
            }
            codepoint = (codepoint << 6) | (next & 0x3fU);
        }

        if (codepoint < minimum || codepoint > 0x10ffffU ||
            (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
            return 0;
        }

        index += continuation_count + 1;
    }

    return 1;
}

/* Skip insignificant JSON whitespace. */
static void nya_json_whitespace(nya_json_parser *parser)
{
    while (parser->position < parser->length &&
           (parser->data[parser->position] == ' ' || parser->data[parser->position] == '\t' ||
            parser->data[parser->position] == '\r' || parser->data[parser->position] == '\n')) {
        parser->position += 1;
    }
}

/* Consume one exact punctuation byte after optional whitespace. */
static int nya_json_take(nya_json_parser *parser, unsigned char expected)
{
    nya_json_whitespace(parser);
    if (parser->position >= parser->length || parser->data[parser->position] != expected) {
        return -1;
    }

    parser->position += 1;
    return 0;
}

/* Convert one hexadecimal JSON escape digit. */
static int nya_json_hex(unsigned char character)
{
    if (character >= '0' && character <= '9') {
        return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
        return character - 'a' + 10;
    }
    if (character >= 'A' && character <= 'F') {
        return character - 'A' + 10;
    }
    return -1;
}

/* Read one UTF-16 code unit from a JSON \u escape. */
static int nya_json_code_unit(nya_json_parser *parser, uint32_t *value)
{
    unsigned int index;
    uint32_t result = 0;

    for (index = 0; index < 4; ++index) {
        int digit;
        if (parser->position >= parser->length) return -1;
        digit = nya_json_hex(parser->data[parser->position++]);
        if (digit < 0) return -1;
        result = (result << 4) | (uint32_t)digit;
    }
    *value = result;
    return 0;
}

/* Decode into the already-consumed part of the header. Escapes always contract,
 * so writes cannot overwrite bytes the parser still needs. This also lets names
 * be compared exactly without allocating another copy of each string. */
static int nya_json_decode_string(nya_json_parser *parser, nya_json_string *output)
{
    size_t start;
    size_t written;

    nya_json_whitespace(parser);
    if (parser->position >= parser->length || parser->data[parser->position] != '\"') {
        return -1;
    }
    parser->position += 1;
    start = parser->position;
    written = start;

    while (parser->position < parser->length) {
        unsigned char character;

        character = parser->data[parser->position];
        parser->position += 1;

        if (character == '\"') {
            if (output != NULL) {
                output->data = parser->data + start;
                output->length = written - start;
            }
            return 0;
        }

        if (character < 0x20) {
            return -1;
        }

        if (character == '\\') {
            if (parser->position >= parser->length) {
                return -1;
            }

            character = parser->data[parser->position];
            parser->position += 1;
            switch (character) {
                case '\"': character = '\"'; break;
                case '\\': character = '\\'; break;
                case '/': character = '/'; break;
                case 'b': character = '\b'; break;
                case 'f': character = '\f'; break;
                case 'n': character = '\n'; break;
                case 'r': character = '\r'; break;
                case 't': character = '\t'; break;
                case 'u': {
                    uint32_t codepoint;

                    if (nya_json_code_unit(parser, &codepoint) != 0) return -1;
                    if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                        uint32_t low;
                        if (parser->length - parser->position < 6 ||
                            parser->data[parser->position] != '\\' ||
                            parser->data[parser->position + 1] != 'u') return -1;
                        parser->position += 2;
                        if (nya_json_code_unit(parser, &low) != 0 || low < 0xdc00U || low > 0xdfffU) return -1;
                        codepoint = 0x10000U + ((codepoint - 0xd800U) << 10) + low - 0xdc00U;
                    } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                        return -1;
                    }
                    if (codepoint < 0x80U) {
                        parser->data[written++] = (unsigned char)codepoint;
                    } else if (codepoint < 0x800U) {
                        parser->data[written++] = (unsigned char)(0xc0U | (codepoint >> 6));
                        parser->data[written++] = (unsigned char)(0x80U | (codepoint & 0x3fU));
                    } else if (codepoint < 0x10000U) {
                        parser->data[written++] = (unsigned char)(0xe0U | (codepoint >> 12));
                        parser->data[written++] = (unsigned char)(0x80U | ((codepoint >> 6) & 0x3fU));
                        parser->data[written++] = (unsigned char)(0x80U | (codepoint & 0x3fU));
                    } else {
                        parser->data[written++] = (unsigned char)(0xf0U | (codepoint >> 18));
                        parser->data[written++] = (unsigned char)(0x80U | ((codepoint >> 12) & 0x3fU));
                        parser->data[written++] = (unsigned char)(0x80U | ((codepoint >> 6) & 0x3fU));
                        parser->data[written++] = (unsigned char)(0x80U | (codepoint & 0x3fU));
                    }
                    continue;
                }
                default:
                    return -1;
            }
        }

        parser->data[written++] = character;
    }

    return -1;
}

/* Capture short ASCII schema keys while preventing truncation and embedded NUL
 * from aliasing a known key. Unknown long keys remain syntactically valid JSON. */
static int nya_json_parse_string(nya_json_parser *parser, char *output, size_t capacity)
{
    nya_json_string value;

    if (nya_json_decode_string(parser, &value) != 0) return -1;
    if (output != NULL && capacity != 0) {
        output[0] = '\0';
        if (value.length < capacity && memchr(value.data, 0, value.length) == NULL) {
            memcpy(output, value.data, value.length);
            output[value.length] = '\0';
        }
    }
    return 0;
}

/* Parse the complete JSON number grammar when an unknown value is skipped. */
static int nya_json_skip_number(nya_json_parser *parser)
{
    size_t start;

    start = parser->position;
    if (parser->position < parser->length && parser->data[parser->position] == '-') {
        parser->position += 1;
    }

    if (parser->position >= parser->length) {
        return -1;
    }

    if (parser->data[parser->position] == '0') {
        parser->position += 1;
    } else {
        if (!isdigit(parser->data[parser->position])) {
            return -1;
        }
        while (parser->position < parser->length && isdigit(parser->data[parser->position])) {
            parser->position += 1;
        }
    }

    if (parser->position < parser->length && parser->data[parser->position] == '.') {
        parser->position += 1;
        if (parser->position >= parser->length || !isdigit(parser->data[parser->position])) {
            return -1;
        }
        while (parser->position < parser->length && isdigit(parser->data[parser->position])) {
            parser->position += 1;
        }
    }

    if (parser->position < parser->length &&
        (parser->data[parser->position] == 'e' || parser->data[parser->position] == 'E')) {
        parser->position += 1;
        if (parser->position < parser->length &&
            (parser->data[parser->position] == '+' || parser->data[parser->position] == '-')) {
            parser->position += 1;
        }
        if (parser->position >= parser->length || !isdigit(parser->data[parser->position])) {
            return -1;
        }
        while (parser->position < parser->length && isdigit(parser->data[parser->position])) {
            parser->position += 1;
        }
    }

    return parser->position > start ? 0 : -1;
}

/* Forward declaration permits recursive arrays and objects. */
static int nya_json_skip_value(nya_json_parser *parser);

/* Skip one arbitrary JSON array with an explicit nesting limit. */
static int nya_json_skip_array(nya_json_parser *parser)
{
    if (parser->depth >= NYA_JSON_DEPTH_LIMIT || nya_json_take(parser, '[') != 0) {
        return -1;
    }
    parser->depth += 1;

    nya_json_whitespace(parser);
    if (parser->position < parser->length && parser->data[parser->position] == ']') {
        parser->position += 1;
        parser->depth -= 1;
        return 0;
    }

    for (;;) {
        if (nya_json_skip_value(parser) != 0) {
            return -1;
        }
        nya_json_whitespace(parser);
        if (parser->position < parser->length && parser->data[parser->position] == ']') {
            parser->position += 1;
            parser->depth -= 1;
            return 0;
        }
        if (nya_json_take(parser, ',') != 0) {
            return -1;
        }
    }
}

/* Skip one arbitrary JSON object with an explicit nesting limit. */
static int nya_json_skip_object(nya_json_parser *parser)
{
    if (parser->depth >= NYA_JSON_DEPTH_LIMIT || nya_json_take(parser, '{') != 0) {
        return -1;
    }
    parser->depth += 1;

    nya_json_whitespace(parser);
    if (parser->position < parser->length && parser->data[parser->position] == '}') {
        parser->position += 1;
        parser->depth -= 1;
        return 0;
    }

    for (;;) {
        if (nya_json_parse_string(parser, NULL, 0) != 0 ||
            nya_json_take(parser, ':') != 0 ||
            nya_json_skip_value(parser) != 0) {
            return -1;
        }
        nya_json_whitespace(parser);
        if (parser->position < parser->length && parser->data[parser->position] == '}') {
            parser->position += 1;
            parser->depth -= 1;
            return 0;
        }
        if (nya_json_take(parser, ',') != 0) {
            return -1;
        }
    }
}

/* Skip any syntactically valid JSON value. */
static int nya_json_skip_value(nya_json_parser *parser)
{
    static const char true_text[] = "true";
    static const char false_text[] = "false";
    static const char null_text[] = "null";
    const unsigned char *remaining;

    nya_json_whitespace(parser);
    if (parser->position >= parser->length) {
        return -1;
    }

    remaining = parser->data + parser->position;
    if (*remaining == '\"') {
        return nya_json_parse_string(parser, NULL, 0);
    }
    if (*remaining == '{') {
        return nya_json_skip_object(parser);
    }
    if (*remaining == '[') {
        return nya_json_skip_array(parser);
    }
    if (*remaining == '-' || isdigit(*remaining)) {
        return nya_json_skip_number(parser);
    }
    if (parser->length - parser->position >= sizeof(true_text) - 1 &&
        memcmp(remaining, true_text, sizeof(true_text) - 1) == 0) {
        parser->position += sizeof(true_text) - 1;
        return 0;
    }
    if (parser->length - parser->position >= sizeof(false_text) - 1 &&
        memcmp(remaining, false_text, sizeof(false_text) - 1) == 0) {
        parser->position += sizeof(false_text) - 1;
        return 0;
    }
    if (parser->length - parser->position >= sizeof(null_text) - 1 &&
        memcmp(remaining, null_text, sizeof(null_text) - 1) == 0) {
        parser->position += sizeof(null_text) - 1;
        return 0;
    }

    return -1;
}

/* Parse one non-negative JSON integer into an exact 64-bit value. */
static int nya_json_u64_exact(nya_json_parser *parser, uint64_t *value)
{
    uint64_t result;

    nya_json_whitespace(parser);
    if (parser->position >= parser->length || !isdigit(parser->data[parser->position])) {
        return -1;
    }

    /* JSON permits zero itself, but never an integer with a leading zero. */
    if (parser->data[parser->position] == '0' && parser->position + 1 < parser->length &&
        isdigit(parser->data[parser->position + 1])) return -1;

    result = 0;
    while (parser->position < parser->length && isdigit(parser->data[parser->position])) {
        unsigned int digit;

        digit = parser->data[parser->position] - '0';
        if (result > (UINT64_MAX - digit) / 10) {
            return -1;
        }
        result = result * 10 + digit;
        parser->position += 1;
    }

    *value = result;
    return 0;
}

/* Return the number of bits in one currently specified SafeTensors element. */
static unsigned int nya_safetensors_dtype_bits(const char *dtype)
{
    if (strcmp(dtype, "F4") == 0) return 4;
    if (strcmp(dtype, "F6_E2M3") == 0 || strcmp(dtype, "F6_E3M2") == 0) return 6;
    if (strcmp(dtype, "BOOL") == 0 || strcmp(dtype, "U8") == 0 ||
        strcmp(dtype, "I8") == 0 || strcmp(dtype, "F8_E5M2") == 0 ||
        strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E8M0") == 0 ||
        strcmp(dtype, "F8_E4M3FNUZ") == 0 || strcmp(dtype, "F8_E5M2FNUZ") == 0) return 8;
    if (strcmp(dtype, "I16") == 0 || strcmp(dtype, "U16") == 0 ||
        strcmp(dtype, "F16") == 0 || strcmp(dtype, "BF16") == 0) return 16;
    if (strcmp(dtype, "I32") == 0 || strcmp(dtype, "U32") == 0 ||
        strcmp(dtype, "F32") == 0) return 32;
    if (strcmp(dtype, "I64") == 0 || strcmp(dtype, "U64") == 0 ||
        strcmp(dtype, "F64") == 0 || strcmp(dtype, "C64") == 0) return 64;
    return 0;
}

/* Parse a tensor shape and safely calculate its element count. */
static int nya_safetensors_shape(nya_json_parser *parser, uint64_t *element_count)
{
    uint64_t product;

    if (nya_json_take(parser, '[') != 0) {
        return -1;
    }

    product = 1;
    nya_json_whitespace(parser);
    if (parser->position < parser->length && parser->data[parser->position] == ']') {
        parser->position += 1;
        *element_count = 1;
        return 0;
    }

    for (;;) {
        uint64_t extent;

        if (nya_json_u64_exact(parser, &extent) != 0) {
            return -1;
        }
        if (extent != 0 && product > UINT64_MAX / extent) {
            return -1;
        }
        product *= extent;

        nya_json_whitespace(parser);
        if (parser->position < parser->length && parser->data[parser->position] == ']') {
            parser->position += 1;
            *element_count = product;
            return 0;
        }
        if (nya_json_take(parser, ',') != 0) {
            return -1;
        }
    }
}

/* Parse exactly two tensor data offsets. */
static int nya_safetensors_offsets(nya_json_parser *parser, uint64_t *begin, uint64_t *end)
{
    if (nya_json_take(parser, '[') != 0 ||
        nya_json_u64_exact(parser, begin) != 0 ||
        nya_json_take(parser, ',') != 0 ||
        nya_json_u64_exact(parser, end) != 0 ||
        nya_json_take(parser, ']') != 0) {
        return -1;
    }

    return 0;
}

/* Append one interval while enforcing the global tensor-count limit. */
static int nya_interval_append(
    nya_interval_list *list, uint64_t begin, uint64_t end, const nya_json_string *name
)
{
    if (list->count >= NYA_SAFETENSORS_TENSOR_LIMIT) {
        return NYA_FORMAT_RESOURCE_LIMIT;
    }

    if (list->count == list->capacity) {
        size_t new_capacity;
        nya_tensor_interval *new_items;

        new_capacity = list->capacity == 0 ? 64 : list->capacity * 2;
        if (new_capacity > NYA_SAFETENSORS_TENSOR_LIMIT) {
            new_capacity = (size_t)NYA_SAFETENSORS_TENSOR_LIMIT;
        }
        if (new_capacity > SIZE_MAX / sizeof(*new_items)) return NYA_FORMAT_RESOURCE_LIMIT;

        new_items = (nya_tensor_interval *)realloc(list->items, new_capacity * sizeof(*new_items));
        if (new_items == NULL) {
            return NYA_FORMAT_RESOURCE_LIMIT;
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }

    list->items[list->count].begin = begin;
    list->items[list->count].end = end;
    list->items[list->count].name = *name;
    list->count += 1;
    return 0;
}

/* Parse and validate one tensor metadata object. */
static int nya_safetensors_tensor(
    nya_json_parser *parser, nya_interval_list *intervals, const nya_json_string *name
)
{
    char dtype[32];
    int has_dtype;
    int has_shape;
    int has_offsets;
    uint64_t element_count;
    uint64_t begin;
    uint64_t end;

    dtype[0] = '\0';
    has_dtype = 0;
    has_shape = 0;
    has_offsets = 0;
    element_count = 0;
    begin = 0;
    end = 0;

    if (nya_json_take(parser, '{') != 0) {
        return -1;
    }

    for (;;) {
        char key[32];

        /* Each iteration starts with a required member. The closing brace is
         * consumed only after a member, so a trailing comma cannot be accepted. */
        if (nya_json_parse_string(parser, key, sizeof(key)) != 0 || nya_json_take(parser, ':') != 0) {
            return -1;
        }

        if (strcmp(key, "dtype") == 0) {
            if (has_dtype || nya_json_parse_string(parser, dtype, sizeof(dtype)) != 0) return -1;
            has_dtype = 1;
        } else if (strcmp(key, "shape") == 0) {
            if (has_shape || nya_safetensors_shape(parser, &element_count) != 0) return -1;
            has_shape = 1;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (has_offsets || nya_safetensors_offsets(parser, &begin, &end) != 0) return -1;
            has_offsets = 1;
        } else if (nya_json_skip_value(parser) != 0) {
            return -1;
        }

        nya_json_whitespace(parser);
        if (parser->position < parser->length && parser->data[parser->position] == '}') {
            parser->position += 1;
            break;
        }
        if (nya_json_take(parser, ',') != 0) {
            return -1;
        }
    }

    if (!has_dtype || !has_shape || !has_offsets || begin > end) {
        return -1;
    }

    {
        unsigned int bits;
        uint64_t total_bits;
        uint64_t expected_bytes;

        bits = nya_safetensors_dtype_bits(dtype);
        if (bits == 0 || (element_count != 0 && element_count > UINT64_MAX / bits)) {
            return -1;
        }
        total_bits = element_count * bits;
        /* Sub-byte types must still occupy complete bytes; padding a partial
         * element group would disagree with SafeTensors' reference validator. */
        if (total_bits % 8 != 0) return -1;
        expected_bytes = total_bits / 8;
        if (end - begin != expected_bytes) {
            return -1;
        }
    }

    return nya_interval_append(intervals, begin, end, name);
}

/* SafeTensors reserves __metadata__ for a map of string values. Arbitrary JSON
 * is valid for unknown tensor fields, but is not a valid metadata value. */
static int nya_safetensors_metadata(nya_json_parser *parser)
{
    if (nya_json_take(parser, '{') != 0) return -1;
    nya_json_whitespace(parser);
    if (parser->position < parser->length && parser->data[parser->position] == '}') {
        parser->position += 1;
        return 0;
    }
    for (;;) {
        if (nya_json_parse_string(parser, NULL, 0) != 0 || nya_json_take(parser, ':') != 0 ||
            nya_json_parse_string(parser, NULL, 0) != 0) return -1;
        nya_json_whitespace(parser);
        if (parser->position < parser->length && parser->data[parser->position] == '}') {
            parser->position += 1;
            return 0;
        }
        if (nya_json_take(parser, ',') != 0) return -1;
    }
}

/* Compare full decoded names before re-sorting by offset. This detects both
 * literal duplicates and equivalent JSON spellings such as x and \u0078. */
static int nya_interval_name_compare(const void *left_pointer, const void *right_pointer)
{
    const nya_json_string *left = &((const nya_tensor_interval *)left_pointer)->name;
    const nya_json_string *right = &((const nya_tensor_interval *)right_pointer)->name;
    size_t common = left->length < right->length ? left->length : right->length;
    int order = memcmp(left->data, right->data, common);
    if (order != 0) return order;
    return left->length < right->length ? -1 : (left->length > right->length ? 1 : 0);
}

/* Sort tensor slices by their declared data position. */
static int nya_interval_compare(const void *left_pointer, const void *right_pointer)
{
    const nya_tensor_interval *left;
    const nya_tensor_interval *right;

    left = (const nya_tensor_interval *)left_pointer;
    right = (const nya_tensor_interval *)right_pointer;
    if (left->begin < right->begin) return -1;
    if (left->begin > right->begin) return 1;
    if (left->end < right->end) return -1;
    if (left->end > right->end) return 1;
    return 0;
}

/* Validate a complete SafeTensors header and its contiguous data ranges. */
nya_format_result nya_safetensors_inspect(nya_reader *reader, nya_format_info *information)
{
    uint64_t header_length;
    uint64_t data_length;
    unsigned char *header;
    nya_json_parser parser;
    nya_interval_list intervals;
    uint64_t metadata_count;
    size_t index;
    uint64_t expected_begin;

    if (nya_reader_u64_le(reader, &header_length) != 0 || header_length < 2 ||
        header_length > NYA_SAFETENSORS_HEADER_LIMIT || header_length > reader->size - 8) {
        return NYA_FORMAT_INVALID;
    }

    header = (unsigned char *)malloc((size_t)header_length);
    if (header == NULL) {
        return NYA_FORMAT_RESOURCE_LIMIT;
    }

    if (nya_reader_read(reader, header, (size_t)header_length) != 0 ||
        header[0] != '{' || !nya_utf8_valid(header, (size_t)header_length)) {
        free(header);
        return NYA_FORMAT_INVALID;
    }

    parser.data = header;
    parser.length = (size_t)header_length;
    parser.position = 0;
    parser.depth = 0;
    memset(&intervals, 0, sizeof(intervals));
    metadata_count = 0;

    if (nya_json_take(&parser, '{') != 0) {
        free(header);
        return NYA_FORMAT_INVALID;
    }

    nya_json_whitespace(&parser);
    while (parser.position < parser.length && parser.data[parser.position] != '}') {
        nya_json_string name;
        int item_result;

        if (nya_json_decode_string(&parser, &name) != 0 || nya_json_take(&parser, ':') != 0) {
            free(intervals.items);
            free(header);
            return NYA_FORMAT_INVALID;
        }

        if (name.length == sizeof("__metadata__") - 1 &&
            memcmp(name.data, "__metadata__", name.length) == 0) {
            item_result = metadata_count == 0 ? nya_safetensors_metadata(&parser) : -1;
            metadata_count += 1;
        } else {
            item_result = nya_safetensors_tensor(&parser, &intervals, &name);
        }

        if (item_result != 0) {
            free(intervals.items);
            free(header);
            return item_result == NYA_FORMAT_RESOURCE_LIMIT ? NYA_FORMAT_RESOURCE_LIMIT : NYA_FORMAT_INVALID;
        }

        nya_json_whitespace(&parser);
        if (parser.position < parser.length && parser.data[parser.position] == '}') {
            break;
        }
        if (nya_json_take(&parser, ',') != 0) {
            free(intervals.items);
            free(header);
            return NYA_FORMAT_INVALID;
        }
        /* Whitespace after a comma is legal; a closing brace is not. */
        nya_json_whitespace(&parser);
        if (parser.position >= parser.length || parser.data[parser.position] == '}') {
            free(intervals.items);
            free(header);
            return NYA_FORMAT_INVALID;
        }
    }

    if (nya_json_take(&parser, '}') != 0) {
        free(intervals.items);
        free(header);
        return NYA_FORMAT_INVALID;
    }
    nya_json_whitespace(&parser);
    if (parser.position != parser.length) {
        free(intervals.items);
        free(header);
        return NYA_FORMAT_INVALID;
    }

    data_length = reader->size - reader->position;
    /* Sorting is unnecessary for zero or one item and avoids a null base pointer. */
    if (intervals.count > 1) {
        qsort(intervals.items, intervals.count, sizeof(intervals.items[0]), nya_interval_name_compare);
        for (index = 1; index < intervals.count; ++index) {
            if (nya_interval_name_compare(&intervals.items[index - 1], &intervals.items[index]) == 0) {
                free(intervals.items);
                free(header);
                return NYA_FORMAT_INVALID;
            }
        }
        qsort(intervals.items, intervals.count, sizeof(intervals.items[0]), nya_interval_compare);
    }
    expected_begin = 0;
    for (index = 0; index < intervals.count; ++index) {
        if (intervals.items[index].begin != expected_begin || intervals.items[index].end > data_length) {
            free(intervals.items);
            free(header);
            return NYA_FORMAT_INVALID;
        }
        expected_begin = intervals.items[index].end;
    }

    if (expected_begin != data_length) {
        free(intervals.items);
        free(header);
        return NYA_FORMAT_INVALID;
    }

    information->format = NYA_FORMAT_SAFETENSORS;
    information->format_version = 1;
    information->tensor_count = (uint64_t)intervals.count;
    information->metadata_count = metadata_count;
    information->data_offset = reader->position;

    free(intervals.items);
    free(header);
    return NYA_FORMAT_OK;
}
