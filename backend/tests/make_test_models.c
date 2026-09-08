#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Write an exact byte span and turn every short write into a test failure. */
static int write_bytes(FILE *file, const void *data, size_t length)
{
    return fwrite(data, 1, length, file) == length ? 0 : -1;
}

/* Write one unsigned little-endian 32-bit integer. */
static int write_u32(FILE *file, uint32_t value)
{
    unsigned char bytes[4];

    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8);
    bytes[2] = (unsigned char)(value >> 16);
    bytes[3] = (unsigned char)(value >> 24);
    return write_bytes(file, bytes, sizeof(bytes));
}

/* Write one unsigned little-endian 64-bit integer. */
static int write_u64(FILE *file, uint64_t value)
{
    unsigned char bytes[8];
    size_t index;

    for (index = 0; index < sizeof(bytes); ++index) {
        bytes[index] = (unsigned char)(value >> (index * 8));
    }
    return write_bytes(file, bytes, sizeof(bytes));
}

/* Join a fixture directory and filename without platform-specific allocation. */
static int fixture_path(char *output, size_t capacity, const char *directory, const char *name)
{
    int written;

#ifdef _WIN32
    written = snprintf(output, capacity, "%s\\%s", directory, name);
#else
    written = snprintf(output, capacity, "%s/%s", directory, name);
#endif
    return written >= 0 && (size_t)written < capacity ? 0 : -1;
}

/* Produce a metadata-only GGUF v3 container with a llama architecture key. */
static int make_gguf(const char *directory)
{
    static const char key[] = "general.architecture";
    static const char architecture[] = "llama";
    unsigned char zeroes[27];
    char path[1024];
    FILE *file;
    int result;

    if (fixture_path(path, sizeof(path), directory, "minimal.gguf") != 0) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;

    memset(zeroes, 0, sizeof(zeroes));
    result = 0;

    /* Separate statements preserve binary field order on every C compiler. */
    if (write_bytes(file, "GGUF", 4) != 0) result = -1;
    if (write_u32(file, 3) != 0) result = -1;
    if (write_u64(file, 0) != 0) result = -1;
    if (write_u64(file, 1) != 0) result = -1;
    if (write_u64(file, sizeof(key) - 1) != 0) result = -1;
    if (write_bytes(file, key, sizeof(key) - 1) != 0) result = -1;
    if (write_u32(file, 8) != 0) result = -1;
    if (write_u64(file, sizeof(architecture) - 1) != 0) result = -1;
    if (write_bytes(file, architecture, sizeof(architecture) - 1) != 0) result = -1;
    if (write_bytes(file, zeroes, sizeof(zeroes)) != 0) result = -1;

    if (fclose(file) != 0) result = -1;
    return result;
}

/* Produce one F32 SafeTensors entry whose data buffer contains 3.5. */
static int make_safetensors(const char *directory)
{
    static const char header[] = "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    float value;
    char path[1024];
    FILE *file;
    int result;

    if (fixture_path(path, sizeof(path), directory, "minimal.safetensors") != 0) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;

    value = 3.5f;
    result = 0;

    /* The header length, header, and tensor data must be emitted in order. */
    if (write_u64(file, sizeof(header) - 1) != 0) result = -1;
    if (write_bytes(file, header, sizeof(header) - 1) != 0) result = -1;
    if (write_bytes(file, &value, sizeof(value)) != 0) result = -1;

    if (fclose(file) != 0) result = -1;
    return result;
}

/* Produce an ONNX Identity graph with one [1] float input and output. */
static int make_onnx(const char *directory)
{
    static const unsigned char model[] = {
        /* ModelProto.ir_version = 8. */
        0x08, 0x08,
        /* ModelProto.producer_name = "fyodor". */
        0x12, 0x06, 'f', 'y', 'o', 'd', 'o', 'r',
        /* ModelProto.graph, 62 bytes. */
        0x3a, 0x3e,
            /* GraphProto.node: Identity(x -> y). */
            0x0a, 0x10,
                0x0a, 0x01, 'x',
                0x12, 0x01, 'y',
                0x22, 0x08, 'I', 'd', 'e', 'n', 't', 'i', 't', 'y',
            /* GraphProto.name = "identity". */
            0x12, 0x08, 'i', 'd', 'e', 'n', 't', 'i', 't', 'y',
            /* GraphProto.input: x, tensor(float, [1]). */
            0x5a, 0x0f,
                0x0a, 0x01, 'x',
                0x12, 0x0a,
                    0x0a, 0x08,
                        0x08, 0x01,
                        0x12, 0x04,
                            0x0a, 0x02, 0x08, 0x01,
            /* GraphProto.output: y, tensor(float, [1]). */
            0x62, 0x0f,
                0x0a, 0x01, 'y',
                0x12, 0x0a,
                    0x0a, 0x08,
                        0x08, 0x01,
                        0x12, 0x04,
                            0x0a, 0x02, 0x08, 0x01,
        /* ModelProto.opset_import: default domain version 13. */
        0x42, 0x02, 0x10, 0x0d
    };
    char path[1024];
    FILE *file;
    int result;

    if (fixture_path(path, sizeof(path), directory, "identity.onnx") != 0) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;

    result = write_bytes(file, model, sizeof(model));
    if (fclose(file) != 0) result = -1;
    return result;
}

/* Produce one deliberately truncated example of every recognized container. */
static int make_invalid_models(const char *directory)
{
    static const char safetensors_header[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    static const unsigned char onnx_without_graph[] = {0x08, 0x08};
    char path[1024];
    FILE *file;
    int result;

    if (fixture_path(path, sizeof(path), directory, "invalid.gguf") != 0) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;
    result = write_bytes(file, "GGUF", 4);
    if (fclose(file) != 0) result = -1;
    if (result != 0) return -1;

    if (fixture_path(path, sizeof(path), directory, "invalid.safetensors") != 0) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;
    result = write_u64(file, sizeof(safetensors_header) - 1);
    if (write_bytes(file, safetensors_header, sizeof(safetensors_header) - 1) != 0) result = -1;
    if (fclose(file) != 0) result = -1;
    if (result != 0) return -1;

    if (fixture_path(path, sizeof(path), directory, "invalid.onnx") != 0) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;
    result = write_bytes(file, onnx_without_graph, sizeof(onnx_without_graph));
    if (fclose(file) != 0) result = -1;
    if (result != 0) return -1;

    return 0;
}

/* Every malformed header has a matching data length, so these regressions fail
 * on their intended grammar/identity check rather than incidental truncation. */
static int make_safetensors_regressions(const char *directory)
{
    static const struct {
        const char *name;
        const char *header;
        size_t bytes;
    } cases[] = {
        {"escaped.safetensors", "{\"\\u0078\":{\"d\\u0074ype\":\"\\u004632\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4},
        {"empty.safetensors", "{}", 0},
        {"zero.safetensors", "{\"z\":{\"dtype\":\"F32\",\"shape\":[0,9],\"data_offsets\":[0,0]}}", 0},
        {"subbyte.safetensors", "{\"x\":{\"dtype\":\"F4\",\"shape\":[2],\"data_offsets\":[0,1]}}", 1},
        {"unicode.safetensors", "{\"\\ud83d\\ude00\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"\\u00e9\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}", 2},
        {"metadata.safetensors", "{\"__metadata__\":{\"author\":\"engine\",\"note\":\"\\ud83d\\ude00\"}}", 0},
        {"trailing-root.safetensors", "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},}", 4},
        {"trailing-tensor.safetensors", "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],}}", 4},
        {"trailing-metadata.safetensors", "{\"__metadata__\":{\"a\":\"b\",}}", 0},
        {"leading-zero.safetensors", "{\"x\":{\"dtype\":\"F32\",\"shape\":[01],\"data_offsets\":[0,4]}}", 4},
        {"duplicate-name.safetensors", "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8},
        {"duplicate-escaped.safetensors", "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"\\u0078\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8},
        {"duplicate-field.safetensors", "{\"x\":{\"dtype\":\"F32\",\"d\\u0074ype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4},
        {"nul-dtype.safetensors", "{\"x\":{\"dtype\":\"F32\\u0000suffix\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4},
        {"high-surrogate.safetensors", "{\"__metadata__\":{\"a\":\"\\ud800\"}}", 0},
        {"low-surrogate.safetensors", "{\"__metadata__\":{\"a\":\"\\udfff\"}}", 0},
        {"invalid-whitespace.safetensors", "{\f}", 0},
        {"non-string-metadata.safetensors", "{\"__metadata__\":{\"a\":123}}", 0},
        {"duplicate-metadata.safetensors", "{\"__metadata__\":{},\"__metadata__\":{}}", 0},
        {"partial-byte.safetensors", "{\"x\":{\"dtype\":\"F4\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1},
        {"overlap.safetensors", "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"y\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 8},
        {"hole.safetensors", "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8},
        {"overflow-shape.safetensors", "{\"x\":{\"dtype\":\"F32\",\"shape\":[18446744073709551615,2],\"data_offsets\":[0,4]}}", 4},
        {"overflow-integer.safetensors", "{\"x\":{\"dtype\":\"U8\",\"shape\":[18446744073709551616],\"data_offsets\":[0,1]}}", 1},
        {"invalid-utf8.safetensors", "{\"__metadata__\":{\"a\":\"\xc0\x80\"}}", 0}
    };
    unsigned char data[8] = {0};
    size_t index;

    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        char path[1024];
        FILE *file;
        int result = 0;
        if (fixture_path(path, sizeof(path), directory, cases[index].name) != 0) return -1;
        file = fopen(path, "wb");
        if (file == NULL) return -1;
        if (write_u64(file, (uint64_t)strlen(cases[index].header)) != 0) result = -1;
        if (write_bytes(file, cases[index].header, strlen(cases[index].header)) != 0) result = -1;
        if (write_bytes(file, data, cases[index].bytes) != 0) result = -1;
        if (fclose(file) != 0) result = -1;
        if (result != 0) return -1;
    }
    return 0;
}

/* Build tiny tensor directories with independently chosen declared shapes and
 * physical payloads. The data is zero-filled: inspection never executes it. */
static int make_gguf_regressions(const char *directory)
{
    static const struct {
        const char *name;
        uint32_t type;
        uint32_t dimensions;
        uint64_t extent;
        size_t payload;
        unsigned int flags; /* 1: two tensors; 2: same name; 4: same offset; 8: align 24 */
    } cases[] = {
        {"tensor.gguf", 0, 1, 1, 4, 0},
        {"q4.gguf", 2, 1, 32, 18, 0},
        {"alignment24.gguf", 0, 1, 1, 4, 8},
        {"truncated-tensor.gguf", 0, 1, 1, 0, 0},
        {"truncated-q4.gguf", 2, 1, 32, 17, 0},
        {"overflow-tensor.gguf", 0, 2, UINT64_MAX / 2, 4, 0},
        {"zero-dimensions.gguf", 0, 0, 1, 4, 0},
        {"row-block.gguf", 2, 2, 16, 18, 0},
        {"duplicate-tensor.gguf", 0, 1, 1, 36, 3},
        {"overlap-tensor.gguf", 0, 1, 1, 36, 5},
        {"unknown-type.gguf", 63, 1, 1, 4, 0}
    };
    static const char alignment_key[] = "general.alignment";
    unsigned char data[36] = {0};
    size_t index;

    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        char path[1024];
        FILE *file;
        unsigned int tensor;
        unsigned int count = (cases[index].flags & 1U) != 0 ? 2U : 1U;
        uint32_t alignment = (cases[index].flags & 8U) != 0 ? 24U : 32U;
        int result = 0;
        long position;
        if (fixture_path(path, sizeof(path), directory, cases[index].name) != 0) return -1;
        file = fopen(path, "wb");
        if (file == NULL) return -1;
        if (write_bytes(file, "GGUF", 4) != 0) result = -1;
        if (write_u32(file, 3) != 0) result = -1;
        if (write_u64(file, count) != 0) result = -1;
        if (write_u64(file, alignment == 24 ? 1 : 0) != 0) result = -1;
        if (alignment == 24) {
            if (write_u64(file, sizeof(alignment_key) - 1) != 0) result = -1;
            if (write_bytes(file, alignment_key, sizeof(alignment_key) - 1) != 0) result = -1;
            if (write_u32(file, 4) != 0) result = -1;
            if (write_u32(file, alignment) != 0) result = -1;
        }
        for (tensor = 0; tensor < count; ++tensor) {
            unsigned char name = tensor == 0 || (cases[index].flags & 2U) != 0 ? 'x' : 'y';
            uint32_t dimension;
            if (write_u64(file, 1) != 0) result = -1;
            if (write_bytes(file, &name, 1) != 0) result = -1;
            if (write_u32(file, cases[index].dimensions) != 0) result = -1;
            for (dimension = 0; dimension < cases[index].dimensions; ++dimension) {
                if (write_u64(file, dimension == 0 ? cases[index].extent : 2) != 0) result = -1;
            }
            if (write_u32(file, cases[index].type) != 0) result = -1;
            if (write_u64(file, tensor == 0 || (cases[index].flags & 4U) != 0 ? 0 : 32) != 0) result = -1;
        }
        position = ftell(file);
        if (position < 0) result = -1;
        else if (write_bytes(file, data, (size_t)((alignment - (uint64_t)position % alignment) % alignment)) != 0) result = -1;
        if (write_bytes(file, data, cases[index].payload) != 0) result = -1;
        if (fclose(file) != 0) result = -1;
        if (result != 0) return -1;
    }
    return 0;
}

/* Append invalid tags to an otherwise complete ONNX envelope. A parser that
 * only checks tag != 0 used to accept these illegal field-number encodings. */
static int make_onnx_regressions(const char *directory)
{
    static const unsigned char envelope[] = {0x08, 0x08, 0x3a, 0x03, 0x12, 0x01, 'g', 0x42, 0x02, 0x10, 0x0d};
    static const struct {
        const char *name;
        unsigned char tag[6];
        size_t length;
    } cases[] = {
        {"zero-field.onnx", {0x02, 0x00}, 2},
        {"oversized-field.onnx", {0x80, 0x80, 0x80, 0x80, 0x10, 0x00}, 6},
        {"truncated-varint.onnx", {0x80}, 1}
    };
    size_t index;
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        char path[1024];
        FILE *file;
        int result = 0;
        if (fixture_path(path, sizeof(path), directory, cases[index].name) != 0) return -1;
        file = fopen(path, "wb");
        if (file == NULL) return -1;
        if (write_bytes(file, envelope, sizeof(envelope)) != 0) result = -1;
        if (write_bytes(file, cases[index].tag, cases[index].length) != 0) result = -1;
        if (fclose(file) != 0) result = -1;
        if (result != 0) return -1;
    }
    {
        char path[1024];
        unsigned char zeros[4096] = {0};
        /* Unknown length-delimited field 100, 8192 bytes, forces the reader's
         * large-skip path. A valid envelope afterward verifies cursor sync. */
        static const unsigned char tag[] = {0xa2, 0x06, 0x80, 0x40};
        FILE *file;
        int result = 0;
        if (fixture_path(path, sizeof(path), directory, "large-field.onnx") != 0) return -1;
        file = fopen(path, "wb");
        if (file == NULL) return -1;
        if (write_bytes(file, tag, sizeof(tag)) != 0) result = -1;
        if (write_bytes(file, zeros, sizeof(zeros)) != 0) result = -1;
        if (write_bytes(file, zeros, sizeof(zeros)) != 0) result = -1;
        if (write_bytes(file, envelope, sizeof(envelope)) != 0) result = -1;
        if (fclose(file) != 0) result = -1;
        if (result != 0) return -1;
    }
    return 0;
}

/* Generate every deterministic fixture into an already existing directory. */
int main(int argument_count, char **arguments)
{
    if (argument_count != 2) {
        fprintf(stderr, "usage: make_test_models OUTPUT_DIRECTORY\n");
        return 2;
    }

    if (make_gguf(arguments[1]) != 0 ||
        make_safetensors(arguments[1]) != 0 ||
        make_onnx(arguments[1]) != 0 ||
        make_invalid_models(arguments[1]) != 0 ||
        make_safetensors_regressions(arguments[1]) != 0 ||
        make_gguf_regressions(arguments[1]) != 0 ||
        make_onnx_regressions(arguments[1]) != 0) {
        fprintf(stderr, "failed to generate test models\n");
        return 1;
    }

    return 0;
}
