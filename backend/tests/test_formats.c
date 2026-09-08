#include "format.h"
#include "model.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <sys/stat.h>
#else
#include <sys/stat.h>
#endif

/* Read one fixture size using the same 64-bit platform choice as the engine. */
static int file_size(const char *path, uint64_t *size)
{
#ifdef _WIN32
    struct _stat64 information;

    if (_stat64(path, &information) != 0 || information.st_size < 0) return -1;
#else
    struct stat information;

    if (stat(path, &information) != 0 || information.st_size < 0) return -1;
#endif
    *size = (uint64_t)information.st_size;
    return 0;
}

/* Inspect one fixture and require its important common metadata. */
static int inspect(
    const char *path,
    nya_model_format format,
    uint64_t version,
    uint64_t tensors
)
{
    uint64_t size;
    nya_format_info information;

    if (file_size(path, &size) != 0 ||
        nya_format_inspect(path, size, &information) != NYA_FORMAT_OK ||
        information.format != format ||
        information.format_version != version ||
        information.tensor_count != tensors) {
        fprintf(stderr, "format inspection failed: %s\n", path);
        return -1;
    }

    return 0;
}

/* Require one recognized but malformed container to fail structural validation. */
static int reject(const char *path)
{
    uint64_t size;
    nya_format_info information;

    if (file_size(path, &size) != 0 ||
        nya_format_inspect(path, size, &information) != NYA_FORMAT_INVALID) {
        fprintf(stderr, "invalid container was accepted: %s\n", path);
        return -1;
    }

    return 0;
}

/* Resolve additional fixture names beside the supplied original GGUF fixture,
 * preserving the existing command-line interface and CTest fixture lifecycle. */
static int sibling_path(char *output, size_t capacity, const char *base, const char *name)
{
    const char *slash = strrchr(base, '/');
    const char *backslash = strrchr(base, '\\');
    size_t prefix;
    size_t name_length = strlen(name);
    if (backslash != NULL && (slash == NULL || backslash > slash)) slash = backslash;
    prefix = slash == NULL ? 0 : (size_t)(slash - base) + 1;
    if (prefix >= capacity || name_length >= capacity - prefix) return -1;
    memcpy(output, base, prefix);
    memcpy(output + prefix, name, name_length + 1);
    return 0;
}

/* Create a non-ASCII filename with the OS Unicode API, then load it through
 * the public UTF-8 model API. This catches accidental narrow fopen/stat calls
 * that work for ASCII fixtures but fail under Windows legacy code pages. */
static int unicode_model_path(const char *base)
{
    char path[1024];
    FILE *source = NULL, *destination = NULL;
    nya_model_registry registry;
    const nya_model *model = NULL;
    unsigned char bytes[4096];
    size_t count;
    int result = -1;
#ifdef _WIN32
    wchar_t wide_path[1024];
#endif
    if (sibling_path(path, sizeof(path), base, "model-\xed\x95\x9c\xea\xb8\x80-\xf0\x9f\x98\x80.gguf") != 0) return -1;
#ifdef _WIN32
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide_path, 1024) <= 0) return -1;
    destination = _wfopen(wide_path, L"wb");
#else
    destination = fopen(path, "wb");
#endif
    source = fopen(base, "rb");
    if (source == NULL || destination == NULL) goto cleanup;
    while ((count = fread(bytes, 1, sizeof(bytes), source)) != 0) {
        if (fwrite(bytes, 1, count, destination) != count) goto cleanup;
    }
    if (ferror(source)) goto cleanup;
    fclose(source); source = NULL;
    if (fclose(destination) != 0) { destination = NULL; goto cleanup; }
    destination = NULL;
    nya_model_registry_init(&registry);
    if (nya_model_load(&registry, path, &model) == NYA_MODEL_OK &&
        model != NULL && strcmp(model->path, path) == 0 && model->format == NYA_FORMAT_GGUF) result = 0;
    nya_model_registry_shutdown(&registry);
cleanup:
    if (source != NULL) fclose(source);
    if (destination != NULL) fclose(destination);
#ifdef _WIN32
    _wremove(wide_path);
#else
    remove(path);
#endif
    if (result != 0) fprintf(stderr, "UTF-8 model path failed\n");
    return result;
}

/* Check malformed fixtures individually so one failure reports the exact
 * violated contract, including the requirement to clear partial metadata. */
static int regressions(const char *base)
{
    static const char *invalid[] = {
        "trailing-root.safetensors", "trailing-tensor.safetensors", "trailing-metadata.safetensors",
        "leading-zero.safetensors", "duplicate-name.safetensors", "duplicate-escaped.safetensors",
        "duplicate-field.safetensors", "nul-dtype.safetensors", "high-surrogate.safetensors",
        "low-surrogate.safetensors", "invalid-whitespace.safetensors", "non-string-metadata.safetensors",
        "duplicate-metadata.safetensors", "partial-byte.safetensors", "overlap.safetensors",
        "hole.safetensors", "overflow-shape.safetensors", "overflow-integer.safetensors",
        "invalid-utf8.safetensors", "truncated-tensor.gguf", "truncated-q4.gguf",
        "overflow-tensor.gguf", "zero-dimensions.gguf", "row-block.gguf",
        "duplicate-tensor.gguf", "overlap-tensor.gguf", "zero-field.onnx",
        "oversized-field.onnx", "truncated-varint.onnx"
    };
    static const struct {
        const char *name;
        nya_model_format format;
        uint64_t version;
        uint64_t tensors;
    } valid[] = {
        {"escaped.safetensors", NYA_FORMAT_SAFETENSORS, 1, 1},
        {"empty.safetensors", NYA_FORMAT_SAFETENSORS, 1, 0},
        {"zero.safetensors", NYA_FORMAT_SAFETENSORS, 1, 1},
        {"subbyte.safetensors", NYA_FORMAT_SAFETENSORS, 1, 1},
        {"unicode.safetensors", NYA_FORMAT_SAFETENSORS, 1, 2},
        {"metadata.safetensors", NYA_FORMAT_SAFETENSORS, 1, 0},
        {"tensor.gguf", NYA_FORMAT_GGUF, 3, 1},
        {"q4.gguf", NYA_FORMAT_GGUF, 3, 1},
        {"alignment24.gguf", NYA_FORMAT_GGUF, 3, 1},
        {"large-field.onnx", NYA_FORMAT_ONNX, 8, 0}
    };
    char path[1024];
    size_t index;
    nya_format_info information;
    nya_format_info empty;
    uint64_t size;

    memset(&empty, 0, sizeof(empty));
    memset(&information, 0xa5, sizeof(information));
    if (nya_format_inspect(NULL, 0, &information) != NYA_FORMAT_INVALID ||
        memcmp(&information, &empty, sizeof(empty)) != 0 ||
        nya_format_inspect(base, 0, NULL) != NYA_FORMAT_INVALID) return -1;
    if (file_size(base, &size) != 0 || size == 0 ||
        nya_format_inspect(base, size - 1, &information) != NYA_FORMAT_IO_ERROR ||
        memcmp(&information, &empty, sizeof(empty)) != 0) {
        fprintf(stderr, "stale file size was accepted\n");
        return -1;
    }
    for (index = 0; index < sizeof(invalid) / sizeof(invalid[0]); ++index) {
        if (sibling_path(path, sizeof(path), base, invalid[index]) != 0 || file_size(path, &size) != 0) return -1;
        memset(&information, 0xa5, sizeof(information));
        if (nya_format_inspect(path, size, &information) != NYA_FORMAT_INVALID ||
            memcmp(&information, &empty, sizeof(empty)) != 0) {
            fprintf(stderr, "malformed regression failed: %s\n", path);
            return -1;
        }
    }
    for (index = 0; index < sizeof(valid) / sizeof(valid[0]); ++index) {
        if (sibling_path(path, sizeof(path), base, valid[index].name) != 0 ||
            inspect(path, valid[index].format, valid[index].version, valid[index].tensors) != 0) return -1;
    }
    if (sibling_path(path, sizeof(path), base, "unknown-type.gguf") != 0 || file_size(path, &size) != 0 ||
        nya_format_inspect(path, size, &information) != NYA_FORMAT_UNSUPPORTED) {
        fprintf(stderr, "unknown GGUF encoding was not reported as unsupported\n");
        return -1;
    }
    return 0;
}

/* Require all three deterministic fixtures to validate. */
int main(int argument_count, char **arguments)
{
    if (argument_count != 7) {
        fprintf(stderr, "usage: test_formats GGUF SAFETENSORS ONNX BAD_GGUF BAD_SAFETENSORS BAD_ONNX\n");
        return 2;
    }

    if (inspect(arguments[1], NYA_FORMAT_GGUF, 3, 0) != 0 ||
        inspect(arguments[2], NYA_FORMAT_SAFETENSORS, 1, 1) != 0 ||
        inspect(arguments[3], NYA_FORMAT_ONNX, 8, 0) != 0 ||
        reject(arguments[4]) != 0 ||
        reject(arguments[5]) != 0 ||
        reject(arguments[6]) != 0 || regressions(arguments[1]) != 0 || unicode_model_path(arguments[1]) != 0) {
        return 1;
    }

    return 0;
}
