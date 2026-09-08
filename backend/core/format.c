#include "format_internal.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include "file.h"

/* Read exact data and advance the logical 64-bit cursor only on success. */
int nya_reader_read(nya_reader *reader, void *output, size_t length)
{
    if (reader == NULL || reader->file == NULL || reader->position > reader->size ||
        (length != 0 && output == NULL) ||
        (uint64_t)length > reader->size - reader->position) {
        return -1;
    }

    if (length > 0 && fread(output, 1, length, reader->file) != length) {
        return -1;
    }

    reader->position += (uint64_t)length;
    return 0;
}

/* Seeking over large opaque payloads avoids reading every tensor byte during
 * metadata inspection. Small skips stay buffered; narrow-offset platforms keep
 * a bounded read fallback. The range and actual file size are checked separately. */
int nya_reader_skip(nya_reader *reader, uint64_t length)
{
    unsigned char discard[4096];

    if (reader == NULL || reader->file == NULL || reader->position > reader->size ||
        length > reader->size - reader->position) {
        return -1;
    }

    if (length >= sizeof(discard) && length <= INT64_MAX) {
#ifdef _WIN32
        if (_fseeki64(reader->file, (int64_t)length, SEEK_CUR) != 0) return -1;
        reader->position += length;
        return 0;
#else
        if (sizeof(off_t) >= sizeof(int64_t)) {
            if (fseeko(reader->file, (off_t)length, SEEK_CUR) != 0) return -1;
            reader->position += length;
            return 0;
        }
#endif
    }

    while (length > 0) {
        size_t chunk;

        chunk = length > sizeof(discard) ? sizeof(discard) : (size_t)length;
        if (nya_reader_read(reader, discard, chunk) != 0) {
            return -1;
        }
        length -= (uint64_t)chunk;
    }

    return 0;
}

/* Decode one little-endian 32-bit field without host alignment assumptions. */
int nya_reader_u32_le(nya_reader *reader, uint32_t *value)
{
    unsigned char bytes[4];

    if (value == NULL || nya_reader_read(reader, bytes, sizeof(bytes)) != 0) {
        return -1;
    }

    *value = (uint32_t)bytes[0] |
             ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) |
             ((uint32_t)bytes[3] << 24);
    return 0;
}

/* Decode one little-endian 64-bit field without host alignment assumptions. */
int nya_reader_u64_le(nya_reader *reader, uint64_t *value)
{
    unsigned char bytes[8];
    size_t index;
    uint64_t result;

    if (value == NULL || nya_reader_read(reader, bytes, sizeof(bytes)) != 0) {
        return -1;
    }

    result = 0;
    for (index = 0; index < sizeof(bytes); ++index) {
        result |= (uint64_t)bytes[index] << (index * 8);
    }

    *value = result;
    return 0;
}

/* Compare a path extension using ASCII case folding. */
static int nya_extension_is(const char *path, const char *extension)
{
    size_t path_length;
    size_t extension_length;
    size_t index;

    path_length = strlen(path);
    extension_length = strlen(extension);
    if (path_length < extension_length) {
        return 0;
    }

    for (index = 0; index < extension_length; ++index) {
        unsigned char left;
        unsigned char right;

        left = (unsigned char)path[path_length - extension_length + index];
        right = (unsigned char)extension[index];
        if (tolower(left) != tolower(right)) {
            return 0;
        }
    }

    return 1;
}

/* Return names that remain stable even when internal enums grow. */
const char *nya_model_format_name(nya_model_format format)
{
    switch (format) {
        case NYA_FORMAT_GGUF: return "gguf";
        case NYA_FORMAT_SAFETENSORS: return "safetensors";
        case NYA_FORMAT_ONNX: return "onnx";
        default: return "unknown";
    }
}

/* Detect formats conservatively and delegate full validation. */
nya_format_result nya_format_inspect(
    const char *path,
    uint64_t file_size,
    nya_format_info *information
)
{
    FILE *file;
    unsigned char prefix[9];
    size_t prefix_length;
    nya_reader reader;
    nya_format_result result;
    uint64_t actual_size;

    if (information == NULL) {
        return NYA_FORMAT_INVALID;
    }

    memset(information, 0, sizeof(*information));
    if (path == NULL) return NYA_FORMAT_INVALID;
    file = nya_file_open_read(path);
    if (file == NULL) {
        return NYA_FORMAT_IO_ERROR;
    }
    if (nya_file_regular_size(file, &actual_size) != 0 || actual_size != file_size) {
        fclose(file);
        return NYA_FORMAT_IO_ERROR;
    }

    prefix_length = file_size < sizeof(prefix) ? (size_t)file_size : sizeof(prefix);
    if (prefix_length > 0 && fread(prefix, 1, prefix_length, file) != prefix_length) {
        fclose(file);
        return NYA_FORMAT_IO_ERROR;
    }
    rewind(file);

    reader.file = file;
    reader.size = file_size;
    reader.position = 0;

    if (prefix_length >= 4 && memcmp(prefix, "GGUF", 4) == 0) {
        result = nya_gguf_inspect(&reader, information);
    } else if (nya_extension_is(path, ".safetensors")) {
        result = nya_safetensors_inspect(&reader, information);
    } else if (nya_extension_is(path, ".onnx")) {
        result = nya_onnx_inspect(&reader, information);
    } else {
        result = NYA_FORMAT_UNSUPPORTED;
    }

    /* A concurrent truncate/append invalidates bounds captured before parsing,
     * including ranges skipped with seek. Never publish that stale inspection. */
    if (nya_file_regular_size(file, &actual_size) != 0 || actual_size != file_size) {
        result = NYA_FORMAT_IO_ERROR;
    }
    fclose(file);
    /* Failed inspection must not publish plausible but partially validated metadata. */
    if (result != NYA_FORMAT_OK) {
        memset(information, 0, sizeof(*information));
    }
    return result;
}
