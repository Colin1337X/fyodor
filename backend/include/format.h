#ifndef NYA_FORMAT_H
#define NYA_FORMAT_H

#include <stdint.h>

/* These are the model containers understood by the metadata layer. */
typedef enum nya_model_format {
    NYA_FORMAT_UNKNOWN = 0,
    NYA_FORMAT_GGUF = 1,
    NYA_FORMAT_SAFETENSORS = 2,
    NYA_FORMAT_ONNX = 3
} nya_model_format;

/* Inspection results keep invalid data distinct from an unknown extension. */
typedef enum nya_format_result {
    NYA_FORMAT_OK = 0,
    NYA_FORMAT_IO_ERROR = -1,
    NYA_FORMAT_UNSUPPORTED = -2,
    NYA_FORMAT_INVALID = -3,
    NYA_FORMAT_RESOURCE_LIMIT = -4
} nya_format_result;

/* Common metadata lets the UI inspect containers through one stable API. */
typedef struct nya_format_info {
    nya_model_format format;
    uint64_t format_version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t data_offset;
    char architecture[128];
    char producer[128];
} nya_format_info;

/* Return the stable lowercase API name for one format. */
const char *nya_model_format_name(nya_model_format format);

/* Inspect a readable regular file whose observed size must match file_size.
 * Container validation is structural; execution providers validate graph/model
 * semantics separately. Any failure clears information, so partial fields are
 * never mistaken for validated metadata. Unknown GGUF layouts are unsupported. */
nya_format_result nya_format_inspect(
    const char *path,
    uint64_t file_size,
    nya_format_info *information
);

#endif
