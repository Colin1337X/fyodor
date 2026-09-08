#ifndef NYA_FORMAT_INTERNAL_H
#define NYA_FORMAT_INTERNAL_H

#include "format.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* All binary format readers use the same checked file cursor. */
typedef struct nya_reader {
    FILE *file;
    uint64_t size;
    uint64_t position;
} nya_reader;

/* Read exactly the requested bytes without crossing the known file size. */
int nya_reader_read(nya_reader *reader, void *output, size_t length);

/* Skip bounded data with 64-bit seek where available and a buffered read fallback. */
int nya_reader_skip(nya_reader *reader, uint64_t length);

/* Decode fixed-width little-endian integers. */
int nya_reader_u32_le(nya_reader *reader, uint32_t *value);
int nya_reader_u64_le(nya_reader *reader, uint64_t *value);

/* Each format-specific inspector receives a fresh reader at file position zero. */
nya_format_result nya_gguf_inspect(nya_reader *reader, nya_format_info *information);
nya_format_result nya_safetensors_inspect(nya_reader *reader, nya_format_info *information);
nya_format_result nya_onnx_inspect(nya_reader *reader, nya_format_info *information);

#endif
