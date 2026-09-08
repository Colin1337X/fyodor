#include "format_internal.h"

#include <stdint.h>
#include <string.h>

/* Protobuf varints contain at most ten bytes for a 64-bit value. */
static int nya_protobuf_varint(nya_reader *reader, uint64_t *value)
{
    uint64_t result;
    unsigned int shift;

    result = 0;
    for (shift = 0; shift < 70; shift += 7) {
        unsigned char byte;

        if (nya_reader_read(reader, &byte, 1) != 0) {
            return -1;
        }

        if (shift == 63 && (byte & 0xfeU) != 0) {
            return -1;
        }

        result |= (uint64_t)(byte & 0x7fU) << shift;
        if ((byte & 0x80U) == 0) {
            *value = result;
            return 0;
        }
    }

    return -1;
}

/* Skip one protobuf field payload for the four supported modern wire types. */
static int nya_protobuf_skip(nya_reader *reader, unsigned int wire_type)
{
    uint64_t length;

    switch (wire_type) {
        case 0:
            return nya_protobuf_varint(reader, &length);
        case 1:
            return nya_reader_skip(reader, 8);
        case 2:
            if (nya_protobuf_varint(reader, &length) != 0) {
                return -1;
            }
            return nya_reader_skip(reader, length);
        case 5:
            return nya_reader_skip(reader, 4);
        default:
            return -1;
    }
}

/* Read a bounded protobuf string and skip any value too large for common metadata. */
static int nya_protobuf_string(nya_reader *reader, char *output, size_t capacity)
{
    uint64_t length;

    if (nya_protobuf_varint(reader, &length) != 0) {
        return -1;
    }

    if (length >= capacity) {
        output[0] = '\0';
        return nya_reader_skip(reader, length);
    }

    if (nya_reader_read(reader, output, (size_t)length) != 0) {
        return -1;
    }
    output[length] = '\0';
    return 0;
}

/* Validate the top-level ONNX ModelProto envelope without requiring protobuf. */
nya_format_result nya_onnx_inspect(nya_reader *reader, nya_format_info *information)
{
    int has_ir_version;
    int has_graph;
    uint64_t opset_count;

    has_ir_version = 0;
    has_graph = 0;
    opset_count = 0;
    information->format = NYA_FORMAT_ONNX;

    while (reader->position < reader->size) {
        uint64_t tag;
        uint64_t field_number;
        unsigned int wire_type;

        if (nya_protobuf_varint(reader, &tag) != 0 || tag == 0) {
            return NYA_FORMAT_INVALID;
        }

        field_number = tag >> 3;
        wire_type = (unsigned int)(tag & 7U);
        /* Protobuf tags have a nonzero 29-bit field number. Checking only tag != 0
         * lets wire types 1/2/5 masquerade as the forbidden field number zero. */
        if (field_number == 0 || field_number > 0x1fffffffU) {
            return NYA_FORMAT_INVALID;
        }

        if (field_number == 1) {
            uint64_t version;

            if (wire_type != 0 || has_ir_version || nya_protobuf_varint(reader, &version) != 0 ||
                version == 0 || version > INT64_MAX) {
                return NYA_FORMAT_INVALID;
            }
            information->format_version = version;
            has_ir_version = 1;
        } else if (field_number == 2) {
            if (wire_type != 2 || nya_protobuf_string(reader, information->producer, sizeof(information->producer)) != 0) {
                return NYA_FORMAT_INVALID;
            }
        } else if (field_number == 7) {
            uint64_t graph_length;

            if (wire_type != 2 || has_graph || nya_protobuf_varint(reader, &graph_length) != 0 || graph_length == 0 ||
                nya_reader_skip(reader, graph_length) != 0) {
                return NYA_FORMAT_INVALID;
            }
            has_graph = 1;
        } else if (field_number == 8) {
            uint64_t opset_length;

            if (wire_type != 2 || nya_protobuf_varint(reader, &opset_length) != 0 ||
                nya_reader_skip(reader, opset_length) != 0 || opset_count == UINT64_MAX) {
                return NYA_FORMAT_INVALID;
            }
            opset_count += 1;
        } else if (nya_protobuf_skip(reader, wire_type) != 0) {
            return NYA_FORMAT_INVALID;
        }
    }

    /* IR version 3 introduced mandatory operator-set imports. */
    if (!has_ir_version || !has_graph || (information->format_version >= 3 && opset_count == 0)) {
        return NYA_FORMAT_INVALID;
    }

    information->metadata_count = opset_count;
    return NYA_FORMAT_OK;
}
