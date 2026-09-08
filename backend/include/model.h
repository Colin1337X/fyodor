#ifndef NYA_MODEL_H
#define NYA_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "format.h"

/* A small fixed limit keeps the first registry allocation-free and obvious. */
#define NYA_MODEL_LIMIT 32

/* Paths are copied into owned storage so callers may release their input. */
#define NYA_MODEL_PATH_LIMIT 1024

/* One record describes one model file known to the current process. */
typedef struct nya_model {
    /* IDs are stable for the lifetime of a loaded model. */
    uint64_t id;

    /* The loader owns the validated container path for later provider access. */
    char path[NYA_MODEL_PATH_LIMIT];

    /* File size is useful metadata and proves the file was readable at load time. */
    uint64_t file_size;

    /* Container format is detected from content plus a conservative extension rule. */
    nya_model_format format;

    /* This is GGUF version, SafeTensors format version, or ONNX IR version. */
    uint64_t format_version;

    /* Common container counts help the UI inspect large files without loading weights. */
    uint64_t tensor_count;
    uint64_t metadata_count;

    /* Data offset points to tensor bytes for containers that define one. */
    uint64_t data_offset;

    /* Formats may expose an architecture or producing tool. */
    char architecture[128];
    char producer[128];

    /* This flag is true only when a configured execution provider can run the model. */
    int inference_supported;

    /* Execution providers may attach one private session without leaking their ABI. */
    void *execution_context;

    /* Session creation failures are visible without making container loading fail. */
    char execution_error[256];

    /* Text generation is separate from the generic named-tensor interface. */
    int generation_supported;
    /* MTP assistants require a target's hidden states and KV cache. They may be
       selected as a draft but cannot generate independently. */
    int draft_supported;

    /* A supported GGUF model owns one private tokenizer and transformer context. */
    void *generation_context;

    /* Unsupported architectures and quantizations remain inspectable with this reason. */
    char generation_error[256];
} nya_model;

/* The registry owns all loaded model records. */
typedef struct nya_model_registry {
    /* Fixed storage avoids hidden allocations in this early engine layer. */
    nya_model models[NYA_MODEL_LIMIT];

    /* Count is the number of valid entries at the start of the array. */
    size_t count;

    /* IDs increase monotonically and are never reused during one process run. */
    uint64_t next_id;
} nya_model_registry;

/* Model operations return one of these explicit results. */
typedef enum nya_model_result {
    NYA_MODEL_OK = 0,
    NYA_MODEL_INVALID_ARGUMENT = -1,
    NYA_MODEL_PATH_TOO_LONG = -2,
    NYA_MODEL_NOT_FOUND = -3,
    NYA_MODEL_LIMIT_REACHED = -4,
    NYA_MODEL_ID_NOT_FOUND = -5,
    NYA_MODEL_FORMAT_UNSUPPORTED = -6,
    NYA_MODEL_FORMAT_INVALID = -7,
    NYA_MODEL_RESOURCE_LIMIT = -8
} nya_model_result;

/* Initialize an empty registry. */
void nya_model_registry_init(nya_model_registry *registry);

/* Forget every registry entry during process shutdown. */
void nya_model_registry_shutdown(nya_model_registry *registry);

/* Register a readable model file and return its record. */
nya_model_result nya_model_load(
    nya_model_registry *registry,
    const char *path,
    const nya_model **loaded_model
);

/* Remove a model record by its stable ID. */
nya_model_result nya_model_unload(nya_model_registry *registry, uint64_t model_id);

/* Find an already loaded model by ID. */
const nya_model *nya_model_find(const nya_model_registry *registry, uint64_t model_id);

#endif
