#include "model.h"

#include "execution.h"
#include "generation.h"
#include "file.h"

#include <stdio.h>
#include <string.h>

/* Read size and readability from one opened file, using the same UTF-8 path
 * conversion as the container inspector. No separate path stat is required. */
static int nya_model_file_size(const char *path, uint64_t *file_size)
{
    FILE *file = nya_file_open_read(path);
    int result;
    if (file == NULL) return -1;
    result = nya_file_regular_size(file, file_size);
    fclose(file);
    return result;
}
/* Reset a registry to its empty initial state. */
void nya_model_registry_init(nya_model_registry *registry)
{
    /* Cleanup paths may pass null, so keep this operation defensive. */
    if (registry == NULL) {
        return;
    }

    /* Zeroing clears paths, IDs, and the active entry count. */
    memset(registry, 0, sizeof(*registry));

    /* Model ID zero is reserved to mean "no model." */
    registry->next_id = 1;
}

/* Detach every live execution session before resetting the complete registry. */
void nya_model_registry_shutdown(nya_model_registry *registry)
{
    size_t index;

    if (registry == NULL) {
        return;
    }

    for (index = 0; index < registry->count; ++index) {
        nya_generation_detach(&registry->models[index]);
        nya_execution_detach(&registry->models[index]);
    }
    nya_execution_global_shutdown();
    nya_model_registry_init(registry);
}

/* Find a record with a stable ID using a deliberately simple linear scan. */
const nya_model *nya_model_find(const nya_model_registry *registry, uint64_t model_id)
{
    size_t index;

    /* Zero is never a valid model ID. */
    if (registry == NULL || model_id == 0) {
        return NULL;
    }

    /* At most 32 records exist, so a linear scan is clearer than a hash table. */
    for (index = 0; index < registry->count; ++index) {
        if (registry->models[index].id == model_id) {
            return &registry->models[index];
        }
    }

    return NULL;
}

/* Validate one model container and attach a compatible execution provider. */
nya_model_result nya_model_load(
    nya_model_registry *registry,
    const char *path,
    const nya_model **loaded_model
)
{
    size_t path_length;
    size_t index;
    uint64_t file_size;
    nya_format_info format_information;
    nya_format_result format_result;
    nya_model *model;

    /* Clear the optional output before any failure can occur. */
    if (loaded_model != NULL) {
        *loaded_model = NULL;
    }

    /* The registry and path are both mandatory inputs. */
    if (registry == NULL || path == NULL || path[0] == '\0') {
        return NYA_MODEL_INVALID_ARGUMENT;
    }

    /* Measure once and reject paths that cannot fit with a terminating byte. */
    path_length = strlen(path);
    if (path_length >= NYA_MODEL_PATH_LIMIT) {
        return NYA_MODEL_PATH_TOO_LONG;
    }

    /* Repeated loads are idempotent and return the existing record. */
    for (index = 0; index < registry->count; ++index) {
        if (strcmp(registry->models[index].path, path) == 0) {
            if (loaded_model != NULL) {
                *loaded_model = &registry->models[index];
            }

            return NYA_MODEL_OK;
        }
    }

    /* Never write past storage or reuse ID zero after the 64-bit sequence wraps. */
    if (registry->count >= NYA_MODEL_LIMIT || registry->next_id == 0) {
        return NYA_MODEL_LIMIT_REACHED;
    }

    /* Confirm the path names a readable regular file and capture its size. */
    if (nya_model_file_size(path, &file_size) != 0) {
        return NYA_MODEL_NOT_FOUND;
    }

    /* Validate the actual container before publishing it in the registry. */
    format_result = nya_format_inspect(path, file_size, &format_information);
    if (format_result == NYA_FORMAT_UNSUPPORTED) {
        return NYA_MODEL_FORMAT_UNSUPPORTED;
    }
    if (format_result == NYA_FORMAT_RESOURCE_LIMIT) {
        return NYA_MODEL_RESOURCE_LIMIT;
    }
    if (format_result != NYA_FORMAT_OK) {
        return NYA_MODEL_FORMAT_INVALID;
    }

    /* Fill the next unused record only after validation has succeeded. */
    model = &registry->models[registry->count];
    memset(model, 0, sizeof(*model));
    model->id = registry->next_id;
    model->file_size = file_size;
    model->format = format_information.format;
    model->format_version = format_information.format_version;
    model->tensor_count = format_information.tensor_count;
    model->metadata_count = format_information.metadata_count;
    model->data_offset = format_information.data_offset;
    memcpy(model->architecture, format_information.architecture, sizeof(model->architecture));
    memcpy(model->producer, format_information.producer, sizeof(model->producer));
    memcpy(model->path, path, path_length + 1);
    model->inference_supported = 0;
    model->execution_context = NULL;
    model->execution_error[0] = '\0';
    nya_execution_attach(model, model->execution_error, sizeof(model->execution_error));
    model->generation_supported = 0;
    model->draft_supported = 0;
    model->generation_context = NULL;
    model->generation_error[0] = '\0';
    nya_generation_attach(model, model->generation_error, sizeof(model->generation_error));

    /* Publish the record by increasing the count last. */
    registry->count += 1;
    registry->next_id += 1;

    if (loaded_model != NULL) {
        *loaded_model = model;
    }

    return NYA_MODEL_OK;
}

/* Remove one model while keeping the compact array representation. */
nya_model_result nya_model_unload(nya_model_registry *registry, uint64_t model_id)
{
    size_t index;

    /* Reject invalid inputs before scanning. */
    if (registry == NULL || model_id == 0) {
        return NYA_MODEL_INVALID_ARGUMENT;
    }

    /* Locate the record with the requested stable ID. */
    for (index = 0; index < registry->count; ++index) {
        if (registry->models[index].id == model_id) {
            size_t remaining;

            /* Release provider state before moving or erasing the record. */
            nya_generation_detach(&registry->models[index]);
            nya_execution_detach(&registry->models[index]);

            /* Shift later records left so the valid range stays contiguous. */
            remaining = registry->count - index - 1;
            if (remaining > 0) {
                memmove(
                    &registry->models[index],
                    &registry->models[index + 1],
                    remaining * sizeof(registry->models[0])
                );
            }

            /* Erase the now-unused last record before reducing the count. */
            memset(&registry->models[registry->count - 1], 0, sizeof(registry->models[0]));
            registry->count -= 1;
            return NYA_MODEL_OK;
        }
    }

    return NYA_MODEL_ID_NOT_FOUND;
}
