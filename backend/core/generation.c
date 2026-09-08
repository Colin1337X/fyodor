#include "generation.h"

#include "llm_internal.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Attach only the native GGUF LLaMA provider; other formats remain inspectable. */
int nya_generation_attach(nya_model *model, char *error, size_t error_capacity)
{
    nya_llm_context *context;

    if (model == NULL || model->format != NYA_FORMAT_GGUF ||
        (strcmp(model->architecture, "llama") != 0 && strcmp(model->architecture, "gemma4") != 0 &&
         strcmp(model->architecture, "gemma4-assistant") != 0)) {
        return 1;
    }
    /* Repeated attachment is idempotent; never leak a live mapped model. */
    if (model->generation_context != NULL) return model->generation_supported || model->draft_supported ? 0 : -1;

    context = NULL;
    if (nya_llm_load(model->path, model->file_size, &context, error, error_capacity) != 0) {
        return -1;
    }
    model->generation_context = context;
    model->generation_supported = !context->is_assistant;
    model->draft_supported = 1;
    if (context->is_assistant && error != NULL && error_capacity > 0)
        snprintf(error, error_capacity, "MTP assistant: use as draft_model_id with its compatible Gemma 4 target");
    return 0;
}

/* Release a native generation context without touching another execution provider. */
void nya_generation_detach(nya_model *model)
{
    if (model == NULL || model->generation_context == NULL) return;
    nya_llm_free((nya_llm_context *)model->generation_context);
    model->generation_context = NULL;
    model->generation_supported = 0;
    model->draft_supported = 0;
}

/* Dispatch one request only when the model owns a valid generation context. */
int nya_generation_run(
    nya_model *model,
    const nya_generation_request *request,
    nya_generation_response *response,
    char *error,
    size_t error_capacity
)
{
    if (response != NULL) memset(response, 0, sizeof(*response));
    if (model == NULL || request == NULL || response == NULL ||
        !model->generation_supported || model->generation_context == NULL) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "this model has no active text-generation provider");
            error[error_capacity - 1] = '\0';
        }
        return -1;
    }
    if (request->draft_model != NULL && (!request->draft_model->draft_supported ||
        request->draft_model->generation_context == NULL)) {
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "the draft has no native generation provider");
        return -1;
    }
    return nya_llm_generate_draft(
        (const nya_llm_context *)model->generation_context,
        request->draft_model == NULL ? NULL : (const nya_llm_context *)request->draft_model->generation_context,
        request,
        response,
        error,
        error_capacity
    );
}

/* Release caller-owned generated text and clear stale metadata. */
void nya_generation_response_free(nya_generation_response *response)
{
    if (response == NULL) return;
    free(response->text);
    memset(response, 0, sizeof(*response));
}

/* Return stable protocol text for every generation stop reason. */
const char *nya_generation_stop_reason_name(nya_generation_stop_reason reason)
{
    switch (reason) {
        case NYA_GENERATION_STOP_EOS: return "eos";
        case NYA_GENERATION_STOP_CONTEXT: return "context";
        case NYA_GENERATION_STOP_LENGTH:
        default:
            return "length";
    }
}
