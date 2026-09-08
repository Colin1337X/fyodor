#ifndef NYA_MULTIMODAL_INTERNAL_H
#define NYA_MULTIMODAL_INTERNAL_H

#include "execution.h"
#include "model.h"

/* Native GGUF projector provider behind the existing named-tensor C API. */
int nya_multimodal_attach(nya_model *model, char *error, size_t capacity);
void nya_multimodal_detach(nya_model *model);
int nya_multimodal_run(nya_model *model, const nya_execution_request *request,
    nya_execution_response *response, char *error, size_t capacity);

#endif
