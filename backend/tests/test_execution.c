#include "execution.h"
#include "model.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Load and execute the tiny Identity graph through the real provider. */
int main(int argument_count, char **arguments)
{
    nya_model_registry registry;
    const nya_model *loaded_model;
    nya_execution_request request;
    nya_execution_response response;
    int64_t shape[1];
    float input;
    float output;
    char error[512];
    int result;

    if (argument_count != 2) {
        fprintf(stderr, "usage: test_execution IDENTITY_ONNX\n");
        return 2;
    }

    nya_model_registry_init(&registry);
    loaded_model = NULL;
    if (nya_model_load(&registry, arguments[1], &loaded_model) != NYA_MODEL_OK ||
        loaded_model == NULL || !loaded_model->inference_supported) {
        fprintf(
            stderr,
            "ONNX execution session did not load: %s\n",
            loaded_model == NULL ? "no model" : loaded_model->execution_error
        );
        nya_model_registry_shutdown(&registry);
        return 1;
    }

    shape[0] = 1;
    input = 42.25f;
    memset(&request, 0, sizeof(request));
    request.input_name = "x";
    request.input_type = NYA_TENSOR_FLOAT32;
    request.input_shape = shape;
    request.input_rank = 1;
    request.input_data = &input;
    request.input_data_size = sizeof(input);
    request.output_name = "y";
    request.max_output_bytes = 1024;
    memset(&response, 0, sizeof(response));
    error[0] = '\0';

    result = nya_execution_run(
        (nya_model *)loaded_model,
        &request,
        &response,
        error,
        sizeof(error)
    );
    if (result != 0 || response.output_type != NYA_TENSOR_FLOAT32 ||
        response.output_rank != 1 || response.output_shape[0] != 1 ||
        response.output_data_size != sizeof(output)) {
        fprintf(stderr, "ONNX execution failed: %s\n", error);
        nya_execution_response_free(&response);
        nya_model_registry_shutdown(&registry);
        return 1;
    }

    memcpy(&output, response.output_data, sizeof(output));
    nya_execution_response_free(&response);
    nya_model_registry_shutdown(&registry);

    if (memcmp(&input, &output, sizeof(input)) != 0) {
        fprintf(stderr, "Identity output did not equal its input\n");
        return 1;
    }

    return 0;
}
