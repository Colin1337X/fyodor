#ifndef NYA_EXECUTION_H
#define NYA_EXECUTION_H

#include <stddef.h>
#include <stdint.h>

/* Forward declaration avoids a circular public-header dependency. */
struct nya_model;

/* Values intentionally match ONNX TensorProto element type identifiers. */
typedef enum nya_tensor_data_type {
    NYA_TENSOR_UNDEFINED = 0,
    NYA_TENSOR_FLOAT32 = 1,
    NYA_TENSOR_UINT8 = 2,
    NYA_TENSOR_INT8 = 3,
    NYA_TENSOR_UINT16 = 4,
    NYA_TENSOR_INT16 = 5,
    NYA_TENSOR_INT32 = 6,
    NYA_TENSOR_INT64 = 7,
    NYA_TENSOR_STRING = 8,
    NYA_TENSOR_BOOL = 9,
    NYA_TENSOR_FLOAT16 = 10,
    NYA_TENSOR_FLOAT64 = 11,
    NYA_TENSOR_UINT32 = 12,
    NYA_TENSOR_UINT64 = 13,
    NYA_TENSOR_COMPLEX64 = 14,
    NYA_TENSOR_COMPLEX128 = 15,
    NYA_TENSOR_BFLOAT16 = 16
} nya_tensor_data_type;

/* One request currently carries one named dense CPU tensor to one named output. */
typedef struct nya_execution_request {
    const char *input_name;
    nya_tensor_data_type input_type;
    const int64_t *input_shape;
    size_t input_rank;
    const void *input_data;
    size_t input_data_size;
    const char *output_name;

    /* Providers must reject an output larger than this caller-owned limit. */
    size_t max_output_bytes;
} nya_execution_request;

/* The execution provider allocates one dense output tensor owned by the caller. */
typedef struct nya_execution_response {
    nya_tensor_data_type output_type;
    int64_t output_shape[16];
    size_t output_rank;
    void *output_data;
    size_t output_data_size;
} nya_execution_response;

/* Attach a real execution session when the model and provider are compatible. */
int nya_execution_attach(struct nya_model *model, char *error, size_t error_capacity);

/* Release one model execution session. */
void nya_execution_detach(struct nya_model *model);

/* Run one named input tensor through one named model output. */
int nya_execution_run(
    struct nya_model *model,
    const nya_execution_request *request,
    nya_execution_response *response,
    char *error,
    size_t error_capacity
);

/* Release response storage allocated by the execution provider. */
void nya_execution_response_free(nya_execution_response *response);

/* Release process-wide execution-provider resources after all models detach. */
void nya_execution_global_shutdown(void);

/* Convert public tensor types to and from stable HTTP header names. */
const char *nya_tensor_data_type_name(nya_tensor_data_type type);
int nya_tensor_data_type_parse(const char *name, nya_tensor_data_type *type);

#endif
