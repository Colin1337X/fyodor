#include "execution.h"

#include "model.h"
#include "multimodal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Return one stable lowercase name for binary inference headers. */
const char *nya_tensor_data_type_name(nya_tensor_data_type type)
{
    switch (type) {
        case NYA_TENSOR_FLOAT32: return "float32";
        case NYA_TENSOR_UINT8: return "uint8";
        case NYA_TENSOR_INT8: return "int8";
        case NYA_TENSOR_UINT16: return "uint16";
        case NYA_TENSOR_INT16: return "int16";
        case NYA_TENSOR_INT32: return "int32";
        case NYA_TENSOR_INT64: return "int64";
        case NYA_TENSOR_BOOL: return "bool";
        case NYA_TENSOR_FLOAT16: return "float16";
        case NYA_TENSOR_FLOAT64: return "float64";
        case NYA_TENSOR_UINT32: return "uint32";
        case NYA_TENSOR_UINT64: return "uint64";
        case NYA_TENSOR_COMPLEX64: return "complex64";
        case NYA_TENSOR_COMPLEX128: return "complex128";
        case NYA_TENSOR_BFLOAT16: return "bfloat16";
        default: return "undefined";
    }
}

/* Parse one exact lowercase name without aliases that could become ambiguous. */
int nya_tensor_data_type_parse(const char *name, nya_tensor_data_type *type)
{
    nya_tensor_data_type candidate;

    if (name == NULL || type == NULL) {
        return -1;
    }

    for (candidate = NYA_TENSOR_FLOAT32; candidate <= NYA_TENSOR_BFLOAT16; ++candidate) {
        if (candidate == NYA_TENSOR_STRING) {
            continue;
        }
        if (strcmp(name, nya_tensor_data_type_name(candidate)) == 0) {
            *type = candidate;
            return 0;
        }
    }

    return -1;
}

#ifdef NYA_ENABLE_ONNXRUNTIME
/* ONNX Runtime is consumed strictly through its stable plain-C ABI. */
#include <onnxruntime_c_api.h>

#ifdef _WIN32
#include <windows.h>
#include <wchar.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

/* One context owns one ONNX Runtime inference session. */
typedef struct nya_onnx_context {
    OrtSession *session;
} nya_onnx_context;

/* The process shares one API table and environment across model sessions. */
static const OrtApi *nya_ort_api = NULL;
static OrtEnv *nya_ort_environment = NULL;
static void *nya_ort_library = NULL;
static int nya_ort_initialization_attempted = 0;

/* Load only the packaged runtime beside the executable to avoid search-path ambiguity. */
#ifdef _WIN32
static HMODULE nya_ort_load_library(void)
{
    wchar_t executable_path[32768];
    static const wchar_t library_name[] = L"onnxruntime.dll";
    DWORD length;
    size_t name_length;

    length = GetModuleFileNameW(
        NULL,
        executable_path,
        (DWORD)(sizeof(executable_path) / sizeof(executable_path[0]))
    );
    if (length == 0 || length >= sizeof(executable_path) / sizeof(executable_path[0])) {
        return NULL;
    }

    /* Keep the final separator and replace only the executable filename. */
    while (length > 0 && executable_path[length - 1] != L'\\' && executable_path[length - 1] != L'/') {
        length -= 1;
    }
    name_length = sizeof(library_name) / sizeof(library_name[0]);
    if (length == 0 || (size_t)length + name_length > sizeof(executable_path) / sizeof(executable_path[0])) {
        return NULL;
    }

    memcpy(executable_path + length, library_name, name_length * sizeof(library_name[0]));
    return LoadLibraryW(executable_path);
}
#else
#define NYA_EXECUTABLE_PATH_LIMIT 4096

static void *nya_ort_load_library(void)
{
    char executable_path[NYA_EXECUTABLE_PATH_LIMIT];
#ifdef __APPLE__
    uint32_t capacity;

    capacity = (uint32_t)sizeof(executable_path);
    if (_NSGetExecutablePath(executable_path, &capacity) != 0) {
        return NULL;
    }
#else
    ssize_t path_length;

    path_length = readlink("/proc/self/exe", executable_path, sizeof(executable_path) - 1);
    if (path_length <= 0 || (size_t)path_length >= sizeof(executable_path)) {
        return NULL;
    }
    executable_path[path_length] = '\0';
#endif

    {
        const char *library_name;
        char *separator;
        size_t directory_length;
        size_t library_length;

#ifdef __APPLE__
        library_name = "libonnxruntime.dylib";
#else
        library_name = "libonnxruntime.so";
#endif
        separator = strrchr(executable_path, '/');
        if (separator == NULL) {
            return NULL;
        }

        directory_length = (size_t)(separator + 1 - executable_path);
        library_length = strlen(library_name) + 1;
        if (directory_length + library_length > sizeof(executable_path)) {
            return NULL;
        }

        memcpy(executable_path + directory_length, library_name, library_length);
        return dlopen(executable_path, RTLD_NOW | RTLD_LOCAL);
    }
}
#endif

/* Store one provider error without overflowing caller storage. */
static void nya_execution_error(char *error, size_t capacity, const char *format, ...)
{
    va_list arguments;

    if (error == NULL || capacity == 0) {
        return;
    }

    va_start(arguments, format);
    vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
    error[capacity - 1] = '\0';
}

/* Convert an ONNX Runtime status into a copied message and release the status. */
static int nya_ort_status(OrtStatus *status, char *error, size_t error_capacity)
{
    const char *message;

    if (status == NULL) {
        return 0;
    }

    message = nya_ort_api->GetErrorMessage(status);
    nya_execution_error(error, error_capacity, "%s", message == NULL ? "ONNX Runtime error" : message);
    nya_ort_api->ReleaseStatus(status);
    return -1;
}

/* Load the shared library and create the process-wide CPU runtime environment. */
static int nya_ort_initialize(char *error, size_t error_capacity)
{
    const OrtApiBase *api_base;
    typedef const OrtApiBase *(ORT_API_CALL *get_api_base_function)(void);
    get_api_base_function get_api_base;

    if (nya_ort_api != NULL && nya_ort_environment != NULL) {
        return 0;
    }
    if (nya_ort_initialization_attempted) {
        nya_execution_error(error, error_capacity, "ONNX Runtime is not available");
        return -1;
    }
    nya_ort_initialization_attempted = 1;

#ifdef _WIN32
    {
        HMODULE library;
        FARPROC symbol;

        library = nya_ort_load_library();
        if (library == NULL) {
            nya_execution_error(error, error_capacity, "onnxruntime.dll was not found beside the backend");
            return -1;
        }
        symbol = GetProcAddress(library, "OrtGetApiBase");
        if (symbol == NULL || sizeof(symbol) != sizeof(get_api_base)) {
            FreeLibrary(library);
            nya_execution_error(error, error_capacity, "onnxruntime.dll does not export OrtGetApiBase");
            return -1;
        }
        memcpy(&get_api_base, &symbol, sizeof(get_api_base));
        nya_ort_library = library;
    }
#else
    {
        void *symbol;

        nya_ort_library = nya_ort_load_library();
        if (nya_ort_library == NULL) {
            nya_execution_error(error, error_capacity, "the ONNX Runtime shared library was not found beside the backend");
            return -1;
        }
        symbol = dlsym(nya_ort_library, "OrtGetApiBase");
        if (symbol == NULL || sizeof(symbol) != sizeof(get_api_base)) {
            dlclose(nya_ort_library);
            nya_ort_library = NULL;
            nya_execution_error(error, error_capacity, "ONNX Runtime does not export OrtGetApiBase");
            return -1;
        }
        memcpy(&get_api_base, &symbol, sizeof(get_api_base));
    }
#endif

    api_base = get_api_base();
    if (api_base == NULL) {
        nya_execution_error(error, error_capacity, "OrtGetApiBase returned null");
        return -1;
    }

    nya_ort_api = api_base->GetApi(ORT_API_VERSION);
    if (nya_ort_api == NULL) {
        nya_execution_error(error, error_capacity, "ONNX Runtime does not support the compiled C API version");
        return -1;
    }
    if (nya_ort_status(
            nya_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "fyodor", &nya_ort_environment),
            error,
            error_capacity
        ) != 0) {
        nya_ort_api = NULL;
        return -1;
    }

    return 0;
}

/* Convert the UTF-8 API path into the path type required by ONNX Runtime. */
#ifdef _WIN32
static wchar_t *nya_ort_path(const char *path)
{
    int length;
    wchar_t *wide_path;

    length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (length <= 0) {
        return NULL;
    }

    wide_path = (wchar_t *)malloc((size_t)length * sizeof(*wide_path));
    if (wide_path == NULL) {
        return NULL;
    }

    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide_path, length) != length) {
        free(wide_path);
        return NULL;
    }

    return wide_path;
}
#endif

/* Return the byte width of fixed-size tensor element types supported by this API. */
static size_t nya_tensor_element_size(nya_tensor_data_type type)
{
    switch (type) {
        case NYA_TENSOR_UINT8:
        case NYA_TENSOR_INT8:
        case NYA_TENSOR_BOOL:
            return 1;
        case NYA_TENSOR_UINT16:
        case NYA_TENSOR_INT16:
        case NYA_TENSOR_FLOAT16:
        case NYA_TENSOR_BFLOAT16:
            return 2;
        case NYA_TENSOR_FLOAT32:
        case NYA_TENSOR_INT32:
        case NYA_TENSOR_UINT32:
            return 4;
        case NYA_TENSOR_INT64:
        case NYA_TENSOR_FLOAT64:
        case NYA_TENSOR_UINT64:
        case NYA_TENSOR_COMPLEX64:
            return 8;
        case NYA_TENSOR_COMPLEX128:
            return 16;
        default:
            return 0;
    }
}

/* Attach ONNX Runtime only to ONNX containers; other containers remain inspectable. */
static int nya_ort_attach(nya_model *model, char *error, size_t error_capacity)
{
    OrtSessionOptions *options;
    OrtSession *session;
    nya_onnx_context *context;
    OrtStatus *status;

    if (model == NULL || model->format != NYA_FORMAT_ONNX) {
        return 1;
    }
    /* Loading an already attached model must not orphan its existing session. */
    if (model->execution_context != NULL) return model->inference_supported ? 0 : -1;
    if (nya_ort_initialize(error, error_capacity) != 0) {
        return -1;
    }

    options = NULL;
    session = NULL;
    status = nya_ort_api->CreateSessionOptions(&options);
    if (nya_ort_status(status, error, error_capacity) != 0) {
        return -1;
    }

    status = nya_ort_api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
    if (nya_ort_status(status, error, error_capacity) != 0) {
        nya_ort_api->ReleaseSessionOptions(options);
        return -1;
    }

#ifdef _WIN32
    {
        wchar_t *wide_path;

        wide_path = nya_ort_path(model->path);
        if (wide_path == NULL) {
            nya_ort_api->ReleaseSessionOptions(options);
            nya_execution_error(error, error_capacity, "the ONNX path is not valid UTF-8");
            return -1;
        }
        status = nya_ort_api->CreateSession(nya_ort_environment, wide_path, options, &session);
        free(wide_path);
    }
#else
    status = nya_ort_api->CreateSession(nya_ort_environment, model->path, options, &session);
#endif
    nya_ort_api->ReleaseSessionOptions(options);
    if (nya_ort_status(status, error, error_capacity) != 0) {
        return -1;
    }

    context = (nya_onnx_context *)malloc(sizeof(*context));
    if (context == NULL) {
        nya_ort_api->ReleaseSession(session);
        nya_execution_error(error, error_capacity, "execution session allocation failed");
        return -1;
    }

    context->session = session;
    model->execution_context = context;
    model->inference_supported = 1;
    return 0;
}

/* Release one attached ONNX Runtime session. */
static void nya_ort_detach(nya_model *model)
{
    nya_onnx_context *context;

    if (model == NULL || model->execution_context == NULL) {
        return;
    }

    context = (nya_onnx_context *)model->execution_context;
    nya_ort_api->ReleaseSession(context->session);
    free(context);
    model->execution_context = NULL;
    model->inference_supported = 0;
}

/* Execute one dense named input and copy one dense named output into owned memory. */
static int nya_ort_run(
    nya_model *model,
    const nya_execution_request *request,
    nya_execution_response *response,
    char *error,
    size_t error_capacity
)
{
    nya_onnx_context *context;
    OrtMemoryInfo *memory_info;
    OrtValue *input_value;
    OrtValue *output_value;
    OrtTensorTypeAndShapeInfo *shape_info;
    OrtStatus *status;
    const char *input_names[1];
    const char *output_names[1];
    const OrtValue *input_values[1];
    size_t element_size;
    size_t element_count;
    size_t output_size;
    void *output_pointer;
    size_t index;
    uint64_t input_elements;
    ONNXTensorElementDataType output_type;

    /* A failure at validation must leave the same safely releasable response
       shape as a failure after provider allocation. */
    if (response != NULL) memset(response, 0, sizeof(*response));
    if (model == NULL || request == NULL || response == NULL || !model->inference_supported ||
        model->execution_context == NULL || request->input_name == NULL || request->output_name == NULL ||
        request->input_shape == NULL || request->input_rank == 0 || request->input_rank > 16 ||
        request->input_data == NULL) {
        nya_execution_error(error, error_capacity, "invalid or unsupported execution request");
        return -1;
    }

    element_size = nya_tensor_element_size(request->input_type);
    input_elements = 1;
    if (element_size == 0) {
        nya_execution_error(error, error_capacity, "unsupported input tensor type");
        return -1;
    }
    for (index = 0; index < request->input_rank; ++index) {
        if (request->input_shape[index] <= 0 ||
            input_elements > UINT64_MAX / (uint64_t)request->input_shape[index]) {
            nya_execution_error(error, error_capacity, "invalid input tensor shape");
            return -1;
        }
        input_elements *= (uint64_t)request->input_shape[index];
    }
    if (input_elements > SIZE_MAX / element_size ||
        (size_t)input_elements * element_size != request->input_data_size) {
        nya_execution_error(error, error_capacity, "input byte count does not match shape and type");
        return -1;
    }

    memset(response, 0, sizeof(*response));
    context = (nya_onnx_context *)model->execution_context;
    memory_info = NULL;
    input_value = NULL;
    output_value = NULL;
    shape_info = NULL;

    status = nya_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;

    status = nya_ort_api->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void *)request->input_data,
        request->input_data_size,
        request->input_shape,
        request->input_rank,
        (ONNXTensorElementDataType)request->input_type,
        &input_value
    );
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;

    input_names[0] = request->input_name;
    output_names[0] = request->output_name;
    input_values[0] = input_value;
    status = nya_ort_api->Run(
        context->session,
        NULL,
        input_names,
        input_values,
        1,
        output_names,
        1,
        &output_value
    );
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;

    status = nya_ort_api->GetTensorTypeAndShape(output_value, &shape_info);
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;
    /* Use the ABI's enum storage rather than aliasing a pointer to our own enum;
       compatible numeric identifiers do not make distinct C enum types aliasable. */
    status = nya_ort_api->GetTensorElementType(shape_info, &output_type);
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;
    response->output_type = (nya_tensor_data_type)output_type;
    status = nya_ort_api->GetDimensionsCount(shape_info, &response->output_rank);
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;
    if (response->output_rank > 16) {
        nya_execution_error(error, error_capacity, "output tensor rank exceeds the supported limit of 16");
        goto failure;
    }
    status = nya_ort_api->GetDimensions(shape_info, response->output_shape, response->output_rank);
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;
    status = nya_ort_api->GetTensorShapeElementCount(shape_info, &element_count);
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;

    element_size = nya_tensor_element_size(response->output_type);
    if (element_size == 0 || element_count > SIZE_MAX / element_size) {
        nya_execution_error(error, error_capacity, "unsupported or oversized output tensor");
        goto failure;
    }
    output_size = element_count * element_size;
    if (request->max_output_bytes == 0 || output_size > request->max_output_bytes) {
        nya_execution_error(error, error_capacity, "output tensor exceeds server.max_output_bytes");
        goto failure;
    }
    response->output_data = malloc(output_size == 0 ? 1 : output_size);
    if (response->output_data == NULL) {
        nya_execution_error(error, error_capacity, "output allocation failed");
        goto failure;
    }

    status = nya_ort_api->GetTensorMutableData(output_value, &output_pointer);
    if (nya_ort_status(status, error, error_capacity) != 0) goto failure;
    if (output_size > 0) {
        memcpy(response->output_data, output_pointer, output_size);
    }
    response->output_data_size = output_size;

    nya_ort_api->ReleaseTensorTypeAndShapeInfo(shape_info);
    nya_ort_api->ReleaseValue(output_value);
    nya_ort_api->ReleaseValue(input_value);
    nya_ort_api->ReleaseMemoryInfo(memory_info);
    return 0;

failure:
    free(response->output_data);
    memset(response, 0, sizeof(*response));
    if (shape_info != NULL) nya_ort_api->ReleaseTensorTypeAndShapeInfo(shape_info);
    if (output_value != NULL) nya_ort_api->ReleaseValue(output_value);
    if (input_value != NULL) nya_ort_api->ReleaseValue(input_value);
    if (memory_info != NULL) nya_ort_api->ReleaseMemoryInfo(memory_info);
    return -1;
}

/* Release provider-allocated output bytes. */
void nya_execution_response_free(nya_execution_response *response)
{
    if (response == NULL) {
        return;
    }

    free(response->output_data);
    memset(response, 0, sizeof(*response));
}

/* Release the shared environment and dynamically loaded library. */
void nya_execution_global_shutdown(void)
{
    if (nya_ort_api != NULL && nya_ort_environment != NULL) {
        nya_ort_api->ReleaseEnv(nya_ort_environment);
    }
    nya_ort_environment = NULL;
    nya_ort_api = NULL;

    if (nya_ort_library != NULL) {
#ifdef _WIN32
        FreeLibrary((HMODULE)nya_ort_library);
#else
        dlclose(nya_ort_library);
#endif
    }
    nya_ort_library = NULL;
    nya_ort_initialization_attempted = 0;
}

#else
/* Builds without ONNX Runtime still expose honest inspect-only capability state. */
static int nya_ort_attach(nya_model *model, char *error, size_t error_capacity)
{
    (void)model;
    if (error != NULL && error_capacity > 0) {
        snprintf(error, error_capacity, "backend was built without ONNX Runtime");
        error[error_capacity - 1] = '\0';
    }
    return 1;
}

static void nya_ort_detach(nya_model *model)
{
    if (model != NULL) {
        model->execution_context = NULL;
        model->inference_supported = 0;
    }
}

static int nya_ort_run(
    nya_model *model,
    const nya_execution_request *request,
    nya_execution_response *response,
    char *error,
    size_t error_capacity
)
{
    (void)model;
    (void)request;
    if (response != NULL) memset(response, 0, sizeof(*response));
    if (error != NULL && error_capacity > 0) {
        snprintf(error, error_capacity, "backend was built without ONNX Runtime");
        error[error_capacity - 1] = '\0';
    }
    return -1;
}

void nya_execution_response_free(nya_execution_response *response)
{
    if (response != NULL) {
        free(response->output_data);
        memset(response, 0, sizeof(*response));
    }
}

void nya_execution_global_shutdown(void)
{
}
#endif

/* Provider selection is based on the inspected container, never on a caller's
   input tensor name. GGUF projectors remain available without ONNX or CUDA. */
static int nya_execution_is_projector(const nya_model *model)
{
    return model != NULL && model->format == NYA_FORMAT_GGUF && strcmp(model->architecture, "clip") == 0;
}

int nya_execution_attach(nya_model *model, char *error, size_t capacity)
{
    if (nya_execution_is_projector(model)) return nya_multimodal_attach(model, error, capacity);
    return nya_ort_attach(model, error, capacity);
}

void nya_execution_detach(nya_model *model)
{
    if (nya_execution_is_projector(model)) nya_multimodal_detach(model);
    else nya_ort_detach(model);
}

int nya_execution_run(nya_model *model, const nya_execution_request *request,
    nya_execution_response *response, char *error, size_t capacity)
{
    if (nya_execution_is_projector(model)) return nya_multimodal_run(model, request, response, error, capacity);
    return nya_ort_run(model, request, response, error, capacity);
}
