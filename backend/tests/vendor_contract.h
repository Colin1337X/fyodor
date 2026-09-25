/* Test-side ABI contract snapshot, checked against upstream public headers.
   Deliberately independent of the production adapter function tables. */
typedef enum hipError_t {
hipSuccess = 0,
hipErrorInvalidValue = 1,
hipErrorOutOfMemory = 2,
hipErrorMemoryAllocation = 2,
hipErrorNotInitialized = 3,
hipErrorInitializationError = 3,
hipErrorDeinitialized = 4,
hipErrorProfilerDisabled = 5,
hipErrorProfilerNotInitialized = 6,
hipErrorProfilerAlreadyStarted = 7,
hipErrorProfilerAlreadyStopped = 8,
hipErrorInvalidConfiguration = 9,
hipErrorInvalidPitchValue = 12,
hipErrorInvalidSymbol = 13,
hipErrorInvalidDevicePointer = 17,
hipErrorInvalidMemcpyDirection = 21,
hipErrorInsufficientDriver = 35,
hipErrorMissingConfiguration = 52,
hipErrorPriorLaunchFailure = 53,
hipErrorInvalidDeviceFunction = 98,
hipErrorNoDevice = 100,
hipErrorInvalidDevice = 101,
hipErrorInvalidImage = 200,
hipErrorInvalidContext = 201,
hipErrorContextAlreadyCurrent = 202,
hipErrorMapFailed = 205,
hipErrorMapBufferObjectFailed = 205,
hipErrorUnmapFailed = 206,
hipErrorArrayIsMapped = 207,
hipErrorAlreadyMapped = 208,
hipErrorNoBinaryForGpu = 209,
hipErrorAlreadyAcquired = 210,
hipErrorNotMapped = 211,
hipErrorNotMappedAsArray = 212,
hipErrorNotMappedAsPointer = 213,
hipErrorECCNotCorrectable = 214,
hipErrorUnsupportedLimit = 215,
hipErrorContextAlreadyInUse = 216,
hipErrorPeerAccessUnsupported = 217,
hipErrorInvalidKernelFile = 218,
hipErrorInvalidGraphicsContext = 219,
hipErrorInvalidSource = 300,
hipErrorFileNotFound = 301,
hipErrorSharedObjectSymbolNotFound = 302,
hipErrorSharedObjectInitFailed = 303,
hipErrorOperatingSystem = 304,
hipErrorInvalidHandle = 400,
hipErrorInvalidResourceHandle = 400,
hipErrorIllegalState = 401,
hipErrorNotFound = 500,
hipErrorNotReady = 600,
hipErrorIllegalAddress = 700,
hipErrorLaunchOutOfResources = 701,
hipErrorLaunchTimeOut = 702,
hipErrorPeerAccessAlreadyEnabled = 704,
hipErrorPeerAccessNotEnabled = 705,
hipErrorSetOnActiveProcess = 708,
hipErrorContextIsDestroyed = 709,
hipErrorAssert = 710,
hipErrorHostMemoryAlreadyRegistered = 712,
hipErrorHostMemoryNotRegistered = 713,
hipErrorLaunchFailure = 719,
hipErrorCooperativeLaunchTooLarge = 720,
hipErrorNotSupported = 801,
hipErrorStreamCaptureUnsupported = 900,
hipErrorStreamCaptureInvalidated = 901,
hipErrorStreamCaptureMerge = 902,
hipErrorStreamCaptureUnmatched = 903,
hipErrorStreamCaptureUnjoined = 904,
hipErrorStreamCaptureIsolation = 905,
hipErrorStreamCaptureImplicit = 906,
hipErrorCapturedEvent = 907,
hipErrorStreamCaptureWrongThread = 908,
hipErrorGraphExecUpdateFailure = 910,
hipErrorInvalidChannelDescriptor = 911,
hipErrorInvalidTexture = 912,
hipErrorUnknown = 999,
hipErrorRuntimeMemory = 1052,
hipErrorRuntimeOther = 1053,
hipErrorTbd
} hipError_t;
typedef enum hipMemcpyKind {
hipMemcpyHostToHost = 0,
hipMemcpyHostToDevice = 1,
hipMemcpyDeviceToHost = 2,
hipMemcpyDeviceToDevice = 3,
hipMemcpyDefault = 4,
hipMemcpyDeviceToDeviceNoCU = 1024
} hipMemcpyKind;
typedef enum rocblas_operation_
{
rocblas_operation_none      = 111,
rocblas_operation_transpose = 112,
rocblas_operation_conjugate_transpose
= 113
} rocblas_operation;
typedef enum rocblas_status_
{
rocblas_status_success         = 0,
rocblas_status_invalid_handle  = 1,
rocblas_status_not_implemented = 2,
rocblas_status_invalid_pointer = 3,
rocblas_status_invalid_size    = 4,
rocblas_status_memory_error    = 5,
rocblas_status_internal_error  = 6,
rocblas_status_perf_degraded   = 7,
rocblas_status_size_query_mismatch = 8,
rocblas_status_size_increased      = 9,
rocblas_status_size_unchanged      = 10,
rocblas_status_invalid_value       = 11,
rocblas_status_continue            = 12,
rocblas_status_check_numerics_fail
= 13,
rocblas_status_excluded_from_build
= 14,
rocblas_status_arch_mismatch
= 15,
} rocblas_status;
typedef enum rocblas_pointer_mode_
{
rocblas_pointer_mode_host = 0,
rocblas_pointer_mode_device = 1
} rocblas_pointer_mode;
typedef struct ihipStream_t *hipStream_t;
typedef struct _rocblas_handle *rocblas_handle;
typedef struct mlx_array_ { void *ctx; } mlx_array;
typedef struct mlx_device_ { void *ctx; } mlx_device;
typedef struct mlx_stream_ { void *ctx; } mlx_stream;
typedef enum mlx_device_type_ { MLX_CPU, MLX_GPU } mlx_device_type;
typedef enum mlx_dtype_ {
MLX_BOOL,
MLX_UINT8,
MLX_UINT16,
MLX_UINT32,
MLX_UINT64,
MLX_INT8,
MLX_INT16,
MLX_INT32,
MLX_INT64,
MLX_FLOAT16,
MLX_FLOAT32,
MLX_FLOAT64,
MLX_BFLOAT16,
MLX_COMPLEX64,
} mlx_dtype;
