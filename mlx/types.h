#ifndef NYA_MLX_TYPES_H
#define NYA_MLX_TYPES_H
/* Private C ABI types retain upstream tags/enumerators, not merely sizes.
   This is required for compatible C function types and UBSan indirect calls.
   Sources/versions and licenses are documented in the provider README. */
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
#endif
