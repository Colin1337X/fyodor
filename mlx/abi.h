#ifndef NYA_MLX_ABI_H
#define NYA_MLX_ABI_H
#include <stdbool.h>
#include <stddef.h>
#include "types.h"
/* MLX C uses distinct by-value, one-pointer structs for its opaque handles.
   Preserve that ABI (these are not pointer-to-handle typedefs). Values and
   signatures are checked against the upstream version documented in README. */
typedef mlx_array nya_mlx_array;
typedef mlx_device nya_mlx_device;
typedef mlx_stream nya_mlx_stream;
typedef struct nya_mlx_api {
    void (*error_handler)(void (*)(const char *,void *),void *,void (*)(void *));
    nya_mlx_device (*device_new)(mlx_device_type,int);
    int (*device_available)(bool *,nya_mlx_device);
    int (*device_free)(nya_mlx_device);
    nya_mlx_stream (*stream_new)(nya_mlx_device);
    int (*stream_free)(nya_mlx_stream);
    int (*synchronize)(nya_mlx_stream);
    nya_mlx_array (*array_data)(const void *,const int *,int,mlx_dtype);
    int (*array_free)(nya_mlx_array);
    int (*eval)(nya_mlx_array);
    const float *(*data_f32)(nya_mlx_array);
    int (*transpose)(nya_mlx_array *,nya_mlx_array,nya_mlx_stream);
    int (*contiguous)(nya_mlx_array *,nya_mlx_array,bool,nya_mlx_stream);
    int (*copy)(nya_mlx_array *,nya_mlx_array,nya_mlx_stream);
    int (*matmul)(nya_mlx_array *,nya_mlx_array,nya_mlx_array,nya_mlx_stream);
    int (*memory_limit)(size_t *);
    int (*active_memory)(size_t *);
    int (*cache_memory)(size_t *);
} nya_mlx_api;
#define NYA_MLX_CPU MLX_CPU
#define NYA_MLX_GPU MLX_GPU
#define NYA_MLX_F32 MLX_FLOAT32
#endif
