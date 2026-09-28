#ifndef NYA_TRAINING_DEVICE_H
#define NYA_TRAINING_DEVICE_H
#include <stddef.h>
#include <stdint.h>

/* Private building block for accelerator graph execution. This is not yet a
   complete training backend. A device owns a bounded persistent arena; buffers
   are zero-initialized and remain valid until device destruction or the end of
   their scratch scope. One caller
   serializes operations. Handles cannot be shared between devices. */
typedef struct nya_train_device nya_train_device;
typedef uint64_t nya_train_buffer;
typedef uint64_t nya_train_scope;
typedef struct nya_train_device_stats {
    size_t capacity_bytes, used_bytes, buffers, peak_bytes;
    uint64_t kernel_launches, uploads, downloads, upload_bytes, download_bytes, synchronizations;
    uint64_t scratch_resets;
    int failed;
} nya_train_device_stats;

nya_train_device *nya_train_device_create(const char *backend, size_t capacity_bytes);
void nya_train_device_free(nya_train_device *device);
const char *nya_train_device_error(const nya_train_device *device);
void nya_train_device_get_stats(const nya_train_device *device, nya_train_device_stats *stats);
nya_train_buffer nya_train_device_alloc(nya_train_device *device, size_t bytes);
int nya_train_device_write(nya_train_device *device, nya_train_buffer buffer, size_t offset, const void *source, size_t bytes);
int nya_train_device_read(nya_train_device *device, nya_train_buffer buffer, size_t offset, void *destination, size_t bytes);
int nya_train_device_zero(nya_train_device *device, nya_train_buffer buffer);
int nya_train_device_finish(nya_train_device *device);

/* One non-nesting scratch scope at a time. Allocations made after begin belong
   to that scope; end invalidates their handles and reclaims arena/descriptors.
   Earlier allocations (parameters, optimizer state, accumulated gradients)
   survive. Zero is never a valid scope; stale/foreign tokens are rejected.
   End adds no fence: queued uses and later zero-initialization/reuse are ordered
   on the same stream. Call finish explicitly to observe asynchronous failure.
   A failed device cannot begin/end scopes or recycle its state. */
nya_train_scope nya_train_device_scratch_begin(nya_train_device *device);
int nya_train_device_scratch_end(nya_train_device *device, nya_train_scope scope);

/* Row-major W[outputs,inputs], X[tokens,inputs], Y[tokens,outputs]. Storage
   type IDs are the existing Fyodor/GGUF IDs. Activations/gradients are F32.
   Linear overwrites Y; gradient operations accumulate into initialized dX/dW.
   Outputs must not alias inputs. No implicit transfers or fences occur here. */
int nya_train_device_linear(nya_train_device *device, nya_train_buffer y, nya_train_buffer w,
    unsigned type, size_t outputs, size_t inputs, nya_train_buffer x, size_t tokens);
int nya_train_device_linear_dx(nya_train_device *device, nya_train_buffer dx, nya_train_buffer w,
    unsigned type, size_t outputs, size_t inputs, nya_train_buffer dy, size_t tokens);
int nya_train_device_linear_dw(nya_train_device *device, nya_train_buffer dw, nya_train_buffer x,
    nya_train_buffer dy, size_t outputs, size_t inputs, size_t tokens);
#endif
