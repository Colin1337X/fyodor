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
typedef uint64_t nya_train_indices;
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

/* Bind validated host token IDs to immutable resident metadata. This explicit
   setup allocates arena storage and uploads once; it uses host sorting/staging
   proportional to count, not vocabulary size. rows/count must fit uint32_t.
   The map follows buffer scratch lifetime but is opaque: ordinary buffer APIs
   cannot read, write or use it as tensor storage. Caller IDs may change after
   this returns. Repeated forward/backward adds no allocation, copy or fence. */
nya_train_indices nya_train_device_indices(nya_train_device *device, const uint32_t *ids, size_t count, size_t rows);
/* Gather table[rows,columns] into output[count,columns], overwriting output.
   Table may use any supported matrix storage type; output is dense F32. */
int nya_train_device_embedding(nya_train_device *device, nya_train_buffer output, nya_train_buffer table,
    unsigned type, size_t columns, nya_train_indices indices);
/* Accumulate into a dense F32 table gradient, touching only referenced rows.
   Repeated IDs accumulate in original token order with one writer per element.
   Output/input aliasing is rejected; no floating-point atomics are used. */
int nya_train_device_embedding_backward(nya_train_device *device, nya_train_buffer dtable,
    nya_train_buffer dy, size_t columns, nya_train_indices indices);

/* One non-nesting scratch scope at a time. Allocations made after begin belong
   to that scope; end invalidates their handles and reclaims arena/descriptors.
   Earlier allocations (parameters, optimizer state, accumulated gradients)
   survive. Zero is never a valid scope; stale/foreign tokens are rejected.
   End adds no fence: queued uses and later zero-initialization/reuse are ordered
   on the same stream. Call finish explicitly to observe asynchronous failure.
   A failed device cannot begin/end scopes or recycle its state. */
nya_train_scope nya_train_device_scratch_begin(nya_train_device *device);
int nya_train_device_scratch_end(nya_train_device *device, nya_train_scope scope);

/* Queue a read-only F32 finite check. status holds a uint32_t, initialized to
   zero by the caller. The first failing check writes its nonzero tag; later
   checks preserve it. status must not alias data and must outlive queued checks
   (allocate before scratch_begin to retain errors across scope reuse).
   Return zero means successfully queued, not numerically valid. Read status
   explicitly at a step boundary: zero means all queued checks passed. No copy,
   allocation or fence occurs here. Numerical failure does not poison the device;
   the graph/optimizer must reject the step before mutating persistent state. */
int nya_train_device_check_finite(nya_train_device *device, nya_train_buffer status,
    nya_train_buffer data, size_t count, uint32_t tag);

typedef enum nya_train_unary_op {
    NYA_TRAIN_UNARY_SCALE=0, NYA_TRAIN_UNARY_SILU=1, NYA_TRAIN_UNARY_GELU=2, NYA_TRAIN_UNARY_SOFTCAP=3
} nya_train_unary_op;
typedef enum nya_train_binary_op { NYA_TRAIN_BINARY_ADD=0, NYA_TRAIN_BINARY_MUL=1 } nya_train_binary_op;
typedef struct nya_train_view { nya_train_buffer buffer; size_t rows, columns; } nya_train_view;

/* Dense F32 elementwise operations. Forward overwrites; backward accumulates.
   Destinations cannot alias any input. The scalar must be finite and is used
   by SCALE/SOFTCAP; SOFTCAP additionally requires a positive scalar. GELU uses
   the same tanh approximation as the CPU graph. Numerical status checks remain
   explicit, as for matrices. No allocation, transfer or fence is added. */
int nya_train_device_unary(nya_train_device *device, nya_train_buffer y, nya_train_buffer x,
    size_t count, nya_train_unary_op operation, double scalar);
int nya_train_device_unary_backward(nya_train_device *device, nya_train_buffer dx, nya_train_buffer x,
    nya_train_buffer dy, size_t count, nya_train_unary_op operation, double scalar);
/* Each dimension broadcasts independently when equal to one. Both gradients
   are optional (zero), but at least one is required for backward. da == db is
   supported only for equal operand shapes, preserving both contributions in
   CPU row-major order. Distinct input views may share read-only storage. */
int nya_train_device_binary(nya_train_device *device, nya_train_buffer y,
    nya_train_view a, nya_train_view b, nya_train_binary_op operation);
int nya_train_device_binary_backward(nya_train_device *device, nya_train_buffer da, nya_train_buffer db,
    nya_train_view a, nya_train_view b, nya_train_buffer dy, nya_train_binary_op operation);

/* RMSNorm over each row of dense F32 x. Optional weight has columns F32 values.
   Forward overwrites y and saves one double inverse RMS per row in inverse
   (at least rows*8 bytes). Backward requires this exact forward state, with x,
   weight and inverse unchanged, and accumulates dx/dweight. Either gradient
   may be omitted; dweight requires weight. dx == dweight requires rows == 1.
   Other output/input aliases are rejected. Epsilon is positive and finite.
   No allocation, transfer or fence is added. Numerical checks stay explicit. */
int nya_train_device_rms_norm(nya_train_device *device, nya_train_buffer y, nya_train_buffer inverse,
    nya_train_view x, nya_train_buffer weight, float epsilon);
int nya_train_device_rms_norm_backward(nya_train_device *device, nya_train_buffer dx, nya_train_buffer dweight,
    nya_train_buffer inverse, nya_train_view x, nya_train_buffer weight, nya_train_buffer dy);

/* RoPE on dense F32 rows, each holding heads*dimension values. dimension is
   positive and even; split_half selects adjacent pairs (0) or head halves (1).
   frequencies holds dimension/2 F32 angular frequencies, shared by heads;
   position is the zero-based row number, matching the CPU training graph.
   Forward overwrites y; backward adds the inverse rotation of dy to dx.
   Frequencies must remain unchanged between forward/backward. Inputs and
   frequencies require explicit finite checks at the graph boundary. No host
   validation of device values, allocation, transfer or fence occurs here.
   Destinations cannot alias source or frequencies; read-only inputs may share. */
int nya_train_device_rope(nya_train_device *device, nya_train_buffer y, nya_train_view x,
    size_t heads, size_t dimension, nya_train_buffer frequencies, int split_half);
int nya_train_device_rope_backward(nya_train_device *device, nya_train_buffer dx, nya_train_view dy,
    size_t heads, size_t dimension, nya_train_buffer frequencies, int split_half);

/* Dense causal/local grouped-query attention. q=[tokens,heads*dimension],
   k/v=[tokens,kv_heads*dimension]. Optional groups is tokens U32 IDs: for a
   nonzero window, matching nonzero IDs permit future keys as on the CPU graph.
   scale must be finite and positive. Saved state holds two doubles per query
   row/head (maximum and mass), not a square probability matrix.
   Backward recomputes probabilities in bounded query tiles; workspace size is
   returned below. Inputs, groups and state must remain unchanged between calls.
   Gradients accumulate in row/head/key order without floating-point atomics.
   At least one gradient is required. Outputs must be distinct and cannot alias
   any input, state or workspace. Numerical checks remain explicit. Dispatch adds
   no allocation, transfer or fence. Zero workspace size signals invalid/overflow. */
typedef struct nya_train_attention_desc {
    nya_train_view q, k, v;
    size_t heads, kv_heads, dimension, window;
    float scale;
    nya_train_buffer groups;
} nya_train_attention_desc;
size_t nya_train_attention_workspace_bytes(size_t tokens, size_t heads);
int nya_train_device_attention(nya_train_device *device, nya_train_buffer y,
    nya_train_buffer state, nya_train_attention_desc descriptor);
int nya_train_device_attention_backward(nya_train_device *device, nya_train_buffer dq,
    nya_train_buffer dk, nya_train_buffer dv, nya_train_buffer state, nya_train_buffer dy,
    nya_train_buffer workspace, nya_train_attention_desc descriptor);

/* Row-major W[outputs,inputs], X[tokens,inputs], Y[tokens,outputs]. Storage
   type IDs are the existing Fyodor/GGUF IDs. Activations/gradients are F32.
   Linear overwrites Y; gradient operations accumulate each F32-rounded product
   in CPU reduction order into initialized dX/dW, without fused multiply-add.
   Outputs must not alias inputs. No implicit transfers or fences occur here. */
int nya_train_device_linear(nya_train_device *device, nya_train_buffer y, nya_train_buffer w,
    unsigned type, size_t outputs, size_t inputs, nya_train_buffer x, size_t tokens);
int nya_train_device_linear_dx(nya_train_device *device, nya_train_buffer dx, nya_train_buffer w,
    unsigned type, size_t outputs, size_t inputs, nya_train_buffer dy, size_t tokens);
int nya_train_device_linear_dw(nya_train_device *device, nya_train_buffer dw, nya_train_buffer x,
    nya_train_buffer dy, size_t outputs, size_t inputs, size_t tokens);
#endif
