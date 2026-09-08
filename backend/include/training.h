#ifndef NYA_TRAINING_H
#define NYA_TRAINING_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* A small eager reverse-mode C API. Graphs own intermediate tensors; parameters
   persist across graphs/steps. Dense tensors use row-major F32 storage. No
   Python runtime, CUDA toolkit, or C++ ABI is involved. */
typedef struct nya_train_graph nya_train_graph;
typedef struct nya_train_tensor nya_train_tensor;
typedef struct nya_train_parameter nya_train_parameter;

nya_train_graph *nya_train_graph_create(size_t memory_limit);
void nya_train_graph_free(nya_train_graph *graph);
const char *nya_train_error(const nya_train_graph *graph);
size_t nya_train_memory_used(const nya_train_graph *graph);

nya_train_parameter *nya_train_parameter_create(size_t rows, size_t columns, const float *initial);
void nya_train_parameter_free(nya_train_parameter *parameter);
float *nya_train_parameter_data(nya_train_parameter *parameter);
const float *nya_train_parameter_gradient(const nya_train_parameter *parameter);
void nya_train_zero_grad(nya_train_parameter *parameter);

/* Inputs are copied. Parameter leaves borrow parameter storage, which must
   remain alive and unchanged until backward and graph destruction finish. */
nya_train_tensor *nya_train_input(nya_train_graph *graph, size_t rows, size_t columns, const float *data);
nya_train_tensor *nya_train_leaf(nya_train_graph *graph, nya_train_parameter *parameter);
const float *nya_train_data(const nya_train_tensor *tensor);
size_t nya_train_rows(const nya_train_tensor *tensor);
size_t nya_train_columns(const nya_train_tensor *tensor);

/* Binary elementwise operations support a scalar or [1,columns] broadcast. */
nya_train_tensor *nya_train_add(nya_train_tensor *a, nya_train_tensor *b);
nya_train_tensor *nya_train_mul(nya_train_tensor *a, nya_train_tensor *b);
nya_train_tensor *nya_train_scale(nya_train_tensor *a, float scale);
/* y = x W^T; x=[tokens,input], W=[output,input]. */
nya_train_tensor *nya_train_linear(nya_train_tensor *x, nya_train_tensor *weight);
nya_train_tensor *nya_train_gelu(nya_train_tensor *x);
nya_train_tensor *nya_train_silu(nya_train_tensor *x);
/* cap*tanh(x/cap), with a positive finite cap. */
nya_train_tensor *nya_train_softcap(nya_train_tensor *x, float cap);
nya_train_tensor *nya_train_rms_norm(nya_train_tensor *x, nya_train_tensor *weight, float epsilon);
nya_train_tensor *nya_train_embedding(nya_train_tensor *table, const uint32_t *ids, size_t count);
nya_train_tensor *nya_train_reshape(nya_train_tensor *x, size_t rows, size_t columns);
/* Copy a column interval from every row; backward scatters into that interval. */
nya_train_tensor *nya_train_slice_columns(nya_train_tensor *x, size_t first, size_t count);
/* Full-head rotary position encoding; frequencies has head_dimension/2 values.
   Position zero is the first row. split_half selects Gemma's NeoX layout. */
nya_train_tensor *nya_train_rope(nya_train_tensor *x, size_t heads,
    size_t head_dimension, const float *frequencies, int split_half);
/* Causal grouped-query self-attention. A nonzero local window permits future
   positions only inside matching positive vision group IDs. NULL groups means
   plain text. A zero window always retains full causal attention. */
nya_train_tensor *nya_train_attention(nya_train_tensor *query, nya_train_tensor *key,
    nya_train_tensor *value, size_t heads, size_t kv_heads, size_t head_dimension,
    float scale, size_t sliding_window, const uint32_t *vision_groups);

/* Target labels refer to the NEXT token at each logit row. A zero mask excludes
   that row from both loss and gradient. SFT masks prompt rows; CPT/pretraining
   train all non-padding rows. Logprob is a SUM (required by standard DPO). */
nya_train_tensor *nya_train_cross_entropy(nya_train_tensor *logits,
    const uint32_t *labels, const unsigned char *mask, size_t count);
nya_train_tensor *nya_train_logprob(nya_train_tensor *logits,
    const uint32_t *labels, const unsigned char *mask, size_t count);
nya_train_tensor *nya_train_dpo(nya_train_tensor *chosen, nya_train_tensor *rejected,
    double reference_chosen, double reference_rejected, float beta);

/* One scalar backward per graph. Gradients accumulate into parameter leaves;
   call zero_grad explicitly before a new optimizer step or accumulation group. */
int nya_train_backward(nya_train_tensor *loss);

typedef struct nya_train_adamw {
    float learning_rate, beta1, beta2, epsilon, weight_decay, max_grad_norm;
    uint64_t step;
} nya_train_adamw;
void nya_train_adamw_defaults(nya_train_adamw *optimizer);
/* Validates the entire update before changing ANY parameter or optimizer state.
   Duplicate parameter pointers are rejected. Moments belong to each parameter. */
int nya_train_adamw_step(nya_train_adamw *optimizer,
    nya_train_parameter *const *parameters, size_t count, char *error, size_t error_capacity);

/* Versioned little-endian checkpoints include weights, accumulated gradients,
   Adam moments, settings and step. Streams must be binary. The caller owns file
   creation/atomic replacement. Read validates the complete checkpoint before
   committing any state; parameter order and shapes must match the model. */
int nya_train_checkpoint_write(FILE *file, const nya_train_adamw *optimizer,
    nya_train_parameter *const *parameters, size_t count);
int nya_train_checkpoint_read(FILE *file, nya_train_adamw *optimizer,
    nya_train_parameter *const *parameters, size_t count);

#endif
