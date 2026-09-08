#ifndef NYA_TRAINING_INTERNAL_H
#define NYA_TRAINING_INTERNAL_H

#include "training.h"
#include "llm_internal.h"

/* Frozen quantized matrices remain mapped while gradients propagate through
   their inputs. The mapping must outlive the graph and its backward pass. */
nya_train_tensor *nya_train_linear_mapped(nya_train_tensor *input, const nya_llm_tensor *weight);
nya_train_tensor *nya_train_embedding_mapped(nya_train_graph *graph, const nya_llm_tensor *table,
    const uint32_t *ids, size_t count);
nya_train_tensor *nya_train_constant_mapped(nya_train_graph *graph, const nya_llm_tensor *vector);
void nya_train_graph_fail(nya_train_graph *graph, const char *message);

#endif
