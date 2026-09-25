#ifndef NYA_TRAIN_EXECUTOR_INTERNAL_H
#define NYA_TRAIN_EXECUTOR_INTERNAL_H
#include "training.h"

typedef void (*nya_train_task)(void *argument, size_t worker, size_t workers);
/* Private synchronous dispatch. Kernels own disjoint destinations; no graph
   allocation, error mutation or nested dispatch is performed by workers. */
void nya_train_execute(nya_train_executor *executor, nya_train_task task, void *argument, int parallel);
void nya_train_partition(size_t count, size_t worker, size_t workers, size_t *begin, size_t *end);
#endif
