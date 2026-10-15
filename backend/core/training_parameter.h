#ifndef NYA_TRAINING_PARAMETER_H
#define NYA_TRAINING_PARAMETER_H
#include "training.h"

/* Private storage shared by CPU graphs and the resident session bridge. A
   session holds an exclusive lease; CPU leaves retain storage until graph
   destruction. One controlling thread serializes all access. */
struct nya_train_parameter {
    size_t rows, columns, count;
    float *data, *gradient, *moment, *variance;
    size_t references, cpu_leaves;
    const void *resident_owner;
};
#endif
