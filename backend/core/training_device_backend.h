#ifndef NYA_TRAINING_DEVICE_BACKEND_H
#define NYA_TRAINING_DEVICE_BACKEND_H
#include "training_device.h"
typedef struct nya_train_device_interface {
    void *(*create)(size_t capacity);
    void (*destroy)(void *context);
    const char *(*error)(const void *context);
    void (*stats)(const void *context, nya_train_device_stats *stats);
    nya_train_buffer (*allocate)(void *context, size_t bytes);
    int (*transfer)(void *context, nya_train_buffer buffer, size_t offset, void *host, size_t bytes, int upload);
    int (*zero)(void *context, nya_train_buffer buffer);
    int (*finish)(void *context);
    int (*matrix)(void *context, nya_train_buffer destination, nya_train_buffer a, nya_train_buffer b,
        unsigned type, size_t outputs, size_t inputs, size_t tokens, unsigned operation);
} nya_train_device_interface;
#endif
