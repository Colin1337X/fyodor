#ifndef NYA_RUNTIME_H
#define NYA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

/* The runtime currently owns process-wide state shared by future engine parts. */
typedef struct nya_runtime {
    /* The flag makes initialization and shutdown safe to call once each. */
    int initialized;

    /* The start time lets the health endpoint report a simple uptime value. */
    uint64_t started_at_seconds;

    /* Monotonic time avoids uptime jumps when the system clock is adjusted. */
    uint64_t started_at_milliseconds;
} nya_runtime;

/* Initialize the portable runtime state. */
int nya_runtime_init(nya_runtime *runtime);

/* Release runtime state after the server has stopped. */
void nya_runtime_shutdown(nya_runtime *runtime);

/* Return the number of whole seconds since runtime initialization. */
uint64_t nya_runtime_uptime_seconds(const nya_runtime *runtime);

/* A process-independent monotonic clock for elapsed times and I/O deadlines. */
uint64_t nya_runtime_monotonic_milliseconds(void);

/* Fill a buffer with random bytes supplied by the operating system. */
int nya_runtime_random_bytes(void *buffer, size_t length);

#endif
