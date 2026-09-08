#include "runtime.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
/* Windows CNG provides cryptographically secure random bytes. */
#include <windows.h>
#include <bcrypt.h>
#endif

/* Initialize the tiny process runtime. */
int nya_runtime_init(nya_runtime *runtime)
{
    time_t started_at;

    /* A null output pointer is always a caller error. */
    if (runtime == NULL) {
        return -1;
    }

    /* Start from deterministic zeroed state. */
    memset(runtime, 0, sizeof(*runtime));

    /* Record wall-clock seconds; monotonic precision is unnecessary for health. */
    started_at = time(NULL);
    if (started_at == (time_t)-1) {
        return -1;
    }
    runtime->started_at_seconds = (uint64_t)started_at;
    runtime->started_at_milliseconds = nya_runtime_monotonic_milliseconds();

    /* Mark initialization only after every required step has succeeded. */
    runtime->initialized = 1;

    return 0;
}

/* Shut down runtime state without owning any external resources yet. */
void nya_runtime_shutdown(nya_runtime *runtime)
{
    /* A null pointer is harmless during cleanup. */
    if (runtime == NULL) {
        return;
    }

    /* Clear the flag so accidental reuse is easy to detect. */
    runtime->initialized = 0;
}

/* Read a clock whose value cannot jump when wall-clock time is corrected. */
uint64_t nya_runtime_monotonic_milliseconds(void)
{
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000) + (uint64_t)now.tv_nsec / UINT64_C(1000000);
#endif
}

/* Calculate elapsed whole seconds without depending on civil time. */
uint64_t nya_runtime_uptime_seconds(const nya_runtime *runtime)
{
    uint64_t now;

    /* An uninitialized runtime has no meaningful uptime. */
    if (runtime == NULL || !runtime->initialized) {
        return 0;
    }

    now = nya_runtime_monotonic_milliseconds();

    /* Protect against a wall clock that moved backward. */
    if (now < runtime->started_at_milliseconds) {
        return 0;
    }

    return (now - runtime->started_at_milliseconds) / UINT64_C(1000);
}

/* Ask the host operating system for unpredictable bytes. */
int nya_runtime_random_bytes(void *buffer, size_t length)
{
    /* Reject an invalid destination while allowing an empty request. */
    if (buffer == NULL && length != 0) {
        return -1;
    }
    if (length == 0) {
        return 0;
    }

#ifdef _WIN32
    /* The Windows function accepts a 32-bit byte count. */
    if (length > (size_t)ULONG_MAX) {
        return -1;
    }

    /* BCryptGenRandom with this flag does not require a provider handle. */
    if (BCryptGenRandom(
            NULL,
            (PUCHAR)buffer,
            (ULONG)length,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG
        ) != 0) {
        return -1;
    }

    return 0;
#else
    FILE *random_file;
    unsigned char *output;
    size_t total;

    /* Open the kernel random source for binary input. */
    random_file = fopen("/dev/urandom", "rb");
    if (random_file == NULL) {
        return -1;
    }

    /* fread may legally return a short count, so fill the buffer in a loop. */
    output = (unsigned char *)buffer;
    total = 0;
    while (total < length) {
        size_t received;

        received = fread(output + total, 1, length - total, random_file);
        if (received == 0) {
            fclose(random_file);
            return -1;
        }

        total += received;
    }

    fclose(random_file);
    return 0;
#endif
}
