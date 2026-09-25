#ifndef NYA_TRAIN_CLOCK_H
#define NYA_TRAIN_CLOCK_H

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* Diagnostic elapsed time, never used for training state or scheduling. */
static double tr_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER now, frequency;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&now)) return 0;
    return (double)now.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
#endif
}
#endif
