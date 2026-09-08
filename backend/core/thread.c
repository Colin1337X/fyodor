#include "thread.h"

#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
/* Win32 supplies threads, critical sections, and condition variables directly. */
#include <windows.h>
#include <process.h>

/* The startup record transfers a C function and argument into CreateThread. */
typedef struct nya_thread_start {
    nya_thread_function function;
    void *argument;
} nya_thread_start;

/* Adapt the project's small thread signature to the Win32 callback ABI. */
static unsigned int __stdcall nya_thread_entry(void *pointer)
{
    nya_thread_start *start;
    nya_thread_function function;
    void *argument;
    int result;

    start = (nya_thread_start *)pointer;
    function = start->function;
    argument = start->argument;
    free(start);

    result = function(argument);
    return (unsigned int)result;
}

/* Create one joinable Windows thread and retain its kernel handle. */
int nya_thread_create(nya_thread *thread, nya_thread_function function, void *argument)
{
    nya_thread_start *start;
    HANDLE handle;

    if (thread == NULL || function == NULL) {
        return -1;
    }

    start = (nya_thread_start *)malloc(sizeof(*start));
    if (start == NULL) {
        return -1;
    }
    start->function = function;
    start->argument = argument;

    /* Workers use malloc, stdio, and locale helpers. _beginthreadex gives the
       C runtime ownership of thread-local initialization and cleanup. */
    handle = (HANDLE)_beginthreadex(NULL, 0, nya_thread_entry, start, 0, NULL);
    if (handle == NULL) {
        free(start);
        return -1;
    }

    thread->internal = handle;
    return 0;
}

/* Wait for a Windows worker and close the owned kernel handle. */
int nya_thread_join(nya_thread *thread)
{
    HANDLE handle;

    if (thread == NULL || thread->internal == NULL) {
        return -1;
    }

    handle = (HANDLE)thread->internal;
    if (WaitForSingleObject(handle, INFINITE) != WAIT_OBJECT_0) {
        return -1;
    }

    CloseHandle(handle);
    thread->internal = NULL;
    return 0;
}

/* Allocate and initialize one Win32 critical section. */
int nya_mutex_init(nya_mutex *mutex)
{
    CRITICAL_SECTION *critical_section;

    if (mutex == NULL) return -1;
    mutex->internal = NULL;

    critical_section = (CRITICAL_SECTION *)malloc(sizeof(*critical_section));
    if (critical_section == NULL) {
        return -1;
    }

    InitializeCriticalSection(critical_section);
    mutex->internal = critical_section;
    return 0;
}

/* Destroy and free one Win32 critical section. */
void nya_mutex_destroy(nya_mutex *mutex)
{
    CRITICAL_SECTION *critical_section;

    if (mutex == NULL || mutex->internal == NULL) {
        return;
    }

    critical_section = (CRITICAL_SECTION *)mutex->internal;
    DeleteCriticalSection(critical_section);
    free(critical_section);
    mutex->internal = NULL;
}

/* Enter and leave the Win32 critical section. */
void nya_mutex_lock(nya_mutex *mutex)
{
    EnterCriticalSection((CRITICAL_SECTION *)mutex->internal);
}

void nya_mutex_unlock(nya_mutex *mutex)
{
    LeaveCriticalSection((CRITICAL_SECTION *)mutex->internal);
}

/* Allocate a Win32 condition variable; the OS object needs no destruction call. */
int nya_condition_init(nya_condition *condition)
{
    CONDITION_VARIABLE *condition_variable;

    if (condition == NULL) return -1;
    condition->internal = NULL;

    condition_variable = (CONDITION_VARIABLE *)malloc(sizeof(*condition_variable));
    if (condition_variable == NULL) {
        return -1;
    }

    InitializeConditionVariable(condition_variable);
    condition->internal = condition_variable;
    return 0;
}

/* Release the memory that stores one condition variable. */
void nya_condition_destroy(nya_condition *condition)
{
    if (condition == NULL) {
        return;
    }

    free(condition->internal);
    condition->internal = NULL;
}

/* Atomically release the mutex while sleeping, then acquire it before returning. */
void nya_condition_wait(nya_condition *condition, nya_mutex *mutex)
{
    SleepConditionVariableCS(
        (CONDITION_VARIABLE *)condition->internal,
        (CRITICAL_SECTION *)mutex->internal,
        INFINITE
    );
}

/* Wake one waiter or every waiter. */
void nya_condition_signal(nya_condition *condition)
{
    WakeConditionVariable((CONDITION_VARIABLE *)condition->internal);
}

void nya_condition_broadcast(nya_condition *condition)
{
    WakeAllConditionVariable((CONDITION_VARIABLE *)condition->internal);
}

#else
/* POSIX supplies the same small concepts through pthreads. */
#include <pthread.h>

/* POSIX already uses a compatible thread entry pointer shape except for return type. */
typedef struct nya_thread_start {
    nya_thread_function function;
    void *argument;
} nya_thread_start;

/* Adapt the project signature and transfer ownership of the startup record. */
static void *nya_thread_entry(void *pointer)
{
    nya_thread_start *start;
    nya_thread_function function;
    void *argument;
    int result;

    start = (nya_thread_start *)pointer;
    function = start->function;
    argument = start->argument;
    free(start);

    result = function(argument);
    (void)result;
    return NULL;
}

/* Create one joinable POSIX worker. */
int nya_thread_create(nya_thread *thread, nya_thread_function function, void *argument)
{
    pthread_t *handle;
    nya_thread_start *start;

    if (thread == NULL || function == NULL) {
        return -1;
    }

    handle = (pthread_t *)malloc(sizeof(*handle));
    start = (nya_thread_start *)malloc(sizeof(*start));
    if (handle == NULL || start == NULL) {
        free(handle);
        free(start);
        return -1;
    }

    start->function = function;
    start->argument = argument;
    if (pthread_create(handle, NULL, nya_thread_entry, start) != 0) {
        free(start);
        free(handle);
        return -1;
    }

    thread->internal = handle;
    return 0;
}

/* Join one POSIX worker and release its stored pthread_t. */
int nya_thread_join(nya_thread *thread)
{
    pthread_t *handle;

    if (thread == NULL || thread->internal == NULL) {
        return -1;
    }

    handle = (pthread_t *)thread->internal;
    if (pthread_join(*handle, NULL) != 0) {
        return -1;
    }

    free(handle);
    thread->internal = NULL;
    return 0;
}

/* Allocate and initialize one POSIX mutex. */
int nya_mutex_init(nya_mutex *mutex)
{
    pthread_mutex_t *native_mutex;

    if (mutex == NULL) return -1;
    mutex->internal = NULL;

    native_mutex = (pthread_mutex_t *)malloc(sizeof(*native_mutex));
    if (native_mutex == NULL) {
        return -1;
    }

    if (pthread_mutex_init(native_mutex, NULL) != 0) {
        free(native_mutex);
        return -1;
    }

    mutex->internal = native_mutex;
    return 0;
}

/* Destroy and free one POSIX mutex. */
void nya_mutex_destroy(nya_mutex *mutex)
{
    pthread_mutex_t *native_mutex;

    if (mutex == NULL || mutex->internal == NULL) {
        return;
    }

    native_mutex = (pthread_mutex_t *)mutex->internal;
    pthread_mutex_destroy(native_mutex);
    free(native_mutex);
    mutex->internal = NULL;
}

/* Lock and unlock one POSIX mutex. */
void nya_mutex_lock(nya_mutex *mutex)
{
    pthread_mutex_lock((pthread_mutex_t *)mutex->internal);
}

void nya_mutex_unlock(nya_mutex *mutex)
{
    pthread_mutex_unlock((pthread_mutex_t *)mutex->internal);
}

/* Allocate and initialize one POSIX condition variable. */
int nya_condition_init(nya_condition *condition)
{
    pthread_cond_t *native_condition;

    if (condition == NULL) return -1;
    condition->internal = NULL;

    native_condition = (pthread_cond_t *)malloc(sizeof(*native_condition));
    if (native_condition == NULL) {
        return -1;
    }

    if (pthread_cond_init(native_condition, NULL) != 0) {
        free(native_condition);
        return -1;
    }

    condition->internal = native_condition;
    return 0;
}

/* Destroy and free one POSIX condition variable. */
void nya_condition_destroy(nya_condition *condition)
{
    pthread_cond_t *native_condition;

    if (condition == NULL || condition->internal == NULL) {
        return;
    }

    native_condition = (pthread_cond_t *)condition->internal;
    pthread_cond_destroy(native_condition);
    free(native_condition);
    condition->internal = NULL;
}

/* Wait, wake one, or wake all using the associated POSIX primitives. */
void nya_condition_wait(nya_condition *condition, nya_mutex *mutex)
{
    pthread_cond_wait(
        (pthread_cond_t *)condition->internal,
        (pthread_mutex_t *)mutex->internal
    );
}

void nya_condition_signal(nya_condition *condition)
{
    pthread_cond_signal((pthread_cond_t *)condition->internal);
}

void nya_condition_broadcast(nya_condition *condition)
{
    pthread_cond_broadcast((pthread_cond_t *)condition->internal);
}
#endif
