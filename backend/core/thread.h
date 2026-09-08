#ifndef NYA_THREAD_H
#define NYA_THREAD_H

/* Thread entry points return a small process-local status value. */
typedef int (*nya_thread_function)(void *argument);

/* Opaque pointers keep Windows and POSIX synchronization types out of callers. */
typedef struct nya_thread {
    void *internal;
} nya_thread;

typedef struct nya_mutex {
    void *internal;
} nya_mutex;

typedef struct nya_condition {
    void *internal;
} nya_condition;

/* Create and join one owned worker thread. */
int nya_thread_create(nya_thread *thread, nya_thread_function function, void *argument);
int nya_thread_join(nya_thread *thread);

/* Initialize, use, and release a non-recursive mutex. */
int nya_mutex_init(nya_mutex *mutex);
void nya_mutex_destroy(nya_mutex *mutex);
void nya_mutex_lock(nya_mutex *mutex);
void nya_mutex_unlock(nya_mutex *mutex);

/* Condition variables always wait while holding the associated mutex. */
int nya_condition_init(nya_condition *condition);
void nya_condition_destroy(nya_condition *condition);
void nya_condition_wait(nya_condition *condition, nya_mutex *mutex);
void nya_condition_signal(nya_condition *condition);
void nya_condition_broadcast(nya_condition *condition);

#endif
