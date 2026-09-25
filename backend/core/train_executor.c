#include "train_executor.h"
#include "thread.h"
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

typedef struct train_worker {
    nya_train_executor *executor;
    nya_thread thread;
    size_t index;
} train_worker;

struct nya_train_executor {
    nya_mutex mutex;
    nya_condition work, done;
    train_worker workers[63];
    size_t count, pending;
    unsigned long long epoch;
    int stopping;
    nya_train_task task;
    void *argument;
};

static int train_worker_main(void *opaque)
{
    train_worker *worker = opaque;
    nya_train_executor *e = worker->executor;
    unsigned long long seen = 0;
    nya_mutex_lock(&e->mutex);
    for (;;) {
        while (!e->stopping && e->epoch == seen) nya_condition_wait(&e->work,&e->mutex);
        if (e->stopping) break;
        seen = e->epoch;
        nya_mutex_unlock(&e->mutex);
        e->task(e->argument,worker->index,e->count+1);
        nya_mutex_lock(&e->mutex);
        if (--e->pending == 0) nya_condition_signal(&e->done);
    }
    nya_mutex_unlock(&e->mutex);
    return 0;
}

static size_t train_core_count(void)
{
#ifdef _WIN32
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore,NULL,&bytes);
    unsigned char *data = bytes ? malloc(bytes) : NULL;
    size_t count = 0;
    if (data && GetLogicalProcessorInformationEx(RelationProcessorCore,
        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(void *)data,&bytes)) {
        for (size_t offset = 0; bytes-offset >= offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,Processor);) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX entry = (void *)(data+offset);
            if (!entry->Size || entry->Size > bytes-offset) break;
            ++count; offset += entry->Size;
        }
    }
    free(data);
    return count ? count : 1;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? (size_t)count : 1;
#endif
}

nya_train_executor *nya_train_executor_create(size_t threads)
{
    if (threads > 64) return NULL;
    if (!threads) { threads = train_core_count(); if (threads > 64) threads = 64; }
    nya_train_executor *e = calloc(1,sizeof(*e));
    if (!e) return NULL;
    if (nya_mutex_init(&e->mutex)) { free(e); return NULL; }
    if (nya_condition_init(&e->work)) { nya_mutex_destroy(&e->mutex); free(e); return NULL; }
    if (nya_condition_init(&e->done)) {
        nya_condition_destroy(&e->work); nya_mutex_destroy(&e->mutex); free(e); return NULL;
    }
    for (size_t i = 1; i < threads; ++i) {
        train_worker *w = &e->workers[e->count]; w->executor = e; w->index = i;
        if (nya_thread_create(&w->thread,train_worker_main,w)) { nya_train_executor_free(e); return NULL; }
        ++e->count;
    }
    return e;
}

void nya_train_executor_free(nya_train_executor *e)
{
    if (!e) return;
    nya_mutex_lock(&e->mutex); e->stopping = 1;
    nya_condition_broadcast(&e->work); nya_mutex_unlock(&e->mutex);
    for (size_t i = 0; i < e->count; ++i) nya_thread_join(&e->workers[i].thread);
    nya_condition_destroy(&e->done); nya_condition_destroy(&e->work);
    nya_mutex_destroy(&e->mutex); free(e);
}

size_t nya_train_executor_threads(const nya_train_executor *e) { return e ? e->count+1 : 1; }

void nya_train_execute(nya_train_executor *e, nya_train_task task, void *argument, int parallel)
{
    if (!e || !e->count || !parallel) { task(argument,0,1); return; }
    nya_mutex_lock(&e->mutex);
    e->task = task; e->argument = argument; e->pending = e->count; ++e->epoch;
    nya_condition_broadcast(&e->work); nya_mutex_unlock(&e->mutex);
    task(argument,0,e->count+1);
    nya_mutex_lock(&e->mutex);
    while (e->pending) nya_condition_wait(&e->done,&e->mutex);
    e->argument = NULL; e->task = NULL;
    nya_mutex_unlock(&e->mutex);
}

void nya_train_partition(size_t count, size_t worker, size_t workers, size_t *begin, size_t *end)
{
    size_t base = count/workers, tail = count%workers;
    *begin = worker*base+(worker < tail ? worker : tail);
    *end = *begin+base+(worker < tail);
}
