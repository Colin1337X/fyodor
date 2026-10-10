#ifndef NYA_TRAINING_SESSION_H
#define NYA_TRAINING_SESSION_H
#include "training.h"
#include "training_device.h"

/* Private ownership/checkpoint bridge for accelerator graph integration, not
   a public tensor ownership type. The CLI uses this private bridge. One controlling thread
   serializes the session, its parameters and its borrowed device. Creation
   validates and copies all four host arrays plus optimizer settings/counter,
   and exclusively leases unique parameters with no live CPU leaves. Failure
   leaves parameters unchanged and unleased. The session owns the device and
   retains parameters even if the caller releases its references.

   CPU parameter arrays remain the pre-session recovery snapshot. No operation
   silently synchronizes them. Successful detach publishes ALL four arrays and
   optimizer settings/counter only after a complete validated device snapshot.
   Free without detach aborts: device updates are discarded, original host
   state survives. Failed detach keeps the lease and host state unchanged.
   A poisoned device must be discarded; recover from a saved checkpoint. */
typedef struct nya_train_session nya_train_session;
typedef struct nya_train_session_metrics {
    uint64_t step;
    double gradient_norm;
    uint32_t status;
} nya_train_session_metrics;

nya_train_session *nya_train_session_create(const char *backend, size_t capacity_bytes,
    const nya_train_adamw *optimizer, nya_train_parameter *const *parameters, size_t count,
    char *error, size_t error_capacity);
void nya_train_session_free(nya_train_session *session);
const char *nya_train_session_error(const nya_train_session *session);
nya_train_device *nya_train_session_device(nya_train_session *session);
/* Borrowed persistent tensor descriptors and sticky status for graph dispatch.
   Never free the device, reset persistent storage or mutate the optimizer
   counter. Graph scratch must end before detach. Tensor descriptors are copied
   to caller output; they expire at successful detach/free. */
int nya_train_session_tensor(nya_train_session *session, size_t index, nya_train_adamw_tensor *tensor);
nya_train_buffer nya_train_session_status(nya_train_session *session);
/* Queue gradient/status reset before an accumulation group. Queue one AdamW
   update at a time; observe its result before another update/reset. Dispatch
   has no host copies, allocation or implicit fence. A zero return from step
   means queued, not committed. Settings become current only on success. */
int nya_train_session_zero_grad(nya_train_session *session);
int nya_train_session_step(nya_train_session *session, nya_train_adamw_config settings);
/* Explicit synchronization boundary. Returns -1 on numerical/device failure;
   numerical failure still supplies metrics and can be recovered by zero_grad
   after correcting the offending inputs. Caller output is unchanged on device
   failure. Status zero plus the expected counter is required for success. */
int nya_train_session_observe(nya_train_session *session, nya_train_session_metrics *metrics);
/* The last successful gradient norm is diagnostic; it starts at zero for each
   new session and is not part of the existing portable checkpoint format. */
/* Snapshot operations use temporary host staging proportional to all four
   parameter arrays. They reject pending numeric errors/nonfinite state or
   negative variance before publishing host state or writing checkpoint bytes.
   Checkpoint bytes use the existing portable CPU format and can be loaded
   before creating a new session for exact GPU continuation. Caller owns file
   atomic replacement. File write failures may leave a partial stream. */
int nya_train_session_checkpoint_write(nya_train_session *session, FILE *file);
int nya_train_session_detach(nya_train_session *session, nya_train_adamw *optimizer);

/* Internal graph integration. A graph retains the session and occupies its
   single scratch scope. Update/reset/checkpoint/detach reject an active graph.
   Register immutable model data before beginning a graph; keys and extents
   identify bindings for the session lifetime, and repeated registration reuses
   the original copy. Lookup never uploads. Caller owns model/key lifetime. */
nya_train_scope nya_train_session_graph_begin(nya_train_session *session, int evaluation);
int nya_train_session_graph_end(nya_train_session *session, nya_train_scope scope);
void nya_train_session_graph_fail(nya_train_session *session);
/* Explicitly discard a completed/failed evaluation's diagnostic status without
   changing parameters, gradients or optimizer state. Training failures and
   poisoned devices cannot use this recovery path. Observe/snapshot still
   validates persistent state before reporting success or saving it. */
int nya_train_session_discard_evaluation(nya_train_session *session);
int nya_train_session_parameter(nya_train_session *session, const nya_train_parameter *parameter,
    nya_train_adamw_tensor *tensor);
nya_train_buffer nya_train_session_register(nya_train_session *session, const void *key,
    const void *data, size_t bytes);
nya_train_buffer nya_train_session_lookup(nya_train_session *session, const void *key, size_t bytes);
#endif
