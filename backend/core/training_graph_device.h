#ifndef NYA_TRAINING_GRAPH_DEVICE_H
#define NYA_TRAINING_GRAPH_DEVICE_H
#include "training_session.h"
#include "pretraining.h"

/* Private accelerator graph entry point. Existing tensor operations record a
   graph; explicit forward/data/backward executes its nodes on the session's
   ordered stream. Inputs/metadata are copied during construction, before the
   forward/backward region. Data access explicitly downloads a host snapshot.
   Graph memory_limit charges graph metadata and device intermediates together;
   persistent session buffers and driver/context overhead are separate.
   One graph occupies the session scratch scope and retains its lifetime.
   Free the graph before optimizer/reset/checkpoint/detach. Numerical failures
   are sticky in the session status and gate the optimizer. No CPU replay. */
nya_train_graph *nya_train_graph_create_resident(size_t memory_limit,
    nya_train_session *session, int evaluation);
int nya_train_graph_forward_resident(nya_train_graph *graph);
/* Upload immutable frozen model tensors before constructing graphs. Fully
   trainable decoders need no extra model uploads. Repeated calls reuse copies.
   Keep the decoder and its source mapping alive until all graphs and the
   session have been released. Frozen tensor descriptors are borrowed keys. */
int nya_train_decoder_prepare_resident(nya_train_decoder *decoder,
    nya_train_session *session, char *error, size_t error_capacity);
#endif
