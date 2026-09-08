#ifndef NYA_SERVER_H
#define NYA_SERVER_H

#include <signal.h>
#include <stdint.h>

#ifdef _WIN32
/* MSVC's C23 mode still requires an experimental switch for stdatomic.h. */
typedef volatile long nya_atomic_int;
#else
/* Other supported C23 compilers provide the standard atomic integer type. */
#include <stdatomic.h>
typedef atomic_int nya_atomic_int;
#endif

#include "config.h"
#include "model.h"
#include "runtime.h"

/* The bearer token contains 32 random bytes encoded as 64 hexadecimal digits. */
#define NYA_AUTH_TOKEN_LENGTH 64

/* Configuration is borrowed by the server for the duration of its run loop. */
typedef struct nya_server_config {
    /* Compiled server behavior is supplied by the loaded YAML configuration. */
    const nya_config *settings;

    /* The token is required by every API request except CORS preflight. */
    const char *auth_token;

    /* The server reads runtime state for health information. */
    nya_runtime *runtime;

    /* The server mutates the model registry through model endpoints. */
    nya_model_registry *models;

    /* A signal handler may set this optional C-standard atomic stop flag. */
    const volatile sig_atomic_t *external_stop_requested;
    /* Opt-in stderr diagnostics. Never include credentials or prompt bodies. */
    int verbose;
} nya_server_config;

/* The server object contains only the state needed by one listening process. */
typedef struct nya_server {
    /* A pointer-sized integer can safely hold either a POSIX fd or Windows SOCKET. */
    uintptr_t socket_handle;

    /* The actual port differs from the requested port when port zero is used. */
    unsigned short port;

    /* The run loop checks this flag between short socket waits. */
    nya_atomic_int stop_requested;

    /* Windows requires matching WSAStartup and WSACleanup calls. */
    int networking_initialized;

    /* The server borrows its configuration instead of copying engine state. */
    nya_server_config config;

    /* Platform-neutral worker internals remain private to server.c. */
    void *internal;
} nya_server;

/* Create and bind the configured IPv4 listening socket. */
int nya_server_init(nya_server *server, const nya_server_config *config);

/* Process requests until shutdown is requested or an unrecoverable error occurs. */
int nya_server_run(nya_server *server);

/* Thread-safe stop request; the listener checks within 250 ms. */
void nya_server_request_stop(nya_server *server);

/* Call after nya_server_run returns. Interrupt active socket I/O, close queued
   clients, and join workers before freeing shared registry/runtime state.
   A provider call already computing finishes before its worker can join. */
void nya_server_shutdown(nya_server *server);

/* Return the bound port for the startup handshake. */
unsigned short nya_server_port(const nya_server *server);

#endif
