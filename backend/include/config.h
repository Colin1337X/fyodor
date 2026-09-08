#ifndef NYA_CONFIG_H
#define NYA_CONFIG_H

#include <stddef.h>

/* IPv4 text, including its terminator, always fits in 16 bytes; extra room is clear. */
#define NYA_CONFIG_ADDRESS_LIMIT 64

/* Environment variable names are kept small and bounded. */
#define NYA_CONFIG_ENV_NAME_LIMIT 128

/* Remote browser deployments may allow a short explicit list of origins. */
#define NYA_CONFIG_ORIGIN_LIMIT 16
#define NYA_CONFIG_ORIGIN_LENGTH 256

/* Configuration results distinguish an optional missing file from invalid input. */
typedef enum nya_config_result {
    NYA_CONFIG_OK = 0,
    NYA_CONFIG_NOT_FOUND = 1,
    NYA_CONFIG_ERROR = -1
} nya_config_result;

/* One structure contains every server setting that may change after compilation. */
typedef struct nya_config {
    /* The server accepts one explicit IPv4 bind address. */
    char bind_address[NYA_CONFIG_ADDRESS_LIMIT];

    /* Port zero asks the operating system to select an available port. */
    unsigned short port;

    /* Non-loopback binding requires this deliberate safety switch. */
    int allow_remote;

    /* A named environment variable may provide a stable 64-digit token. */
    char auth_token_env[NYA_CONFIG_ENV_NAME_LIMIT];

    /* Slow or abandoned request bodies are disconnected after this interval. */
    unsigned int request_timeout_ms;

    /* The complete request, headers plus body, may not exceed this many bytes. */
    size_t max_request_bytes;

    /* One inference response may not allocate or return more than this many bytes. */
    size_t max_output_bytes;

    /* One generation request cannot sample more than this many new tokens. */
    unsigned int max_generation_tokens;

    /* listen() uses this bounded pending-connection backlog. */
    unsigned int backlog;

    /* A fixed worker pool handles clients without creating unbounded threads. */
    unsigned int worker_threads;

    /* Accepted sockets wait in this bounded in-process queue. */
    unsigned int queue_capacity;

    /* These origins are added to the built-in local Tauri origins. */
    char allowed_origins[NYA_CONFIG_ORIGIN_LIMIT][NYA_CONFIG_ORIGIN_LENGTH];
    size_t allowed_origin_count;
} nya_config;

/* Fill a configuration with conservative localhost defaults. */
void nya_config_defaults(nya_config *config);

/* Load the documented strict YAML subset over existing defaults. */
nya_config_result nya_config_load(
    nya_config *config,
    const char *path,
    char *error,
    size_t error_capacity
);

/* Validate values after both file settings and command-line overrides. */
int nya_config_validate(const nya_config *config, char *error, size_t error_capacity);

#endif
