#include "config.h"
#include "model.h"
#include "nya.h"
#include "runtime.h"
#include "server.h"
#include "compute.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* sig_atomic_t is the C-standard type that a signal handler may safely assign. */
static volatile sig_atomic_t nya_stop_signal = 0;

/* Ask the polling server loop to stop after Ctrl+C or a termination signal. */
static void nya_signal_handler(int signal_number)
{
    /* The actual signal value is irrelevant; every registered signal means stop. */
    (void)signal_number;

    nya_stop_signal = 1;
}

/* Convert 32 random bytes into the 64-character lowercase bearer token. */
static int nya_create_auth_token(char token[NYA_AUTH_TOKEN_LENGTH + 1])
{
    static const char hexadecimal[] = "0123456789abcdef";
    unsigned char random_bytes[NYA_AUTH_TOKEN_LENGTH / 2];
    size_t index;

    if (nya_runtime_random_bytes(random_bytes, sizeof(random_bytes)) != 0) {
        return -1;
    }

    for (index = 0; index < sizeof(random_bytes); ++index) {
        token[index * 2] = hexadecimal[random_bytes[index] >> 4];
        token[index * 2 + 1] = hexadecimal[random_bytes[index] & 0x0f];
    }

    token[NYA_AUTH_TOKEN_LENGTH] = '\0';
    return 0;
}

/* Parse a base-10 TCP port in the inclusive range 0 through 65535. */
static int nya_parse_port(const char *text, unsigned short *port)
{
    unsigned long value;
    char *end;

    if (text == NULL || text[0] < '0' || text[0] > '9') {
        return -1;
    }

    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || *end != '\0' || value > 65535UL) {
        return -1;
    }

    *port = (unsigned short)value;
    return 0;
}

/* Copy one command-line string only when it fits in the destination. */
static int nya_copy_option(char *destination, size_t capacity, const char *source)
{
    size_t length;

    length = strlen(source);
    if (length >= capacity) {
        return -1;
    }

    memcpy(destination, source, length + 1);
    return 0;
}

/* Return true only for the exact 64-digit lowercase or uppercase hex token syntax. */
static int nya_auth_token_valid(const char *token)
{
    size_t index;

    if (token == NULL || strlen(token) != NYA_AUTH_TOKEN_LENGTH) {
        return 0;
    }

    for (index = 0; index < NYA_AUTH_TOKEN_LENGTH; ++index) {
        if (!((token[index] >= '0' && token[index] <= '9') ||
              (token[index] >= 'a' && token[index] <= 'f') ||
              (token[index] >= 'A' && token[index] <= 'F'))) {
            return 0;
        }
    }

    return 1;
}

/* Print the intentionally tiny command-line surface. */
static void nya_print_usage(const char *program_name)
{
    printf(
        "Usage: %s [--config PATH] [--bind IPV4] [--port PORT] [--allow-remote]\n"
        "\n"
        "  --config PATH  Read this file instead of ./config.yaml.\n"
        "  --bind IPV4    Override server.bind.\n"
        "  --port PORT    Override server.port; zero selects a free port.\n"
        "  --allow-remote Permit a non-loopback bind address.\n"
        "  --verbose      Log runtime, request and generation diagnostics to stderr.\n"
        "  --help         Show this help text.\n",
        program_name
    );
}

/* Own all engine state and release it in the reverse order of initialization. */
int main(int argument_count, char **arguments)
{
    nya_config configuration;
    char config_path[1024];
    int config_explicit;
    char config_error[512];
    nya_config_result config_result;
    char auth_token[NYA_AUTH_TOKEN_LENGTH + 1];
    nya_runtime runtime;
    nya_model_registry models;
    nya_server server;
    nya_server_config server_config;
    int server_initialized;
    int exit_code;
    int index;
    int verbose = 0;

    /* Load conservative defaults before applying the optional YAML file. */
    nya_config_defaults(&configuration);
    memcpy(config_path, "config.yaml", sizeof("config.yaml"));
    config_explicit = 0;

    /* Find the configuration path first so command-line overrides can win later. */
    for (index = 1; index < argument_count; ++index) {
        if (strcmp(arguments[index], "--help") == 0) {
            /* Help must remain available even beside a broken config file. */
            nya_print_usage(arguments[0]);
            return 0;
        }
        if (strcmp(arguments[index], "--config") == 0) {
            if (index + 1 >= argument_count ||
                nya_copy_option(config_path, sizeof(config_path), arguments[index + 1]) != 0) {
                fprintf(stderr, "fyodor: --config requires a bounded path\n");
                return 2;
            }
            config_explicit = 1;
            index += 1;
        }
    }

    config_result = nya_config_load(
        &configuration,
        config_path,
        config_error,
        sizeof(config_error)
    );
    if (config_result == NYA_CONFIG_ERROR ||
        (config_result == NYA_CONFIG_NOT_FOUND && config_explicit)) {
        fprintf(
            stderr,
            "fyodor: configuration error: %s\n",
            config_result == NYA_CONFIG_NOT_FOUND ? "requested file does not exist" : config_error
        );
        return 2;
    }

    /* Apply explicit command-line overrides after file configuration. */
    for (index = 1; index < argument_count; ++index) {
        if (strcmp(arguments[index], "--help") == 0) {
            nya_print_usage(arguments[0]);
            return 0;
        } else if (strcmp(arguments[index], "--config") == 0) {
            index += 1;
        } else if (strcmp(arguments[index], "--bind") == 0) {
            if (index + 1 >= argument_count ||
                nya_copy_option(
                    configuration.bind_address,
                    sizeof(configuration.bind_address),
                    arguments[index + 1]
                ) != 0) {
                fprintf(stderr, "fyodor: --bind requires a bounded IPv4 address\n");
                return 2;
            }
            index += 1;
        } else if (strcmp(arguments[index], "--port") == 0) {
            if (index + 1 >= argument_count ||
                nya_parse_port(arguments[index + 1], &configuration.port) != 0) {
                fprintf(stderr, "fyodor: --port requires a number from 0 through 65535\n");
                return 2;
            }
            index += 1;
        } else if (strcmp(arguments[index], "--allow-remote") == 0) {
            configuration.allow_remote = 1;
        } else if (strcmp(arguments[index], "--verbose") == 0) {
            verbose = 1;
        } else {
            fprintf(stderr, "fyodor: unknown option: %s\n", arguments[index]);
            nya_print_usage(arguments[0]);
            return 2;
        }
    }

    if (nya_config_validate(&configuration, config_error, sizeof(config_error)) != 0) {
        fprintf(stderr, "fyodor: configuration error: %s\n", config_error);
        return 2;
    }

    /* Initialize local variables before entering shared cleanup paths. */
    memset(&runtime, 0, sizeof(runtime));
    memset(&models, 0, sizeof(models));
    memset(&server, 0, sizeof(server));
    server_initialized = 0;
    exit_code = 1;

    if (nya_runtime_init(&runtime) != 0) {
        fprintf(stderr, "fyodor: runtime initialization failed\n");
        goto cleanup;
    }

    nya_model_registry_init(&models);

    /* Read a configured secret from the environment or generate a fresh one. */
    if (configuration.auth_token_env[0] != '\0') {
        const char *configured_token;

        configured_token = getenv(configuration.auth_token_env);
        if (!nya_auth_token_valid(configured_token)) {
            fprintf(stderr, "fyodor: %s must contain exactly 64 hexadecimal digits\n", configuration.auth_token_env);
            goto cleanup;
        }
        memcpy(auth_token, configured_token, NYA_AUTH_TOKEN_LENGTH + 1);
    } else if (nya_create_auth_token(auth_token) != 0) {
        fprintf(stderr, "fyodor: secure token generation failed\n");
        goto cleanup;
    }

    server_config.settings = &configuration;
    server_config.auth_token = auth_token;
    server_config.runtime = &runtime;
    server_config.models = &models;
    server_config.external_stop_requested = &nya_stop_signal;
    server_config.verbose = verbose;
    if (verbose) fprintf(stderr,"[runtime] compiled cpu=1 cuda=%d vulkan=%d workers=%u queue=%u request_limit=%zu output_limit=%zu generation_limit=%u timeout_ms=%u\n",
        nya_compute_cuda_compiled(),nya_compute_vulkan_compiled(),configuration.worker_threads,
        configuration.queue_capacity,configuration.max_request_bytes,configuration.max_output_bytes,
        configuration.max_generation_tokens,configuration.request_timeout_ms);

    if (nya_server_init(&server, &server_config) != 0) {
        fprintf(stderr, "fyodor: could not bind the configured server address\n");
        goto cleanup;
    }
    server_initialized = 1;

    /* Keep human logs on stderr so stdout remains a machine-readable handshake. */
    fprintf(
        stderr,
        "fyodor %d.%d.%d: ready on %s:%u with %u workers\n",
        NYA_VERSION_MAJOR,
        NYA_VERSION_MINOR,
        NYA_VERSION_PATCH,
        configuration.bind_address,
        (unsigned int)nya_server_port(&server),
        configuration.worker_threads
    );

    /* Tauri reads this single flushed line to discover the port and token. */
    printf(
        "FYODOR_READY %s %u %s\n",
        strcmp(configuration.bind_address, "0.0.0.0") == 0 ? "127.0.0.1" : configuration.bind_address,
        (unsigned int)nya_server_port(&server),
        auth_token
    );
    fflush(stdout);

    /* The polling run loop makes these simple signal handlers sufficient. */
    signal(SIGINT, nya_signal_handler);
#ifdef SIGTERM
    signal(SIGTERM, nya_signal_handler);
#endif

    if (nya_server_run(&server) != 0) {
        fprintf(stderr, "fyodor: server loop failed\n");
        goto cleanup;
    }

    exit_code = 0;

cleanup:
    if (verbose) fprintf(stderr,"[runtime] shutdown starting exit_code=%d\n",exit_code);
    if (server_initialized) {
        nya_server_shutdown(&server);
    }

    nya_model_registry_shutdown(&models);
    nya_runtime_shutdown(&runtime);
    if (verbose) fprintf(stderr,"[runtime] shutdown complete\n");

    /* Erase the secret before the process returns. */
    memset(auth_token, 0, sizeof(auth_token));
    return exit_code;
}
