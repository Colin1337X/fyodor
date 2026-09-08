#include "config.h"

#include <stdio.h>
#include <string.h>

/* Require the checked-in configuration to preserve its documented defaults. */
static int test_repository_config(const char *path)
{
    nya_config config;
    char error[512];

    nya_config_defaults(&config);
    error[0] = '\0';
    if (nya_config_load(&config, path, error, sizeof(error)) != NYA_CONFIG_OK ||
        nya_config_validate(&config, error, sizeof(error)) != 0 ||
        strcmp(config.bind_address, "127.0.0.1") != 0 ||
        config.port != 0 || config.allow_remote ||
        config.request_timeout_ms != 5000 ||
        config.max_request_bytes != 1048576 ||
        config.max_output_bytes != 67108864 ||
        config.max_generation_tokens != 256 ||
        config.backlog != 64 || config.worker_threads != 4 ||
        config.queue_capacity != 128 || config.allowed_origin_count != 1) {
        fprintf(stderr, "repository config failed validation: %s\n", error);
        return -1;
    }

    return 0;
}

/* Prove remote binding is denied until the explicit safety switch is enabled. */
static int test_remote_gate(void)
{
    nya_config config;
    char error[512];

    nya_config_defaults(&config);
    memcpy(config.bind_address, "0.0.0.0", sizeof("0.0.0.0"));
    if (nya_config_validate(&config, error, sizeof(error)) == 0) {
        fprintf(stderr, "remote bind passed without allow_remote\n");
        return -1;
    }

    config.allow_remote = 1;
    if (nya_config_validate(&config, error, sizeof(error)) != 0) {
        fprintf(stderr, "explicit remote bind failed: %s\n", error);
        return -1;
    }

    return 0;
}

/* Write an ambiguous file and prove duplicate keys fail closed. */
static int test_duplicate_rejected(const char *path)
{
    static const char duplicate_config[] =
        "server:\n"
        "  port: 0\n"
        "  port: 1\n";
    nya_config config;
    char error[512];
    FILE *file;
    nya_config_result result;

    file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "could not create duplicate config fixture\n");
        return -1;
    }
    if (fwrite(duplicate_config, 1, sizeof(duplicate_config) - 1, file) != sizeof(duplicate_config) - 1) {
        fclose(file);
        remove(path);
        fprintf(stderr, "could not create duplicate config fixture\n");
        return -1;
    }
    if (fclose(file) != 0) {
        remove(path);
        fprintf(stderr, "could not create duplicate config fixture\n");
        return -1;
    }

    nya_config_defaults(&config);
    result = nya_config_load(&config, path, error, sizeof(error));
    remove(path);
    if (result != NYA_CONFIG_ERROR) {
        fprintf(stderr, "duplicate config key was accepted\n");
        return -1;
    }

    return 0;
}

/* Exercise failures after a valid setting to verify all-or-nothing loading.
 * Length is explicit because a NUL in a file must not truncate parser input. */
static int test_invalid_files(const char *path)
{
    static const char nul_config[] = "server:\n  port: 12\n\0  bind: 0.0.0.0\n";
    static const char malformed[] = "server:\n  port: 12\n  unknown: true\n";
    static const char trailing_origin[] = "server:\n  allowed_origins: http://localhost:5173,\n";
    const struct { const char *data; size_t length; } cases[] = {
        {nul_config, sizeof(nul_config) - 1},
        {malformed, sizeof(malformed) - 1},
        {trailing_origin, sizeof(trailing_origin) - 1}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        FILE *file = fopen(path, "wb");
        nya_config config;
        nya_config original;
        char error[512];
        if (file == NULL) return -1;
        if (fwrite(cases[i].data, 1, cases[i].length, file) != cases[i].length) {
            fclose(file); remove(path); return -1;
        }
        if (fclose(file) != 0) { remove(path); return -1; }
        nya_config_defaults(&config);
        original = config;
        if (nya_config_load(&config, path, error, sizeof(error)) != NYA_CONFIG_ERROR ||
            memcmp(&config, &original, sizeof(config)) != 0) {
            remove(path);
            fprintf(stderr, "invalid configuration accepted or partially applied: %zu\n", i);
            return -1;
        }
    }
    remove(path);
    return 0;
}

static int test_invalid_structures(void)
{
    static const char *addresses[] = {"", "localhost", "127.1", "256.0.0.1", "127.00.0.1", "1.2.3.4.5"};
    nya_config config;
    char error[512];
    for (size_t i = 0; i < sizeof(addresses) / sizeof(addresses[0]); ++i) {
        nya_config_defaults(&config);
        config.allow_remote = 1;
        snprintf(config.bind_address, sizeof(config.bind_address), "%s", addresses[i]);
        if (nya_config_validate(&config, error, sizeof(error)) == 0) return -1;
    }
    nya_config_defaults(&config);
    config.allowed_origin_count = NYA_CONFIG_ORIGIN_LIMIT + 1;
    if (nya_config_validate(&config, error, sizeof(error)) == 0) return -1;
    nya_config_defaults(&config);
    memset(config.auth_token_env, 'a', sizeof(config.auth_token_env));
    if (nya_config_validate(&config, error, sizeof(error)) == 0) return -1;
    nya_config_defaults(&config);
    config.allowed_origin_count = 1;
    snprintf(config.allowed_origins[0], sizeof(config.allowed_origins[0]), "http://localhost\r\nInjected: true");
    return nya_config_validate(&config, error, sizeof(error)) != 0 ? 0 : -1;
}

/* Run every deterministic configuration check. */
int main(int argument_count, char **arguments)
{
    if (argument_count != 3) {
        fprintf(stderr, "usage: test_config CONFIG_PATH TEMP_PATH\n");
        return 2;
    }

    if (test_repository_config(arguments[1]) != 0 ||
        test_remote_gate() != 0 ||
        test_duplicate_rejected(arguments[2]) != 0 ||
        test_invalid_files(arguments[2]) != 0 || test_invalid_structures() != 0) {
        return 1;
    }

    return 0;
}
