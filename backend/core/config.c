#include "config.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Configuration files are control data, so a 64 KiB limit is generous. */
#define NYA_CONFIG_FILE_LIMIT 65536

/* Individual lines remain readable and bounded. */
#define NYA_CONFIG_LINE_LIMIT 4096

/* Store one formatted error without writing outside caller-owned memory. */
static void nya_config_error(char *error, size_t capacity, const char *format, ...)
{
    va_list arguments;

    if (error == NULL || capacity == 0) {
        return;
    }

    va_start(arguments, format);
    vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
    error[capacity - 1] = '\0';
}

/* Trim ASCII whitespace from both ends of a mutable string. */
static char *nya_config_trim(char *text)
{
    char *end;

    while (*text != '\0' && isspace((unsigned char)*text)) {
        text += 1;
    }

    end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1])) {
        end -= 1;
    }
    *end = '\0';

    return text;
}

/* Remove a YAML comment while respecting simple single- and double-quoted values. */
static void nya_config_remove_comment(char *line)
{
    char quote;
    size_t index;

    quote = '\0';
    for (index = 0; line[index] != '\0'; ++index) {
        if (quote == '\0' && (line[index] == '\'' || line[index] == '\"')) {
            quote = line[index];
            continue;
        }

        if (quote != '\0' && line[index] == quote && (index == 0 || line[index - 1] != '\\')) {
            quote = '\0';
            continue;
        }

        if (quote == '\0' && line[index] == '#') {
            line[index] = '\0';
            return;
        }
    }
}

/* Remove matching outer quotes; complex YAML scalar features are intentionally absent. */
static int nya_config_unquote(char *value)
{
    size_t length;

    length = strlen(value);
    if (length == 0) {
        return 0;
    }

    if (value[0] != '\'' && value[0] != '\"') {
        return 0;
    }

    if (length < 2 || value[length - 1] != value[0]) {
        return -1;
    }

    memmove(value, value + 1, length - 2);
    value[length - 2] = '\0';
    return 0;
}

/* Parse a strict YAML boolean. */
static int nya_config_boolean(const char *text, int *value)
{
    if (strcmp(text, "true") == 0) {
        *value = 1;
        return 0;
    }

    if (strcmp(text, "false") == 0) {
        *value = 0;
        return 0;
    }

    return -1;
}

/* Parse a non-negative decimal integer without accepting suffixes or signs. */
static int nya_config_unsigned(const char *text, unsigned long long maximum, unsigned long long *value)
{
    unsigned long long parsed;
    char *end;

    if (text[0] == '\0' || !isdigit((unsigned char)text[0])) {
        return -1;
    }

    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > maximum) {
        return -1;
    }

    *value = parsed;
    return 0;
}

/* Copy one scalar only when it fits with its null terminator. */
static int nya_config_copy(char *destination, size_t capacity, const char *value)
{
    size_t length;

    length = strlen(value);
    if (length >= capacity) {
        return -1;
    }

    memcpy(destination, value, length + 1);
    return 0;
}

/* Split the documented comma-separated origin scalar into bounded entries. */
static int nya_config_origins(nya_config *config, char *value)
{
    char *cursor;

    config->allowed_origin_count = 0;
    cursor = value;

    while (*cursor != '\0') {
        char *comma;
        char *origin;

        comma = strchr(cursor, ',');
        if (comma != NULL) {
            *comma = '\0';
        }

        origin = nya_config_trim(cursor);
        if (origin[0] == '\0' || config->allowed_origin_count >= NYA_CONFIG_ORIGIN_LIMIT) {
            return -1;
        }

        if (nya_config_copy(
                config->allowed_origins[config->allowed_origin_count],
                NYA_CONFIG_ORIGIN_LENGTH,
                origin
            ) != 0) {
            return -1;
        }
        config->allowed_origin_count += 1;

        if (comma == NULL) {
            break;
        }
        cursor = comma + 1;
        if (*cursor == '\0') {
            /* A trailing delimiter is an empty origin, not a second spelling
             * for the previous list. Keep configuration errors explicit. */
            return -1;
        }
    }

    return 0;
}

/* Give each supported key one bit so duplicate settings are unambiguous errors. */
static unsigned int nya_config_key_bit(const char *key)
{
    if (strcmp(key, "bind") == 0) return 1U << 0;
    if (strcmp(key, "port") == 0) return 1U << 1;
    if (strcmp(key, "allow_remote") == 0) return 1U << 2;
    if (strcmp(key, "auth_token_env") == 0) return 1U << 3;
    if (strcmp(key, "request_timeout_ms") == 0) return 1U << 4;
    if (strcmp(key, "max_request_bytes") == 0) return 1U << 5;
    if (strcmp(key, "max_output_bytes") == 0) return 1U << 6;
    if (strcmp(key, "backlog") == 0) return 1U << 7;
    if (strcmp(key, "worker_threads") == 0) return 1U << 8;
    if (strcmp(key, "queue_capacity") == 0) return 1U << 9;
    if (strcmp(key, "allowed_origins") == 0) return 1U << 10;
    if (strcmp(key, "max_generation_tokens") == 0) return 1U << 11;
    return 0;
}

/* Apply one known server key and reject every unknown key. */
static int nya_config_apply(nya_config *config, const char *key, char *value)
{
    unsigned long long number;

    if (strcmp(key, "bind") == 0) {
        return nya_config_copy(config->bind_address, sizeof(config->bind_address), value);
    }

    if (strcmp(key, "port") == 0) {
        if (nya_config_unsigned(value, 65535, &number) != 0) {
            return -1;
        }
        config->port = (unsigned short)number;
        return 0;
    }

    if (strcmp(key, "allow_remote") == 0) {
        return nya_config_boolean(value, &config->allow_remote);
    }

    if (strcmp(key, "auth_token_env") == 0) {
        return nya_config_copy(config->auth_token_env, sizeof(config->auth_token_env), value);
    }

    if (strcmp(key, "request_timeout_ms") == 0) {
        if (nya_config_unsigned(value, 300000, &number) != 0) {
            return -1;
        }
        config->request_timeout_ms = (unsigned int)number;
        return 0;
    }

    if (strcmp(key, "max_request_bytes") == 0) {
        if (nya_config_unsigned(value, 16ULL * 1024ULL * 1024ULL, &number) != 0) {
            return -1;
        }
        config->max_request_bytes = (size_t)number;
        return 0;
    }

    if (strcmp(key, "max_output_bytes") == 0) {
        if (nya_config_unsigned(value, 1024ULL * 1024ULL * 1024ULL, &number) != 0) {
            return -1;
        }
        config->max_output_bytes = (size_t)number;
        return 0;
    }

    if (strcmp(key, "max_generation_tokens") == 0) {
        if (nya_config_unsigned(value, 4096, &number) != 0) {
            return -1;
        }
        config->max_generation_tokens = (unsigned int)number;
        return 0;
    }

    if (strcmp(key, "backlog") == 0) {
        if (nya_config_unsigned(value, 512, &number) != 0) {
            return -1;
        }
        config->backlog = (unsigned int)number;
        return 0;
    }

    if (strcmp(key, "worker_threads") == 0) {
        if (nya_config_unsigned(value, 64, &number) != 0) {
            return -1;
        }
        config->worker_threads = (unsigned int)number;
        return 0;
    }

    if (strcmp(key, "queue_capacity") == 0) {
        if (nya_config_unsigned(value, 4096, &number) != 0) {
            return -1;
        }
        config->queue_capacity = (unsigned int)number;
        return 0;
    }

    if (strcmp(key, "allowed_origins") == 0) {
        return nya_config_origins(config, value);
    }

    return -1;
}

/* Start from settings safe enough for a local Tauri child process. */
void nya_config_defaults(nya_config *config)
{
    if (config == NULL) {
        return;
    }

    memset(config, 0, sizeof(*config));
    memcpy(config->bind_address, "127.0.0.1", sizeof("127.0.0.1"));
    config->port = 0;
    config->allow_remote = 0;
    config->request_timeout_ms = 5000;
    config->max_request_bytes = 1024 * 1024;
    config->max_output_bytes = 64 * 1024 * 1024;
    config->max_generation_tokens = 256;
    config->backlog = 64;
    config->worker_threads = 4;
    config->queue_capacity = 128;
}

/* Load a deliberately small, strict, and dependency-free subset of YAML. */
nya_config_result nya_config_load(
    nya_config *config,
    const char *path,
    char *error,
    size_t error_capacity
)
{
    FILE *file;
    char line[NYA_CONFIG_LINE_LIMIT];
    size_t total_bytes;
    unsigned long line_number;
    int in_server;
    int server_seen;
    unsigned int seen_keys;
    nya_config staged;
    nya_config *destination;

    if (config == NULL || path == NULL) {
        nya_config_error(error, error_capacity, "invalid configuration arguments");
        return NYA_CONFIG_ERROR;
    }

    /* Loading is transactional. A malformed later line cannot leave a caller
     * with half-applied settings, which matters to future configuration reloads. */
    destination = config;
    staged = *config;
    config = &staged;

    file = fopen(path, "rb");
    if (file == NULL) {
        if (errno == ENOENT) {
            return NYA_CONFIG_NOT_FOUND;
        }
        nya_config_error(error, error_capacity, "cannot open %s", path);
        return NYA_CONFIG_ERROR;
    }

    total_bytes = 0;
    line_number = 0;
    in_server = 0;
    server_seen = 0;
    seen_keys = 0;

    for (;;) {
        size_t length;
        size_t indent;
        char *content;
        int character;

        line_number += 1;
        length = 0;
        /* Count physical bytes rather than strlen(fgets(...)): embedded NUL
         * must not hide settings or bypass the file-size bound. stdio buffers
         * fgetc internally, so this tiny control file needs no custom reader. */
        while ((character = fgetc(file)) != EOF) {
            total_bytes += 1;
            if (total_bytes > NYA_CONFIG_FILE_LIMIT || character == 0 ||
                length == sizeof(line) - 1) {
                nya_config_error(error, error_capacity, "%s:%lu invalid byte or size limit exceeded", path, line_number);
                fclose(file);
                return NYA_CONFIG_ERROR;
            }
            line[length++] = (char)character;
            if (character == '\n') break;
        }
        if (length == 0) break;
        line[length] = '\0';

        /* A full buffer without a newline means one unsupported oversized line. */
        if (length == sizeof(line) - 1 && line[length - 1] != '\n') {
            nya_config_error(error, error_capacity, "%s:%lu line is too long", path, line_number);
            fclose(file);
            return NYA_CONFIG_ERROR;
        }

        if (strchr(line, '\t') != NULL) {
            nya_config_error(error, error_capacity, "%s:%lu tabs are not allowed", path, line_number);
            fclose(file);
            return NYA_CONFIG_ERROR;
        }

        nya_config_remove_comment(line);
        length = strlen(line);
        while (length > 0 && (line[length - 1] == '\r' || line[length - 1] == '\n')) {
            line[length - 1] = '\0';
            length -= 1;
        }

        indent = 0;
        while (line[indent] == ' ') {
            indent += 1;
        }
        content = nya_config_trim(line + indent);
        if (content[0] == '\0') {
            continue;
        }

        if (indent == 0) {
            if (strcmp(content, "server:") != 0 || server_seen) {
                nya_config_error(error, error_capacity, "%s:%lu expected server:", path, line_number);
                fclose(file);
                return NYA_CONFIG_ERROR;
            }
            in_server = 1;
            server_seen = 1;
            continue;
        }

        if (!in_server || indent != 2) {
            nya_config_error(error, error_capacity, "%s:%lu server keys require two spaces", path, line_number);
            fclose(file);
            return NYA_CONFIG_ERROR;
        }

        {
            char *colon;
            char *key;
            char *value;
            unsigned int key_bit;

            colon = strchr(content, ':');
            if (colon == NULL) {
                nya_config_error(error, error_capacity, "%s:%lu expected key: value", path, line_number);
                fclose(file);
                return NYA_CONFIG_ERROR;
            }

            *colon = '\0';
            key = nya_config_trim(content);
            value = nya_config_trim(colon + 1);
            key_bit = nya_config_key_bit(key);
            if (key[0] == '\0' || key_bit == 0 || (seen_keys & key_bit) != 0 ||
                value[0] == '\0' || nya_config_unquote(value) != 0 ||
                nya_config_apply(config, key, value) != 0) {
                nya_config_error(error, error_capacity, "%s:%lu invalid or unknown server setting", path, line_number);
                fclose(file);
                return NYA_CONFIG_ERROR;
            }
            seen_keys |= key_bit;
        }
    }

    if (ferror(file)) {
        nya_config_error(error, error_capacity, "failed while reading %s", path);
        fclose(file);
        return NYA_CONFIG_ERROR;
    }

    fclose(file);
    *destination = staged;
    return NYA_CONFIG_OK;
}

/* Accept only four decimal IPv4 octets. This avoids both DNS resolution and
 * platform-dependent legacy forms such as shortened or octal addresses. */
static int nya_config_ipv4(const char *text)
{
    for (unsigned int octet = 0; octet < 4; ++octet) {
        unsigned int value = 0;
        size_t digits = 0;
        const char *start = text;
        while (*text >= '0' && *text <= '9') {
            value = value * 10U + (unsigned int)(*text++ - '0');
            if (++digits > 3 || value > 255) return 0;
        }
        if (digits == 0 || (digits > 1 && *start == '0')) return 0;
        if (octet == 3) return *text == '\0';
        if (*text++ != '.') return 0;
    }
    return 0;
}

/* Validate cross-field constraints after all configuration sources are applied. */
int nya_config_validate(const nya_config *config, char *error, size_t error_capacity)
{
    size_t index;

    if (config == NULL ||
        memchr(config->bind_address, '\0', sizeof(config->bind_address)) == NULL ||
        !nya_config_ipv4(config->bind_address)) {
        nya_config_error(error, error_capacity, "server.bind must be an explicit dotted-decimal IPv4 address");
        return -1;
    }
    if (memchr(config->auth_token_env, '\0', sizeof(config->auth_token_env)) == NULL ||
        config->allowed_origin_count > NYA_CONFIG_ORIGIN_LIMIT ||
        (config->allow_remote != 0 && config->allow_remote != 1)) {
        nya_config_error(error, error_capacity, "invalid bounded configuration structure");
        return -1;
    }

    if (strcmp(config->bind_address, "127.0.0.1") != 0 && !config->allow_remote) {
        nya_config_error(error, error_capacity, "non-loopback binding requires server.allow_remote: true");
        return -1;
    }

    if (config->auth_token_env[0] != '\0') {
        if (!(isalpha((unsigned char)config->auth_token_env[0]) || config->auth_token_env[0] == '_')) {
            nya_config_error(error, error_capacity, "server.auth_token_env is not a valid environment variable name");
            return -1;
        }
        for (index = 1; config->auth_token_env[index] != '\0'; ++index) {
            if (!(isalnum((unsigned char)config->auth_token_env[index]) || config->auth_token_env[index] == '_')) {
                nya_config_error(error, error_capacity, "server.auth_token_env is not a valid environment variable name");
                return -1;
            }
        }
    }

    if (config->request_timeout_ms < 100 || config->request_timeout_ms > 300000) {
        nya_config_error(error, error_capacity, "server.request_timeout_ms must be 100..300000");
        return -1;
    }

    if (config->max_request_bytes < 4096 || config->max_request_bytes > 16U * 1024U * 1024U) {
        nya_config_error(error, error_capacity, "server.max_request_bytes must be 4096..16777216");
        return -1;
    }

    if (config->max_output_bytes < 4096 || config->max_output_bytes > 1024ULL * 1024ULL * 1024ULL) {
        nya_config_error(error, error_capacity, "server.max_output_bytes must be 4096..1073741824");
        return -1;
    }

    if (config->max_generation_tokens == 0 || config->max_generation_tokens > 4096) {
        nya_config_error(error, error_capacity, "server.max_generation_tokens must be 1..4096");
        return -1;
    }

    if (config->backlog == 0 || config->backlog > 512 ||
        config->worker_threads == 0 || config->worker_threads > 64 ||
        config->queue_capacity < config->worker_threads || config->queue_capacity > 4096) {
        nya_config_error(error, error_capacity, "invalid backlog, worker_threads, or queue_capacity");
        return -1;
    }

    for (index = 0; index < config->allowed_origin_count; ++index) {
        const char *origin;

        origin = config->allowed_origins[index];
        if (memchr(origin, '\0', NYA_CONFIG_ORIGIN_LENGTH) == NULL) {
            nya_config_error(error, error_capacity, "allowed origin exceeds length limit");
            return -1;
        }
        if (strncmp(origin, "http://", 7) != 0 &&
            strncmp(origin, "https://", 8) != 0 &&
            strcmp(origin, "tauri://localhost") != 0) {
            nya_config_error(error, error_capacity, "allowed origins must use http, https, or tauri://localhost");
            return -1;
        }
        /* Origins are serialized into HTTP headers. Never permit whitespace or
         * control bytes, even when the configuration came from a C caller. */
        for (const unsigned char *cursor = (const unsigned char *)origin; *cursor != 0; ++cursor) {
            if (*cursor <= 32 || *cursor == 127) {
                nya_config_error(error, error_capacity, "allowed origins cannot contain whitespace/control bytes");
                return -1;
            }
        }
    }

    return 0;
}
