#include "server.h"

#include "nya.h"
#include "execution.h"
#include "generation.h"
#include "thread.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "llm_internal.h"

#ifdef _WIN32
/* Winsock is the native Windows sockets interface. */
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
typedef SOCKET nya_socket;
typedef int nya_socket_length;
#define NYA_INVALID_SOCKET INVALID_SOCKET
#else
/* These headers provide the small POSIX sockets surface used below. */
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int nya_socket;
typedef socklen_t nya_socket_length;
#define NYA_INVALID_SOCKET (-1)
#endif

/* Header blocks have a tighter limit than configurable complete request bodies. */
#define NYA_HTTP_HEADER_BLOCK_LIMIT 32768

/* JSON metadata responses have a fixed 128 KiB serialization boundary. */
#define NYA_HTTP_RESPONSE_LIMIT 131072

/* Individual HTTP fields have small explicit bounds. */
#define NYA_HTTP_METHOD_LIMIT 16
#define NYA_HTTP_PATH_LIMIT 256
#define NYA_HTTP_HEADER_LIMIT 256

/* A parsed request borrows its body from the receive buffer. */
typedef struct nya_http_request {
    char method[NYA_HTTP_METHOD_LIMIT];
    char path[NYA_HTTP_PATH_LIMIT];
    char authorization[NYA_HTTP_HEADER_LIMIT];
    char api_key[NYA_HTTP_HEADER_LIMIT];
    char anthropic_version[64];
    int api_style; /* 0 native, 1 OpenAI chat, 2 Anthropic, 3 OpenAI completion */
    char origin[NYA_HTTP_HEADER_LIMIT];
    char content_type[NYA_HTTP_HEADER_LIMIT];
    char model_id[32];
    char input_name[128];
    char input_type[32];
    char input_shape[256];
    char output_name[128];
    const char *body;
    size_t body_length;
} nya_http_request;

/* This tiny builder makes every bounded response append explicit and checked. */
typedef struct nya_text_builder {
    char *data;
    size_t capacity;
    size_t length;
    int failed;
} nya_text_builder;

struct nya_server_internal;

/* Each worker publishes its current socket under queue_mutex. Shutdown may
   interrupt I/O with shutdown(), while only the worker ever closes the socket. */
typedef struct nya_server_worker_state {
    struct nya_server_internal *internal;
    nya_socket client;
} nya_server_worker_state;

/* The private worker pool owns all synchronization and accepted-socket queues. */
typedef struct nya_server_internal {
    nya_server *server;
    nya_thread *threads;
    nya_server_worker_state *workers;
    unsigned int started_threads;
    nya_socket *queue;
    size_t queue_capacity;
    size_t queue_head;
    size_t queue_tail;
    size_t queue_count;
    int shutting_down;
    nya_mutex queue_mutex;
    nya_condition queue_ready;
    nya_mutex model_mutex;
    nya_mutex execution_mutex;
    int queue_mutex_initialized;
    int queue_condition_initialized;
    int model_mutex_initialized;
    int execution_mutex_initialized;
} nya_server_internal;

/* Worker threads call the same bounded one-client handler declared below. */
static void nya_server_handle_client(nya_socket client, nya_server *server);

/* Initialize, read, and write the one cross-thread server stop flag. */
static void nya_server_stop_initialize(nya_server *server)
{
#ifdef _WIN32
    InterlockedExchange(&server->stop_requested, 0);
#else
    atomic_init(&server->stop_requested, 0);
#endif
}

static int nya_server_stop_requested(nya_server *server)
{
#ifdef _WIN32
    return InterlockedCompareExchange(&server->stop_requested, 0, 0) != 0;
#else
    return atomic_load(&server->stop_requested) != 0;
#endif
}

static void nya_server_stop_request(nya_server *server)
{
#ifdef _WIN32
    InterlockedExchange(&server->stop_requested, 1);
#else
    atomic_store(&server->stop_requested, 1);
#endif
}

/* Convert the public pointer-sized storage back to the native socket type. */
static nya_socket nya_socket_from_handle(uintptr_t handle)
{
    return (nya_socket)handle;
}

/* Store either kind of native socket without exposing platform headers publicly. */
static uintptr_t nya_socket_to_handle(nya_socket socket_value)
{
    return (uintptr_t)socket_value;
}

/* Close one connected or listening socket using the host API. */
static void nya_socket_close(nya_socket socket_value)
{
    if (socket_value == NYA_INVALID_SOCKET) {
        return;
    }

#ifdef _WIN32
    closesocket(socket_value);
#else
    close(socket_value);
#endif
}

/* Pop accepted sockets and process them until the server begins shutdown. */
static int nya_server_worker(void *argument)
{
    nya_server_internal *internal;
    nya_server_worker_state *worker;

    worker = (nya_server_worker_state *)argument;
    internal = worker->internal;
    for (;;) {
        nya_socket client;

        nya_mutex_lock(&internal->queue_mutex);
        while (internal->queue_count == 0 && !internal->shutting_down) {
            nya_condition_wait(&internal->queue_ready, &internal->queue_mutex);
        }

        if (internal->shutting_down) {
            nya_mutex_unlock(&internal->queue_mutex);
            return 0;
        }

        client = internal->queue[internal->queue_head];
        internal->queue_head = (internal->queue_head + 1) % internal->queue_capacity;
        internal->queue_count -= 1;
        worker->client = client;
        nya_mutex_unlock(&internal->queue_mutex);

        nya_server_handle_client(client, internal->server);
        /* Serialize close and clearing the active slot with shutdown's scan.
           Otherwise a recycled descriptor could interrupt an unrelated socket. */
        nya_mutex_lock(&internal->queue_mutex);
        nya_socket_close(client);
        worker->client = NYA_INVALID_SOCKET;
        nya_mutex_unlock(&internal->queue_mutex);
    }
}

/* Allocate synchronization, the socket queue, and a fixed number of workers. */
static int nya_server_workers_init(nya_server *server)
{
    nya_server_internal *internal;
    unsigned int index;

    internal = (nya_server_internal *)calloc(1, sizeof(*internal));
    if (internal == NULL) {
        return -1;
    }

    internal->server = server;
    internal->queue_capacity = server->config.settings->queue_capacity;
    internal->queue = (nya_socket *)calloc(internal->queue_capacity, sizeof(*internal->queue));
    internal->threads = (nya_thread *)calloc(
        server->config.settings->worker_threads,
        sizeof(*internal->threads)
    );
    internal->workers = (nya_server_worker_state *)calloc(
        server->config.settings->worker_threads, sizeof(*internal->workers));
    if (internal->queue == NULL || internal->threads == NULL || internal->workers == NULL) {
        free(internal->queue);
        free(internal->threads);
        free(internal->workers);
        free(internal);
        return -1;
    }

    if (nya_mutex_init(&internal->queue_mutex) != 0) {
        free(internal->queue);
        free(internal->threads);
        free(internal->workers);
        free(internal);
        return -1;
    }
    internal->queue_mutex_initialized = 1;

    if (nya_condition_init(&internal->queue_ready) != 0) {
        nya_mutex_destroy(&internal->queue_mutex);
        free(internal->queue);
        free(internal->threads);
        free(internal->workers);
        free(internal);
        return -1;
    }
    internal->queue_condition_initialized = 1;

    if (nya_mutex_init(&internal->model_mutex) != 0) {
        nya_condition_destroy(&internal->queue_ready);
        nya_mutex_destroy(&internal->queue_mutex);
        free(internal->queue);
        free(internal->threads);
        free(internal->workers);
        free(internal);
        return -1;
    }
    internal->model_mutex_initialized = 1;

    server->internal = internal;
    if (nya_mutex_init(&internal->execution_mutex) != 0) {
        return -1;
    }
    internal->execution_mutex_initialized = 1;
    for (index = 0; index < server->config.settings->worker_threads; ++index) {
        internal->workers[index].internal = internal;
        internal->workers[index].client = NYA_INVALID_SOCKET;
        if (nya_thread_create(&internal->threads[index], nya_server_worker, &internal->workers[index]) != 0) {
            return -1;
        }
        internal->started_threads += 1;
    }

    return 0;
}

/* Queue one accepted client or report that bounded capacity has been reached. */
static int nya_server_queue_client(nya_server *server, nya_socket client)
{
    nya_server_internal *internal;
    int result;

    internal = (nya_server_internal *)server->internal;
    result = -1;

    nya_mutex_lock(&internal->queue_mutex);
    if (!internal->shutting_down && internal->queue_count < internal->queue_capacity) {
        internal->queue[internal->queue_tail] = client;
        internal->queue_tail = (internal->queue_tail + 1) % internal->queue_capacity;
        internal->queue_count += 1;
        nya_condition_signal(&internal->queue_ready);
        result = 0;
    }
    nya_mutex_unlock(&internal->queue_mutex);

    return result;
}

/* Stop accepting queued work, close queued sockets, and join active workers. */
static void nya_server_workers_shutdown(nya_server *server)
{
    nya_server_internal *internal;
    unsigned int index;

    internal = (nya_server_internal *)server->internal;
    if (internal == NULL) {
        return;
    }

    if (internal->queue_mutex_initialized) {
        nya_mutex_lock(&internal->queue_mutex);
        internal->shutting_down = 1;
        for (index = 0; index < internal->started_threads; ++index) {
            if (internal->workers[index].client != NYA_INVALID_SOCKET) {
#ifdef _WIN32
                shutdown(internal->workers[index].client, SD_BOTH);
#else
                shutdown(internal->workers[index].client, SHUT_RDWR);
#endif
            }
        }
        while (internal->queue_count > 0) {
            nya_socket client;

            client = internal->queue[internal->queue_head];
            internal->queue_head = (internal->queue_head + 1) % internal->queue_capacity;
            internal->queue_count -= 1;
            nya_socket_close(client);
        }
        if (internal->queue_condition_initialized) {
            nya_condition_broadcast(&internal->queue_ready);
        }
        nya_mutex_unlock(&internal->queue_mutex);
    }

    for (index = 0; index < internal->started_threads; ++index) {
        nya_thread_join(&internal->threads[index]);
    }

    if (internal->model_mutex_initialized) {
        nya_mutex_destroy(&internal->model_mutex);
    }
    if (internal->execution_mutex_initialized) {
        nya_mutex_destroy(&internal->execution_mutex);
    }
    if (internal->queue_condition_initialized) {
        nya_condition_destroy(&internal->queue_ready);
    }
    if (internal->queue_mutex_initialized) {
        nya_mutex_destroy(&internal->queue_mutex);
    }

    free(internal->queue);
    free(internal->threads);
    free(internal->workers);
    free(internal);
    server->internal = NULL;
}

/* Serialize model registry operations across otherwise concurrent requests. */
static void nya_server_models_lock(nya_server *server)
{
    nya_server_internal *internal;

    internal = (nya_server_internal *)server->internal;
    nya_mutex_lock(&internal->model_mutex);
}

/* Release the model registry after one complete operation or serialization. */
static void nya_server_models_unlock(nya_server *server)
{
    nya_server_internal *internal;

    internal = (nya_server_internal *)server->internal;
    nya_mutex_unlock(&internal->model_mutex);
}

/* Provider sessions contain mutable inference state. This lock also pins their
   lifetime against unload, separately from the short metadata registry lock. */
static int nya_server_execution_lock(nya_server *server)
{
    nya_server_internal *internal = (nya_server_internal *)server->internal;
    nya_mutex_lock(&internal->execution_mutex);
    if (nya_server_stop_requested(server)) {
        nya_mutex_unlock(&internal->execution_mutex);
        return -1;
    }
    return 0;
}

static void nya_server_execution_unlock(nya_server *server)
{
    nya_server_internal *internal = (nya_server_internal *)server->internal;
    nya_mutex_unlock(&internal->execution_mutex);
}

/* Return true when two ASCII spans are equal without case sensitivity. */
static int nya_ascii_equal_case(const char *left, size_t left_length, const char *right)
{
    size_t index;
    size_t right_length;

    right_length = strlen(right);
    if (left_length != right_length) {
        return 0;
    }

    for (index = 0; index < left_length; ++index) {
        unsigned char left_character;
        unsigned char right_character;

        left_character = (unsigned char)left[index];
        right_character = (unsigned char)right[index];
        if (tolower(left_character) != tolower(right_character)) {
            return 0;
        }
    }

    return 1;
}

/* Copy one HTTP header value from the header block into bounded output. */
static int nya_http_header_value(
    const char *headers,
    const char *headers_end,
    const char *name,
    char *output,
    size_t output_capacity
)
{
    const char *line;
    int found;

    /* An empty output consistently represents a missing header. */
    if (output_capacity == 0) {
        return -1;
    }
    output[0] = '\0';
    found = 0;

    /* Skip the request line before scanning header lines. */
    line = strstr(headers, "\r\n");
    if (line == NULL || line > headers_end) {
        return -1;
    }
    line += 2;

    while (line < headers_end) {
        const char *line_end;
        const char *colon;
        const char *value;
        const char *value_end;
        size_t value_length;

        /* Each header line must end before the blank header terminator. */
        line_end = strstr(line, "\r\n");
        if (line_end == NULL || line_end > headers_end) {
            return -1;
        }

        /* The first empty line marks the end of headers. */
        if (line_end == line) {
            break;
        }

        /* A colon separates the ASCII header name from its value. */
        colon = memchr(line, ':', (size_t)(line_end - line));
        if (colon == NULL || colon == line) return -1;
        /* Field names must be HTTP tokens, and values must not contain controls.
           In particular, accepting whitespace before ':' permits ambiguous
           framing when a request passes through another HTTP implementation. */
        {
            const char *cursor;
            for (cursor = line; cursor < colon; ++cursor) {
                unsigned char ch = (unsigned char)*cursor;
                if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                      (ch >= '0' && ch <= '9') || strchr("!#$%&'*+-.^_`|~", ch) != NULL)) return -1;
            }
            for (cursor = colon + 1; cursor < line_end; ++cursor) {
                unsigned char ch = (unsigned char)*cursor;
                if ((ch < 32 && ch != '\t') || ch == 127) return -1;
            }
        }
        if (colon != NULL && nya_ascii_equal_case(line, (size_t)(colon - line), name)) {
            /* API fields have single values; duplicates are never resolved by
               choosing whichever one a parser happened to encounter first. */
            if (found) return -1;
            value = colon + 1;
            while (value < line_end && (*value == ' ' || *value == '\t')) {
                value += 1;
            }

            value_end = line_end;
            while (value_end > value && (value_end[-1] == ' ' || value_end[-1] == '\t')) {
                value_end -= 1;
            }

            value_length = (size_t)(value_end - value);
            if (value_length >= output_capacity) {
                return -1;
            }

            memcpy(output, value, value_length);
            output[value_length] = '\0';
            found = 1;
        }

        line = line_end + 2;
    }

    return found;
}

/* Parse a decimal size while rejecting signs, junk, and overflow. */
static int nya_parse_size(const char *text, size_t *value)
{
    unsigned long long parsed;
    char *end;

    if (text == NULL || text[0] == '\0' || !isdigit((unsigned char)text[0])) {
        return -1;
    }

    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > (unsigned long long)SIZE_MAX) {
        return -1;
    }

    *value = (size_t)parsed;
    return 0;
}

/* Parse the request line and the few headers used by this API. */
static int nya_http_parse_request(
    char *buffer,
    size_t total_length,
    const char *headers_end,
    size_t body_length,
    nya_http_request *request
)
{
    const char *request_line_end;
    size_t request_line_length;
    char request_line[NYA_HTTP_PATH_LIMIT + 64];
    char version[16];
    char extra;

    memset(request, 0, sizeof(*request));

    /* Bound and copy the request line before using the standard parser. */
    request_line_end = strstr(buffer, "\r\n");
    if (request_line_end == NULL || request_line_end > headers_end) {
        return -1;
    }

    request_line_length = (size_t)(request_line_end - buffer);
    if (request_line_length >= sizeof(request_line)) {
        return -1;
    }

    memcpy(request_line, buffer, request_line_length);
    request_line[request_line_length] = '\0';

    /* Exactly three whitespace-separated request-line fields are accepted. */
    if (sscanf(
            request_line,
            "%15s %255s %15s %c",
            request->method,
            request->path,
            version,
            &extra
        ) != 3) {
        return -1;
    }

    /* HTTP/1.0 and HTTP/1.1 are enough for a connection-close local API. */
    if (strcmp(version, "HTTP/1.1") != 0 && strcmp(version, "HTTP/1.0") != 0) {
        return -1;
    }
    if (request->path[0] != '/') return -1;
    {
        size_t index;
        for (index = 0; index < request_line_length; ++index) {
            if ((unsigned char)request_line[index] < 32 || (unsigned char)request_line[index] == 127) return -1;
        }
    }

    /* Copy optional headers; malformed oversized values reject the request. */
    if (nya_http_header_value(
            buffer,
            headers_end,
            "Authorization",
            request->authorization,
            sizeof(request->authorization)
        ) < 0) {
        return -1;
    }

    if (nya_http_header_value(buffer, headers_end, "x-api-key", request->api_key, sizeof(request->api_key)) < 0 ||
        nya_http_header_value(buffer, headers_end, "anthropic-version", request->anthropic_version, sizeof(request->anthropic_version)) < 0 ||
        nya_http_header_value(buffer, headers_end, "X-Fyodor-Model-Id", request->model_id, sizeof(request->model_id)) < 0 ||
        nya_http_header_value(buffer, headers_end, "X-Fyodor-Input-Name", request->input_name, sizeof(request->input_name)) < 0 ||
        nya_http_header_value(buffer, headers_end, "X-Fyodor-Input-Type", request->input_type, sizeof(request->input_type)) < 0 ||
        nya_http_header_value(buffer, headers_end, "X-Fyodor-Input-Shape", request->input_shape, sizeof(request->input_shape)) < 0 ||
        nya_http_header_value(buffer, headers_end, "X-Fyodor-Output-Name", request->output_name, sizeof(request->output_name)) < 0) {
        return -1;
    }

    if (nya_http_header_value(
            buffer,
            headers_end,
            "Origin",
            request->origin,
            sizeof(request->origin)
        ) < 0) {
        return -1;
    }

    if (nya_http_header_value(
            buffer,
            headers_end,
            "Content-Type",
            request->content_type,
            sizeof(request->content_type)
        ) < 0) {
        return -1;
    }

    /* The body begins immediately after the four-byte header terminator. */
    request->body = headers_end + 4;
    request->body_length = body_length;

    /* Keep the compiler and reader aware that total_length was validated upstream. */
    (void)total_length;
    return 0;
}

/* Wait only until a fixed deadline. Unlike SO_RCVTIMEO alone, this prevents a
   peer from retaining a worker forever by dripping one byte per timeout. */
static int nya_socket_wait(nya_socket client, int writing, uint64_t deadline, nya_server *server)
{
    for (;;) {
        fd_set ready;
        struct timeval timeout;
        uint64_t now = nya_runtime_monotonic_milliseconds();
        uint64_t remaining;
        int result;
        if (now >= deadline || (server != NULL && nya_server_stop_requested(server))) return -1;
#ifndef _WIN32
        /* select cannot represent larger descriptors; reject without FD_SET UB. */
        if (client < 0 || client >= FD_SETSIZE) return -1;
#endif
        remaining = deadline - now;
        if (remaining > 100) remaining = 100;
        timeout.tv_sec = (long)(remaining / 1000);
        timeout.tv_usec = (long)(remaining % 1000) * 1000L;
        FD_ZERO(&ready);
        FD_SET(client, &ready);
#ifdef _WIN32
        result = select(0, writing ? NULL : &ready, writing ? &ready : NULL, NULL, &timeout);
#else
        result = select(client + 1, writing ? NULL : &ready, writing ? &ready : NULL, NULL, &timeout);
#endif
        if (result > 0) return 0;
        if (result == 0) continue;
#ifndef _WIN32
        if (result < 0 && errno == EINTR) continue;
#endif
        return -1;
    }
}

/* Nonblocking I/O can race readiness notifications, so retry only transient
   errors; the unchanged deadline still bounds the whole transfer. */
static int nya_socket_retryable(void)
{
#ifdef _WIN32
    int error = WSAGetLastError();
    return error == WSAEINTR || error == WSAEWOULDBLOCK;
#else
    return errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

/* Receive exactly one bounded HTTP request from a connection-close client. */
static int nya_http_receive(
    nya_socket client,
    char *buffer,
    size_t capacity,
    nya_http_request *request,
    unsigned int timeout_ms,
    nya_server *server
)
{
    size_t total;
    const char *headers_end;
    size_t header_bytes;
    size_t body_length;
    uint64_t deadline;

    total = 0;
    headers_end = NULL;
    header_bytes = 0;
    body_length = 0;
    deadline = nya_runtime_monotonic_milliseconds() + timeout_ms;
    if (capacity < 2) return -2;

    while (total < capacity - 1) {
        int received;
        size_t available = capacity - total - 1;

        if (available > INT_MAX) available = INT_MAX;
        if (nya_socket_wait(client, 0, deadline, server) != 0) return -4;
        received = (int)recv(client, buffer + total, (int)available, 0);
        if (received < 0 && nya_socket_retryable()) continue;
        if (received <= 0) {
            return -1;
        }

        total += (size_t)received;
        buffer[total] = '\0';

        /* Locate the header terminator once enough bytes have arrived. */
        if (headers_end == NULL) {
            char content_length[64];
            char transfer_encoding[64];
            int header_result;
            int transfer_result;

            headers_end = strstr(buffer, "\r\n\r\n");
            if (headers_end == NULL) {
                if (memchr(buffer, '\0', total) != NULL) return -1;
                if (total >= NYA_HTTP_HEADER_BLOCK_LIMIT) return -3;
                continue;
            }

            header_bytes = (size_t)((headers_end + 4) - buffer);
            /* A large body may share the very first recv with tiny headers.
               Limit the actual header span, never the total received bytes. */
            if (header_bytes > NYA_HTTP_HEADER_BLOCK_LIMIT) return -3;
            if (memchr(buffer, '\0', header_bytes) != NULL) return -1;

            /* A missing Content-Length means an empty body. */
            header_result = nya_http_header_value(
                buffer,
                headers_end,
                "Content-Length",
                content_length,
                sizeof(content_length)
            );
            if (header_result < 0) {
                return -1;
            }

            if (header_result > 0 && nya_parse_size(content_length, &body_length) != 0) {
                return -1;
            }

            /* Chunked bodies add ambiguity and are unnecessary for this local API. */
            transfer_result = nya_http_header_value(
                buffer,
                headers_end,
                "Transfer-Encoding",
                transfer_encoding,
                sizeof(transfer_encoding)
            );
            if (transfer_result != 0) {
                return -1;
            }

            /* Refuse any request whose declared body exceeds the fixed buffer. */
            if (body_length > capacity - header_bytes - 1) {
                return -2;
            }
        }

        /* Stop as soon as the declared request body is complete. */
        if (headers_end != NULL && total >= header_bytes + body_length) {
            buffer[header_bytes + body_length] = '\0';
            return nya_http_parse_request(
                buffer,
                header_bytes + body_length,
                headers_end,
                body_length,
                request
            );
        }
    }

    return -2;
}

/* Initialize a bounded text builder over caller-owned storage. */
static void nya_builder_init(nya_text_builder *builder, char *data, size_t capacity)
{
    builder->data = data;
    builder->capacity = capacity;
    builder->length = 0;
    builder->failed = capacity == 0;

    if (capacity > 0) {
        data[0] = '\0';
    }
}

/* Append formatted text while detecting every truncation. */
static void nya_builder_append(nya_text_builder *builder, const char *format, ...)
{
    va_list arguments;
    int written;
    size_t remaining;

    if (builder->failed) {
        return;
    }

    remaining = builder->capacity - builder->length;
    va_start(arguments, format);
    written = vsnprintf(builder->data + builder->length, remaining, format, arguments);
    va_end(arguments);

    if (written < 0 || (size_t)written >= remaining) {
        builder->failed = 1;
        return;
    }

    builder->length += (size_t)written;
}

/* Append a valid JSON string with all control characters escaped. */
static void nya_builder_append_json_string(nya_text_builder *builder, const char *text)
{
    const unsigned char *cursor;

    nya_builder_append(builder, "\"");
    cursor = (const unsigned char *)text;

    while (*cursor != '\0' && !builder->failed) {
        switch (*cursor) {
            case '\"':
                nya_builder_append(builder, "\\\"");
                break;
            case '\\':
                nya_builder_append(builder, "\\\\");
                break;
            case '\b':
                nya_builder_append(builder, "\\b");
                break;
            case '\f':
                nya_builder_append(builder, "\\f");
                break;
            case '\n':
                nya_builder_append(builder, "\\n");
                break;
            case '\r':
                nya_builder_append(builder, "\\r");
                break;
            case '\t':
                nya_builder_append(builder, "\\t");
                break;
            default:
                if (*cursor < 0x20) {
                    nya_builder_append(builder, "\\u%04x", (unsigned int)*cursor);
                } else {
                    nya_builder_append(builder, "%c", (char)*cursor);
                }
                break;
        }

        cursor += 1;
    }

    nya_builder_append(builder, "\"");
}

/* Send an entire buffer even when the socket accepts only a partial write. */
static int nya_socket_send_all(nya_socket socket_value, const char *data, size_t length, nya_server *server)
{
    size_t total;
    uint64_t timeout_ms;
    uint64_t deadline;
#ifdef _WIN32
    DWORD timeout = 0;
#else
    struct timeval timeout = {0, 0};
#endif
    nya_socket_length option_length = (nya_socket_length)sizeof(timeout);

    /* The accepted socket carries its configured timeout through every handler.
       One deadline spans all partial writes of this buffer, so a slow reader
       cannot extend a large tensor transfer indefinitely. */
    timeout_ms = 5000;
    if (getsockopt(socket_value, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, &option_length) == 0) {
#ifdef _WIN32
        if (timeout != 0) timeout_ms = timeout;
#else
        if (timeout.tv_sec != 0 || timeout.tv_usec != 0)
            timeout_ms = (uint64_t)timeout.tv_sec * 1000 + (uint64_t)timeout.tv_usec / 1000;
#endif
    }
    deadline = nya_runtime_monotonic_milliseconds() + timeout_ms;

    total = 0;
    while (total < length) {
        int sent;
        int flags;
        size_t remaining = length - total;

        flags = 0;
#ifdef MSG_NOSIGNAL
        /* Linux can suppress SIGPIPE for this individual send operation. */
        flags = MSG_NOSIGNAL;
#endif
        if (remaining > INT_MAX) remaining = INT_MAX;
        if (nya_socket_wait(socket_value, 1, deadline, server) != 0) return -1;
        sent = (int)send(socket_value, data + total, (int)remaining, flags);
        if (sent < 0 && nya_socket_retryable()) continue;
        if (sent <= 0) {
            return -1;
        }

        total += (size_t)sent;
    }

    return 0;
}

/* Permit only the known Tauri origins and local development servers. */
static int nya_origin_allowed(const nya_server *server, const char *origin)
{
    const char *port;
    size_t index;

    /* Non-browser clients normally omit Origin and are allowed through auth. */
    if (origin[0] == '\0') {
        return 1;
    }

    if (strcmp(origin, "tauri://localhost") == 0 ||
        strcmp(origin, "http://tauri.localhost") == 0 ||
        strcmp(origin, "https://tauri.localhost") == 0) {
        return 1;
    }

    /* Remote deployments must name every additional browser origin explicitly. */
    for (index = 0; index < server->config.settings->allowed_origin_count; ++index) {
        if (strcmp(origin, server->config.settings->allowed_origins[index]) == 0) {
            return 1;
        }
    }

    /* Development origins may use localhost with a numeric port. */
    if (strncmp(origin, "http://localhost", 16) != 0) {
        return 0;
    }

    port = origin + 16;
    if (*port == '\0') {
        return 1;
    }

    if (*port != ':') {
        return 0;
    }

    port += 1;
    if (*port == '\0') {
        return 0;
    }

    while (*port != '\0') {
        if (!isdigit((unsigned char)*port)) {
            return 0;
        }
        port += 1;
    }

    return 1;
}

/* Compare bearer credentials without exiting early on the first wrong byte. */
static int nya_auth_valid(const char *authorization, const char *token)
{
    static const char prefix[] = "Bearer ";
    size_t authorization_length;
    size_t token_length;
    size_t expected_length;
    size_t index;
    unsigned char difference;

    authorization_length = strlen(authorization);
    token_length = strlen(token);
    expected_length = sizeof(prefix) - 1 + token_length;
    if (authorization_length != expected_length) {
        return 0;
    }

    difference = 0;
    for (index = 0; index < sizeof(prefix) - 1; ++index) {
        difference |= (unsigned char)(authorization[index] ^ prefix[index]);
    }
    for (index = 0; index < token_length; ++index) {
        difference |= (unsigned char)(authorization[sizeof(prefix) - 1 + index] ^ token[index]);
    }

    return difference == 0;
}

/* Build and send one connection-close JSON response. */
static int nya_http_send_response(
    nya_server *server,
    nya_socket client,
    int status,
    const char *reason,
    const char *body,
    const char *origin
)
{
    char headers[2048];
    int written;
    size_t body_length;
    const char *origin_header;

    body_length = body == NULL ? 0 : strlen(body);
    if (server->config.verbose) fprintf(stderr,"[http] socket=%" PRIuPTR " status=%d response_bytes=%zu\n",(uintptr_t)client,status,body_length);
    origin_header = origin != NULL && origin[0] != '\0' ? origin : NULL;

    if (origin_header != NULL) {
        written = snprintf(
            headers,
            sizeof(headers),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n"
            "Cache-Control: no-store\r\n"
            "X-Content-Type-Options: nosniff\r\n"
            "Access-Control-Allow-Origin: %s\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type, X-Api-Key, Anthropic-Version, X-Fyodor-Model-Id, X-Fyodor-Input-Name, X-Fyodor-Input-Type, X-Fyodor-Input-Shape, X-Fyodor-Output-Name\r\n"
            "Vary: Origin\r\n"
            "\r\n",
            status,
            reason,
            body_length,
            origin_header
        );
    } else {
        written = snprintf(
            headers,
            sizeof(headers),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n"
            "Cache-Control: no-store\r\n"
            "X-Content-Type-Options: nosniff\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type, X-Api-Key, Anthropic-Version, X-Fyodor-Model-Id, X-Fyodor-Input-Name, X-Fyodor-Input-Type, X-Fyodor-Input-Shape, X-Fyodor-Output-Name\r\n"
            "Vary: Origin\r\n"
            "\r\n",
            status,
            reason,
            body_length
        );
    }

    if (written < 0 || (size_t)written >= sizeof(headers)) {
        return -1;
    }

    if (nya_socket_send_all(client, headers, (size_t)written, server) != 0) {
        return -1;
    }

    if (body_length > 0 && nya_socket_send_all(client, body, body_length, server) != 0) {
        return -1;
    }

    return 0;
}

/* Send one raw dense tensor with its type and shape described by response headers. */
static int nya_http_send_tensor(
    nya_server *server,
    nya_socket client,
    const nya_execution_response *response,
    const char *origin
)
{
    char headers[4096];
    char shape[512];
    nya_text_builder shape_builder;
    int written;
    size_t index;
    const char *origin_header;

    nya_builder_init(&shape_builder, shape, sizeof(shape));
    for (index = 0; index < response->output_rank; ++index) {
        nya_builder_append(
            &shape_builder,
            "%s%" PRId64,
            index == 0 ? "" : ",",
            response->output_shape[index]
        );
    }
    if (shape_builder.failed) {
        return -1;
    }

    origin_header = origin != NULL && origin[0] != '\0' ? origin : NULL;
    if (origin_header != NULL) {
        written = snprintf(
            headers,
            sizeof(headers),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/octet-stream\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n"
            "Cache-Control: no-store\r\n"
            "X-Content-Type-Options: nosniff\r\n"
            "X-Fyodor-Output-Type: %s\r\n"
            "X-Fyodor-Output-Shape: %s\r\n"
            "Access-Control-Allow-Origin: %s\r\n"
            "Access-Control-Expose-Headers: X-Fyodor-Output-Type, X-Fyodor-Output-Shape\r\n"
            "Vary: Origin\r\n"
            "\r\n",
            response->output_data_size,
            nya_tensor_data_type_name(response->output_type),
            shape,
            origin_header
        );
    } else {
        written = snprintf(
            headers,
            sizeof(headers),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/octet-stream\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n"
            "Cache-Control: no-store\r\n"
            "X-Content-Type-Options: nosniff\r\n"
            "X-Fyodor-Output-Type: %s\r\n"
            "X-Fyodor-Output-Shape: %s\r\n"
            "Vary: Origin\r\n"
            "\r\n",
            response->output_data_size,
            nya_tensor_data_type_name(response->output_type),
            shape
        );
    }

    if (written < 0 || (size_t)written >= sizeof(headers) ||
        nya_socket_send_all(client, headers, (size_t)written, server) != 0) {
        return -1;
    }

    if (response->output_data_size > 0 &&
        nya_socket_send_all(
            client,
            (const char *)response->output_data,
            response->output_data_size,
            server
        ) != 0) {
        return -1;
    }

    return 0;
}

/* Send a compact, stable JSON error shape. */
static int nya_http_send_error(
    nya_server *server,
    nya_socket client,
    int status,
    const char *reason,
    const char *code,
    const char *message,
    const char *origin
)
{
    char body[1024];
    nya_text_builder builder;

    nya_builder_init(&builder, body, sizeof(body));
    nya_builder_append(&builder, "{\"version\":%d,\"ok\":false,\"error\":{\"code\":", NYA_API_VERSION);
    nya_builder_append_json_string(&builder, code);
    nya_builder_append(&builder, ",\"message\":");
    nya_builder_append_json_string(&builder, message);
    nya_builder_append(&builder, "}}");

    if (builder.failed) {
        return -1;
    }

    return nya_http_send_response(server, client, status, reason, body, origin);
}

/* JSON parsing is length-bounded and validates the complete document before any
   route uses it. Values are scanned structurally, so keys inside prompts or
   nested objects cannot impersonate top-level API parameters. */
static void nya_json_space(const char **cursor, const char *end)
{
    while (*cursor < end && (**cursor == ' ' || **cursor == '\t' ||
           **cursor == '\r' || **cursor == '\n')) *cursor += 1;
}

static int nya_json_hex4(const char **cursor, const char *end, uint32_t *value)
{
    unsigned int index;
    uint32_t result = 0;
    if ((size_t)(end - *cursor) < 4) return -1;
    for (index = 0; index < 4; ++index) {
        unsigned char ch = (unsigned char)*(*cursor)++;
        unsigned int digit;
        if (ch >= '0' && ch <= '9') digit = (unsigned int)(ch - '0');
        else if (ch >= 'a' && ch <= 'f') digit = (unsigned int)(ch - 'a') + 10;
        else if (ch >= 'A' && ch <= 'F') digit = (unsigned int)(ch - 'A') + 10;
        else return -1;
        result = result * 16 + digit;
    }
    *value = result;
    return 0;
}

/* Decode escapes and validate UTF-8 even when the caller only skips a string.
   Embedded NUL is rejected because the engine's path/prompt APIs use C strings. */
static int nya_json_decode_string(const char **cursor, const char *end, char *output, size_t capacity)
{
    size_t used = 0;
    if (*cursor >= end || *(*cursor)++ != '"') return -1;
    while (*cursor < end) {
        uint32_t cp = (unsigned char)*(*cursor)++;
        unsigned char encoded[4];
        size_t bytes = 1;
        size_t index;
        if (cp == '"') {
            if (output != NULL) {
                if (used >= capacity) return -1;
                output[used] = '\0';
            }
            return 0;
        }
        if (cp < 32) return -1;
        if (cp == '\\') {
            if (*cursor >= end) return -1;
            cp = (unsigned char)*(*cursor)++;
            switch (cp) {
                case '"': case '\\': case '/': break;
                case 'b': cp = '\b'; break;
                case 'f': cp = '\f'; break;
                case 'n': cp = '\n'; break;
                case 'r': cp = '\r'; break;
                case 't': cp = '\t'; break;
                case 'u':
                    if (nya_json_hex4(cursor, end, &cp) != 0) return -1;
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        uint32_t low;
                        if ((size_t)(end - *cursor) < 6 || (*cursor)[0] != '\\' || (*cursor)[1] != 'u') return -1;
                        *cursor += 2;
                        if (nya_json_hex4(cursor, end, &low) != 0 || low < 0xdc00 || low > 0xdfff) return -1;
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                    } else if (cp >= 0xdc00 && cp <= 0xdfff) return -1;
                    if (cp == 0) return -1;
                    break;
                default: return -1;
            }
        } else if (cp >= 128) {
            size_t continuation;
            uint32_t minimum;
            if (cp >= 0xc2 && cp <= 0xdf) { continuation = 1; cp &= 0x1f; minimum = 0x80; }
            else if (cp >= 0xe0 && cp <= 0xef) { continuation = 2; cp &= 0x0f; minimum = 0x800; }
            else if (cp >= 0xf0 && cp <= 0xf4) { continuation = 3; cp &= 7; minimum = 0x10000; }
            else return -1;
            if ((size_t)(end - *cursor) < continuation) return -1;
            for (index = 0; index < continuation; ++index) {
                uint32_t ch = (unsigned char)*(*cursor)++;
                if ((ch & 0xc0) != 0x80) return -1;
                cp = (cp << 6) | (ch & 0x3f);
            }
            if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return -1;
        }
        /* Re-encode the validated scalar. This also normalizes escaped keys,
           so "model_id" and "model_\\u0069d" are detected as duplicates. */
        if (cp < 0x80) encoded[0] = (unsigned char)cp;
        else if (cp < 0x800) {
            bytes = 2; encoded[0] = (unsigned char)(0xc0 | (cp >> 6));
            encoded[1] = (unsigned char)(0x80 | (cp & 0x3f));
        } else if (cp < 0x10000) {
            bytes = 3; encoded[0] = (unsigned char)(0xe0 | (cp >> 12));
            encoded[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
            encoded[2] = (unsigned char)(0x80 | (cp & 0x3f));
        } else {
            bytes = 4; encoded[0] = (unsigned char)(0xf0 | (cp >> 18));
            encoded[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3f));
            encoded[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
            encoded[3] = (unsigned char)(0x80 | (cp & 0x3f));
        }
        if (output != NULL) {
            if (used >= capacity || bytes >= capacity - used) return -1;
            memcpy(output + used, encoded, bytes);
        }
        used += bytes;
    }
    return -1;
}

static int nya_json_skip_value(const char **cursor, const char *end, unsigned int depth)
{
    const char *start;
    if (depth > 32) return -1;
    nya_json_space(cursor, end);
    if (*cursor >= end) return -1;
    if (**cursor == '"') return nya_json_decode_string(cursor, end, NULL, 0);
    if (**cursor == '{' || **cursor == '[') {
        int object = **cursor == '{';
        char close = object ? '}' : ']';
        *cursor += 1;
        nya_json_space(cursor, end);
        if (*cursor < end && **cursor == close) { *cursor += 1; return 0; }
        for (;;) {
            if (object) {
                if (nya_json_decode_string(cursor, end, NULL, 0) != 0) return -1;
                nya_json_space(cursor, end);
                if (*cursor >= end || *(*cursor)++ != ':') return -1;
            }
            if (nya_json_skip_value(cursor, end, depth + 1) != 0) return -1;
            nya_json_space(cursor, end);
            if (*cursor >= end) return -1;
            if (**cursor == close) { *cursor += 1; return 0; }
            if (*(*cursor)++ != ',') return -1;
            nya_json_space(cursor, end);
        }
    }
    start = *cursor;
    if ((size_t)(end - start) >= 4 && (memcmp(start, "null", 4) == 0 || memcmp(start, "true", 4) == 0)) {
        *cursor += 4; return 0;
    }
    if ((size_t)(end - start) >= 5 && memcmp(start, "false", 5) == 0) { *cursor += 5; return 0; }
    /* Follow JSON's number grammar before strtod can accept non-JSON forms
       such as +1, 0x1p0, NaN, leading zeroes, or a missing fractional digit. */
    if (**cursor == '-') *cursor += 1;
    if (*cursor >= end) return -1;
    if (**cursor == '0') *cursor += 1;
    else {
        if (**cursor < '1' || **cursor > '9') return -1;
        do { *cursor += 1; } while (*cursor < end && **cursor >= '0' && **cursor <= '9');
    }
    if (*cursor < end && **cursor == '.') {
        *cursor += 1;
        if (*cursor >= end || **cursor < '0' || **cursor > '9') return -1;
        do { *cursor += 1; } while (*cursor < end && **cursor >= '0' && **cursor <= '9');
    }
    if (*cursor < end && (**cursor == 'e' || **cursor == 'E')) {
        *cursor += 1;
        if (*cursor < end && (**cursor == '+' || **cursor == '-')) *cursor += 1;
        if (*cursor >= end || **cursor < '0' || **cursor > '9') return -1;
        do { *cursor += 1; } while (*cursor < end && **cursor >= '0' && **cursor <= '9');
    }
    return 0;
}

/* API objects contain at most 64 fields, with decoded keys of at most 127
   bytes. Unknown values may be nested up to 32 levels and are safely skipped. */
static int nya_json_validate(const char *json, size_t length)
{
    char keys[64][128];
    size_t count = 0;
    const char *cursor = json;
    const char *end = json + length;
    nya_json_space(&cursor, end);
    if (cursor >= end || *cursor++ != '{') return -1;
    nya_json_space(&cursor, end);
    if (cursor < end && *cursor == '}') cursor += 1;
    else for (;;) {
        size_t index;
        if (count == 64 || nya_json_decode_string(&cursor, end, keys[count], sizeof(keys[count])) != 0) return -1;
        for (index = 0; index < count; ++index) if (strcmp(keys[index], keys[count]) == 0) return -1;
        count += 1;
        nya_json_space(&cursor, end);
        if (cursor >= end || *cursor++ != ':') return -1;
        if (nya_json_skip_value(&cursor, end, 0) != 0) return -1;
        nya_json_space(&cursor, end);
        if (cursor >= end) return -1;
        if (*cursor == '}') { cursor += 1; break; }
        if (*cursor++ != ',') return -1;
        nya_json_space(&cursor, end);
    }
    nya_json_space(&cursor, end);
    return cursor == end ? 0 : -1;
}

/* Lookup operates only on the previously validated top-level object. */
static const char *nya_json_value(const char *json, size_t length, const char *key)
{
    const char *cursor = json;
    const char *end = json + length;
    nya_json_space(&cursor, end);
    if (cursor >= end || *cursor++ != '{') return NULL;
    for (;;) {
        char decoded[128];
        nya_json_space(&cursor, end);
        if (nya_json_decode_string(&cursor, end, decoded, sizeof(decoded)) != 0) return NULL;
        nya_json_space(&cursor, end);
        if (cursor >= end || *cursor++ != ':') return NULL;
        nya_json_space(&cursor, end);
        if (strcmp(decoded, key) == 0) return cursor;
        if (nya_json_skip_value(&cursor, end, 0) != 0) return NULL;
        nya_json_space(&cursor, end);
        if (cursor >= end || *cursor++ != ',') return NULL;
    }
}

static int nya_json_string(const char *json, size_t length, const char *key, char *output, size_t capacity)
{
    const char *cursor = nya_json_value(json, length, key);
    if (cursor == NULL) return -1;
    return nya_json_decode_string(&cursor, json + length, output, capacity);
}
/* Parse one unsigned JSON integer without accepting floating-point syntax. */
static int nya_json_u64(const char *json, size_t length, const char *key, uint64_t *output)
{
    const char *value;
    const char *end;
    uint64_t number;

    value = nya_json_value(json, length, key);
    end = json + length;
    if (value == NULL || value >= end || !isdigit((unsigned char)*value)) {
        return -1;
    }

    number = 0;
    while (value < end && isdigit((unsigned char)*value)) {
        unsigned int digit;

        digit = (unsigned int)(*value - '0');
        if (number > (UINT64_MAX - digit) / 10) {
            return -1;
        }

        number = number * 10 + digit;
        value += 1;
    }

    /* Whitespace followed by a comma or object end completes this numeric field. */
    while (value < end && isspace((unsigned char)*value)) {
        value += 1;
    }
    if (value >= end || (*value != '}' && *value != ',')) {
        return -1;
    }

    *output = number;
    return 0;
}

/* Parse an optional unsigned field while distinguishing absence from bad syntax. */
static int nya_json_optional_u64(
    const char *json,
    size_t length,
    const char *key,
    uint64_t *output
)
{
    if (nya_json_value(json, length, key) == NULL) return 0;
    return nya_json_u64(json, length, key, output) == 0 ? 1 : -1;
}

/* Parse one optional finite JSON number into a double. */
static int nya_json_optional_double(
    const char *json,
    size_t length,
    const char *key,
    double *output
)
{
    const char *value;
    const char *end;
    const char *number_end;
    char number[64];
    size_t number_length;
    char *parsed_end;
    double parsed;

    value = nya_json_value(json, length, key);
    if (value == NULL) return 0;
    end = json + length;
    number_end = value;
    while (number_end < end && *number_end != ',' && *number_end != '}' &&
           !isspace((unsigned char)*number_end)) {
        number_end += 1;
    }
    number_length = (size_t)(number_end - value);
    if (number_length == 0 || number_length >= sizeof(number)) return -1;
    memcpy(number, value, number_length);
    number[number_length] = '\0';

    errno = 0;
    parsed = strtod(number, &parsed_end);
    if (errno != 0 || parsed_end != number + number_length || !isfinite(parsed)) return -1;
    *output = parsed;
    return 1;
}

/* Return true when a POST body declares the JSON media type. */
static int nya_content_type_is_json(const char *content_type)
{
    static const char json_type[] = "application/json";
    size_t type_length;

    type_length = sizeof(json_type) - 1;
    if (strncmp(content_type, json_type, type_length) != 0) {
        return 0;
    }

    return content_type[type_length] == '\0' || content_type[type_length] == ';';
}

/* Send current process health. */
static int nya_handle_health(nya_socket client, nya_server *server, const char *origin)
{
    char body[512];
    int written;

    written = snprintf(
        body,
        sizeof(body),
        "{\"version\":%d,\"ok\":true,\"status\":\"ready\","
        "\"engine_version\":\"%d.%d.%d\",\"uptime_seconds\":%" PRIu64
        ",\"bind\":\"%s\",\"port\":%u,\"worker_threads\":%u,\"remote_enabled\":%s}",
        NYA_API_VERSION,
        NYA_VERSION_MAJOR,
        NYA_VERSION_MINOR,
        NYA_VERSION_PATCH,
        nya_runtime_uptime_seconds(server->config.runtime),
        server->config.settings->bind_address,
        (unsigned int)server->port,
        server->config.settings->worker_threads,
        server->config.settings->allow_remote ? "true" : "false"
    );

    if (written < 0 || (size_t)written >= sizeof(body)) {
        return -1;
    }

    return nya_http_send_response(server, client, 200, "OK", body, origin);
}

/* Serialize every loaded model into one bounded JSON array. */
static int nya_handle_models(nya_socket client, nya_server *server, const char *origin)
{
    char body[NYA_HTTP_RESPONSE_LIMIT];
    nya_text_builder builder;
    size_t index;

    nya_builder_init(&builder, body, sizeof(body));
    nya_builder_append(&builder, "{\"version\":%d,\"ok\":true,\"models\":[", NYA_API_VERSION);

    nya_server_models_lock(server);
    for (index = 0; index < server->config.models->count; ++index) {
        const nya_model *model;

        model = &server->config.models->models[index];
        if (index > 0) {
            nya_builder_append(&builder, ",");
        }

        nya_builder_append(&builder, "{\"id\":%" PRIu64 ",\"path\":", model->id);
        nya_builder_append_json_string(&builder, model->path);
        nya_builder_append(&builder, ",\"file_size\":%" PRIu64 ",\"format\":", model->file_size);
        nya_builder_append_json_string(&builder, nya_model_format_name(model->format));
        nya_builder_append(
            &builder,
            ",\"format_version\":%" PRIu64
            ",\"tensor_count\":%" PRIu64
            ",\"metadata_count\":%" PRIu64
            ",\"data_offset\":%" PRIu64
            ",\"architecture\":",
            model->format_version,
            model->tensor_count,
            model->metadata_count,
            model->data_offset
        );
        if (model->architecture[0] == '\0') {
            nya_builder_append(&builder, "null");
        } else {
            nya_builder_append_json_string(&builder, model->architecture);
        }
        nya_builder_append(&builder, ",\"producer\":");
        if (model->producer[0] == '\0') {
            nya_builder_append(&builder, "null");
        } else {
            nya_builder_append_json_string(&builder, model->producer);
        }
        nya_builder_append(
            &builder,
            ",\"inference_supported\":%s,\"generation_supported\":%s,\"draft_supported\":%s,"
            "\"capabilities\":[\"inspect\"%s%s%s],\"execution_error\":",
            model->inference_supported ? "true" : "false",
            model->generation_supported ? "true" : "false",
            model->draft_supported ? "true" : "false",
            model->inference_supported ? ",\"inference\"" : "",
            model->generation_supported ? ",\"generation\"" : "",
            model->draft_supported ? ",\"draft\"" : ""
        );
        if (model->execution_error[0] == '\0') {
            nya_builder_append(&builder, "null");
        } else {
            nya_builder_append_json_string(&builder, model->execution_error);
        }
        nya_builder_append(&builder, ",\"generation_error\":");
        if (model->generation_error[0] == '\0') {
            nya_builder_append(&builder, "null}");
        } else {
            nya_builder_append_json_string(&builder, model->generation_error);
            nya_builder_append(&builder, "}");
        }
    }
    /* The response buffer now owns all metadata. A slow network peer must
       never retain the registry lock while its response is being sent. */
    nya_server_models_unlock(server);

    nya_builder_append(&builder, "]}");
    if (builder.failed) {
        return nya_http_send_error(
            server,
            client,
            500,
            "Internal Server Error",
            "response_too_large",
            "The model list exceeded the response limit.",
            origin
        );
    }

    return nya_http_send_response(server, client, 200, "OK", body, origin);
}

/* Validate and register the model path from a JSON request body. */
static int nya_handle_model_load(
    nya_socket client,
    nya_server *server,
    const nya_http_request *request,
    const char *origin
)
{
    char path[NYA_MODEL_PATH_LIMIT];
    const nya_model *model;
    nya_model snapshot;
    nya_model_result result;
    char body[4096];
    nya_text_builder builder;

    if (nya_json_string(request->body, request->body_length, "path", path, sizeof(path)) != 0) {
        return nya_http_send_error(server, client, 400, "Bad Request", "invalid_json", "A valid string field named path is required.", origin);
    }

    if (nya_server_execution_lock(server) != 0) return -1;
    nya_server_models_lock(server);
    result = nya_model_load(server->config.models, path, &model);
    if (result == NYA_MODEL_OK && model != NULL) {
        snapshot = *model;
        model = &snapshot;
    }
    nya_server_models_unlock(server);
    nya_server_execution_unlock(server);
    if (result == NYA_MODEL_NOT_FOUND) {
        return nya_http_send_error(server, client, 404, "Not Found", "model_file_not_found", "The path is not a readable regular file.", origin);
    }
    if (result == NYA_MODEL_PATH_TOO_LONG) {
        return nya_http_send_error(server, client, 400, "Bad Request", "path_too_long", "The model path is too long.", origin);
    }
    if (result == NYA_MODEL_LIMIT_REACHED) {
        return nya_http_send_error(server, client, 409, "Conflict", "model_limit_reached", "Unload a model before loading another one.", origin);
    }
    if (result == NYA_MODEL_FORMAT_UNSUPPORTED) {
        return nya_http_send_error(server, client, 415, "Unsupported Media Type", "model_format_unsupported", "Supported containers are GGUF v2/v3, SafeTensors, and .onnx ModelProto files.", origin);
    }
    if (result == NYA_MODEL_FORMAT_INVALID) {
        return nya_http_send_error(server, client, 422, "Unprocessable Content", "model_format_invalid", "The model container failed structural validation.", origin);
    }
    if (result == NYA_MODEL_RESOURCE_LIMIT) {
        return nya_http_send_error(server, client, 413, "Content Too Large", "model_metadata_too_large", "The model metadata exceeds a safe parser limit.", origin);
    }
    if (result != NYA_MODEL_OK || model == NULL) {
        return nya_http_send_error(server, client, 400, "Bad Request", "invalid_model", "The model could not be registered.", origin);
    }

    nya_builder_init(&builder, body, sizeof(body));
    nya_builder_append(&builder, "{\"version\":%d,\"ok\":true,\"model\":{\"id\":%" PRIu64 ",\"path\":", NYA_API_VERSION, model->id);
    nya_builder_append_json_string(&builder, model->path);
    nya_builder_append(&builder, ",\"file_size\":%" PRIu64 ",\"format\":", model->file_size);
    nya_builder_append_json_string(&builder, nya_model_format_name(model->format));
    nya_builder_append(
        &builder,
        ",\"format_version\":%" PRIu64
        ",\"tensor_count\":%" PRIu64
        ",\"metadata_count\":%" PRIu64
        ",\"data_offset\":%" PRIu64
        ",\"architecture\":",
        model->format_version,
        model->tensor_count,
        model->metadata_count,
        model->data_offset
    );
    if (model->architecture[0] == '\0') {
        nya_builder_append(&builder, "null");
    } else {
        nya_builder_append_json_string(&builder, model->architecture);
    }
    nya_builder_append(&builder, ",\"producer\":");
    if (model->producer[0] == '\0') {
        nya_builder_append(&builder, "null");
    } else {
        nya_builder_append_json_string(&builder, model->producer);
    }
    nya_builder_append(
        &builder,
        ",\"inference_supported\":%s,\"generation_supported\":%s,\"draft_supported\":%s,"
        "\"capabilities\":[\"inspect\"%s%s%s],\"execution_error\":",
        model->inference_supported ? "true" : "false",
        model->generation_supported ? "true" : "false",
        model->draft_supported ? "true" : "false",
        model->inference_supported ? ",\"inference\"" : "",
        model->generation_supported ? ",\"generation\"" : "",
        model->draft_supported ? ",\"draft\"" : ""
    );
    if (model->execution_error[0] == '\0') {
        nya_builder_append(&builder, "null");
    } else {
        nya_builder_append_json_string(&builder, model->execution_error);
    }
    nya_builder_append(&builder, ",\"generation_error\":");
    if (model->generation_error[0] == '\0') {
        nya_builder_append(&builder, "null}}");
    } else {
        nya_builder_append_json_string(&builder, model->generation_error);
        nya_builder_append(&builder, "}}");
    }
    if (builder.failed) {
        return -1;
    }

    return nya_http_send_response(server, client, 200, "OK", body, origin);
}

/* Validate and remove a model ID from a JSON request body. */
static int nya_handle_model_unload(
    nya_socket client,
    nya_server *server,
    const nya_http_request *request,
    const char *origin
)
{
    uint64_t model_id;
    nya_model_result result;
    char body[256];
    int written;

    if (nya_json_u64(request->body, request->body_length, "model_id", &model_id) != 0 || model_id == 0) {
        return nya_http_send_error(server, client, 400, "Bad Request", "invalid_json", "A positive integer field named model_id is required.", origin);
    }

    if (nya_server_execution_lock(server) != 0) return -1;
    nya_server_models_lock(server);
    result = nya_model_unload(server->config.models, model_id);
    nya_server_models_unlock(server);
    nya_server_execution_unlock(server);
    if (result == NYA_MODEL_ID_NOT_FOUND) {
        return nya_http_send_error(server, client, 404, "Not Found", "model_not_loaded", "No loaded model has that ID.", origin);
    }
    if (result != NYA_MODEL_OK) {
        return nya_http_send_error(server, client, 400, "Bad Request", "invalid_model_id", "The model ID is invalid.", origin);
    }

    written = snprintf(
        body,
        sizeof(body),
        "{\"version\":%d,\"ok\":true,\"unloaded_model_id\":%" PRIu64 "}",
        NYA_API_VERSION,
        model_id
    );
    if (written < 0 || (size_t)written >= sizeof(body)) {
        return -1;
    }

    return nya_http_send_response(server, client, 200, "OK", body, origin);
}

/* Parse an exact positive decimal model ID from a request header. */
static int nya_http_model_id(const char *text, uint64_t *model_id)
{
    unsigned long long value;
    char *end;

    if (text[0] == '\0' || !isdigit((unsigned char)text[0])) {
        return -1;
    }

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || value == 0) {
        return -1;
    }

    *model_id = (uint64_t)value;
    return 0;
}

/* Parse a comma-separated positive tensor shape into at most sixteen dimensions. */
static int nya_http_tensor_shape(const char *text, int64_t shape[16], size_t *rank)
{
    const char *cursor;
    size_t count;

    cursor = text;
    count = 0;
    while (*cursor != '\0') {
        unsigned long long extent;
        char *end;

        while (*cursor == ' ' || *cursor == '\t') {
            cursor += 1;
        }
        if (!isdigit((unsigned char)*cursor) || count >= 16) {
            return -1;
        }

        errno = 0;
        extent = strtoull(cursor, &end, 10);
        if (errno != 0 || extent == 0 || extent > INT64_MAX) {
            return -1;
        }
        shape[count] = (int64_t)extent;
        count += 1;
        cursor = end;

        while (*cursor == ' ' || *cursor == '\t') {
            cursor += 1;
        }
        if (*cursor == '\0') {
            break;
        }
        if (*cursor != ',') {
            return -1;
        }
        cursor += 1;
        /* Each separator must introduce another dimension; a trailing comma
         * would otherwise silently describe a different shape than supplied. */
        if (*cursor == '\0') return -1;
    }

    if (count == 0) {
        return -1;
    }

    *rank = count;
    return 0;
}

/* Return true only for the raw tensor request media type. */
static int nya_content_type_is_binary(const char *content_type)
{
    static const char binary_type[] = "application/octet-stream";
    size_t type_length;

    type_length = sizeof(binary_type) - 1;
    if (strncmp(content_type, binary_type, type_length) != 0) {
        return 0;
    }

    return content_type[type_length] == '\0' || content_type[type_length] == ';';
}

/* Run a real single-input, single-output model using raw tensor bytes. */
static int nya_handle_model_run(
    nya_socket client,
    nya_server *server,
    const nya_http_request *http_request,
    const char *origin
)
{
    uint64_t model_id;
    int64_t input_shape[16];
    size_t input_rank;
    nya_tensor_data_type input_type;
    nya_execution_request execution_request;
    nya_execution_response execution_response;
    nya_model *model;
    nya_model snapshot;
    int execution_result;
    char execution_error[512];
    int send_result;

    if (!nya_content_type_is_binary(http_request->content_type)) {
        return nya_http_send_error(server, client, 415, "Unsupported Media Type", "binary_tensor_required", "Use Content-Type: application/octet-stream.", origin);
    }

    if (nya_http_model_id(http_request->model_id, &model_id) != 0 ||
        http_request->input_name[0] == '\0' || http_request->output_name[0] == '\0' ||
        nya_tensor_data_type_parse(http_request->input_type, &input_type) != 0 ||
        nya_http_tensor_shape(http_request->input_shape, input_shape, &input_rank) != 0) {
        return nya_http_send_error(server, client, 400, "Bad Request", "invalid_tensor_headers", "Model ID, input name/type/shape, and output name headers are required.", origin);
    }

    execution_request.input_name = http_request->input_name;
    execution_request.input_type = input_type;
    execution_request.input_shape = input_shape;
    execution_request.input_rank = input_rank;
    execution_request.input_data = http_request->body;
    execution_request.input_data_size = http_request->body_length;
    execution_request.output_name = http_request->output_name;
    execution_request.max_output_bytes = server->config.settings->max_output_bytes;
    memset(&execution_response, 0, sizeof(execution_response));
    execution_error[0] = '\0';

    /* Copy the record while locked, then pin its provider through execution.
       Metadata listing stays responsive while a long inference call runs. */
    if (nya_server_execution_lock(server) != 0) return -1;
    nya_server_models_lock(server);
    model = (nya_model *)nya_model_find(server->config.models, model_id);
    if (model == NULL) {
        nya_server_models_unlock(server);
        nya_server_execution_unlock(server);
        return nya_http_send_error(server, client, 404, "Not Found", "model_not_loaded", "No loaded model has that ID.", origin);
    }

    if (!model->inference_supported) {
        snprintf(
            execution_error,
            sizeof(execution_error),
            "%s",
            model->execution_error[0] == '\0' ? "This container has no active execution provider." : model->execution_error
        );
        nya_server_models_unlock(server);
        nya_server_execution_unlock(server);
        return nya_http_send_error(server, client, 409, "Conflict", "inference_not_supported", execution_error, origin);
    }

    snapshot = *model;
    model = &snapshot;
    nya_server_models_unlock(server);
    execution_result = nya_execution_run(
        model,
        &execution_request,
        &execution_response,
        execution_error,
        sizeof(execution_error)
    );
    nya_server_execution_unlock(server);

    if (execution_result != 0) {
        return nya_http_send_error(server, client, 422, "Unprocessable Content", "inference_failed", execution_error, origin);
    }

    send_result = nya_http_send_tensor(server, client, &execution_response, origin);
    nya_execution_response_free(&execution_response);
    return send_result;
}

/* Generate one complete text response from a supported local GGUF LLaMA model. */
#include "server_compat.inc"
#include "server_runtime.inc"

static int nya_handle_generate(
    nya_socket client,
    nya_server *server,
    const nya_http_request *http_request,
    const char *origin
)
{
    uint64_t model_id, draft_model_id = 0;
    char *prompt;
    uint64_t integer_value;
    double number_value;
    nya_generation_request request;
    nya_generation_response response;
    nya_model *model;
    nya_model snapshot, draft_snapshot;
    char generation_error[512];
    int parse_result;
    int generation_result;
    size_t response_capacity;
    char *body;
    nya_text_builder builder;
    int send_result;

    if (!nya_content_type_is_json(http_request->content_type)) {
        return nya_api_error(server, client, http_request->api_style, 415, "Unsupported Media Type", "json_required", "Use Content-Type: application/json.", origin);
    }
    if (http_request->body_length == SIZE_MAX) {
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "invalid_json", "The generation request is too large.", origin);
    }

    prompt = (char *)malloc(http_request->body_length + 1);
    if (prompt == NULL) {
        return nya_api_error(server, client, http_request->api_style, 503, "Service Unavailable", "memory_unavailable", "Prompt allocation failed.", origin);
    }
    if (nya_json_u64(http_request->body, http_request->body_length, "model_id", &model_id) != 0 || model_id == 0 ||
        nya_json_string(
            http_request->body,
            http_request->body_length,
            "prompt",
            prompt,
            http_request->body_length + 1
        ) != 0) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "invalid_json", "model_id and a valid prompt string are required.", origin);
    }

    memset(&request, 0, sizeof(request));
    request.prompt = prompt;
    request.max_tokens = server->config.settings->max_generation_tokens < 128 ?
        server->config.settings->max_generation_tokens : 128;
    request.temperature = 0.8f;
    request.top_p = 0.95f;
    request.top_k = 40;
    request.max_output_bytes = (server->config.settings->max_output_bytes - 1024) / 6;
    /* Native clients may tighten their raw UTF-8 output budget. The server's
       serialization limit remains an upper bound and cannot be bypassed. */
    parse_result = nya_json_optional_u64(http_request->body,http_request->body_length,"max_output_bytes",&integer_value);
    if (parse_result < 0 || (parse_result > 0 && (integer_value == 0 || integer_value > request.max_output_bytes))) {
        free(prompt);
        return nya_api_error(server,client,http_request->api_style,400,"Bad Request","invalid_output_limit","max_output_bytes must be positive and fit the server's escaped output budget.",origin);
    }
    if (parse_result > 0) request.max_output_bytes = (size_t)integer_value;

    parse_result = nya_json_optional_u64(http_request->body, http_request->body_length, "draft_model_id", &draft_model_id);
    if (parse_result < 0 || (parse_result > 0 && draft_model_id == 0)) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "invalid_draft", "draft_model_id must identify a loaded draft model.", origin);
    }
    parse_result = nya_json_optional_u64(http_request->body, http_request->body_length, "speculative_tokens", &integer_value);
    if (parse_result < 0 || (parse_result > 0 && (draft_model_id == 0 || integer_value == 0 || integer_value > 32))) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "invalid_draft", "speculative_tokens requires draft_model_id and must be from 1 through 32.", origin);
    }
    if (parse_result > 0) request.speculative_tokens = (size_t)integer_value;

    parse_result = nya_json_optional_u64(
        http_request->body,
        http_request->body_length,
        "max_tokens",
        &integer_value
    );
    if (parse_result < 0 ||
        (parse_result > 0 && (integer_value == 0 ||
         integer_value > server->config.settings->max_generation_tokens))) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "generation_limit", "max_tokens is outside the configured range.", origin);
    }
    if (parse_result > 0) request.max_tokens = (size_t)integer_value;

    parse_result = nya_json_optional_u64(
        http_request->body,
        http_request->body_length,
        "top_k",
        &integer_value
    );
    if (parse_result < 0 || (parse_result > 0 && integer_value > 1000000)) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "invalid_sampling", "top_k must be an integer from 0 through 1000000.", origin);
    }
    if (parse_result > 0) request.top_k = (size_t)integer_value;

    parse_result = nya_json_optional_u64(
        http_request->body,
        http_request->body_length,
        "seed",
        &integer_value
    );
    if (parse_result < 0) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "invalid_sampling", "seed must be an unsigned integer.", origin);
    }
    if (parse_result > 0) {
        request.seed = integer_value;
    } else if (nya_runtime_random_bytes(&request.seed, sizeof(request.seed)) != 0) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 500, "Internal Server Error", "random_unavailable", "A secure sampling seed could not be generated.", origin);
    }

    parse_result = nya_json_optional_double(
        http_request->body,
        http_request->body_length,
        "temperature",
        &number_value
    );
    if (parse_result < 0 || (parse_result > 0 && (number_value < 0.0 || number_value > 5.0))) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "invalid_sampling", "temperature must be a finite number from 0 through 5.", origin);
    }
    if (parse_result > 0) request.temperature = (float)number_value;

    parse_result = nya_json_optional_double(
        http_request->body,
        http_request->body_length,
        "top_p",
        &number_value
    );
    if (parse_result < 0 || (parse_result > 0 && (number_value <= 0.0 || number_value > 1.0))) {
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 400, "Bad Request", "invalid_sampling", "top_p must be a finite number greater than 0 and at most 1.", origin);
    }
    if (parse_result > 0) request.top_p = (float)number_value;

    memset(&response, 0, sizeof(response));
    generation_error[0] = '\0';
    if (nya_server_execution_lock(server) != 0) { free(prompt); return -1; }
    nya_server_models_lock(server);
    model = (nya_model *)nya_model_find(server->config.models, model_id);
    if (model == NULL) {
        nya_server_models_unlock(server);
        nya_server_execution_unlock(server);
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 404, "Not Found", "model_not_loaded", "No loaded model has that ID.", origin);
    }
    if (!model->generation_supported) {
        snprintf(
            generation_error,
            sizeof(generation_error),
            "%s",
            model->generation_error[0] == '\0' ?
                "This container has no active text-generation provider." : model->generation_error
        );
        nya_server_models_unlock(server);
        nya_server_execution_unlock(server);
        free(prompt);
        return nya_api_error(server, client, http_request->api_style, 409, "Conflict", "generation_not_supported", generation_error, origin);
    }

    snapshot = *model;
    model = &snapshot;
    if (draft_model_id != 0) {
        const nya_model *draft_model = nya_model_find(server->config.models, draft_model_id);
        if (draft_model == NULL) {
            nya_server_models_unlock(server);
            nya_server_execution_unlock(server);
            free(prompt);
            return nya_api_error(server, client, http_request->api_style, 404, "Not Found", "draft_not_loaded", "No loaded draft model has that ID.", origin);
        }
        /* The execution lock pins both mappings through inference. Copying the
           registry records lets model listing proceed without exposing a draft
           pointer that could move when an unrelated registry slot is unloaded. */
        draft_snapshot = *draft_model;
        request.draft_model = &draft_snapshot;
    }
    nya_server_models_unlock(server);
    uint64_t generation_started = nya_runtime_monotonic_milliseconds();
    if (server->config.verbose) fprintf(stderr,"[generation] model=%" PRIu64 " compute=%s prompt_bytes=%zu max_tokens=%zu top_k=%zu temperature=%.3g top_p=%.3g draft=%" PRIu64 "\n",
        model_id,nya_compute_name(((nya_llm_context *)model->generation_context)->compute),strlen(prompt),request.max_tokens,
        request.top_k,(double)request.temperature,(double)request.top_p,draft_model_id);
    generation_result = nya_generation_run(
        model,
        &request,
        &response,
        generation_error,
        sizeof(generation_error)
    );
    if (server->config.verbose) fprintf(stderr,"[generation] model=%" PRIu64 " result=%d elapsed_ms=%" PRIu64 " prompt_tokens=%zu output_tokens=%zu accepted_draft=%zu\n",
        model_id,generation_result,nya_runtime_monotonic_milliseconds()-generation_started,response.prompt_tokens,response.generated_tokens,response.accepted_draft_tokens);
    nya_server_execution_unlock(server);
    free(prompt);
    if (generation_result != 0) {
        nya_generation_response_free(&response);
        return nya_api_error(server, client, http_request->api_style, 422, "Unprocessable Content", "generation_failed", generation_error, origin);
    }

    if (response.text_length > (SIZE_MAX - 1024) / 6) {
        nya_generation_response_free(&response);
        return nya_api_error(server, client, http_request->api_style, 500, "Internal Server Error", "response_too_large", "Generated text could not be serialized.", origin);
    }
    response_capacity = response.text_length * 6 + 1024;
    if (response_capacity > server->config.settings->max_output_bytes) {
        nya_generation_response_free(&response);
        return nya_api_error(server, client, http_request->api_style, 500, "Internal Server Error", "response_too_large", "Generated JSON exceeded max_output_bytes.", origin);
    }
    body = (char *)malloc(response_capacity);
    if (body == NULL) {
        nya_generation_response_free(&response);
        return nya_api_error(server, client, http_request->api_style, 503, "Service Unavailable", "memory_unavailable", "Response allocation failed.", origin);
    }

    nya_builder_init(&builder, body, response_capacity);
    if (http_request->api_style != 0) {
        nya_compat_response(&builder,http_request->api_style,model_id,&response);
    } else {
    nya_builder_append(
        &builder,
        "{\"version\":%d,\"ok\":true,\"model_id\":%" PRIu64 ",\"text\":",
        NYA_API_VERSION,
        model_id
    );
    nya_builder_append_json_string(&builder, response.text);
    nya_builder_append(
        &builder,
        ",\"prompt_tokens\":%zu,\"generated_tokens\":%zu,\"seed\":%" PRIu64 ",\"stop_reason\":",
        response.prompt_tokens,
        response.generated_tokens,
        response.seed
    );
    nya_builder_append_json_string(&builder, nya_generation_stop_reason_name(response.stop_reason));
    nya_builder_append(&builder, ",\"draft_tokens\":%zu,\"accepted_draft_tokens\":%zu,\"target_steps\":%zu",
        response.draft_tokens, response.accepted_draft_tokens, response.target_steps);
    nya_builder_append(&builder, "}");
    }
    nya_generation_response_free(&response);
    if (builder.failed) {
        free(body);
        return nya_api_error(server, client, http_request->api_style, 500, "Internal Server Error", "response_too_large", "Generated JSON exceeded its checked capacity.", origin);
    }

    send_result = nya_http_send_response(server, client, 200, "OK", body, origin);
    free(body);
    return send_result;
}

/* Route one authenticated request to a small explicit endpoint handler. */
static int nya_http_dispatch_request(
    nya_socket client,
    nya_server *server,
    const nya_http_request *request
)
{
    const char *origin;
    int known_path;

    /* Namespaced aliases let SDKs choose a base URL without colliding on the
       two providers' different /v1/models response schemas. Native aliases
       retain the original paths for existing desktop clients. */
    nya_http_request normalized = *request;
    const char *path = request->path;
    if (strncmp(path,"/openai/",8) == 0) { path += 7; normalized.api_style = 1; }
    else if (strncmp(path,"/anthropic/",11) == 0) { path += 10; normalized.api_style = 2; }
    else if (strncmp(path,"/api/v1/",8) == 0) path += 7;
    snprintf(normalized.path,sizeof(normalized.path),"%s",path);
    if (strcmp(path,"/v1/messages") == 0 || strcmp(path,"/v1/messages/count_tokens") == 0) normalized.api_style = 2;
    else if (strcmp(path,"/v1/completions") == 0) normalized.api_style = 3;
    else if (strncmp(path,"/v1/",4) == 0 && normalized.api_style == 0) normalized.api_style = 1;
    request = &normalized;

    /* Never reflect an origin until it passes the narrow allowlist. */
    if (!nya_origin_allowed(server, request->origin)) {
        return nya_api_error(server, client, request->api_style, 403, "Forbidden", "origin_not_allowed", "The browser origin is not allowed.", NULL);
    }
    origin = request->origin[0] == '\0' ? NULL : request->origin;

    /* Browsers send preflight without credentials, so answer it before auth. */
    if (strcmp(request->method, "OPTIONS") == 0) {
        return nya_http_send_response(server, client, 204, "No Content", NULL, origin);
    }

    /* Every actual API operation requires the random per-process token. */
    char key_authorization[NYA_HTTP_HEADER_LIMIT+8];
    snprintf(key_authorization,sizeof(key_authorization),"Bearer %.*s",(int)sizeof(request->api_key)-1,request->api_key);
    int authenticated = nya_auth_valid(request->authorization,server->config.auth_token);
    if (request->api_style == 2 && request->api_key[0] != '\0') {
        int key_valid = nya_auth_valid(key_authorization,server->config.auth_token);
        authenticated = key_valid && (request->authorization[0] == '\0' || authenticated);
    }
    if (!authenticated) {
        return nya_api_error(server, client, request->api_style, 401, "Unauthorized", "unauthorized", "Use the local Fyodor token as the bearer token (or Anthropic x-api-key).", origin);
    }

    /* Check the complete JSON once, before interpreting any field or mutating
       model state. Trailing junk, duplicate keys, and invalid Unicode fail as
       one request instead of letting individual field searches disagree. */
    if (strcmp(request->method, "POST") == 0 && nya_content_type_is_json(request->content_type) &&
        nya_json_validate(request->body, request->body_length) != 0) {
        return nya_api_error(server, client, request->api_style, 400, "Bad Request", "invalid_json", "A valid JSON object with unique top-level fields is required.", origin);
    }

    if (strcmp(request->path,"/v1/models") == 0 && strcmp(request->method,"GET") == 0)
        return nya_handle_compat_models(client,server,request,origin);
    if ((strcmp(request->path,"/v1/chat/completions") == 0 || strcmp(request->path,"/v1/completions") == 0 ||
         strcmp(request->path,"/v1/messages") == 0 || strcmp(request->path,"/v1/messages/count_tokens") == 0) && strcmp(request->method,"POST") == 0)
        return nya_handle_compat_generate(client,server,request,origin);
    if (strcmp(request->path,"/runtime") == 0 && strcmp(request->method,"GET") == 0)
        return nya_handle_runtime(client,server,origin);
    if (strcmp(request->path,"/model/compute") == 0 && strcmp(request->method,"POST") == 0)
        return nya_handle_model_compute(client,server,request,origin);
    if (strcmp(request->path,"/model/info") == 0 && strcmp(request->method,"POST") == 0)
        return nya_handle_model_info(client,server,request,origin);
    if (strcmp(request->path,"/capabilities") == 0 && strcmp(request->method,"GET") == 0)
        return nya_http_send_response(server,client,200,"OK",
            "{\"version\":1,\"ok\":true,\"api\":{\"native\":\"/api/v1\",\"openai\":\"/openai/v1\",\"anthropic\":\"/anthropic\"},"
            "\"compatibility\":{\"text\":true,\"streaming\":false,\"tools\":false,\"media\":false,\"stop_sequences\":false,\"multiple_choices\":false},"
            "\"native_controls\":[\"model_load\",\"model_unload\",\"model_compute\",\"runtime\",\"tensor_execution\",\"tokenize\",\"top_k\",\"seed\",\"draft_model_id\",\"speculative_tokens\"]}",origin);
    if (strcmp(request->path,"/tokenize") == 0 && strcmp(request->method,"POST") == 0) {
        uint64_t id; char *text = (char *)malloc(request->body_length+1);
        if (text == NULL) return -1;
        if (!nya_content_type_is_json(request->content_type) ||
            nya_json_u64(request->body,request->body_length,"model_id",&id) != 0 || id == 0 ||
            nya_json_string(request->body,request->body_length,"text",text,request->body_length+1) != 0) {
            free(text); return nya_api_error(server,client,0,400,"Bad Request","invalid_tokenize","JSON model_id and text are required.",origin);
        }
        int result = nya_handle_tokenize(client,server,request,origin,id,text,0); free(text); return result;
    }

    if (strcmp(request->path, "/health") == 0 && strcmp(request->method, "GET") == 0) {
        return nya_handle_health(client, server, origin);
    }

    if (strcmp(request->path, "/models") == 0 && strcmp(request->method, "GET") == 0) {
        return nya_handle_models(client, server, origin);
    }

    if (strcmp(request->path, "/model/load") == 0 && strcmp(request->method, "POST") == 0) {
        if (!nya_content_type_is_json(request->content_type)) {
            return nya_http_send_error(server, client, 415, "Unsupported Media Type", "json_required", "Use Content-Type: application/json.", origin);
        }
        return nya_handle_model_load(client, server, request, origin);
    }

    if (strcmp(request->path, "/model/unload") == 0 && strcmp(request->method, "POST") == 0) {
        if (!nya_content_type_is_json(request->content_type)) {
            return nya_http_send_error(server, client, 415, "Unsupported Media Type", "json_required", "Use Content-Type: application/json.", origin);
        }
        return nya_handle_model_unload(client, server, request, origin);
    }

    if (strcmp(request->path, "/model/run") == 0 && strcmp(request->method, "POST") == 0) {
        return nya_handle_model_run(client, server, request, origin);
    }

    if (strcmp(request->path, "/generate") == 0 && strcmp(request->method, "POST") == 0) {
        return nya_handle_generate(client, server, request, origin);
    }

    if (strcmp(request->path, "/shutdown") == 0 && strcmp(request->method, "POST") == 0) {
        int result;

        result = nya_http_send_response(server, client, 200, "OK", "{\"version\":1,\"ok\":true,\"status\":\"shutting_down\"}", origin);
        nya_server_request_stop(server);
        return result;
    }

    /* Distinguish a wrong method from a path that does not exist. */
    known_path = strcmp(request->path, "/health") == 0 ||
                 strcmp(request->path, "/models") == 0 ||
                 strcmp(request->path, "/model/load") == 0 ||
                 strcmp(request->path, "/model/unload") == 0 ||
                 strcmp(request->path, "/model/run") == 0 ||
                 strcmp(request->path, "/generate") == 0 ||
                 strcmp(request->path, "/shutdown") == 0;
    known_path = known_path || strcmp(request->path,"/runtime") == 0 || strcmp(request->path,"/model/compute") == 0 || strcmp(request->path,"/model/info") == 0 ||
        strcmp(request->path,"/capabilities") == 0 || strcmp(request->path,"/tokenize") == 0 ||
        strcmp(request->path,"/v1/models") == 0 || strcmp(request->path,"/v1/messages") == 0 ||
        strcmp(request->path,"/v1/messages/count_tokens") == 0 || strcmp(request->path,"/v1/chat/completions") == 0 ||
        strcmp(request->path,"/v1/completions") == 0;

    if (known_path) {
        return nya_api_error(server, client, request->api_style, 405, "Method Not Allowed", "method_not_allowed", "That HTTP method is not supported for this endpoint.", origin);
    }

    return nya_api_error(server, client, request->api_style, 404, "Not Found", "not_found", "The requested endpoint does not exist.", origin);
}

/* Request summaries omit headers and bodies. The socket identifies overlapping
   requests in verbose output without disclosing authentication or user text. */
static int nya_http_handle_request(nya_socket client, nya_server *server, const nya_http_request *request)
{
    uint64_t started = nya_runtime_monotonic_milliseconds();
    if (server->config.verbose) fprintf(stderr,"[http] socket=%" PRIuPTR " %s %s request_bytes=%zu\n",(uintptr_t)client,request->method,request->path,request->body_length);
    int result = nya_http_dispatch_request(client,server,request);
    if (server->config.verbose) fprintf(stderr,"[http] socket=%" PRIuPTR " elapsed_ms=%" PRIu64 " send_result=%d\n",(uintptr_t)client,nya_runtime_monotonic_milliseconds()-started,result);
    return result;
}

/* Nonblocking sockets make select's absolute deadline authoritative; kernel
   timeouts additionally retain the configured send budget for each handler. */
static int nya_socket_configure_client(nya_socket client, unsigned int timeout_ms)
{
#ifdef _WIN32
    DWORD timeout;
    u_long nonblocking = 1;

    timeout = (DWORD)timeout_ms;
    if (ioctlsocket(client, (long)FIONBIO, &nonblocking) != 0 ||
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout, sizeof(timeout)) != 0 ||
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (const char *)&timeout, sizeof(timeout)) != 0) return -1;
#else
    struct timeval timeout;
    int flags = fcntl(client, F_GETFL, 0);

    timeout.tv_sec = (long)(timeout_ms / 1000);
    timeout.tv_usec = (long)(timeout_ms % 1000) * 1000L;
    if (client >= FD_SETSIZE || flags < 0 || fcntl(client, F_SETFL, flags | O_NONBLOCK) != 0 ||
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) return -1;
#ifdef SO_NOSIGPIPE
    /* macOS uses a socket option instead of Linux's per-send MSG_NOSIGNAL. */
    {
        int enabled = 1;
        if (setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0) return -1;
    }
#endif
#endif
    return 0;
}

/* Receive, parse, route, and close one client request. */
static void nya_server_handle_client(nya_socket client, nya_server *server)
{
    char *request_buffer;
    size_t request_capacity;
    nya_http_request request;
    int receive_result;

    request_capacity = server->config.settings->max_request_bytes + 1;
    request_buffer = (char *)malloc(request_capacity);
    if (request_buffer == NULL) {
        nya_http_send_error(server, client, 503, "Service Unavailable", "memory_unavailable", "The server could not allocate a bounded request buffer.", NULL);
        return;
    }

    receive_result = nya_http_receive(client, request_buffer, request_capacity, &request,
        server->config.settings->request_timeout_ms, server);
    if (receive_result == -2) {
        nya_http_send_error(server, client, 413, "Content Too Large", "request_too_large", "The request exceeds server.max_request_bytes.", NULL);
        free(request_buffer);
        return;
    }
    if (receive_result == -3) {
        nya_http_send_error(server, client, 431, "Request Header Fields Too Large", "headers_too_large", "HTTP headers may not exceed 32 KiB.", NULL);
        free(request_buffer);
        return;
    }
    if (receive_result == -4) {
        nya_http_send_error(server, client, 408, "Request Timeout", "request_timeout", "The complete request was not received within request_timeout_ms.", NULL);
        free(request_buffer);
        return;
    }
    if (receive_result != 0) {
        nya_http_send_error(server, client, 400, "Bad Request", "malformed_request", "The HTTP request is malformed or incomplete.", NULL);
        free(request_buffer);
        return;
    }

    nya_http_handle_request(client, server, &request);
    free(request_buffer);
}

/* Create the configured IPv4 listener and determine its actual port. */
int nya_server_init(nya_server *server, const nya_server_config *config)
{
    nya_socket listener;
    struct sockaddr_in address;
    int reuse_address;
    nya_socket_length address_length;
    char config_error[256];

    if (server == NULL || config == NULL || config->settings == NULL || config->auth_token == NULL ||
        config->runtime == NULL || config->models == NULL ||
        strlen(config->auth_token) != NYA_AUTH_TOKEN_LENGTH) {
        return -1;
    }
    if (nya_config_validate(config->settings, config_error, sizeof(config_error)) != 0) return -1;

    memset(server, 0, sizeof(*server));
    nya_server_stop_initialize(server);
    server->socket_handle = nya_socket_to_handle(NYA_INVALID_SOCKET);
    server->config = *config;

#ifdef _WIN32
    {
        WSADATA winsock_data;

        if (WSAStartup(MAKEWORD(2, 2), &winsock_data) != 0) {
            return -1;
        }
        server->networking_initialized = 1;
    }
#endif

    listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener == NYA_INVALID_SOCKET) {
        nya_server_shutdown(server);
        return -1;
    }
    server->socket_handle = nya_socket_to_handle(listener);
#ifndef _WIN32
    if (listener >= FD_SETSIZE) {
        nya_server_shutdown(server);
        return -1;
    }
#endif

    /* Windows SO_REUSEADDR permits another process to steal the bound endpoint.
       Exclusive binding preserves one backend owner; POSIX reuse only permits
       a restart while old connections remain in TIME_WAIT. */
    reuse_address = 1;
#ifdef _WIN32
    setsockopt(listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *)&reuse_address, sizeof(reuse_address));
#else
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse_address, sizeof(reuse_address));
#endif

    /* Convert the already validated textual IPv4 address without DNS ambiguity. */
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    if (inet_pton(AF_INET, config->settings->bind_address, &address.sin_addr) != 1) {
        nya_server_shutdown(server);
        return -1;
    }
    address.sin_port = htons(config->settings->port);

    if (bind(listener, (const struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(listener, (int)config->settings->backlog) != 0) {
        nya_server_shutdown(server);
        return -1;
    }

    /* Query the kernel-selected port when the caller requested port zero. */
    address_length = (nya_socket_length)sizeof(address);
    if (getsockname(listener, (struct sockaddr *)&address, &address_length) != 0) {
        nya_server_shutdown(server);
        return -1;
    }

    server->port = ntohs(address.sin_port);

    /* Start bounded workers only after the listener is fully configured. */
    if (nya_server_workers_init(server) != 0) {
        nya_server_shutdown(server);
        return -1;
    }

    return 0;
}

/* Wait in short intervals so signals and HTTP shutdown remain responsive. */
int nya_server_run(nya_server *server)
{
    nya_socket listener;

    if (server == NULL) {
        return -1;
    }

    listener = nya_socket_from_handle(server->socket_handle);
    if (listener == NYA_INVALID_SOCKET) {
        return -1;
    }

    while (!nya_server_stop_requested(server) &&
           (server->config.external_stop_requested == NULL ||
            !*server->config.external_stop_requested)) {
        fd_set readable;
        struct timeval timeout;
        int selected;

        FD_ZERO(&readable);
        FD_SET(listener, &readable);
        timeout.tv_sec = 0;
        timeout.tv_usec = 250000;

#ifdef _WIN32
        selected = select(0, &readable, NULL, NULL, &timeout);
#else
        selected = select(listener + 1, &readable, NULL, NULL, &timeout);
#endif
        if (selected < 0) {
#ifndef _WIN32
            if (errno == EINTR) {
                continue;
            }
#endif
            return -1;
        }

        if (selected > 0 && FD_ISSET(listener, &readable)) {
            nya_socket client;

            client = accept(listener, NULL, NULL);
            if (client == NYA_INVALID_SOCKET) {
                continue;
            }
            if (nya_socket_configure_client(client, server->config.settings->request_timeout_ms) != 0) {
                nya_socket_close(client);
                continue;
            }

            if (nya_server_queue_client(server, client) != 0) {
                /* One best-effort nonblocking write prevents a full queue's
                   slow reader from stalling the accepting thread. */
                static const char busy[] = "HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
                int flags = 0;
#ifdef MSG_NOSIGNAL
                flags = MSG_NOSIGNAL;
#endif
                (void)send(client, busy, (int)(sizeof(busy) - 1), flags);
                nya_socket_close(client);
            }
        }
    }

    return 0;
}

/* Set the only state observed by the server run loop. */
void nya_server_request_stop(nya_server *server)
{
    if (server != NULL) {
        nya_server_stop_request(server);
    }
}

/* Release networking resources in reverse initialization order. */
void nya_server_shutdown(nya_server *server)
{
    nya_socket listener;

    if (server == NULL) {
        return;
    }

    nya_server_stop_request(server);

    listener = nya_socket_from_handle(server->socket_handle);
    if (listener != NYA_INVALID_SOCKET) {
        nya_socket_close(listener);
        server->socket_handle = nya_socket_to_handle(NYA_INVALID_SOCKET);
    }

    /* Workers finish active requests and release all synchronization state. */
    nya_server_workers_shutdown(server);

#ifdef _WIN32
    if (server->networking_initialized) {
        WSACleanup();
        server->networking_initialized = 0;
    }
#endif
}

/* Expose the actual bound port without exposing the socket itself. */
unsigned short nya_server_port(const nya_server *server)
{
    return server == NULL ? 0 : server->port;
}
