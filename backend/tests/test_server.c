/* Private parser tests intentionally include the implementation rather than
   exporting HTTP internals as part of the backend's public C API. Network
   tests use real loopback sockets to exercise framing and worker shutdown. */
#include "../core/server.c"

#ifndef _WIN32
#include <time.h>
#endif

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return -1; \
} } while (0)

static void pause_ms(unsigned int milliseconds)
{
#ifdef _WIN32
    Sleep(milliseconds);
#else
    struct timespec delay;
    delay.tv_sec = (time_t)(milliseconds / 1000);
    delay.tv_nsec = (long)(milliseconds % 1000) * 1000000L;
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) { }
#endif
}

static int test_json(void)
{
    static const char *invalid[] = {
        "", "[]", "null", "{\"model_id\":1", "{\"model_id\":1}junk",
        "{\"model_id\":01}", "{\"model_id\":+1}", "{\"temperature\":0x1p0}",
        "{\"temperature\":NaN}", "{\"temperature\":1.}", "{\"temperature\":1e}",
        "{\"model_id\":1,}", "{\"model_id\":1,\"model_id\":2}",
        "{\"model_id\":1,\"model_\\u0069d\":2}",
        "{\"prompt\":\"\\u0000\"}", "{\"prompt\":\"\\ud800\"}",
        "{\"prompt\":\"\\udc00\"}", "{\"prompt\":\"\xc0\x80\"}",
        "{\"prompt\":\"\xed\xa0\x80\"}", "{\"prompt\":\"\xf4\x90\x80\x80\"}",
        "{\"extra\":[1,]}", "{\"extra\":{\"a\":true false}}"
    };
    static const char valid[] = "{\"extra\":{\"model_id\":99},\"prompt\":\"inside \\\"model_id\\\": 7\",\"model_\\u0069d\":42,\"temperature\":1.5e-1}";
    static const char unicode[] = "{\"prompt\":\"\\uD55C\\uAE00 \\ud83d\\ude00\"}";
    const char *nested = "{\"extra\":{\"model_id\":99}}";
    char decoded[128];
    uint64_t id = 0;
    double number = 0;
    size_t index;
    for (index = 0; index < sizeof(invalid) / sizeof(invalid[0]); ++index)
        CHECK(nya_json_validate(invalid[index], strlen(invalid[index])) != 0);
    CHECK(nya_json_validate(valid, strlen(valid)) == 0);
    CHECK(nya_json_u64(valid, strlen(valid), "model_id", &id) == 0 && id == 42);
    CHECK(nya_json_optional_double(valid, strlen(valid), "temperature", &number) == 1 && number == 0.15);
    CHECK(nya_json_value(nested, strlen(nested), "model_id") == NULL);
    CHECK(nya_json_validate(unicode, strlen(unicode)) == 0);
    CHECK(nya_json_string(unicode, strlen(unicode), "prompt", decoded, sizeof(decoded)) == 0);
    CHECK(strcmp(decoded, "\xed\x95\x9c\xea\xb8\x80 \xf0\x9f\x98\x80") == 0);
    CHECK(nya_json_string(unicode, strlen(unicode), "prompt", decoded, 2) != 0);
    CHECK(nya_json_validate("{}", 2) == 0);
    return 0;
}

static int parse_http_text(const char *text)
{
    char buffer[2048];
    const char *end;
    nya_http_request request;
    size_t length = strlen(text);
    if (length >= sizeof(buffer)) return -1;
    memcpy(buffer, text, length + 1);
    end = strstr(buffer, "\r\n\r\n");
    if (end == NULL) return -1;
    return nya_http_parse_request(buffer, length, end, 0, &request);
}

static int test_headers(void)
{
    char value[64];
    int64_t shape[16];
    size_t rank;
    const char *duplicate = "POST /generate HTTP/1.1\r\nContent-Length: 1\r\ncontent-length: 2\r\n\r\n";
    CHECK(parse_http_text("GET /health HTTP/1.1\r\n\r\n") == 0);
    CHECK(parse_http_text("GET /health HTTP/1.1 extra\r\n\r\n") != 0);
    CHECK(parse_http_text("GET /health HTTP/1.1\r\nBad Header: x\r\n\r\n") != 0);
    CHECK(parse_http_text("GET /health HTTP/1.1\r\nAuthorization: one\r\nAuthorization: two\r\n\r\n") != 0);
    CHECK(parse_http_text("GET /health HTTP/1.1\r\nX-Test: x\001y\r\n\r\n") != 0);
    CHECK(nya_http_header_value(duplicate, strstr(duplicate, "\r\n\r\n"), "Content-Length", value, sizeof(value)) < 0);
    CHECK(nya_http_tensor_shape("2,3", shape, &rank) == 0 && rank == 2 && shape[0] == 2 && shape[1] == 3);
    CHECK(nya_http_tensor_shape("2,", shape, &rank) != 0);
    CHECK(nya_http_tensor_shape("2, ", shape, &rank) != 0);
    return 0;
}

/* TCP loopback gives the same socket type and framing behavior on both host
   platforms; socketpair is unavailable on supported Windows environments. */
static int connected_pair(nya_socket pair[2])
{
    nya_socket listener;
    struct sockaddr_in address;
    nya_socket_length length = (nya_socket_length)sizeof(address);
    listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    CHECK(listener != NYA_INVALID_SOCKET);
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    CHECK(bind(listener, (struct sockaddr *)&address, sizeof(address)) == 0);
    CHECK(listen(listener, 1) == 0);
    CHECK(getsockname(listener, (struct sockaddr *)&address, &length) == 0);
    pair[0] = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    CHECK(pair[0] != NYA_INVALID_SOCKET);
    CHECK(connect(pair[0], (struct sockaddr *)&address, sizeof(address)) == 0);
    pair[1] = accept(listener, NULL, NULL);
    nya_socket_close(listener);
    CHECK(pair[1] != NYA_INVALID_SOCKET);
    CHECK(nya_socket_configure_client(pair[0], 1000) == 0);
    CHECK(nya_socket_configure_client(pair[1], 1000) == 0);
    return 0;
}

static int test_receive(void)
{
    nya_socket pair[2];
    char *data = (char *)malloc(70000);
    char *received = (char *)malloc(70000);
    nya_http_request request;
    int prefix;
    const char *duplicate = "POST /generate HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n";
    const char *transfer = "POST /generate HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
    CHECK(data != NULL && received != NULL);
    CHECK(connected_pair(pair) == 0);
    prefix = snprintf(data, 70000, "POST /model/run HTTP/1.1\r\nContent-Length: 50000\r\n\r\n");
    CHECK(prefix > 0);
    memset(data + prefix, 0xa5, 50000);
    data[prefix + 1] = '\0'; /* Binary bodies may contain embedded NUL bytes. */
    CHECK(nya_socket_send_all(pair[0], data, (size_t)prefix + 50000, NULL) == 0);
    CHECK(nya_http_receive(pair[1], received, 70000, &request, 1000, NULL) == 0);
    CHECK(request.body_length == 50000 && memcmp(request.body, data + prefix, 50000) == 0);
    nya_socket_close(pair[0]); nya_socket_close(pair[1]);

    CHECK(connected_pair(pair) == 0);
    CHECK(nya_socket_send_all(pair[0], duplicate, strlen(duplicate), NULL) == 0);
    CHECK(nya_http_receive(pair[1], received, 70000, &request, 1000, NULL) == -1);
    nya_socket_close(pair[0]); nya_socket_close(pair[1]);

    CHECK(connected_pair(pair) == 0);
    CHECK(nya_socket_send_all(pair[0], transfer, strlen(transfer), NULL) == 0);
    CHECK(nya_http_receive(pair[1], received, 70000, &request, 1000, NULL) == -1);
    nya_socket_close(pair[0]); nya_socket_close(pair[1]);

    CHECK(connected_pair(pair) == 0);
    memset(data, 'a', 33000);
    memcpy(data, "GET / HTTP/1.1\r\nX:", 18);
    memcpy(data + 33000, "\r\n\r\n", 4);
    CHECK(nya_socket_send_all(pair[0], data, 33004, NULL) == 0);
    CHECK(nya_http_receive(pair[1], received, 70000, &request, 1000, NULL) == -3);
    nya_socket_close(pair[0]); nya_socket_close(pair[1]);
    free(data); free(received);
    return 0;
}

static int drip_sender(void *argument)
{
    nya_socket client = *(nya_socket *)argument;
    unsigned int index;
    for (index = 0; index < 12; ++index) {
        if (nya_socket_send_all(client, "G", 1, NULL) != 0) break;
        pause_ms(25);
    }
    return 0;
}

static int test_deadline(void)
{
    nya_socket pair[2];
    nya_thread sender = {0};
    char buffer[1024];
    nya_http_request request;
    uint64_t started;
    CHECK(connected_pair(pair) == 0);
    CHECK(nya_thread_create(&sender, drip_sender, &pair[0]) == 0);
    started = nya_runtime_monotonic_milliseconds();
    CHECK(nya_http_receive(pair[1], buffer, sizeof(buffer), &request, 80, NULL) == -4);
    CHECK(nya_runtime_monotonic_milliseconds() - started < 1000);
    CHECK(nya_thread_join(&sender) == 0);
    nya_socket_close(pair[0]); nya_socket_close(pair[1]);
    return 0;
}

static int run_server(void *argument)
{
    return nya_server_run((nya_server *)argument);
}

static int test_shutdown(void)
{
    static const char token[] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    nya_config settings;
    nya_runtime runtime;
    nya_model_registry models;
    nya_server_config config;
    nya_server server;
    nya_thread listener = {0};
    nya_socket clients[2];
    struct sockaddr_in address;
    uint64_t started;
    size_t index;
    nya_config_defaults(&settings);
    settings.worker_threads = 1;
    settings.request_timeout_ms = 5000;
    settings.port = 0;
    CHECK(nya_runtime_init(&runtime) == 0);
    nya_model_registry_init(&models);
    memset(&config, 0, sizeof(config));
    config.settings = &settings; config.auth_token = token;
    config.runtime = &runtime; config.models = &models;
    CHECK(nya_server_init(&server, &config) == 0);
    CHECK(nya_thread_create(&listener, run_server, &server) == 0);
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(nya_server_port(&server));
    for (index = 0; index < 2; ++index) {
        clients[index] = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        CHECK(clients[index] != NYA_INVALID_SOCKET);
        CHECK(connect(clients[index], (struct sockaddr *)&address, sizeof(address)) == 0);
        CHECK(send(clients[index], "GET /", 5, 0) == 5);
    }
    pause_ms(50);
    started = nya_runtime_monotonic_milliseconds();
    nya_server_request_stop(&server);
    CHECK(nya_thread_join(&listener) == 0);
    nya_server_shutdown(&server);
    /* Both an active incomplete request and a queued client must close without
       waiting for their five-second request timeout. */
    CHECK(nya_runtime_monotonic_milliseconds() - started < 1500);
    for (index = 0; index < 2; ++index) nya_socket_close(clients[index]);
    nya_model_registry_shutdown(&models);
    nya_runtime_shutdown(&runtime);
    return 0;
}

int main(void)
{
#ifdef _WIN32
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    if (test_json() != 0 || test_headers() != 0 || test_receive() != 0 ||
        test_deadline() != 0 || test_shutdown() != 0) return 1;
#ifdef _WIN32
    WSACleanup();
#endif
    puts("HTTP framing, JSON, deadlines, and worker shutdown checks passed.");
    return 0;
}
