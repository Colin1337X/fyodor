/* Exercise the real route handlers and serializers over loopback TCP. This
   bypasses only the already-tested worker pool, not authentication or model
   execution. The pretraining fixture is a complete learned byte-token model. */
#include "../core/server.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"API:%d: %s\n",__LINE__,#x); return 1; } } while (0)
static char response_text[16384];

static int call(nya_server *server, const char *method, const char *path, const char *body, int auth, int expected)
{
    nya_socket listener = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP), sender, receiver;
    struct sockaddr_in address = {0}; nya_socket_length length = sizeof(address);
    CHECK(listener != NYA_INVALID_SOCKET);
    address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    CHECK(bind(listener,(struct sockaddr *)&address,sizeof(address)) == 0 && listen(listener,1) == 0 &&
        getsockname(listener,(struct sockaddr *)&address,&length) == 0);
    receiver = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP); CHECK(receiver != NYA_INVALID_SOCKET);
    CHECK(connect(receiver,(struct sockaddr *)&address,sizeof(address)) == 0);
    sender = accept(listener,NULL,NULL); nya_socket_close(listener); CHECK(sender != NYA_INVALID_SOCKET);
    CHECK(nya_socket_configure_client(sender,1000) == 0 && nya_socket_configure_client(receiver,1000) == 0);
    nya_http_request r = {0}; snprintf(r.method,sizeof(r.method),"%s",method); snprintf(r.path,sizeof(r.path),"%s",path);
    snprintf(r.content_type,sizeof(r.content_type),"application/json");
    if (auth == 1) snprintf(r.authorization,sizeof(r.authorization),"Bearer %s",server->config.auth_token);
    if (auth == 2) snprintf(r.api_key,sizeof(r.api_key),"%s",server->config.auth_token);
    snprintf(r.anthropic_version,sizeof(r.anthropic_version),"2023-06-01");
    r.body = body == NULL ? "" : body; r.body_length = strlen(r.body);
    CHECK(nya_http_handle_request(sender,server,&r) == 0); nya_socket_close(sender);
    size_t used = 0;
    while (used < sizeof(response_text)-1) {
        fd_set read_set; FD_ZERO(&read_set); FD_SET(receiver,&read_set); struct timeval timeout = {1,0};
#ifdef _WIN32
        /* Winsock ignores nfds; do not narrow its pointer-sized SOCKET to int. */
        CHECK(select(0,&read_set,NULL,NULL,&timeout) > 0);
#else
        CHECK(select(receiver+1,&read_set,NULL,NULL,&timeout) > 0);
#endif
        int received = recv(receiver,response_text+used,(int)(sizeof(response_text)-used-1),0);
        CHECK(received >= 0); if (received == 0) break; used += (size_t)received;
    }
    nya_socket_close(receiver); response_text[used] = '\0';
    int status; CHECK(sscanf(response_text,"HTTP/1.1 %d",&status) == 1);
    if (status != expected) fprintf(stderr,"%s: %s\n",path,response_text);
    CHECK(status == expected);
    char *json = strstr(response_text,"\r\n\r\n"); CHECK(json != NULL && nya_json_validate(json+4,strlen(json+4)) == 0);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    WSADATA data; CHECK(WSAStartup(MAKEWORD(2,2),&data) == 0);
#endif
    CHECK(argc == 2);
    nya_config config; nya_config_defaults(&config);
    nya_model_registry models; nya_model_registry_init(&models); const nya_model *model;
    CHECK(nya_model_load(&models,argv[1],&model) == NYA_MODEL_OK && model->generation_supported && model->id == 1);
    nya_server server = {0}; server.config.settings = &config; server.config.models = &models;
    nya_server_internal internal = {0}; internal.server = &server; server.internal = &internal;
    CHECK(nya_mutex_init(&internal.model_mutex) == 0 && nya_mutex_init(&internal.execution_mutex) == 0);
    server.config.auth_token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    CHECK(call(&server,"GET","/v1/models",NULL,0,401) == 0 && strstr(response_text,"authentication_error") != NULL);
    CHECK(call(&server,"GET","/anthropic/v1/models",NULL,0,401) == 0 && strstr(response_text,"\"type\":\"error\"") != NULL);
    CHECK(call(&server,"GET","/openai/v1/models",NULL,1,200) == 0 && strstr(response_text,"\"object\":\"list\"") != NULL);
    CHECK(call(&server,"GET","/anthropic/v1/models",NULL,2,200) == 0 && strstr(response_text,"\"has_more\":false") != NULL);
    CHECK(call(&server,"GET","/api/v1/capabilities",NULL,1,200) == 0 && strstr(response_text,"\"streaming\":false") != NULL);
    CHECK(call(&server,"GET","/api/v1/runtime",NULL,1,200) == 0 && strstr(response_text,"\"training_compute\":\"cpu\"") != NULL);
    CHECK(call(&server,"POST","/api/v1/model/info","{\"model_id\":1}",1,200) == 0 && strstr(response_text,"\"vocabulary_size\":259") != NULL);
    CHECK(call(&server,"POST","/api/v1/model/info","{\"model_id\":999}",1,404) == 0);
    CHECK(call(&server,"POST","/api/v1/generate","{\"model_id\":1,\"prompt\":\"a\",\"max_output_bytes\":0}",1,400) == 0);
    CHECK(call(&server,"POST","/api/v1/model/compute","{\"model_id\":1,\"compute\":\"cpu\"}",1,200) == 0);
    CHECK(call(&server,"POST","/api/v1/model/compute","{\"model_id\":999,\"compute\":\"cpu\"}",1,404) == 0);
    CHECK(call(&server,"POST","/api/v1/model/compute","{\"model_id\":1,\"compute\":\"invalid\"}",1,400) == 0);
    if (!nya_compute_cuda_compiled()) CHECK(call(&server,"POST","/api/v1/model/compute","{\"model_id\":1,\"compute\":\"cuda\"}",1,409) == 0);
    CHECK(call(&server,"GET","/api/v1/model/compute",NULL,1,405) == 0);
    CHECK(call(&server,"POST","/api/v1/tokenize","{\"model_id\":1,\"text\":\"a\"}",1,200) == 0 && strstr(response_text,"[1,100]") != NULL);
    CHECK(call(&server,"POST","/api/v1/tokenize","{\"model_id\":999,\"text\":\"a\"}",1,422) == 0 && strstr(response_text,"tokenization_failed") != NULL);
    CHECK(call(&server,"POST","/api/v1/tokenize","{\"model_id\":1,\"text\":null}",1,400) == 0);
    CHECK(call(&server,"POST","/v1/completions","{\"model\":\"1\",\"prompt\":\"a\",\"temperature\":0,\"max_tokens\":8}",1,200) == 0 && strstr(response_text,"bcabcabc") != NULL);
    const char *chat = "{\"model\":\"1\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"temperature\":0,\"max_tokens\":2}";
    CHECK(call(&server,"POST","/openai/v1/chat/completions",chat,1,200) == 0 && strstr(response_text,"chat.completion") != NULL);
    CHECK(call(&server,"POST","/anthropic/v1/messages",chat,2,200) == 0 && strstr(response_text,"\"type\":\"message\"") != NULL);
    CHECK(call(&server,"POST","/v1/messages/count_tokens",chat,2,200) == 0 && strstr(response_text,"input_tokens") != NULL);
    const char *bad[] = {
        "{\"model\":\"1\",\"messages\":[],\"stream\":true}",
        "{\"model\":\"1\",\"messages\":[],\"tools\":[]}",
        "{\"model\":\"1\",\"messages\":[{\"role\":\"user\",\"role\":\"assistant\",\"content\":\"a\"}]}",
        "{\"model\":\"1\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com\"}}]}]}",
        "{\"model\":\"1\",\"messages\":[{\"role\":\"assistant\",\"content\":\"prefill\"}]}",
        "{\"model\":\"1\",\"messages\":[],\"n\":2}"
    };
    for (size_t i = 0; i < sizeof(bad)/sizeof(bad[0]); ++i) CHECK(call(&server,"POST","/v1/chat/completions",bad[i],1,400) == 0);
    CHECK(call(&server,"GET","/v1/chat/completions",NULL,1,405) == 0);
    CHECK(call(&server,"POST","/v1/responses","{}",1,404) == 0);
    /* Rendering is tested separately to include Unicode, text blocks, exact
       Gemma markers and rejection of duplicate nested block fields. */
    nya_http_request r = {0}; char prompt[1024], scratch[1024]; nya_text_builder b;
    r.api_style = 1; r.body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi \\ud83d\\ude00\"}]}]}"; r.body_length = strlen(r.body);
    nya_builder_init(&b,prompt,sizeof(prompt));
    CHECK(nya_compat_messages(&r,1,&b,scratch,sizeof(scratch)) == 0 && strcmp(prompt,"<|turn>user\nhi \xf0\x9f\x98\x80<turn|>\n<|turn>model\n<|channel>thought\n<channel|>") == 0);
    nya_model_registry_shutdown(&models);
    nya_mutex_destroy(&internal.model_mutex); nya_mutex_destroy(&internal.execution_mutex);
#ifdef _WIN32
    WSACleanup();
#endif
    puts("Compatible and native API checks passed"); return 0;
}
