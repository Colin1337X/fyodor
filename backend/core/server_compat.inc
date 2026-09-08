/* Private REST adapters included by server.c after its bounded JSON and socket
   helpers. They translate into the native generation handler, so sampling,
   model lifetime locks, limits and execution stay in one implementation. */
static int nya_handle_generate(nya_socket, nya_server *, const nya_http_request *, const char *);

static int nya_api_error(nya_server *server, nya_socket client, int style, int status,
    const char *reason, const char *code, const char *message, const char *origin)
{
    if (style == 0) return nya_http_send_error(server,client,status,reason,code,message,origin);
    char body[8192]; nya_text_builder b; nya_builder_init(&b,body,sizeof(body));
    const char *type = status == 401 ? "authentication_error" : status == 404 ? "not_found_error" :
        status == 403 ? "permission_error" : status >= 500 ? "api_error" : "invalid_request_error";
    nya_builder_append(&b,style == 2 ? "{\"type\":\"error\",\"error\":{\"type\":" : "{\"error\":{\"type\":");
    nya_builder_append_json_string(&b,type);
    nya_builder_append(&b,",\"message\":"); nya_builder_append_json_string(&b,message);
    if (style != 2) { nya_builder_append(&b,",\"param\":null,\"code\":"); nya_builder_append_json_string(&b,code); }
    nya_builder_append(&b,"}}");
    return b.failed ? -1 : nya_http_send_response(server,client,status,reason,body,origin);
}

static void nya_compat_response(nya_text_builder *b, int style, uint64_t id, const nya_generation_response *r)
{
    unsigned char random[16]; char request_id[33];
    if (nya_runtime_random_bytes(random,sizeof(random)) != 0) { b->failed = 1; return; }
    for (size_t i = 0; i < sizeof(random); ++i) snprintf(request_id+i*2,3,"%02x",(unsigned int)random[i]);
    if (style == 2) {
        nya_builder_append(b,"{\"id\":\"msg_%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"%" PRIu64 "\",\"content\":[{\"type\":\"text\",\"text\":",request_id,id);
        nya_builder_append_json_string(b,r->text);
        nya_builder_append(b,"}],\"stop_reason\":\"%s\",\"stop_sequence\":null,\"usage\":{\"input_tokens\":%zu,\"output_tokens\":%zu}}",
            r->stop_reason == NYA_GENERATION_STOP_EOS ? "end_turn" : "max_tokens",r->prompt_tokens,r->generated_tokens);
    } else {
        nya_builder_append(b,"{\"id\":\"%s-%s\",\"object\":\"%s\",\"created\":%" PRIu64 ",\"model\":\"%" PRIu64 "\",\"choices\":[{\"index\":0,",
            style == 3 ? "cmpl" : "chatcmpl",request_id,style == 3 ? "text_completion" : "chat.completion",(uint64_t)time(NULL),id);
        nya_builder_append(b,style == 3 ? "\"text\":" : "\"message\":{\"role\":\"assistant\",\"content\":");
        nya_builder_append_json_string(b,r->text);
        if (style != 3) nya_builder_append(b,"}");
        nya_builder_append(b,",\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu}}",
            r->stop_reason == NYA_GENERATION_STOP_EOS ? "stop" : "length",r->prompt_tokens,r->generated_tokens,r->prompt_tokens+r->generated_tokens);
    }
}

/* Validate each interpreted nested object too: the base parser validates JSON
   syntax recursively, while duplicate-field detection is object-local. */
static int nya_compat_keys(const char *json, size_t length, const char *const *allowed, size_t count)
{
    if (nya_json_validate(json,length) != 0) return -1;
    const char *p = json, *end = json+length; nya_json_space(&p,end); ++p; nya_json_space(&p,end);
    while (p < end && *p != '}') {
        char key[128]; if (nya_json_decode_string(&p,end,key,sizeof(key)) != 0) return -1;
        size_t i; for (i = 0; i < count; ++i) if (strcmp(key,allowed[i]) == 0) break;
        if (i == count) return -1;
        nya_json_space(&p,end); ++p;
        if (nya_json_skip_value(&p,end,0) != 0) return -1;
        nya_json_space(&p,end); if (p < end && *p == ',') { ++p; nya_json_space(&p,end); }
    }
    return 0;
}

static int nya_compat_model_id(const char *json, size_t length, uint64_t *id)
{
    char text[32]; if (nya_json_string(json,length,"model",text,sizeof(text)) != 0 || *text == '\0') return -1;
    uint64_t value = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p < '0' || *p > '9' || value > (UINT64_MAX-(uint64_t)(*p-'0'))/10) return -1;
        value = value*10+(uint64_t)(*p-'0');
    }
    if (value == 0 || (text[0] == '0' && text[1] != '\0')) return -1;
    *id = value; return 0;
}

/* Text strings and arrays of {type:"text",text:"..."} are supported by both
   protocols. Binary media/tool/thinking blocks need different execution paths
   and are rejected explicitly rather than disappearing from the prompt. */
static int nya_compat_content(const char *value, const char *end, nya_text_builder *b, char *scratch, size_t capacity)
{
    if (value == NULL || value >= end) return -1;
    if (*value == '"') {
        if (nya_json_decode_string(&value,end,scratch,capacity) != 0) return -1;
        nya_builder_append(b,"%s",scratch); return b->failed ? -1 : 0;
    }
    if (*value++ != '[') return -1;
    nya_json_space(&value,end); size_t blocks = 0;
    while (value < end && *value != ']') {
        const char *start = value; if (nya_json_skip_value(&value,end,0) != 0) return -1;
        size_t length = (size_t)(value-start); char type[32]; const char *keys[] = {"type","text"};
        if (++blocks > 256 || nya_compat_keys(start,length,keys,2) != 0 ||
            nya_json_string(start,length,"type",type,sizeof(type)) != 0 || strcmp(type,"text") != 0 ||
            nya_json_string(start,length,"text",scratch,capacity) != 0) return -1;
        nya_builder_append(b,"%s",scratch);
        nya_json_space(&value,end); if (value < end && *value == ',') { ++value; nya_json_space(&value,end); }
    }
    return value < end && *value == ']' && blocks != 0 && !b->failed ? 0 : -1;
}

static int nya_compat_messages(const nya_http_request *r, int gemma, nya_text_builder *b, char *scratch, size_t capacity)
{
    const char *end = r->body+r->body_length;
    const char *system = r->api_style == 2 ? nya_json_value(r->body,r->body_length,"system") : NULL;
    int saw_system = system != NULL;
    if (system != NULL) {
        nya_builder_append(b,gemma ? "<|turn>system\n" : "system: ");
        if (nya_compat_content(system,end,b,scratch,capacity) != 0) return -1;
        nya_builder_append(b,gemma ? "<turn|>\n" : "\n");
    }
    const char *p = nya_json_value(r->body,r->body_length,"messages");
    if (p == NULL || p >= end || *p++ != '[') return -1;
    nya_json_space(&p,end); size_t count = 0; char last_role[24] = {0};
    while (p < end && *p != ']') {
        const char *start = p; if (nya_json_skip_value(&p,end,0) != 0) return -1;
        size_t length = (size_t)(p-start); char role[24]; const char *keys[] = {"role","content"};
        if (++count > 256 || nya_compat_keys(start,length,keys,2) != 0 ||
            nya_json_string(start,length,"role",role,sizeof(role)) != 0) return -1;
        int is_system = strcmp(role,"system") == 0 || strcmp(role,"developer") == 0;
        if (is_system) {
            if (r->api_style == 2 || count != 1 || saw_system) return -1;
            saw_system = 1;
        } else if (strcmp(role,"user") != 0 && strcmp(role,"assistant") != 0) return -1;
        const char *render_role = is_system ? "system" : gemma && strcmp(role,"assistant") == 0 ? "model" : role;
        nya_builder_append(b,gemma ? "<|turn>%s\n" : "%s: ",render_role);
        if (nya_compat_content(nya_json_value(start,length,"content"),p,b,scratch,capacity) != 0) return -1;
        nya_builder_append(b,gemma ? "<turn|>\n" : "\n");
        snprintf(last_role,sizeof(last_role),"%s",role);
        nya_json_space(&p,end); if (p < end && *p == ',') { ++p; nya_json_space(&p,end); }
    }
    /* Assistant prefills have protocol/model-specific semantics; require a
       final user message until those semantics are implemented and tested. */
    if (count == 0 || strcmp(last_role,"user") != 0) return -1;
    /* The text-only Gemma template closes an empty thought channel. This is
       the checkpoint template's non-thinking generation prefix, not a request
       for a reasoning trace in the compatibility response. */
    nya_builder_append(b,gemma ? "<|turn>model\n<|channel>thought\n<channel|>" : "assistant: ");
    return b->failed ? -1 : 0;
}

static int nya_handle_tokenize(nya_socket client, nya_server *server, const nya_http_request *r,
    const char *origin, uint64_t id, const char *text, int count_only)
{
    uint32_t *ids = NULL; size_t count = 0; char error[256];
    if (nya_server_execution_lock(server) != 0) return -1;
    nya_server_models_lock(server);
    const nya_model *model = nya_model_find(server->config.models,id);
    const nya_llm_context *context = model == NULL ? NULL : (const nya_llm_context *)model->generation_context;
    nya_server_models_unlock(server);
    /* Capture availability while execution still pins the model. Once that
       lock is released, another worker may unload it; even testing the old
       pointer value afterward would depend on an ended object lifetime. */
    int has_context = context != NULL;
    int result = !has_context ? -1 : nya_llm_tokenize(context,text,&ids,&count,error,sizeof(error));
    nya_server_execution_unlock(server);
    if (result != 0) return nya_api_error(server,client,r->api_style,422,"Unprocessable Content","tokenization_failed",!has_context ? "A loaded native text model is required." : error,origin);
    size_t cap = count_only ? 128 : count <= (SIZE_MAX-128)/12 ? count*12+128 : SIZE_MAX;
    if (cap > server->config.settings->max_output_bytes) { free(ids); return nya_api_error(server,client,r->api_style,413,"Content Too Large","response_limit","Token list exceeds max_output_bytes.",origin); }
    char *body = (char *)malloc(cap); if (body == NULL) { free(ids); return -1; }
    nya_text_builder b; nya_builder_init(&b,body,cap);
    if (count_only) nya_builder_append(&b,"{\"input_tokens\":%zu}",count);
    else {
        nya_builder_append(&b,"{\"version\":1,\"ok\":true,\"model_id\":%" PRIu64 ",\"count\":%zu,\"tokens\":[",id,count);
        for (size_t i = 0; i < count; ++i) nya_builder_append(&b,"%s%" PRIu32,i == 0 ? "" : ",",ids[i]);
        nya_builder_append(&b,"]}");
    }
    free(ids); result = b.failed ? -1 : nya_http_send_response(server,client,200,"OK",body,origin); free(body); return result;
}

static int nya_handle_compat_generate(nya_socket client, nya_server *server, const nya_http_request *r, const char *origin)
{
    int count_only = strcmp(r->path,"/v1/messages/count_tokens") == 0;
    const char *common[] = {"model","messages","max_tokens","max_completion_tokens","temperature","top_p","seed","stream","n"};
    const char *anthropic[] = {"model","messages","max_tokens","temperature","top_p","top_k","stream","system"};
    const char *completion[] = {"model","prompt","max_tokens","temperature","top_p","seed","stream","n"};
    const char *const *allowed = r->api_style == 2 ? anthropic : r->api_style == 3 ? completion : common;
    size_t allowed_count = r->api_style == 2 ? 8 : r->api_style == 3 ? 8 : 9;
    if (!nya_content_type_is_json(r->content_type)) return nya_api_error(server,client,r->api_style,415,"Unsupported Media Type","json_required","Use Content-Type: application/json.",origin);
    if (nya_compat_keys(r->body,r->body_length,allowed,allowed_count) != 0)
        return nya_api_error(server,client,r->api_style,400,"Bad Request","unsupported_field","Unknown, duplicate or unsupported field. Tools, media, stops and streaming are not supported by this text adapter.",origin);
    if (r->api_style == 2 && strcmp(r->anthropic_version,"2023-06-01") != 0)
        return nya_api_error(server,client,2,400,"Bad Request","invalid_version","Use anthropic-version: 2023-06-01.",origin);
    const char *stream = nya_json_value(r->body,r->body_length,"stream"); uint64_t n;
    if ((stream != NULL && strncmp(stream,"false",5) != 0) ||
        (nya_json_value(r->body,r->body_length,"n") != NULL && (nya_json_u64(r->body,r->body_length,"n",&n) != 0 || n != 1)))
        return nya_api_error(server,client,r->api_style,400,"Bad Request","unsupported_option","This adapter requires stream=false and n=1.",origin);
    uint64_t id;
    if (nya_compat_model_id(r->body,r->body_length,&id) != 0)
        return nya_api_error(server,client,r->api_style,400,"Bad Request","invalid_model","model must be a loaded model's decimal ID string, as returned by /v1/models.",origin);
    nya_server_models_lock(server); const nya_model *model = nya_model_find(server->config.models,id);
    int found = model != NULL, gemma = model != NULL && strcmp(model->architecture,"gemma4") == 0;
    nya_server_models_unlock(server);
    if (!found) return nya_api_error(server,client,r->api_style,404,"Not Found","model_not_loaded","No loaded model has that ID.",origin);
    if (r->body_length > (SIZE_MAX-197632)/12) return -1;
    size_t capacity = r->body_length*2+32768, json_capacity = capacity*6+1024;
    char *prompt = (char *)malloc(capacity), *scratch = (char *)malloc(r->body_length+1), *json = (char *)malloc(json_capacity);
    if (prompt == NULL || scratch == NULL || json == NULL) { free(prompt); free(scratch); free(json); return -1; }
    nya_text_builder b; nya_builder_init(&b,prompt,capacity); int result;
    if (r->api_style == 3) {
        result = nya_json_string(r->body,r->body_length,"prompt",prompt,capacity);
    } else result = nya_compat_messages(r,gemma,&b,scratch,r->body_length+1);
    if (result != 0) {
        result = nya_api_error(server,client,r->api_style,400,"Bad Request","invalid_messages","Use text-only messages with role/content, optional leading system, and a final user message (at most 256 messages).",origin); goto done;
    }
    if (count_only) { result = nya_handle_tokenize(client,server,r,origin,id,prompt,1); goto done; }
    nya_builder_init(&b,json,json_capacity);
    nya_builder_append(&b,"{\"model_id\":%" PRIu64 ",\"prompt\":",id); nya_builder_append_json_string(&b,prompt);
    /* Chat compatibility defaults avoid native top-k filtering unless a caller
       using the Anthropic adapter explicitly supplies top_k. */
    nya_builder_append(&b,",\"top_k\":");
    const char *top_k = nya_json_value(r->body,r->body_length,"top_k");
    if (top_k == NULL) nya_builder_append(&b,"0");
    else { const char *end = top_k; if (nya_json_skip_value(&end,r->body+r->body_length,0) != 0 || end-top_k > INT_MAX) goto invalid; nya_builder_append(&b,"%.*s",(int)(end-top_k),top_k); }
    const char *fields[] = {"temperature","top_p","seed","max_tokens","max_completion_tokens"};
    int has_max = 0;
    for (size_t i = 0; i < 5; ++i) {
        const char *value = nya_json_value(r->body,r->body_length,fields[i]); if (value == NULL) continue;
        if (i >= 3 && has_max++) goto invalid;
        const char *end = value; if (nya_json_skip_value(&end,r->body+r->body_length,0) != 0 || end-value > INT_MAX) goto invalid;
        nya_builder_append(&b,",\"%s\":%.*s",i == 4 ? "max_tokens" : fields[i],(int)(end-value),value);
    }
    if (r->api_style == 2 && !has_max) goto invalid;
    nya_builder_append(&b,"}"); if (b.failed) goto invalid;
    nya_http_request translated = *r; translated.body = json; translated.body_length = b.length;
    result = nya_handle_generate(client,server,&translated,origin); goto done;
invalid:
    result = nya_api_error(server,client,r->api_style,400,"Bad Request","invalid_options","Supply valid sampling options and only one max-token field; Anthropic messages require max_tokens.",origin);
done:
    free(prompt); free(scratch); free(json); return result;
}

static int nya_handle_compat_models(nya_socket client, nya_server *server, const nya_http_request *r, const char *origin)
{
    char body[NYA_HTTP_RESPONSE_LIMIT]; nya_text_builder b; nya_builder_init(&b,body,sizeof(body));
    nya_builder_append(&b,r->api_style == 2 ? "{\"data\":[" : "{\"object\":\"list\",\"data\":[");
    nya_server_models_lock(server); size_t emitted = 0; uint64_t first = 0, last = 0;
    for (size_t i = 0; i < server->config.models->count; ++i) {
        const nya_model *m = &server->config.models->models[i]; if (!m->generation_supported) continue;
        if (emitted++ != 0) nya_builder_append(&b,","); else first = m->id;
        last = m->id;
        if (r->api_style == 2) nya_builder_append(&b,"{\"id\":\"%" PRIu64 "\",\"type\":\"model\",\"display_name\":\"Fyodor %" PRIu64 "\",\"created_at\":\"1970-01-01T00:00:00Z\"}",m->id,m->id);
        else nya_builder_append(&b,"{\"id\":\"%" PRIu64 "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"local\"}",m->id);
    }
    nya_server_models_unlock(server);
    if (r->api_style == 2) {
        nya_builder_append(&b,"],\"has_more\":false,\"first_id\":");
        if (emitted != 0) nya_builder_append(&b,"\"%" PRIu64 "\",\"last_id\":\"%" PRIu64 "\"}",first,last);
        else nya_builder_append(&b,"null,\"last_id\":null}");
    } else nya_builder_append(&b,"]}");
    return b.failed ? -1 : nya_http_send_response(server,client,200,"OK",body,origin);
}
