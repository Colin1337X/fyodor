/* Desktop runtime controls share the execution lock with inference and unload.
   A provider swap is transactional: initialization must succeed before the old
   device context is released. No process-wide environment mutation is needed. */
static nya_llm_context *nya_runtime_model_context(const nya_model *model)
{
    if (model == NULL) return NULL;
    if (model->generation_context != NULL) return model->generation_context;
    if (strcmp(model->architecture,"gemma4uv") == 0 || strcmp(model->architecture,"gemma4ua") == 0)
        return model->execution_context;
    return NULL;
}

static int nya_handle_runtime(nya_socket client, nya_server *server, const char *origin)
{
    char body[8192]; nya_text_builder b; nya_builder_init(&b,body,sizeof(body));
    if (nya_server_execution_lock(server) != 0) return -1;
    nya_server_models_lock(server);
    nya_builder_append(&b,"{\"version\":1,\"ok\":true,\"compiled\":{\"cpu\":true,\"cuda\":%s,\"vulkan\":%s},\"training_compute\":\"cpu\",\"verbose\":%s,\"limits\":{\"request_bytes\":%zu,\"output_bytes\":%zu,\"generation_tokens\":%u},\"models\":[",
        nya_compute_cuda_compiled() ? "true" : "false", nya_compute_vulkan_compiled() ? "true" : "false",
        server->config.verbose ? "true" : "false",server->config.settings->max_request_bytes,server->config.settings->max_output_bytes,server->config.settings->max_generation_tokens);
    for (size_t i = 0; i < server->config.models->count; ++i) {
        const nya_model *model = &server->config.models->models[i];
        const nya_llm_context *context = nya_runtime_model_context(model);
        nya_builder_append(&b,"%s{\"model_id\":%" PRIu64 ",\"selectable\":%s,\"compute\":",
            i == 0 ? "" : ",",model->id,context != NULL ? "true" : "false");
        nya_builder_append_json_string(&b,context != NULL ? nya_compute_name(context->compute) : "external");
        nya_builder_append(&b,"}");
    }
    nya_builder_append(&b,"]}");
    nya_server_models_unlock(server);
    nya_server_execution_unlock(server);
    return b.failed ? -1 : nya_http_send_response(server,client,200,"OK",body,origin);
}

static int nya_handle_model_compute(nya_socket client, nya_server *server, const nya_http_request *r, const char *origin)
{
    uint64_t id; char provider[16]; const char *keys[] = {"model_id","compute"};
    if (!nya_content_type_is_json(r->content_type))
        return nya_http_send_error(server,client,415,"Unsupported Media Type","json_required","Use Content-Type: application/json.",origin);
    if (nya_compat_keys(r->body,r->body_length,keys,2) != 0 ||
        nya_json_u64(r->body,r->body_length,"model_id",&id) != 0 || id == 0 ||
        nya_json_string(r->body,r->body_length,"compute",provider,sizeof(provider)) != 0 ||
        (strcmp(provider,"cpu") != 0 && strcmp(provider,"cuda") != 0 && strcmp(provider,"vulkan") != 0))
        return nya_http_send_error(server,client,400,"Bad Request","invalid_compute","Supply model_id and compute: cpu, cuda, or vulkan.",origin);
    if (nya_server_execution_lock(server) != 0) return -1;
    nya_server_models_lock(server);
    const nya_model *model = nya_model_find(server->config.models,id);
    int found = model != NULL;
    nya_llm_context *context = nya_runtime_model_context(model);
    nya_server_models_unlock(server);
    if (context == NULL) {
        nya_server_execution_unlock(server);
        return nya_http_send_error(server,client,found ? 409 : 404,found ? "Conflict" : "Not Found",
            "compute_unavailable","Select a loaded native text model, MTP assistant, or Unified projector.",origin);
    }
    if (strcmp(nya_compute_name(context->compute),provider) != 0) {
        nya_compute_context *next = nya_compute_create_for(provider);
        if (strcmp(nya_compute_name(next),provider) != 0) {
            nya_compute_free(next);
            nya_server_execution_unlock(server);
            return nya_http_send_error(server,client,409,"Conflict","compute_unavailable",
                "Provider initialization failed. Check compiled support, GPU driver and CUDA NVRTC runtime. The current provider was retained.",origin);
        }
        nya_compute_context *previous = context->compute;
        context->compute = next;
        nya_compute_free(previous);
    }
    if (server->config.verbose) fprintf(stderr,"[compute] model=%" PRIu64 " active=%s\n",id,provider);
    nya_server_execution_unlock(server);
    char body[160];
    snprintf(body,sizeof(body),"{\"version\":1,\"ok\":true,\"model_id\":%" PRIu64 ",\"compute\":\"%s\"}",id,provider);
    return nya_http_send_response(server,client,200,"OK",body,origin);
}

/* Shape information lets native clients select context and draft settings from
   validated model data instead of inferring them from filenames. */
static int nya_handle_model_info(nya_socket client, nya_server *server, const nya_http_request *r, const char *origin)
{
    uint64_t id; const char *keys[] = {"model_id"};
    if (!nya_content_type_is_json(r->content_type) || nya_compat_keys(r->body,r->body_length,keys,1) != 0 ||
        nya_json_u64(r->body,r->body_length,"model_id",&id) != 0 || id == 0)
        return nya_http_send_error(server,client,400,"Bad Request","invalid_model","JSON model_id is required.",origin);
    if (nya_server_execution_lock(server) != 0) return -1;
    nya_server_models_lock(server);
    const nya_model *model = nya_model_find(server->config.models,id);
    const nya_llm_context *c = nya_runtime_model_context(model);
    if (c == NULL) {
        nya_server_models_unlock(server); nya_server_execution_unlock(server);
        return nya_http_send_error(server,client,404,"Not Found","native_model_not_found","No native execution context exists for that model.",origin);
    }
    char body[2048]; nya_text_builder b; nya_builder_init(&b,body,sizeof(body));
    nya_builder_append(&b,"{\"version\":1,\"ok\":true,\"model_id\":%" PRIu64 ",\"architecture\":",id);
    nya_builder_append_json_string(&b,model->architecture);
    nya_builder_append(&b,",\"compute\":"); nya_builder_append_json_string(&b,nya_compute_name(c->compute));
    nya_builder_append(&b,",\"context_length\":%" PRIu32 ",\"embedding_length\":%" PRIu32 ",\"layers\":%" PRIu32 ",\"attention_heads\":%" PRIu32 ",\"kv_heads\":%" PRIu32 ",\"vocabulary_size\":%zu,\"experts\":%" PRIu32 ",\"sliding_window\":%" PRIu32 ",\"mtp_assistant\":%s}",
        c->context_length,c->embedding_length,c->block_count,c->head_count,c->key_value_head_count,
        (size_t)c->tokenizer.vocabulary_size,c->expert_count,c->sliding_window,c->is_assistant ? "true" : "false");
    nya_server_models_unlock(server); nya_server_execution_unlock(server);
    return b.failed ? -1 : nya_http_send_response(server,client,200,"OK",body,origin);
}
