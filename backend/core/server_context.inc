/* Local authenticated administrator chooses a principal to evaluate explicit
 * context grants. This is not a remote principal authentication mechanism. */
static int nya_context_refs(const nya_http_request *request,fyodor_resource_ref refs[64],size_t *count)
{ return nya_resource_refs(request,"resources",refs,64,count); }
static int nya_context_layers(const nya_http_request *request,const fyodor_resource_ref refs[64],
    size_t count,fyodor_context_entry entries[64])
{
    for(size_t i=0;i<count;++i) entries[i]=(fyodor_context_entry){refs[i],FYODOR_CONTEXT_EXPLICIT};
    const char *p=nya_json_value(request->body,request->body_length,"layers"),*end=request->body+request->body_length;
    if(p==NULL) return 0;
    if(p==end||*p++!='[') return -1;
    nya_json_space(&p,end);
    for(size_t i=0;i<count;++i) {
        char name[32]; int found=0;
        if(nya_json_decode_string(&p,end,name,sizeof(name))!=0) return -1;
        for(int layer=0;layer<FYODOR_CONTEXT_LAYER_COUNT;++layer)
            if(strcmp(name,fyodor_context_layer_name((fyodor_context_layer)layer))==0) {
                entries[i].layer=(fyodor_context_layer)layer; found=1; break;
            }
        if(!found) return -1;
        nya_json_space(&p,end);
        if(i+1<count) { if(p==end||*p++!=',') return -1; nya_json_space(&p,end); }
    }
    return p<end&&*p==']'?0:-1;
}
static int nya_handle_context(nya_socket client,nya_server *server,const nya_http_request *request,const char *origin)
{
    if(!server->config.resource_store_path||server->config.settings->allow_remote||strcmp(server->config.settings->bind_address,"127.0.0.1")!=0)
        return nya_api_error(server,client,0,403,"Forbidden","workspace_disabled","Local workspace access is not enabled.",origin);
    if(strcmp(request->method,"POST")!=0)
        return nya_api_error(server,client,0,405,"Method Not Allowed","method_not_allowed","Context operations require POST.",origin);
    if(!nya_content_type_is_json(request->content_type))
        return nya_api_error(server,client,0,415,"Unsupported Media Type","content_type","Use application/json.",origin);
    int writing=strcmp(request->path,"/context/writing")==0;
    int permissions=strcmp(request->path,"/context/permissions")==0,
        generate=writing||strcmp(request->path,"/context/generate")==0,receipt=strcmp(request->path,"/context/receipt")==0;
    int receipts=strcmp(request->path,"/context/receipts")==0;
    int search=strcmp(request->path,"/context/search")==0;
    if(!permissions&&!generate&&!receipt&&!receipts&&!search)
        return nya_api_error(server,client,0,404,"Not Found","not_found","Unknown context operation.",origin);
    char name_space[65],principal_text[FYODOR_UUID_TEXT_CAPACITY],identifier[FYODOR_RESOURCE_URI_CAPACITY];
    fyodor_uuid principal,receipt_id;fyodor_resource_ref refs[64];size_t count=0;
    uint64_t mask=0,budget=0,max_tokens=0,model_id=0;
    fyodor_context_entry entries[64];
    fyodor_resource_ref target={0};uint64_t expected_revision=0;fyodor_writing_mode mode=FYODOR_WRITING_GENERATE;
    char query[129];
    if(search&&nya_json_string(request->body,request->body_length,"query",query,sizeof(query))!=0) goto invalid;
    if(nya_json_string(request->body,request->body_length,"namespace",name_space,sizeof(name_space))!=0||
       nya_json_string(request->body,request->body_length,"principal",principal_text,sizeof(principal_text))!=0||
       fyodor_uuid_parse(principal_text,strlen(principal_text),&principal)!=FYODOR_RESOURCE_OK) goto invalid;
    int set=permissions && nya_json_value(request->body,request->body_length,"permissions")!=NULL;
    if(permissions && (nya_json_string(request->body,request->body_length,"uri",identifier,sizeof(identifier))!=0||
        fyodor_resource_parse(identifier,strlen(identifier),refs)!=FYODOR_RESOURCE_OK||
        (set&&(nya_json_u64(request->body,request->body_length,"permissions",&mask)!=0||mask>FYODOR_PERMISSION_ALL)))) goto invalid;
    if(receipt && (nya_json_string(request->body,request->body_length,"receipt_id",identifier,sizeof(identifier))!=0||
        fyodor_uuid_parse(identifier,strlen(identifier),&receipt_id)!=FYODOR_RESOURCE_OK)) goto invalid;
    fyodor_uuid after;int has_after=0;
    if(receipts && nya_json_value(request->body,request->body_length,"after")) {
        if(nya_json_string(request->body,request->body_length,"after",identifier,sizeof(identifier))!=0) goto invalid;
        if(identifier[0]) {if(fyodor_uuid_parse(identifier,strlen(identifier),&after)!=FYODOR_RESOURCE_OK) goto invalid;has_after=1;}
    }
    if(generate && (nya_context_refs(request,refs,&count)!=0||nya_context_layers(request,refs,count,entries)!=0||
        nya_json_u64(request->body,request->body_length,"byte_budget",&budget)!=0||budget>FYODOR_STORE_CONTENT_LIMIT||
        nya_json_u64(request->body,request->body_length,"max_tokens",&max_tokens)!=0||max_tokens==0||max_tokens>server->config.settings->max_generation_tokens||
        nya_json_u64(request->body,request->body_length,"model_id",&model_id)!=0||model_id==0)) goto invalid;
    if(writing) {
        char target_uri[FYODOR_RESOURCE_URI_CAPACITY],mode_text[16];
        if(nya_json_string(request->body,request->body_length,"target",target_uri,sizeof(target_uri))!=0||
            fyodor_resource_parse(target_uri,strlen(target_uri),&target)!=FYODOR_RESOURCE_OK||
            nya_resource_number(request,"expected_revision",INT64_MAX,&expected_revision)!=0||expected_revision==0||
            nya_json_string(request->body,request->body_length,"mode",mode_text,sizeof(mode_text))!=0||count>63)goto invalid;
        if(strcmp(mode_text,"generate")==0)mode=FYODOR_WRITING_GENERATE;
        else if(strcmp(mode_text,"rewrite")==0)mode=FYODOR_WRITING_REWRITE;
        else if(strcmp(mode_text,"continue")==0)mode=FYODOR_WRITING_CONTINUE;
        else goto invalid;
    }
    fyodor_store *store=NULL;fyodor_store_result result=fyodor_store_open(server->config.resource_store_path,&store);
    if(result!=FYODOR_STORE_OK) return nya_resource_error(client,server,origin,result);
    size_t capacity=generate?8u*1024u*1024u:search?262144:receipts?8192:256;
    if(capacity>server->config.settings->max_output_bytes) capacity=server->config.settings->max_output_bytes;
    char *body=malloc(capacity);nya_text_builder builder;
    if(!body){fyodor_store_close(store);return nya_resource_error(client,server,origin,FYODOR_STORE_NOMEM);}
    nya_builder_init(&builder,body,capacity);
    if(search) {
        fyodor_resource_summary *items=NULL;size_t total=0;
        result=fyodor_context_search(store,&principal,name_space,query,20,&items,&total);
        if(result==FYODOR_STORE_OK) {
            nya_builder_append(&builder,"{\"ok\":true,\"resources\":[");
            for(size_t i=0;i<total;++i) {
                (void)fyodor_resource_format(&items[i].ref,identifier,sizeof(identifier));
                nya_builder_append(&builder,"%s{\"uri\":",i?",":"");nya_builder_append_json_string(&builder,identifier);
                nya_builder_append(&builder,",\"title\":");nya_builder_append_json_string(&builder,items[i].title);
                nya_builder_append(&builder,",\"revision\":\"%" PRIu64 "\"}",items[i].revision);
            }
            nya_builder_append(&builder,"],\"next\":\"\"}");
        }
        fyodor_resource_summaries_free(items,total);
    } else if(receipts) {
        fyodor_receipt_summary *items=NULL;size_t total=0;
        result=fyodor_receipt_list(store,&principal,name_space,has_after?&after:NULL,20,&items,&total);
        if(result==FYODOR_STORE_OK) {
            nya_builder_append(&builder,"{\"ok\":true,\"receipts\":[");
            for(size_t i=0;i<total;++i) {
                (void)fyodor_uuid_format(&items[i].id,identifier,sizeof(identifier));
                nya_builder_append(&builder,"%s{\"id\":\"%s\",\"created_ms\":\"%" PRId64 "\"}",i?",":"",identifier,items[i].created_ms);
            }
            nya_builder_append(&builder,"],\"next\":\"%s\"}",total==20?identifier:"");
        }
        fyodor_receipt_summaries_free(items);
    } else if(permissions) {
        unsigned value=0;
        if(set) result=fyodor_context_permissions_set(store,&principal,name_space,refs,(unsigned)mask);
        if(result==FYODOR_STORE_OK) result=fyodor_context_permissions_get(store,&principal,name_space,refs,&value);
        if(result==FYODOR_STORE_OK) nya_builder_append(&builder,"{\"ok\":true,\"permissions\":%u}",value);
    } else if(receipt) {
        fyodor_receipt record={0};result=fyodor_receipt_read(store,&principal,name_space,&receipt_id,&record);
        if(result==FYODOR_STORE_OK) {
            size_t length=strlen(record.prompt)+strlen(record.output)+strlen(record.metadata)+strlen(record.sources);
            if(length>(SIZE_MAX-256)/6 || length*6+256>server->config.settings->max_output_bytes) result=FYODOR_STORE_INVALID;
            else {
                capacity=length*6+256;char *grown=realloc(body,capacity);
                if(!grown) result=FYODOR_STORE_NOMEM;
                else {body=grown;nya_builder_init(&builder,body,capacity);}
            }
        }
        if(result==FYODOR_STORE_OK) {
            nya_builder_append(&builder,"{\"ok\":true,\"prompt\":");nya_builder_append_json_string(&builder,record.prompt);
            nya_builder_append(&builder,",\"output\":");nya_builder_append_json_string(&builder,record.output);
            nya_builder_append(&builder,",\"metadata_json\":");nya_builder_append_json_string(&builder,record.metadata);
            nya_builder_append(&builder,",\"sources_json\":");nya_builder_append_json_string(&builder,record.sources);nya_builder_append(&builder,"}");
        }
        fyodor_receipt_free(&record);
    } else {
        char *prompt=malloc(FYODOR_STORE_CONTENT_LIMIT+1);
        if(!prompt) result=FYODOR_STORE_NOMEM;
        else if(capacity<=4096||nya_json_string(request->body,request->body_length,"prompt",prompt,FYODOR_STORE_CONTENT_LIMIT+1)!=0) result=FYODOR_STORE_INVALID;
        else {
            nya_generation_request sampling={0};sampling.prompt=prompt;sampling.max_tokens=(size_t)max_tokens;
            sampling.temperature=0;sampling.top_p=1;sampling.seed=42;sampling.max_output_bytes=(capacity-4096)/6;
            if(sampling.max_output_bytes>FYODOR_STORE_CONTENT_LIMIT) sampling.max_output_bytes=FYODOR_STORE_CONTENT_LIMIT;
            fyodor_context_generation generated={0};char error[512];
            if(nya_server_execution_lock(server)!=0) result=FYODOR_STORE_BUSY;
            else {
                nya_server_models_lock(server);
                const nya_model *model=nya_model_find(server->config.models,model_id);
                nya_model snapshot;
                if(model) snapshot=*model;
                nya_server_models_unlock(server);
                if(!model) result=FYODOR_STORE_NOT_FOUND;
                else if(writing) result=fyodor_writing_generate(store,&snapshot,&principal,name_space,&target,expected_revision,mode,
                    entries,count,(size_t)budget,&sampling,&generated,error,sizeof(error));
                else result=fyodor_generate_with_layers(store,&snapshot,&principal,name_space,entries,count,(size_t)budget,&sampling,&generated,error,sizeof(error));
                nya_server_execution_unlock(server);
            }
            if(result==FYODOR_STORE_OK) {
                (void)fyodor_uuid_format(&generated.receipt_id,identifier,sizeof(identifier));
                nya_builder_append(&builder,"{\"ok\":true,\"receipt_id\":");nya_builder_append_json_string(&builder,identifier);
                nya_builder_append(&builder,",\"text\":");nya_builder_append_json_string(&builder,generated.generation.text);
                nya_builder_append(&builder,",\"prompt_tokens\":%zu,\"generated_tokens\":%zu,\"context_bytes\":%zu}",generated.generation.prompt_tokens,generated.generation.generated_tokens,generated.context.length);
            }
            fyodor_context_generation_free(&generated);
        }
        free(prompt);
    }
    fyodor_store_close(store);
    int sent=result!=FYODOR_STORE_OK?nya_resource_error(client,server,origin,result):builder.failed?
        nya_resource_error(client,server,origin,FYODOR_STORE_NOMEM):nya_http_send_response(server,client,200,"OK",body,origin);
    free(body);return sent;
invalid:
    return nya_resource_error(client,server,origin,FYODOR_STORE_INVALID);
}
