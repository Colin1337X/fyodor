/* Included after the HTTP/JSON helpers. All operations reuse fyodor-app.
 * Every request owns its SQLite handle: no transaction crosses worker threads. */
static int nya_resource_error(nya_socket client,nya_server *server,const char *origin,fyodor_store_result result)
{
    int status=500; const char *reason="Internal Server Error",*code="store_failed",*message="Workspace operation failed.";
    if(result==FYODOR_STORE_INVALID) { status=400;reason="Bad Request";code="invalid_resource";message="Invalid resource request."; }
    else if(result==FYODOR_STORE_NOT_FOUND) { status=404;reason="Not Found";code="resource_missing";message="Resource not found."; }
    else if(result==FYODOR_STORE_CONFLICT) { status=409;reason="Conflict";code="revision_conflict";message="Resource revision or parent conflict. Reload before editing."; }
    else if(result==FYODOR_STORE_BUSY) { status=503;reason="Service Unavailable";code="workspace_busy";message="Workspace is busy; retry later."; }
    else if(result==FYODOR_STORE_DENIED) { status=403;reason="Forbidden";code="context_denied";message="Context access denied."; }
    return nya_api_error(server,client,0,status,reason,code,message,origin);
}
static int nya_resource_number(const nya_http_request *request,const char *key,uint64_t maximum,uint64_t *revision)
{
    char text[24];
    if(nya_json_string(request->body,request->body_length,key,text,sizeof(text))!=0 || !text[0]) return -1;
    uint64_t value=0;
    if(text[0]=='0'&&text[1]) return -1;
    for(const char *p=text;*p;++p) {
        if(*p<'0'||*p>'9') return -1;
        unsigned digit=(unsigned)(*p-'0');
        if(value>(maximum-digit)/10) return -1;
        value=value*10+digit;
    }
    *revision=value;return 0;
}
static int nya_resource_refs(const nya_http_request *request,const char *key,fyodor_resource_ref *refs,size_t capacity,size_t *count)
{
    const char *p=nya_json_value(request->body,request->body_length,key),*end=request->body+request->body_length;
    *count=0;
    if(!p||p==end||*p++!='[') return -1;
    nya_json_space(&p,end);
    if(p<end&&*p==']') return 0;
    for(;;) {
        char uri[FYODOR_RESOURCE_URI_CAPACITY];
        if(*count==capacity||nya_json_decode_string(&p,end,uri,sizeof(uri))!=0||fyodor_resource_parse(uri,strlen(uri),&refs[*count])!=FYODOR_RESOURCE_OK) return -1;
        ++*count;nya_json_space(&p,end);
        if(p==end) return -1;
        if(*p++==']') return 0;
        if(p[-1]!=',') return -1;
        nya_json_space(&p,end);
    }
}
static int nya_handle_resources(nya_socket client,nya_server *server,const nya_http_request *request,const char *origin)
{
    if(!server->config.resource_store_path || server->config.settings->allow_remote || strcmp(server->config.settings->bind_address,"127.0.0.1")!=0)
        return nya_api_error(server,client,0,403,"Forbidden","workspace_disabled","Local workspace access is not enabled.",origin);
    if(strcmp(request->method,"POST")!=0)
        return nya_api_error(server,client,0,405,"Method Not Allowed","method_not_allowed","Resource operations require POST.",origin);
    if(!nya_content_type_is_json(request->content_type))
        return nya_api_error(server,client,0,415,"Unsupported Media Type","content_type","Use application/json.",origin);
    int update=strcmp(request->path,"/resources/update")==0,export=strcmp(request->path,"/resources/export")==0;
    int lore=strcmp(request->path,"/resources/lore")==0;
    int history=strcmp(request->path,"/resources/history")==0,move=strcmp(request->path,"/resources/move")==0;
    int list=strcmp(request->path,"/resources/list")==0,get=strcmp(request->path,"/resources/get")==0,
        put=strcmp(request->path,"/resources/put")==0,remove=strcmp(request->path,"/resources/delete")==0;
    if(!list&&!get&&!put&&!remove&&!update&&!export&&!history&&!move&&!lore)
        return nya_api_error(server,client,0,404,"Not Found","not_found","Unknown resource operation.",origin);
    char name_space[65],uri[FYODOR_RESOURCE_URI_CAPACITY],after[FYODOR_RESOURCE_URI_CAPACITY]={0};
    fyodor_resource_ref ref={0},parent={0};char folder[FYODOR_RESOURCE_URI_CAPACITY]={0};
    int filtered=nya_json_value(request->body,request->body_length,"folder")!=NULL;
    uint64_t revision=0,read_revision=0,before=0;
    char scope[16]="all";
    if(nya_json_string(request->body,request->body_length,"namespace",name_space,sizeof(name_space))!=0 ||
       ((get||remove||update||export||history||move||lore) && (nya_json_string(request->body,request->body_length,"uri",uri,sizeof(uri))!=0 || fyodor_resource_parse(uri,strlen(uri),&ref)!=FYODOR_RESOURCE_OK)) ||
       ((put||remove||update||move||lore) && nya_resource_number(request,"expected_revision",INT64_MAX-1,&revision)!=0) || ((remove||update||move||lore)&&revision==0) ||
       (list && nya_json_value(request->body,request->body_length,"after") && nya_json_string(request->body,request->body_length,"after",after,sizeof(after))!=0))
        return nya_resource_error(client,server,origin,FYODOR_STORE_INVALID);
    if(((get||export)&&nya_json_value(request->body,request->body_length,"revision")&&nya_resource_number(request,"revision",INT64_MAX,&read_revision)!=0)||
        (history&&nya_json_value(request->body,request->body_length,"before")&&nya_resource_number(request,"before",INT64_MAX,&before)!=0)||
        (list&&nya_json_value(request->body,request->body_length,"scope")&&(nya_json_string(request->body,request->body_length,"scope",scope,sizeof(scope))!=0||
        (strcmp(scope,"all")!=0&&strcmp(scope,"writing")!=0&&strcmp(scope,"projects")!=0&&strcmp(scope,"lore")!=0)))) return nya_resource_error(client,server,origin,FYODOR_STORE_INVALID);
    if((move&&!filtered)||((move||list)&&filtered&&
        (nya_json_string(request->body,request->body_length,"folder",folder,sizeof(folder))!=0||
        (folder[0]&&(fyodor_resource_parse(folder,strlen(folder),&parent)!=FYODOR_RESOURCE_OK||parent.kind!=FYODOR_RESOURCE_PROJECT)))))
        return nya_resource_error(client,server,origin,FYODOR_STORE_INVALID);
    fyodor_store *store=NULL; fyodor_store_result result=fyodor_store_open(server->config.resource_store_path,&store);
    if(result!=FYODOR_STORE_OK) return nya_resource_error(client,server,origin,result);
    size_t capacity=(get||export)?FYODOR_STORE_PACKAGE_LIMIT+4096:262144;
    char *body=malloc(capacity); nya_text_builder builder;
    if(!body) { fyodor_store_close(store);return nya_resource_error(client,server,origin,FYODOR_STORE_NOMEM); }
    nya_builder_init(&builder,body,capacity);
    if(lore) {
        fyodor_resource_ref refs[32];size_t count=0;
        if(nya_resource_refs(request,"resources",refs,32,&count)!=0)result=FYODOR_STORE_INVALID;
        else result=fyodor_store_set_lore(store,name_space,&ref,revision,refs,count,&revision);
        if(result==FYODOR_STORE_OK)nya_builder_append(&builder,"{\"ok\":true,\"revision\":\"%" PRIu64 "\"}",revision);
    } else if(move) {
        result=fyodor_store_move_writing(store,name_space,&ref,revision,folder[0]?&parent:NULL,&revision);
        if(result==FYODOR_STORE_OK)nya_builder_append(&builder,"{\"ok\":true,\"revision\":\"%" PRIu64 "\"}",revision);
    } else if(export) {
        char *package=NULL;size_t length=0;result=fyodor_store_export_revision(store,name_space,&ref,read_revision,&package,&length);
        if(result==FYODOR_STORE_OK) nya_builder_append(&builder,"%s",package);
        fyodor_resource_package_free(package);
    } else if(history) {
        fyodor_revision_summary *items=NULL;size_t count=0;
        result=fyodor_store_history(store,name_space,&ref,before,20,&items,&count);
        if(result==FYODOR_STORE_OK) {
            nya_builder_append(&builder,"{\"ok\":true,\"revisions\":[");
            for(size_t i=0;i<count;++i)
                nya_builder_append(&builder,"%s{\"revision\":\"%" PRIu64 "\",\"modified_ms\":\"%" PRId64 "\",\"deleted\":%s}",
                    i?",":"",items[i].revision,items[i].modified_ms,items[i].deleted?"true":"false");
            nya_builder_append(&builder,"],\"next\":\"%" PRIu64 "\"}",count==20?items[count-1].revision:0);
        }
        free(items);
    } else if(update) {
        char title[1025];char *content=malloc(FYODOR_STORE_CONTENT_LIMIT+1);
        fyodor_resource_record record={0};
        if(!content) result=FYODOR_STORE_NOMEM;
        else if(nya_json_string(request->body,request->body_length,"title",title,sizeof(title))!=0 ||
                nya_json_string(request->body,request->body_length,"content",content,FYODOR_STORE_CONTENT_LIMIT+1)!=0) result=FYODOR_STORE_INVALID;
        else if(nya_json_value(request->body,request->body_length,"receipt_id")) {
            char receipt_text[FYODOR_UUID_TEXT_CAPACITY],principal_text[FYODOR_UUID_TEXT_CAPACITY];fyodor_uuid receipt_id,principal;
            if(nya_json_string(request->body,request->body_length,"receipt_id",receipt_text,sizeof(receipt_text))!=0||
                nya_json_string(request->body,request->body_length,"principal",principal_text,sizeof(principal_text))!=0||
                fyodor_uuid_parse(receipt_text,strlen(receipt_text),&receipt_id)!=FYODOR_RESOURCE_OK||
                fyodor_uuid_parse(principal_text,strlen(principal_text),&principal)!=FYODOR_RESOURCE_OK)result=FYODOR_STORE_INVALID;
            else result=fyodor_writing_save(store,&principal,name_space,&ref,revision,&receipt_id,title,content,&revision);
            if(result==FYODOR_STORE_OK)nya_builder_append(&builder,"{\"ok\":true,\"revision\":\"%" PRIu64 "\"}",revision);
        }
        else {
            result=fyodor_store_get(store,name_space,&ref,0,&record);
            if(result==FYODOR_STORE_OK && record.revision!=revision) result=FYODOR_STORE_CONFLICT;
            if(result==FYODOR_STORE_OK) {
                fyodor_resource_input input={ref,name_space,title,content,record.metadata,record.provenance};
                result=fyodor_store_put(store,&input,revision,&revision);
                if(result==FYODOR_STORE_OK) nya_builder_append(&builder,"{\"ok\":true,\"revision\":\"%" PRIu64 "\"}",revision);
            }
        }
        free(content);fyodor_resource_record_free(&record);
    } else if(list) {
        fyodor_resource_summary *items=NULL;size_t count=0;
        result=filtered?fyodor_store_list_folder(store,name_space,folder[0]?&parent:NULL,after[0]?after:NULL,20,&items,&count):
            (strcmp(scope,"lore")==0?fyodor_store_list_lore:strcmp(scope,"projects")==0?fyodor_store_list_projects:strcmp(scope,"writing")==0?fyodor_store_list_writing:fyodor_store_list)(store,name_space,after[0]?after:NULL,20,&items,&count);
        if(result==FYODOR_STORE_OK) {
            nya_builder_append(&builder,"{\"ok\":true,\"resources\":[");
            for(size_t i=0;i<count;++i) {
                (void)fyodor_resource_format(&items[i].ref,uri,sizeof(uri));
                nya_builder_append(&builder,"%s{\"uri\":",i?",":"");nya_builder_append_json_string(&builder,uri);
                nya_builder_append(&builder,",\"title\":");nya_builder_append_json_string(&builder,items[i].title);
                nya_builder_append(&builder,",\"revision\":\"%" PRIu64 "\"}",items[i].revision);
            }
            nya_builder_append(&builder,"],\"next\":");nya_builder_append_json_string(&builder,count==20?uri:"");nya_builder_append(&builder,"}");
        }
        fyodor_resource_summaries_free(items,count);
    } else if(get) {
        fyodor_resource_record record={0};result=fyodor_store_get(store,name_space,&ref,read_revision,&record);
        if(result==FYODOR_STORE_OK) {
            /* Serialize one record snapshot, not separate head/export reads. */
            nya_builder_append(&builder,"{\"ok\":true,\"revision\":\"%" PRIu64 "\",\"deleted\":%s,\"resource\":{\"schema\":1,\"uri\":",record.revision,record.deleted?"true":"false");
            nya_builder_append_json_string(&builder,uri);nya_builder_append(&builder,",\"title\":");nya_builder_append_json_string(&builder,record.title);
            nya_builder_append(&builder,",\"content\":");nya_builder_append_json_string(&builder,record.content);
            nya_builder_append(&builder,",\"metadata\":%s,\"provenance\":%s}}",record.metadata,record.provenance);
        }
        fyodor_resource_record_free(&record);
    } else if(put) {
        const char *package=nya_json_value(request->body,request->body_length,"resource"),*end=package;
        if(!package || nya_json_skip_value(&end,request->body+request->body_length,0)!=0) result=FYODOR_STORE_INVALID;
        else result=fyodor_store_import(store,name_space,package,(size_t)(end-package),revision,&ref,&revision);
        if(result==FYODOR_STORE_OK) {
            (void)fyodor_resource_format(&ref,uri,sizeof(uri));nya_builder_append(&builder,"{\"ok\":true,\"uri\":");nya_builder_append_json_string(&builder,uri);
            nya_builder_append(&builder,",\"revision\":\"%" PRIu64 "\"}",revision);
        }
    } else {
        result=fyodor_store_delete(store,name_space,&ref,revision);
        if(result==FYODOR_STORE_OK) nya_builder_append(&builder,"{\"ok\":true,\"deleted\":true}");
    }
    fyodor_store_close(store);
    int sent;
    if(result!=FYODOR_STORE_OK) sent=nya_resource_error(client,server,origin,result);
    else if(builder.failed) sent=nya_resource_error(client,server,origin,FYODOR_STORE_NOMEM);
    else sent=nya_http_send_response(server,client,200,"OK",body,origin);
    free(body);return sent;
}
