#include "fyodor_client.h"
#include "fyodor_store.h"
#include "fyodor_context.h"
#include "fyodor_generate.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const char usage[] =
    "Usage:\n"
    "  fyodor resource id KIND\n"
    "  fyodor --store PATH --namespace NAME resource list\n"
    "  fyodor --store PATH --namespace NAME resource lore URI REVISION [LORE_URI...]\n"
    "  fyodor --store PATH --namespace NAME resource move URI REVISION PROJECT_URI_OR_ROOT\n"
    "  fyodor --store PATH --namespace NAME resource folder PROJECT_URI_OR_ROOT [AFTER_URI]\n"
    "  fyodor --store PATH --namespace NAME resource export URI [REVISION]\n"
    "  fyodor --store PATH --namespace NAME resource history URI [BEFORE_REVISION]\n"
    "  fyodor --store PATH --namespace NAME resource import [EXPECTED_REVISION] < package.json\n"
    "  fyodor --store PATH --namespace NAME resource delete URI EXPECTED_REVISION\n"
    "  fyodor --store PATH --namespace NAME context permissions PRINCIPAL URI [MASK]\n"
    "  fyodor --store PATH --namespace NAME context inspect PRINCIPAL URI\n"
    "  fyodor --store PATH --namespace NAME context search PRINCIPAL QUERY\n"
    "  fyodor --store PATH --namespace NAME context explain PRINCIPAL BYTE_BUDGET URI...\n"
    "  fyodor --store PATH --namespace NAME context generate PRINCIPAL MODEL BYTE_BUDGET MAX_TOKENS PROMPT [URI...]\n"
    "  fyodor --store PATH --namespace NAME context write PRINCIPAL MODEL TARGET REVISION MODE BYTE_BUDGET MAX_TOKENS INSTRUCTIONS [URI...]\n"
    "  fyodor --store PATH --namespace NAME context accept PRINCIPAL URI REVISION RECEIPT_ID TITLE CONTENT\n"
    "    Writing MODE is generate, rewrite or continue. Output is a preview, never saved over the target.\n"
    "    explain/generate/write sources accept [explicit|workspace|session|retrieved|global]=URI\n"
    "  fyodor --store PATH --namespace NAME context receipt PRINCIPAL RECEIPT_ID\n"
    "  fyodor --store PATH --namespace NAME context receipts PRINCIPAL [AFTER_ID]\n"
    "  fyodor --store PATH --namespace NAME context append PRINCIPAL URI EXPECTED_REVISION TEXT\n"
    "  fyodor --store PATH --namespace NAME context delete PRINCIPAL URI EXPECTED_REVISION\n"
    "Import defaults to create-only (revision 0). List emits JSONL; other output is JSON.\n"
    "This build currently provides local resource commands.\n";

static int status(fyodor_store_result result, FILE *error)
{
    const char *message = "storage I/O failed"; int code = 1;
    switch (result) {
    case FYODOR_STORE_OK: return 0;
    case FYODOR_STORE_INVALID: message = "invalid input"; code = 2; break;
    case FYODOR_STORE_NOT_FOUND: message = "resource not found"; code = 3; break;
    case FYODOR_STORE_CONFLICT: message = "resource revision or parent conflict"; code = 4; break;
    case FYODOR_STORE_BUSY: message = "workspace is busy"; code = 5; break;
    case FYODOR_STORE_DENIED: message = "resource access denied"; code = 6; break;
    case FYODOR_STORE_SCHEMA_ERROR: message = "unsupported or damaged workspace schema"; break;
    case FYODOR_STORE_NOMEM: message = "allocation failed"; break;
    default: break;
    }
    fprintf(error,"fyodor: %s\n",message); return code;
}

static int revision_parse(const char *text, uint64_t *value)
{
    uint64_t n=0;
    if (*text==0) return 0;
    for (const char *p=text;*p;++p) {
        if (*p<'0'||*p>'9') return 0;
        unsigned digit=(unsigned)(*p-'0');
        if(n>((uint64_t)INT64_MAX-1-digit)/10) return 0;
        n=n*10+digit;
    }
    *value=n; return 1;
}

static void json_string(FILE *out,const char *text)
{
    fputc('"',out);
    for(const unsigned char *p=(const unsigned char *)text;*p;++p) {
        if(*p=='"'||*p=='\\') { fputc('\\',out); fputc(*p,out); }
        else if(*p<32) fprintf(out,"\\u%04x",(unsigned)*p);
        else fputc(*p,out);
    }
    fputc('"',out);
}

static int list(fyodor_store *store,const char *name_space,FILE *output,FILE *error)
{
    char cursor[FYODOR_RESOURCE_URI_CAPACITY]="";
    for(;;) {
        fyodor_resource_summary *items=NULL; size_t count=0;
        int code=status(fyodor_store_list(store,name_space,cursor[0]?cursor:NULL,100,&items,&count),error);
        if(code) return code;
        for(size_t i=0;i<count;++i) {
            (void)fyodor_resource_format(&items[i].ref,cursor,sizeof(cursor));
            fputs("{\"uri\":",output); json_string(output,cursor);
            fputs(",\"title\":",output); json_string(output,items[i].title);
            fprintf(output,",\"revision\":%llu}\n",(unsigned long long)items[i].revision);
        }
        fyodor_resource_summaries_free(items,count);
        if(ferror(output)) return 1;
        if(count<100) return 0;
    }
}

static int context_entry_parse(const char *text,fyodor_context_entry *entry)
{
    entry->layer=FYODOR_CONTEXT_EXPLICIT;
    const char *equals=strchr(text,'=');
    if(equals) {
        int found=0;
        for(int layer=0;layer<FYODOR_CONTEXT_LAYER_COUNT;++layer) {
            const char *name=fyodor_context_layer_name((fyodor_context_layer)layer);
            if(strlen(name)==(size_t)(equals-text)&&strncmp(text,name,(size_t)(equals-text))==0) {
                entry->layer=(fyodor_context_layer)layer; found=1; break;
            }
        }
        if(!found) return 0;
        text=equals+1;
    }
    return fyodor_resource_parse(text,strlen(text),&entry->ref)==FYODOR_RESOURCE_OK;
}

static int context_command(int argc,char **argv,FILE *output,FILE *error)
{
    const char *command=argv[6];
    int permissions=strcmp(command,"permissions")==0, inspect=strcmp(command,"inspect")==0;
    int search=strcmp(command,"search")==0, explain=strcmp(command,"explain")==0;
    int writing=strcmp(command,"write")==0;
    int generate=writing||strcmp(command,"generate")==0, receipt=strcmp(command,"receipt")==0;
    int accept=strcmp(command,"accept")==0;
    int receipts=strcmp(command,"receipts")==0;
    int append=strcmp(command,"append")==0, remove=strcmp(command,"delete")==0;
    fyodor_uuid principal; fyodor_resource_ref refs[64]; uint64_t value=0;
    fyodor_uuid receipt_id; uint64_t max_tokens=0; fyodor_context_entry entries[64];
    int offset=writing?3:0;fyodor_resource_ref target={0};uint64_t target_revision=0;fyodor_writing_mode mode=FYODOR_WRITING_GENERATE;
    if(argc<8||fyodor_uuid_parse(argv[7],strlen(argv[7]),&principal)!=FYODOR_RESOURCE_OK) goto invalid;
    if((permissions&&(argc!=9&&argc!=10))||(inspect&&argc!=9)||(search&&argc!=9)||
        (explain&&(argc<10||argc>73))||(receipt&&argc!=9)||(generate&&(argc<12+offset||argc>76+offset-(writing?1:0)))||
        (append&&argc!=11)||(remove&&argc!=10)||(accept&&argc!=13)||
        (receipts&&argc!=8&&argc!=9)||
        (!permissions&&!inspect&&!search&&!explain&&!generate&&!receipt&&!append&&!remove&&!receipts&&!accept)) goto invalid;
    if(receipts&&argc==9&&fyodor_uuid_parse(argv[8],strlen(argv[8]),&receipt_id)!=FYODOR_RESOURCE_OK) goto invalid;
    if(append||remove||accept) {
        if(fyodor_resource_parse(argv[8],strlen(argv[8]),&refs[0])!=FYODOR_RESOURCE_OK||
            !revision_parse(argv[9],&value)||value==0) goto invalid;
    }
    if(accept&&fyodor_uuid_parse(argv[10],strlen(argv[10]),&receipt_id)!=FYODOR_RESOURCE_OK)goto invalid;
    if(receipt&&fyodor_uuid_parse(argv[8],strlen(argv[8]),&receipt_id)!=FYODOR_RESOURCE_OK) goto invalid;
    if(writing) {
        if(fyodor_resource_parse(argv[9],strlen(argv[9]),&target)!=FYODOR_RESOURCE_OK||
            !revision_parse(argv[10],&target_revision)||target_revision==0)goto invalid;
        if(strcmp(argv[11],"generate")==0)mode=FYODOR_WRITING_GENERATE;
        else if(strcmp(argv[11],"rewrite")==0)mode=FYODOR_WRITING_REWRITE;
        else if(strcmp(argv[11],"continue")==0)mode=FYODOR_WRITING_CONTINUE;
        else goto invalid;
    }
    if(generate) {
        if(!revision_parse(argv[9+offset],&value)||value>FYODOR_STORE_CONTENT_LIMIT||
            !revision_parse(argv[10+offset],&max_tokens)||max_tokens==0||max_tokens>1048576) goto invalid;
        for(int i=12+offset;i<argc;++i) if(!context_entry_parse(argv[i],&entries[i-12-offset])) goto invalid;
    }
    if((permissions||inspect)&&fyodor_resource_parse(argv[8],strlen(argv[8]),&refs[0])!=FYODOR_RESOURCE_OK) goto invalid;
    if(permissions&&argc==10&&(!revision_parse(argv[9],&value)||value>FYODOR_PERMISSION_ALL)) goto invalid;
    if(explain) {
        if(!revision_parse(argv[8],&value)||value>FYODOR_STORE_CONTENT_LIMIT) goto invalid;
        for(int i=9;i<argc;++i) if(!context_entry_parse(argv[i],&entries[i-9])) goto invalid;
    }
    fyodor_store *store=NULL; int code=status(fyodor_store_open(argv[2],&store),error);
    if(code) return code;
    if(receipts) {
        fyodor_receipt_summary *items=NULL;size_t count=0;
        code=status(fyodor_receipt_list(store,&principal,argv[4],argc==9?&receipt_id:NULL,20,&items,&count),error);
        for(size_t i=0;code==0&&i<count;++i) {
            char id[FYODOR_UUID_TEXT_CAPACITY];(void)fyodor_uuid_format(&items[i].id,id,sizeof(id));
            fprintf(output,"{\"id\":\"%s\",\"created_ms\":\"%lld\"}\n",id,(long long)items[i].created_ms);
        }
        fyodor_receipt_summaries_free(items);
    }
    if(permissions) {
        unsigned mask=0;
        if(argc==10) code=status(fyodor_context_permissions_set(store,&principal,argv[4],&refs[0],(unsigned)value),error);
        if(code==0) code=status(fyodor_context_permissions_get(store,&principal,argv[4],&refs[0],&mask),error);
        if(code==0) fprintf(output,"{\"permissions\":%u}\n",mask);
    }
    if(inspect) {
        fyodor_resource_record record={0};
        code=status(fyodor_context_read(store,&principal,argv[4],&refs[0],&record),error);
        if(code==0) {
            fputs("{\"title\":",output); json_string(output,record.title);
            fputs(",\"content\":",output); json_string(output,record.content);
            fprintf(output,",\"revision\":%llu}\n",(unsigned long long)record.revision);
        }
        fyodor_resource_record_free(&record);
    }
    if(search) {
        fyodor_resource_summary *items=NULL; size_t count=0;
        code=status(fyodor_context_search(store,&principal,argv[4],argv[8],100,&items,&count),error);
        for(size_t i=0;code==0&&i<count;++i) {
            char uri[FYODOR_RESOURCE_URI_CAPACITY]; (void)fyodor_resource_format(&items[i].ref,uri,sizeof(uri));
            fputs("{\"uri\":",output); json_string(output,uri);
            fputs(",\"title\":",output); json_string(output,items[i].title);
            fprintf(output,",\"revision\":%llu}\n",(unsigned long long)items[i].revision);
        }
        fyodor_resource_summaries_free(items,count);
    }
    if(explain) {
        fyodor_context_bundle bundle={0};
        code=status(fyodor_context_assemble_layers(store,&principal,argv[4],entries,(size_t)argc-9,(size_t)value,&bundle),error);
        if(code==0) {
            fputs("{\"text\":",output); json_string(output,bundle.text);
            fprintf(output,",\"bytes\":%zu,\"sources\":[",bundle.length);
            for(size_t i=0;i<bundle.source_count;++i) {
                char uri[FYODOR_RESOURCE_URI_CAPACITY]; (void)fyodor_resource_format(&bundle.sources[i].ref,uri,sizeof(uri));
                if(i) fputc(',',output);
                fputs("{\"uri\":",output); json_string(output,uri);
                fprintf(output,",\"revision\":%llu,\"offset\":%zu,\"length\":%zu,\"original_length\":%zu,\"layer\":\"%s\"}",
                    (unsigned long long)bundle.sources[i].revision,bundle.sources[i].offset,bundle.sources[i].length,bundle.sources[i].original_length,fyodor_context_layer_name(bundle.sources[i].layer));
            }
            fputs("]}\n",output);
        }
        fyodor_context_bundle_free(&bundle);
    }
    if(generate) {
        nya_model_registry registry; nya_model_registry_init(&registry);
        const nya_model *loaded=NULL;
        if(nya_model_load(&registry,argv[8],&loaded)!=NYA_MODEL_OK||loaded==NULL||!loaded->generation_supported) {
            fputs("fyodor: model has no usable native generation provider\n",error); code=1;
        } else {
            nya_generation_request request={0}; request.prompt=argv[11+offset]; request.max_tokens=(size_t)max_tokens;
            request.temperature=0; request.top_p=1; request.seed=42; request.max_output_bytes=FYODOR_STORE_CONTENT_LIMIT;
            fyodor_context_generation result={0}; char detail[512];
            if(writing) code=status(fyodor_writing_generate(store,&registry.models[0],&principal,argv[4],&target,target_revision,mode,
                entries,(size_t)(argc-12-offset),(size_t)value,&request,&result,detail,sizeof(detail)),error);
            else code=status(fyodor_generate_with_layers(store,&registry.models[0],&principal,argv[4],entries,
                (size_t)argc-12,(size_t)value,&request,&result,detail,sizeof(detail)),error);
            if(code==0) {
                char id[FYODOR_UUID_TEXT_CAPACITY]; (void)fyodor_uuid_format(&result.receipt_id,id,sizeof(id));
                fputs("{\"receipt_id\":",output); json_string(output,id);
                fputs(",\"text\":",output); json_string(output,result.generation.text);
                fprintf(output,",\"prompt_tokens\":%zu,\"generated_tokens\":%zu}\n",
                    result.generation.prompt_tokens,result.generation.generated_tokens);
            } else if(detail[0]) fprintf(error,"fyodor: %s\n",detail);
            fyodor_context_generation_free(&result);
        }
        nya_model_registry_shutdown(&registry);
    }
    if(receipt) {
        fyodor_receipt result={0};
        code=status(fyodor_receipt_read(store,&principal,argv[4],&receipt_id,&result),error);
        if(code==0) {
            fputs("{\"prompt\":",output); json_string(output,result.prompt);
            fputs(",\"output\":",output); json_string(output,result.output);
            fprintf(output,",\"metadata\":%s,\"sources\":%s}\n",result.metadata,result.sources);
        }
        fyodor_receipt_free(&result);
    }
    if(accept) {
        uint64_t revision=0;
        code=status(fyodor_writing_save(store,&principal,argv[4],&refs[0],value,&receipt_id,argv[11],argv[12],&revision),error);
        if(code==0)fprintf(output,"{\"revision\":%llu}\n",(unsigned long long)revision);
    }
    if(append) {
        uint64_t revision=0;
        code=status(fyodor_context_append(store,&principal,argv[4],&refs[0],argv[10],value,&revision),error);
        if(code==0) fprintf(output,"{\"revision\":%llu}\n",(unsigned long long)revision);
    }
    if(remove) {
        code=status(fyodor_context_delete(store,&principal,argv[4],&refs[0],value),error);
        if(code==0) fputs("{\"deleted\":true}\n",output);
    }
    fyodor_store_close(store);
    if(fflush(output)!=0) code=1;
    return code;
invalid:
    fputs(usage,error); return 2;
}

int fyodor_command_run(int argc,char **argv,FILE *input,FILE *output,FILE *error)
{
    if(argc<1||argv==NULL||input==NULL||output==NULL||error==NULL) return 2;
    if(argc==2 && (strcmp(argv[1],"--help")==0||strcmp(argv[1],"-h")==0)) {
        fputs(usage,output); return fflush(output)==0?0:1;
    }
    if(argc==4 && strcmp(argv[1],"resource")==0 && strcmp(argv[2],"id")==0) {
        fyodor_resource_ref ref={0};
        for(int kind=FYODOR_RESOURCE_MODEL;kind<=FYODOR_RESOURCE_WORKFLOW;++kind)
            if(strcmp(argv[3],fyodor_resource_kind_name((fyodor_resource_kind)kind))==0) ref.kind=(fyodor_resource_kind)kind;
        /* Child IDs require explicit parent identity, handled via packages. */
        if(ref.kind==0||ref.kind==FYODOR_RESOURCE_WORLD_LORE||ref.kind==FYODOR_RESOURCE_WORLD_SAVE||ref.kind==FYODOR_RESOURCE_AGENT_RUN)
            goto usage_error;
        char uri[FYODOR_RESOURCE_URI_CAPACITY];
        if(fyodor_uuid_generate(&ref.id)!=FYODOR_RESOURCE_OK||fyodor_resource_format(&ref,uri,sizeof(uri))!=FYODOR_RESOURCE_OK) return 1;
        fputs("{\"uri\":",output); json_string(output,uri); fputs("}\n",output);
        return fflush(output)==0?0:1;
    }
    if(argc<7||strcmp(argv[1],"--store")||strcmp(argv[3],"--namespace")) goto usage_error;
    if(strcmp(argv[5],"context")==0) return context_command(argc,argv,output,error);
    if(strcmp(argv[5],"resource")) goto usage_error;
    const char *command=argv[6]; uint64_t expected=0;
    int is_list=strcmp(command,"list")==0, is_export=strcmp(command,"export")==0;
    int is_import=strcmp(command,"import")==0, is_delete=strcmp(command,"delete")==0;
    int is_lore=strcmp(command,"lore")==0;
    int is_history=strcmp(command,"history")==0,is_move=strcmp(command,"move")==0,is_folder=strcmp(command,"folder")==0;
    fyodor_resource_ref ref={0},parent={0};const char *folder=NULL;
    if((is_list && argc!=7)||(is_export && argc!=8 && argc!=9)||(is_import && argc!=7 && argc!=8)||
        (is_delete && argc!=9)||(is_history&&argc!=8&&argc!=9)||(is_lore&&(argc<9||argc>41))||(is_move&&argc!=10)||(is_folder&&argc!=8&&argc!=9)||(!is_list&&!is_export&&!is_import&&!is_delete&&!is_history&&!is_move&&!is_folder&&!is_lore)) goto usage_error;
    if((is_export||is_delete||is_history||is_move||is_lore) && fyodor_resource_parse(argv[7],strlen(argv[7]),&ref)!=FYODOR_RESOURCE_OK) goto usage_error;
    if(is_import && argc==8 && !revision_parse(argv[7],&expected)) goto usage_error;
    if((is_export||is_history)&&argc==9&&!revision_parse(argv[8],&expected)) goto usage_error;
    if((is_delete||is_move||is_lore) && (!revision_parse(argv[8],&expected)||expected==0)) goto usage_error;
    if(is_move||is_folder) {
        folder=argv[is_move?9:7];
        if(strcmp(folder,"root")&&(fyodor_resource_parse(folder,strlen(folder),&parent)!=FYODOR_RESOURCE_OK||parent.kind!=FYODOR_RESOURCE_PROJECT))goto usage_error;
    }
    fyodor_store *store=NULL;
    int code=status(fyodor_store_open(argv[2],&store),error);
    if(code) return code;
    if(is_list) code=list(store,argv[4],output,error);
    if(is_lore) {
        fyodor_resource_ref refs[32];size_t count=(size_t)(argc-9);uint64_t revision=0;
        for(size_t i=0;i<count;++i)if(fyodor_resource_parse(argv[i+9],strlen(argv[i+9]),&refs[i])!=FYODOR_RESOURCE_OK){code=2;break;}
        if(!code)code=status(fyodor_store_set_lore(store,argv[4],&ref,expected,refs,count,&revision),error);
        if(!code)fprintf(output,"{\"revision\":\"%llu\"}\n",(unsigned long long)revision);
    }
    if(is_move) {
        uint64_t revision=0;
        code=status(fyodor_store_move_writing(store,argv[4],&ref,expected,parent.kind?&parent:NULL,&revision),error);
        if(!code)fprintf(output,"{\"revision\":\"%llu\"}\n",(unsigned long long)revision);
    }
    if(is_folder) {
        fyodor_resource_summary *items=NULL;size_t count=0;
        code=status(fyodor_store_list_folder(store,argv[4],parent.kind?&parent:NULL,argc==9?argv[8]:NULL,20,&items,&count),error);
        for(size_t i=0;!code&&i<count;++i) {
            char uri[FYODOR_RESOURCE_URI_CAPACITY];(void)fyodor_resource_format(&items[i].ref,uri,sizeof(uri));
            fputs("{\"uri\":",output);json_string(output,uri);fputs(",\"title\":",output);json_string(output,items[i].title);
            fprintf(output,",\"revision\":\"%llu\"}\n",(unsigned long long)items[i].revision);
        }
        fyodor_resource_summaries_free(items,count);
    }
    if(is_export) {
        char *json=NULL; size_t length=0;
        code=status(fyodor_store_export_revision(store,argv[4],&ref,expected,&json,&length),error);
        if(code==0 && (fwrite(json,1,length,output)!=length || fputc('\n',output)==EOF)) code=1;
        fyodor_resource_package_free(json);
    }
    if(is_history) {
        fyodor_revision_summary *items=NULL;size_t count=0;
        code=status(fyodor_store_history(store,argv[4],&ref,expected,20,&items,&count),error);
        for(size_t i=0;code==0&&i<count;++i)
            fprintf(output,"{\"revision\":\"%llu\",\"modified_ms\":\"%lld\",\"deleted\":%s}\n",
                (unsigned long long)items[i].revision,(long long)items[i].modified_ms,items[i].deleted?"true":"false");
        free(items);
    }
    if(is_delete) {
        code=status(fyodor_store_delete(store,argv[4],&ref,expected),error);
        if(code==0) fputs("{\"deleted\":true}\n",output);
    }
    if(is_import) {
        size_t capacity=4096, length=0; char *json=malloc(capacity);
        if(json==NULL) code=1;
        while(code==0) {
            if(length==capacity) {
                if(capacity==FYODOR_STORE_PACKAGE_LIMIT) {
                    int next=fgetc(input);
                    if(next!=EOF) code=2;
                    break;
                }
                size_t larger=capacity*2;
                char *grown=realloc(json,larger);
                if(grown==NULL) { code=1; break; }
                json=grown; capacity=larger;
            }
            size_t got=fread(json+length,1,capacity-length,input); length+=got;
            if(got==0) break;
        }
        if(ferror(input)) code=1;
        if(code==0) {
            uint64_t revision=0;
            code=status(fyodor_store_import(store,argv[4],json,length,expected,&ref,&revision),error);
            if(code==0) {
                char uri[FYODOR_RESOURCE_URI_CAPACITY]; (void)fyodor_resource_format(&ref,uri,sizeof(uri));
                fputs("{\"uri\":",output); json_string(output,uri);
                fprintf(output,",\"revision\":%llu}\n",(unsigned long long)revision);
            }
        } else fputs("fyodor: input read or size limit failure\n",error);
        free(json);
    }
    fyodor_store_close(store);
    if(fflush(output)!=0) code=1;
    return code;
usage_error:
    fputs(usage,error); return 2;
}
