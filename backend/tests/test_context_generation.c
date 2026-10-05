#include "fyodor_generate.h"
#include "sqlite3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while(0)

static int writing(fyodor_store *store,nya_model *model,const char *path)
{
    fyodor_uuid principal;CHECK(fyodor_uuid_generate(&principal)==FYODOR_RESOURCE_OK);
    fyodor_resource_input input={0};input.ref.kind=FYODOR_RESOURCE_DOCUMENT;
    CHECK(fyodor_uuid_generate(&input.ref.id)==FYODOR_RESOURCE_OK);
    input.name_space="writing";input.title="Draft";input.content="abc";input.metadata="{\"big\":9007199254740993}";input.provenance="{\"origin\":\"manual\",\"big\":9007199254740993}";
    uint64_t revision=0;CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
    nya_generation_request request={0};request.prompt="a";request.max_tokens=2;request.temperature=0;request.top_p=1;request.seed=42;request.max_output_bytes=4096;
    fyodor_context_generation generated={0};char error[512];
    CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,1,FYODOR_WRITING_CONTINUE,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_DENIED);
    CHECK(fyodor_context_permissions_set(store,&principal,"writing",&input.ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,2,FYODOR_WRITING_CONTINUE,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_CONFLICT&&generated.prompt==NULL);
    fyodor_receipt_summary *page=NULL;size_t count=0;
    CHECK(fyodor_receipt_list(store,&principal,"writing",NULL,100,&page,&count)==FYODOR_STORE_OK&&count==0);fyodor_receipt_summaries_free(page);
    const char *expected[]={"abc\n\nWrite:\na","abc\n\nRewrite:\na","abc\n\nContinue:\na"};
    fyodor_uuid accepted_receipt;char accepted_text[8192];
    for(int mode=0;mode<3;++mode) {
        CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,1,(fyodor_writing_mode)mode,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_OK);
        CHECK(strcmp(generated.prompt,expected[mode])==0&&generated.generation.generated_tokens>0&&generated.context.sources[0].revision==1);
        accepted_receipt=generated.receipt_id;snprintf(accepted_text,sizeof(accepted_text),"abc%s",generated.generation.text);
        fyodor_receipt saved={0};CHECK(fyodor_receipt_read(store,&principal,"writing",&generated.receipt_id,&saved)==FYODOR_STORE_OK);
        CHECK(strcmp(saved.prompt,expected[mode])==0);fyodor_receipt_free(&saved);fyodor_context_generation_free(&generated);
        fyodor_resource_record unchanged={0};CHECK(fyodor_store_get(store,"writing",&input.ref,0,&unchanged)==FYODOR_STORE_OK);
        CHECK(unchanged.revision==1&&strcmp(unchanged.content,"abc")==0);fyodor_resource_record_free(&unchanged);
    }
    fyodor_context_entry duplicate={input.ref,FYODOR_CONTEXT_WORKSPACE};
    CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,1,FYODOR_WRITING_REWRITE,&duplicate,1,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_INVALID);
    CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,1,(fyodor_writing_mode)99,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_INVALID);
    CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,1,FYODOR_WRITING_REWRITE,NULL,64,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_INVALID);
    fyodor_uuid stranger;CHECK(fyodor_uuid_generate(&stranger)==FYODOR_RESOURCE_OK);
    uint64_t unchanged=999;
    CHECK(fyodor_writing_save(store,&stranger,"writing",&input.ref,1,&accepted_receipt,"Draft",accepted_text,&unchanged)==FYODOR_STORE_DENIED&&unchanged==999);
    CHECK(fyodor_writing_save(store,&principal,"elsewhere",&input.ref,1,&accepted_receipt,"Draft",accepted_text,&unchanged)==FYODOR_STORE_DENIED);
    CHECK(fyodor_writing_save(store,&principal,"writing",&input.ref,2,&accepted_receipt,"Draft",accepted_text,&unchanged)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_context_permissions_set(store,&principal,"writing",&input.ref,0)==FYODOR_STORE_OK);
    CHECK(fyodor_writing_save(store,&principal,"writing",&input.ref,1,&accepted_receipt,"Draft",accepted_text,&unchanged)==FYODOR_STORE_DENIED);
    CHECK(fyodor_context_permissions_set(store,&principal,"writing",&input.ref,1)==FYODOR_STORE_OK);
    sqlite3 *db=NULL;CHECK(sqlite3_open(path,&db)==SQLITE_OK);
    CHECK(sqlite3_exec(db,"CREATE TRIGGER fail_accept BEFORE INSERT ON revisions BEGIN SELECT RAISE(ABORT,'injected'); END",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(fyodor_writing_save(store,&principal,"writing",&input.ref,1,&accepted_receipt,"Draft",accepted_text,&unchanged)==FYODOR_STORE_CONFLICT&&unchanged==999);
    fyodor_resource_record record={0};CHECK(fyodor_store_get(store,"writing",&input.ref,0,&record)==FYODOR_STORE_OK);
    CHECK(record.revision==1&&strcmp(record.content,"abc")==0&&strcmp(record.provenance,input.provenance)==0);fyodor_resource_record_free(&record);
    CHECK(fyodor_store_get(store,"writing",&input.ref,2,&record)==FYODOR_STORE_NOT_FOUND);
    CHECK(sqlite3_exec(db,"DROP TRIGGER fail_accept",NULL,NULL,NULL)==SQLITE_OK);CHECK(sqlite3_close(db)==SQLITE_OK);
    CHECK(fyodor_writing_save(store,&principal,"writing",&input.ref,1,&accepted_receipt,"Draft",accepted_text,&revision)==FYODOR_STORE_OK&&revision==2);
    CHECK(fyodor_store_get(store,"writing",&input.ref,0,&record)==FYODOR_STORE_OK);
    CHECK(strcmp(record.content,accepted_text)==0&&strcmp(record.metadata,input.metadata)==0);
    CHECK(strstr(record.provenance,"9007199254740993")&&strstr(record.provenance,"\"text_edited\":false")&&strstr(record.provenance,"\"saved_revision\":\"2\""));fyodor_resource_record_free(&record);
    CHECK(fyodor_writing_save(store,&principal,"writing",&input.ref,2,&accepted_receipt,"Draft","a",&unchanged)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,2,FYODOR_WRITING_REWRITE,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_OK);
    CHECK(fyodor_writing_save(store,&principal,"writing",&input.ref,2,&generated.receipt_id,"Edited","user edit",&revision)==FYODOR_STORE_OK&&revision==3);
    fyodor_context_generation_free(&generated);
    CHECK(fyodor_store_get(store,"writing",&input.ref,0,&record)==FYODOR_STORE_OK);
    CHECK(strstr(record.provenance,"\"saved_revision\":\"2\"")&&strstr(record.provenance,"\"saved_revision\":\"3\"")&&strstr(record.provenance,"\"text_edited\":true"));fyodor_resource_record_free(&record);
    CHECK(fyodor_generate_with_context(store,model,&principal,"writing",&input.ref,1,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_OK);
    CHECK(fyodor_writing_save(store,&principal,"writing",&input.ref,3,&generated.receipt_id,"Draft","text",&unchanged)==FYODOR_STORE_INVALID);
    fyodor_context_generation_free(&generated);
    input.provenance="{\"writing_generations\":{}}";input.content="abc";
    CHECK(fyodor_store_put(store,&input,3,&revision)==FYODOR_STORE_OK&&revision==4);
    CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,4,FYODOR_WRITING_REWRITE,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_OK);
    CHECK(fyodor_writing_save(store,&principal,"writing",&input.ref,4,&generated.receipt_id,"Draft","text",&unchanged)==FYODOR_STORE_CONFLICT);
    fyodor_context_generation_free(&generated);
    CHECK(fyodor_store_delete(store,"writing",&input.ref,4)==FYODOR_STORE_OK);
    CHECK(fyodor_writing_generate(store,model,&principal,"writing",&input.ref,1,FYODOR_WRITING_CONTINUE,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_DENIED);
    return 0;
}

static int project_generation(fyodor_store *store,nya_model *model)
{
    fyodor_uuid principal;CHECK(fyodor_uuid_generate(&principal)==FYODOR_RESOURCE_OK);
    fyodor_resource_input project={0};project.ref.kind=FYODOR_RESOURCE_PROJECT;
    CHECK(fyodor_uuid_generate(&project.ref.id)==FYODOR_RESOURCE_OK);
    project.name_space="provider";project.title="Project";project.content="b";project.metadata="{}";project.provenance="{}";
    uint64_t revision=0;CHECK(fyodor_store_put(store,&project,0,&revision)==FYODOR_STORE_OK);
    char uri[FYODOR_RESOURCE_URI_CAPACITY],metadata[256];CHECK(fyodor_resource_format(&project.ref,uri,sizeof(uri))==FYODOR_RESOURCE_OK);
    snprintf(metadata,sizeof(metadata),"{\"writing_parent\":\"%s\"}",uri);
    fyodor_resource_input target=project;target.ref.kind=FYODOR_RESOURCE_DOCUMENT;
    CHECK(fyodor_uuid_generate(&target.ref.id)==FYODOR_RESOURCE_OK);target.content="a";target.metadata=metadata;
    CHECK(fyodor_store_put(store,&target,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_context_permissions_set(store,&principal,"provider",&target.ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_context_permissions_set(store,&principal,"provider",&project.ref,1)==FYODOR_STORE_OK);
    nya_generation_request request={0};request.prompt="a";request.max_tokens=2;request.temperature=0;request.top_p=1;request.seed=42;request.max_output_bytes=4096;
    fyodor_context_generation generated={0};char error[512];
    CHECK(fyodor_writing_generate(store,model,&principal,"provider",&target.ref,1,FYODOR_WRITING_GENERATE,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_OK);
    CHECK(strcmp(generated.prompt,"a\nb\n\nWrite:\na")==0&&generated.context.source_count==2&&generated.generation.generated_tokens>0);
    CHECK(generated.context.sources[1].layer==FYODOR_CONTEXT_WORKSPACE);
    fyodor_uuid receipt_id=generated.receipt_id;fyodor_context_generation_free(&generated);
    project.content="c";CHECK(fyodor_store_put(store,&project,1,&revision)==FYODOR_STORE_OK);
    fyodor_receipt receipt={0};CHECK(fyodor_receipt_read(store,&principal,"provider",&receipt_id,&receipt)==FYODOR_STORE_OK);
    CHECK(strcmp(receipt.prompt,"a\nb\n\nWrite:\na")==0&&strstr(receipt.sources,"workspace"));fyodor_receipt_free(&receipt);
    CHECK(fyodor_context_permissions_set(store,&principal,"provider",&project.ref,0)==FYODOR_STORE_OK);
    CHECK(fyodor_receipt_read(store,&principal,"provider",&receipt_id,&receipt)==FYODOR_STORE_DENIED);
    CHECK(fyodor_writing_generate(store,model,&principal,"provider",&target.ref,1,FYODOR_WRITING_GENERATE,NULL,0,3,&request,&generated,error,sizeof(error))==FYODOR_STORE_OK);
    CHECK(strcmp(generated.prompt,"a\n\nWrite:\na")==0&&generated.context.source_count==1);fyodor_context_generation_free(&generated);
    return 0;
}

int main(int argc,char **argv)
{
    if(argc!=3) return 2;
    (void)remove(argv[1]);
    fyodor_store *store=NULL; CHECK(fyodor_store_open(argv[1],&store)==FYODOR_STORE_OK);
    fyodor_uuid principal; CHECK(fyodor_uuid_generate(&principal)==FYODOR_RESOURCE_OK);
    fyodor_resource_input input={0}; input.ref.kind=FYODOR_RESOURCE_DOCUMENT;
    CHECK(fyodor_uuid_generate(&input.ref.id)==FYODOR_RESOURCE_OK);
    input.name_space="workspace"; input.title="source"; input.content="abc";
    input.metadata="{}"; input.provenance="{}";
    uint64_t revision=0; CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
    nya_model_registry registry; nya_model_registry_init(&registry);
    const nya_model *loaded=NULL; CHECK(nya_model_load(&registry,argv[2],&loaded)==NYA_MODEL_OK);
    CHECK(loaded!=NULL&&loaded->generation_supported);
    nya_model *model=&registry.models[0];
    nya_generation_request request={0}; request.prompt="a"; request.max_tokens=2;
    request.temperature=0; request.top_p=1; request.seed=42; request.max_output_bytes=4096;
    fyodor_context_generation result={0}; char error[512];
    CHECK(fyodor_generate_with_context(store,model,&principal,"workspace",&input.ref,1,100,&request,&result,error,sizeof(error))==FYODOR_STORE_DENIED);
    CHECK(result.prompt==NULL&&result.generation.text==NULL);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,FYODOR_PERMISSION_READ)==FYODOR_STORE_OK);
    CHECK(fyodor_generate_with_context(store,model,&principal,"workspace",&input.ref,1,100,&request,&result,error,sizeof(error))==FYODOR_STORE_OK);
    CHECK(strcmp(result.prompt,"abc\n\na")==0&&result.context.source_count==1);
    CHECK(result.context.sources[0].revision==1&&result.context.sources[0].length==3);
    CHECK(result.generation.prompt_tokens>0&&result.generation.generated_tokens>0&&result.generation.target_steps>0);
    CHECK(result.generation.prompt_tokens+request.max_tokens<=result.model_context_length);
    fyodor_uuid first_receipt=result.receipt_id;
    fyodor_receipt saved={0};
    CHECK(fyodor_receipt_read(store,&principal,"workspace",&first_receipt,&saved)==FYODOR_STORE_OK);
    CHECK(strcmp(saved.prompt,result.prompt)==0&&strcmp(saved.output,result.generation.text)==0);
    CHECK(strstr(saved.sources,"\"revision\":1")!=NULL&&strstr(saved.metadata,"\"schema\":1")!=NULL);
    fyodor_receipt_free(&saved);
    nya_generation_response direct={0}; nya_generation_request exact=request; exact.prompt=result.prompt;
    CHECK(nya_generation_run(model,&exact,&direct,error,sizeof(error))==0);
    CHECK(direct.prompt_tokens==result.generation.prompt_tokens&&strcmp(direct.text,result.generation.text)==0);
    nya_generation_response_free(&direct); fyodor_context_generation_free(&result);
    fyodor_store_close(store); store=NULL;
    CHECK(fyodor_store_open(argv[1],&store)==FYODOR_STORE_OK);
    CHECK(fyodor_receipt_read(store,&principal,"workspace",&first_receipt,&saved)==FYODOR_STORE_OK);
    CHECK(strcmp(saved.prompt,"abc\n\na")==0); fyodor_receipt_free(&saved);
    fyodor_uuid stranger; CHECK(fyodor_uuid_generate(&stranger)==FYODOR_RESOURCE_OK);
    CHECK(fyodor_receipt_read(store,&stranger,"workspace",&first_receipt,&saved)==FYODOR_STORE_DENIED);
    CHECK(fyodor_receipt_read(store,&principal,"other",&first_receipt,&saved)==FYODOR_STORE_DENIED);
    /* Force a genuine tokenizer budget reduction using much more context than
     * this trained tiny transformer's context window can hold. */
    char *long_text=malloc(65537); CHECK(long_text!=NULL);
    for(size_t i=0;i<65536;++i) long_text[i]=i%2?'b':'a';
    long_text[65536]=0;
    input.content=long_text;
    CHECK(fyodor_store_put(store,&input,1,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_receipt_read(store,&principal,"workspace",&first_receipt,&saved)==FYODOR_STORE_OK);
    CHECK(strcmp(saved.prompt,"abc\n\na")==0); fyodor_receipt_free(&saved);
    fyodor_context_entry global={input.ref,FYODOR_CONTEXT_GLOBAL};
    CHECK(fyodor_generate_with_layers(store,model,&principal,"workspace",&global,1,65536,&request,&result,error,sizeof(error))==FYODOR_STORE_OK);
    CHECK(result.context.sources[0].layer==FYODOR_CONTEXT_GLOBAL);
    CHECK(fyodor_receipt_read(store,&principal,"workspace",&result.receipt_id,&saved)==FYODOR_STORE_OK);
    CHECK(strstr(saved.sources,"\"layer\":\"global\"")!=NULL);
    fyodor_receipt_free(&saved);
    CHECK(result.context.length<65536&&result.context.sources[0].length==result.context.length);
    CHECK(result.context.sources[0].original_length==65536&&result.context.sources[0].revision==2);
    CHECK(result.generation.prompt_tokens+request.max_tokens<=result.model_context_length);
    size_t capacity=result.model_context_length;
    fyodor_receipt_summary *page=NULL;size_t page_count=0;
    CHECK(fyodor_receipt_list(store,&principal,"workspace",NULL,1,&page,&page_count)==FYODOR_STORE_OK && page_count==1);
    fyodor_uuid cursor=page[0].id;fyodor_receipt_summaries_free(page);page=NULL;
    CHECK(fyodor_receipt_list(store,&principal,"workspace",&cursor,1,&page,&page_count)==FYODOR_STORE_OK && page_count==1);
    CHECK(memcmp(&cursor,&page[0].id,sizeof(cursor))!=0);cursor=page[0].id;fyodor_receipt_summaries_free(page);page=NULL;
    CHECK(fyodor_receipt_list(store,&principal,"workspace",&cursor,1,&page,&page_count)==FYODOR_STORE_OK && page_count==0);
    fyodor_receipt_summaries_free(page);page=NULL;
    CHECK(fyodor_receipt_list(store,&principal,"workspace",NULL,0,&page,&page_count)==FYODOR_STORE_INVALID && page==NULL);
    fyodor_context_generation_free(&result);
    request.max_tokens=capacity;
    CHECK(fyodor_generate_with_context(store,model,&principal,"workspace",&input.ref,1,100,&request,&result,error,sizeof(error))==FYODOR_STORE_INVALID);
    CHECK(result.prompt==NULL&&strstr(error,"exceed")!=NULL);
    request.max_tokens=2;
    sqlite3 *db=NULL; CHECK(sqlite3_open(argv[1],&db)==SQLITE_OK);
    CHECK(sqlite3_exec(db,"CREATE TRIGGER reject_receipt BEFORE INSERT ON context_receipts "
        "BEGIN SELECT RAISE(ABORT,'injected save failure'); END",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(fyodor_generate_with_context(store,model,&principal,"workspace",&input.ref,1,100,&request,&result,error,sizeof(error))==FYODOR_STORE_CONFLICT);
    CHECK(result.prompt==NULL&&strstr(error,"receipt storage failed")!=NULL);
    sqlite3_stmt *counter=NULL;
    CHECK(sqlite3_prepare_v2(db,"SELECT count(*) FROM context_receipts",-1,&counter,NULL)==SQLITE_OK);
    CHECK(sqlite3_step(counter)==SQLITE_ROW&&sqlite3_column_int(counter,0)==2);
    CHECK(sqlite3_finalize(counter)==SQLITE_OK);
    CHECK(sqlite3_exec(db,"DROP TRIGGER reject_receipt",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(sqlite3_close(db)==SQLITE_OK);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,0)==FYODOR_STORE_OK);
    CHECK(fyodor_receipt_read(store,&principal,"workspace",&first_receipt,&saved)==FYODOR_STORE_DENIED&&saved.prompt==NULL);
    CHECK(fyodor_receipt_list(store,&principal,"workspace",NULL,100,&page,&page_count)==FYODOR_STORE_OK && page_count==0);
    fyodor_receipt_summaries_free(page);
    CHECK(fyodor_generate_with_context(store,model,&principal,"workspace",&input.ref,1,100,&request,&result,error,sizeof(error))==FYODOR_STORE_DENIED);
    CHECK(writing(store,model,argv[1])==0);
    CHECK(project_generation(store,model)==0);
    free(long_text); nya_model_registry_shutdown(&registry); fyodor_store_close(store);
    CHECK(remove(argv[1])==0);
    puts("context generation: native inference, exact-token budget, attribution and permission denial passed");
    return 0;
}
