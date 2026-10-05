#include "fyodor_generate.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void fyodor_context_generation_free(fyodor_context_generation *result)
{
    if(result==NULL) return;
    fyodor_context_bundle_free(&result->context);
    nya_generation_response_free(&result->generation);
    free(result->prompt); memset(result,0,sizeof(*result));
}

static void truncate_context(fyodor_context_bundle *bundle,size_t length)
{
    while(length>0&&((unsigned char)bundle->text[length]&0xc0u)==0x80u) --length;
    bundle->length=length; bundle->text[length]=0;
    for(size_t i=0;i<bundle->source_count;++i) {
        fyodor_context_source *source=&bundle->sources[i];
        if(source->offset>length) { source->offset=length; source->length=0; }
        else if(source->length>length-source->offset) source->length=length-source->offset;
    }
}

static fyodor_store_result generate_layers_checked(fyodor_store *store,nya_model *model,
    const fyodor_uuid *principal,const char *name_space,const fyodor_context_entry *entries,
    size_t count,size_t byte_budget,const nya_generation_request *request,
    fyodor_context_generation *out,char *error,size_t error_capacity,
    const fyodor_resource_ref *target,uint64_t expected_revision,const char *writing_mode)
{
    if(error!=NULL&&error_capacity) error[0]=0;
    if(request==NULL||out==NULL||model==NULL||request->prompt==NULL||
        request->max_tokens==0||request->max_tokens>1048576||request->draft_model!=NULL||
        request->soft_token_spans!=0||request->max_output_bytes==0||request->max_output_bytes>FYODOR_STORE_CONTENT_LIMIT)
        return FYODOR_STORE_INVALID;
    size_t prompt_length=0;
    while(prompt_length<=FYODOR_STORE_CONTENT_LIMIT&&request->prompt[prompt_length]) ++prompt_length;
    if(prompt_length>FYODOR_STORE_CONTENT_LIMIT) return FYODOR_STORE_INVALID;
    fyodor_context_generation result={0};
    fyodor_store_result status=target?
        fyodor_writing_assemble(store,principal,name_space,target,expected_revision,entries+1,count-1,byte_budget,&result.context):
        fyodor_context_assemble_layers(store,principal,name_space,entries,count,byte_budget,&result.context);
    if(status!=FYODOR_STORE_OK) return status;
    if(target&&(result.context.source_count==0||!fyodor_resource_equal(target,&result.context.sources[0].ref)||
        result.context.sources[0].revision!=expected_revision)) {
        fyodor_context_generation_free(&result);return FYODOR_STORE_CONFLICT;
    }
    result.prompt=malloc(result.context.length+prompt_length+3);
    if(result.prompt==NULL) { fyodor_context_generation_free(&result); return FYODOR_STORE_NOMEM; }
    nya_generation_request actual=*request; actual.prompt=result.prompt;
    for(;;) {
        size_t prefix=result.context.length;
        memcpy(result.prompt,result.context.text,prefix);
        if(prefix) { result.prompt[prefix++]='\n'; result.prompt[prefix++]='\n'; }
        memcpy(result.prompt+prefix,request->prompt,prompt_length+1);
        size_t tokens=0,capacity=0;
        if(nya_generation_prompt_info(model,result.prompt,&tokens,&capacity,error,error_capacity)!=0) {
            status=FYODOR_STORE_INVALID; break;
        }
        result.model_context_length=capacity;
        if(request->max_tokens<=capacity&&tokens<=capacity-request->max_tokens) {
            if(nya_generation_run(model,&actual,&result.generation,error,error_capacity)!=0) status=FYODOR_STORE_IO;
            else if(result.generation.prompt_tokens!=tokens) {
                if(error!=NULL&&error_capacity) snprintf(error,error_capacity,"tokenizer count changed during generation");
                status=FYODOR_STORE_IO;
            }
            break;
        }
        if(result.context.length==0) {
            if(error!=NULL&&error_capacity) snprintf(error,error_capacity,"user prompt and requested output exceed model context");
            status=FYODOR_STORE_INVALID; break;
        }
        truncate_context(&result.context,result.context.length/2);
    }
    if(status==FYODOR_STORE_OK) {
        if(result.generation.text==NULL||strlen(result.generation.text)!=result.generation.text_length) {
            if(error!=NULL&&error_capacity) snprintf(error,error_capacity,"native output cannot be represented as a receipt text string");
            fyodor_context_generation_free(&result); return FYODOR_STORE_INVALID;
        }
        fyodor_receipt_input receipt={0};
        receipt.writing_mode=writing_mode;receipt.context=&result.context; receipt.prompt=result.prompt; receipt.output=result.generation.text;
        receipt.model_path=model->path; receipt.model_file_size=model->file_size; receipt.seed=request->seed;
        receipt.temperature=request->temperature; receipt.top_p=request->top_p; receipt.top_k=request->top_k;
        receipt.max_tokens=request->max_tokens; receipt.prompt_tokens=result.generation.prompt_tokens;
        receipt.generated_tokens=result.generation.generated_tokens; receipt.model_context_length=result.model_context_length;
        receipt.stop_reason=nya_generation_stop_reason_name(result.generation.stop_reason);
        status=fyodor_receipt_save(store,principal,name_space,&receipt,&result.receipt_id);
        if(status!=FYODOR_STORE_OK&&error!=NULL&&error_capacity)
            snprintf(error,error_capacity,"generation finished but durable receipt storage failed");
    }
    if(status==FYODOR_STORE_OK) *out=result; else fyodor_context_generation_free(&result);
    return status;
}

fyodor_store_result fyodor_generate_with_layers(fyodor_store *store,nya_model *model,
    const fyodor_uuid *principal,const char *name_space,const fyodor_context_entry *entries,
    size_t count,size_t byte_budget,const nya_generation_request *request,
    fyodor_context_generation *out,char *error,size_t error_capacity)
{
    return generate_layers_checked(store,model,principal,name_space,entries,count,byte_budget,request,out,error,error_capacity,NULL,0,NULL);
}

fyodor_store_result fyodor_writing_generate(fyodor_store *store,nya_model *model,
    const fyodor_uuid *principal,const char *name_space,const fyodor_resource_ref *target,
    uint64_t expected_revision,fyodor_writing_mode mode,const fyodor_context_entry *extra,
    size_t count,size_t byte_budget,const nya_generation_request *request,
    fyodor_context_generation *out,char *error,size_t error_capacity)
{
    if(error&&error_capacity)error[0]=0;
    if(target==NULL||expected_revision==0||expected_revision>INT64_MAX||count>63||(count&&extra==NULL)||
        (unsigned)mode>FYODOR_WRITING_CONTINUE||request==NULL||request->prompt==NULL||
        (target->kind!=FYODOR_RESOURCE_DOCUMENT&&target->kind!=FYODOR_RESOURCE_NOTE&&
         target->kind!=FYODOR_RESOURCE_CHARACTER&&target->kind!=FYODOR_RESOURCE_PROJECT)) return FYODOR_STORE_INVALID;
    static const char *prefixes[]={"Write:\n","Rewrite:\n","Continue:\n"};
    size_t prefix=strlen(prefixes[mode]),length=0;
    while(length<=FYODOR_STORE_CONTENT_LIMIT&&request->prompt[length])++length;
    if(length>FYODOR_STORE_CONTENT_LIMIT-prefix)return FYODOR_STORE_INVALID;
    char *instructions=malloc(prefix+length+1);
    if(instructions==NULL)return FYODOR_STORE_NOMEM;
    memcpy(instructions,prefixes[mode],prefix);memcpy(instructions+prefix,request->prompt,length+1);
    nya_generation_request actual=*request;actual.prompt=instructions;
    fyodor_context_entry entries[64];entries[0]=(fyodor_context_entry){*target,FYODOR_CONTEXT_EXPLICIT};
    for(size_t i=0;i<count;++i)entries[i+1]=extra[i];
    static const char *modes[]={"generate","rewrite","continue"};
    fyodor_store_result result=generate_layers_checked(store,model,principal,name_space,entries,count+1,byte_budget,
        &actual,out,error,error_capacity,target,expected_revision,modes[mode]);
    free(instructions);return result;
}

fyodor_store_result fyodor_generate_with_context(fyodor_store *store,nya_model *model,
    const fyodor_uuid *principal,const char *name_space,const fyodor_resource_ref *refs,
    size_t count,size_t byte_budget,const nya_generation_request *request,
    fyodor_context_generation *out,char *error,size_t error_capacity)
{
    if(error!=NULL&&error_capacity) error[0]=0;
    if(count>64||(count&&refs==NULL)) return FYODOR_STORE_INVALID;
    fyodor_context_entry entries[64];
    for(size_t i=0;i<count;++i) entries[i]=(fyodor_context_entry){refs[i],FYODOR_CONTEXT_EXPLICIT};
    return fyodor_generate_with_layers(store,model,principal,name_space,entries,count,byte_budget,request,out,error,error_capacity);
}
