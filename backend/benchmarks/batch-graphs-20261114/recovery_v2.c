/* Test-only failure recovery for the isolated graph driver wrappers. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "llm_internal.h"
#include "file.h"
#include "pretraining.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);goto done;}}while(0)
static int setting(const char *name,const char *value)
{
#ifdef _WIN32
    return _putenv_s(name,value);
#else
    return setenv(name,value,1);
#endif
}
static int parity(const float *actual,const float *expected,size_t n)
{
    for(size_t i=0;i<n;++i){
        double a=actual[i],b=expected[i];
        if(a==-INFINITY && b==-INFINITY)continue;
        if(!isfinite(a)||!isfinite(b)||fabs(a-b)/(1+fabs(b))>0.0005)return 0;
    }
    return 1;
}
int main(int argc,char **argv)
{
    if(argc!=6 || (strcmp(argv[2],"256") && strcmp(argv[2],"512")) ||
       (strcmp(argv[3],"0") && strcmp(argv[3],"1")) ||
       (strcmp(argv[4],"request") && strcmp(argv[4],"lowering") && strcmp(argv[4],"sync") && strcmp(argv[4],"readback") && strcmp(argv[4],"begin") && strcmp(argv[4],"end") && strcmp(argv[4],"instantiate") && strcmp(argv[4],"launch")) ||
       (strcmp(argv[5],"cold") && strcmp(argv[5],"cached")))return 2;
    int cached=!strcmp(argv[5],"cached");
    if(cached && strcmp(argv[4],"launch") && strcmp(argv[4],"request") && strcmp(argv[4],"sync") && strcmp(argv[4],"readback"))return 2;
    int result=1;char error[512]={0};FILE *file=NULL;
    nya_train_decoder *decoder=NULL;nya_llm_context *model=NULL;
    nya_llm_session *reference=NULL,*candidate=NULL;
    float *expected=NULL,*actual=NULL;
    size_t count=cached?2:4,prefix=8,capacity=32;
    uint32_t tokens[32];
    nya_train_decoder_config cfg;nya_train_decoder_defaults(&cfg);
    cfg.embedding_length=(uint32_t)strtoul(argv[2],NULL,10)*2;
    cfg.feed_forward_length=32;cfg.block_count=2;cfg.head_count=2;cfg.kv_head_count=1;
    cfg.context_length=32;cfg.seed=713;
    decoder=nya_train_decoder_create(&cfg,error,sizeof(error));CHECK(decoder);
    file=fopen(argv[1],"wb");CHECK(file);
    CHECK(!nya_train_decoder_export(decoder,file,error,sizeof(error)));
    {int status=fclose(file);file=NULL;CHECK(!status);}
    nya_train_decoder_free(decoder);decoder=NULL;
    file=nya_file_open_read(argv[1]);CHECK(file);uint64_t bytes=0;
    CHECK(!nya_file_regular_size(file,&bytes));
    {int status=fclose(file);file=NULL;CHECK(!status);}
    CHECK(!nya_llm_load(argv[1],bytes,&model,error,sizeof(error)));
    if(!strcmp(argv[3],"1"))for(size_t i=0;i<model->block_count;++i)model->layers[i].sliding_window=3;
    size_t vocabulary=model->tokenizer.vocabulary_size;
    CHECK(vocabulary && vocabulary<=UINT32_MAX && vocabulary<(SIZE_MAX/sizeof(float)-2)/4);
    expected=malloc(count*vocabulary*sizeof(float));actual=malloc((4*vocabulary+2)*sizeof(float));
    CHECK(expected && actual);
    for(size_t i=0;i<capacity;++i)tokens[i]=(uint32_t)((20+i%128)%vocabulary);
    CHECK(!setting("NYA_TEST_GRAPH_FAULT",""));
    CHECK(!setting("NYA_CUDA_RING_KV",argv[3]));
    CHECK(!setting("NYA_CUDA_BATCH_GRAPHS","0"));CHECK(!setting("NYA_CUDA_GRAPHS","0"));
    reference=nya_llm_session_create(model,capacity,error,sizeof(error));CHECK(reference);
    CHECK(!strcmp(nya_llm_session_execution(reference),"resident"));
    CHECK(!nya_llm_session_run(reference,tokens,prefix,0));
    for(size_t i=0;i<count;++i){
        CHECK(!nya_llm_session_run(reference,tokens+prefix+i,1,1));
        memcpy(expected+i*vocabulary,nya_llm_session_logits(reference),vocabulary*sizeof(float));
    }
    CHECK(!setting("NYA_CUDA_BATCH_GRAPHS","1"));CHECK(!setting("NYA_CUDA_GRAPHS","1"));
    candidate=nya_llm_session_create(model,capacity,error,sizeof(error));CHECK(candidate);
    CHECK(!strcmp(nya_llm_session_execution(candidate),"resident"));
    CHECK(!nya_llm_session_run(candidate,tokens,prefix,0));
    nya_compute_stats before,after;
    nya_llm_session_stats(candidate,&before);
    CHECK(!nya_llm_session_verify(candidate,tokens+prefix,2,actual+1,2*vocabulary));
    nya_llm_session_stats(candidate,&after);
    CHECK(after.graph_captures-before.graph_captures==1 && after.graph_replays-before.graph_replays==1);
    CHECK(!nya_llm_session_rewind(candidate,prefix));
    for(size_t i=0;i<count*vocabulary+2;++i)actual[i]=12345;
    CHECK(!setting("NYA_TEST_GRAPH_FAULT",argv[4]));
    CHECK(nya_llm_session_verify(candidate,tokens+prefix,count,actual+1,count*vocabulary)<0);
    CHECK(!setting("NYA_TEST_GRAPH_FAULT",""));
    CHECK(actual[0]==12345 && actual[count*vocabulary+1]==12345);
    CHECK(!nya_llm_session_logits(candidate));
    CHECK(nya_llm_session_rewind(candidate,0)<0);
    CHECK(nya_llm_session_verify(candidate,tokens+prefix,count,actual+1,count*vocabulary)<0);
    nya_llm_session_reset(candidate);
    CHECK(!strcmp(nya_llm_session_execution(candidate),"cpu-reference"));
    CHECK(!nya_llm_session_run(candidate,tokens,prefix,0));
    CHECK(!nya_llm_session_verify(candidate,tokens+prefix,count,actual+1,count*vocabulary));
    CHECK(parity(actual+1,expected,count*vocabulary));
    nya_llm_session_free(candidate);candidate=NULL;
    candidate=nya_llm_session_create(model,capacity,error,sizeof(error));CHECK(candidate);
    CHECK(!strcmp(nya_llm_session_execution(candidate),"resident"));
    CHECK(!nya_llm_session_run(candidate,tokens,prefix,0));
    CHECK(!nya_llm_session_verify(candidate,tokens+prefix,count,actual+1,count*vocabulary));
    CHECK(parity(actual+1,expected,count*vocabulary));
    CHECK(actual[0]==12345 && actual[count*vocabulary+1]==12345);
    printf("recovery=1 site=%s cache=%s width=%s ring=%s\n",argv[4],argv[5],argv[2],argv[3]);
    result=0;
done:
    setting("NYA_TEST_GRAPH_FAULT","");
    if(result)fprintf(stderr,"%s\n",error);
    if(file)fclose(file);
    nya_train_decoder_free(decoder);nya_llm_session_free(candidate);nya_llm_session_free(reference);
    nya_llm_free(model);free(expected);free(actual);return result;
}
