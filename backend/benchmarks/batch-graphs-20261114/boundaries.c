/* Original Fyodor graph lifecycle test. Small F32 fixtures complement, rather
   than replace, Gemma quantized generation and full-model memory checks. */
#include "llm_internal.h"
#include "file.h"
#include "pretraining.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)) {fprintf(stderr,"line %d: %s\n",__LINE__,#x);goto done;} } while(0)
static int setting(const char *name,const char *value)
{
#ifdef _WIN32
    return _putenv_s(name,value);
#else
    return setenv(name,value,1);
#endif
}
int main(int argc,char **argv)
{
    if(argc!=4 || (strcmp(argv[2],"256") && strcmp(argv[2],"512")) ||
       (strcmp(argv[3],"0") && strcmp(argv[3],"1"))) return 2;
    int result=1;char error[512]={0};FILE *file=NULL;
    nya_train_decoder *decoder=NULL;nya_llm_context *model=NULL;
    nya_llm_session *reference=NULL,*candidate=NULL;
    float *expected=NULL,*actual=NULL;uint32_t *tokens=NULL;
    const size_t capacity=8208;
    nya_train_decoder_config cfg;nya_train_decoder_defaults(&cfg);
    cfg.embedding_length=(uint32_t)strtoul(argv[2],NULL,10)*2;
    cfg.feed_forward_length=32;cfg.block_count=2;cfg.head_count=2;cfg.kv_head_count=1;
    cfg.context_length=(uint32_t)capacity;cfg.seed=713;
    decoder=nya_train_decoder_create(&cfg,error,sizeof(error));CHECK(decoder);
    file=fopen(argv[1],"wb");CHECK(file);
    CHECK(!nya_train_decoder_export(decoder,file,error,sizeof(error)));
    {int status=fclose(file);file=NULL;CHECK(!status);}
    nya_train_decoder_free(decoder);decoder=NULL;
    file=nya_file_open_read(argv[1]);CHECK(file);
    uint64_t bytes=0;CHECK(!nya_file_regular_size(file,&bytes));
    {int status=fclose(file);file=NULL;CHECK(!status);}
    CHECK(!nya_llm_load(argv[1],bytes,&model,error,sizeof(error)));
    if(!strcmp(argv[3],"1")) for(size_t i=0;i<model->block_count;++i)
        model->layers[i].sliding_window=257;
    size_t vocabulary=model->tokenizer.vocabulary_size;
    CHECK(vocabulary && vocabulary<=UINT32_MAX && vocabulary<(SIZE_MAX/sizeof(float)-2)/4);
    expected=malloc(4*vocabulary*sizeof(float));actual=malloc((4*vocabulary+2)*sizeof(float));
    tokens=malloc(capacity*sizeof(*tokens));CHECK(expected && actual && tokens);
    for(size_t i=0;i<capacity;++i) tokens[i]=(uint32_t)((20+i%128)%vocabulary);
    CHECK(!setting("NYA_CUDA_RING_KV",argv[3]));
    CHECK(!setting("NYA_CUDA_BATCH_GRAPHS","0"));
    CHECK(!setting("NYA_CUDA_GRAPHS","0"));
    reference=nya_llm_session_create(model,capacity,error,sizeof(error));CHECK(reference);
    CHECK(!strcmp(nya_llm_session_execution(reference),"resident"));
    CHECK(!setting("NYA_CUDA_BATCH_GRAPHS","1"));
    CHECK(!setting("NYA_CUDA_GRAPHS","1"));
    candidate=nya_llm_session_create(model,capacity,error,sizeof(error));CHECK(candidate);
    CHECK(!strcmp(nya_llm_session_execution(candidate),"resident"));
    const size_t positions[]={250,254,256,258,8190,8192,8196};
    const size_t widths[]={4,2,2,4,2,4,2};
    unsigned seen[6]={0};size_t checked=0;double maximum=0;
    for(unsigned reset=0;reset<2;++reset) {
        nya_llm_session_reset(reference);nya_llm_session_reset(candidate);
        size_t cursor=0;
        for(size_t test=0;test<7;++test) {
            size_t position=positions[test],count=widths[test];
            if(position>cursor) {
                CHECK(!nya_llm_session_run(reference,tokens+cursor,position-cursor,0));
                CHECK(!nya_llm_session_run(candidate,tokens+cursor,position-cursor,0));
            }
            unsigned slot=(count==4?3U:0U)+(position>=8192?2U:position>=256?1U:0U);
            for(unsigned trial=0;trial<3;++trial) {
                if(trial) {
                    CHECK(!nya_llm_session_rewind(reference,position));
                    CHECK(!nya_llm_session_rewind(candidate,position));
                }
                for(size_t q=0;q<count;++q) {
                    tokens[position+q]=(uint32_t)((31+test*7+q*3+trial*11+reset*13)%vocabulary);
                    CHECK(!nya_llm_session_run(reference,tokens+position+q,1,1));
                    memcpy(expected+q*vocabulary,nya_llm_session_logits(reference),vocabulary*sizeof(float));
                }
                for(size_t i=0;i<count*vocabulary+2;++i) actual[i]=12345;
                nya_compute_stats before,after;nya_llm_session_stats(candidate,&before);
                CHECK(!nya_llm_session_verify(candidate,tokens+position,count,actual+1,count*vocabulary));
                nya_llm_session_stats(candidate,&after);
                CHECK(after.graph_replays-before.graph_replays==1);
                CHECK(after.graph_captures-before.graph_captures==(seen[slot]?0U:1U));
                CHECK(after.downloads-before.downloads==1);seen[slot]=1;
                CHECK(actual[0]==12345 && actual[count*vocabulary+1]==12345);
                for(size_t i=0;i<count*vocabulary;++i) {
                    double a=actual[i+1],b=expected[i];
                    if(a==-INFINITY && b==-INFINITY) continue;
                    CHECK(isfinite(a) && isfinite(b));
                    double scaled=fabs(a-b)/(1+fabs(b));CHECK(scaled<=0.0005);
                    if(scaled>maximum)maximum=scaled;
                }
                ++checked;
                printf("reset=%u position=%zu count=%zu trial=%u parity=1\n",reset,position,count,trial);
            }
            cursor=position+count;
        }
    }
    for(unsigned i=0;i<6;++i) CHECK(seen[i]);
    CHECK(checked==42);
    printf("boundary_cases=%zu max_scaled=%.9g head_width=%s ring=%s\n",checked,maximum,argv[2],argv[3]);
    result=0;
done:
    if(result)fprintf(stderr,"%s\n",error);
    if(file)fclose(file);
    nya_train_decoder_free(decoder);nya_llm_session_free(candidate);nya_llm_session_free(reference);
    nya_llm_free(model);free(expected);free(actual);free(tokens);return result;
}
