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
#define CHECK(x) do {if(!(x)) {fprintf(stderr,"line %d: %s\n",__LINE__,#x);goto done;}} while(0)
static int inject_failure(void)
{
#ifdef _WIN32
    return _putenv_s("NYA_CUDA_FAIL_AFTER","15");
#else
    return setenv("NYA_CUDA_FAIL_AFTER","15",1);
#endif
}
static int batch_logits_setting(const char *value)
{
#ifdef _WIN32
    return _putenv_s("NYA_CUDA_BATCH_LOGITS",value);
#else
    return setenv("NYA_CUDA_BATCH_LOGITS",value,1);
#endif
}
static int interleaved_setting(const char *name,const char *value)
{
#ifdef _WIN32
    return _putenv_s(name,value);
#else
    return setenv(name,value,1);
#endif
}
int main(int argc,char **argv)
{
    if(argc<2 || argc>3) return 2;
    char *end=NULL;unsigned long parsed=argc==3?strtoul(argv[2],&end,10):7;
    if(parsed>8192 || (argc==3 && (!*argv[2] || *end))) return 2;
    size_t prefix=(size_t)parsed;
    int test_split=getenv("NYA_TEST_VERIFY_BATCH_SPLIT")!=NULL;
    size_t count=2,total=prefix+count+1;int result=1;
    char fixture[1200];const char *path=argv[1];
    const char *wide=getenv("NYA_TEST_VERIFY_WIDE");
    if(wide && strcmp(wide,"256") && strcmp(wide,"512"))return 2;
    if(getenv("NYA_TEST_VERIFY_SHARED") || wide) {
        if(strlen(path)>1000) return 2;
        snprintf(fixture,sizeof(fixture),"%s.verify-%s.gguf",path,wide?wide:"shared");path=fixture;
        nya_train_decoder_config cfg;nya_train_decoder_defaults(&cfg);
        cfg.embedding_length=16;cfg.feed_forward_length=32;cfg.block_count=2;
        cfg.head_count=2;cfg.kv_head_count=1;cfg.context_length=32;cfg.seed=123;
        if(wide) {cfg.embedding_length=(uint32_t)strtoul(wide,NULL,10)*2;cfg.context_length=total>512?(uint32_t)total:512;}
        char message[512]={0};nya_train_decoder *decoder=nya_train_decoder_create(&cfg,message,sizeof(message));
        if(!decoder) return 2;
        FILE *file=fopen(path,"wb");int failed=file?nya_train_decoder_export(decoder,file,message,sizeof(message)):-1;
        if(file && fclose(file)) failed=1;
        nya_train_decoder_free(decoder);if(failed)return 2;
    }
    FILE *f=nya_file_open_read(path);uint64_t bytes=0;if(!f)return 2;
    int bad=nya_file_regular_size(f,&bytes);fclose(f);if(bad)return 2;
    nya_llm_context *m=NULL;nya_llm_session *s=NULL;char error[512]={0};
    float *reference=NULL,*actual=NULL,*last=NULL;uint32_t *tokens=NULL;
    CHECK(!nya_llm_load(path,bytes,&m,error,sizeof(error)));
    if(getenv("NYA_TEST_VERIFY_RING")) for(size_t i=0;i<m->block_count;++i) m->layers[i].sliding_window=test_split?257:3;
    if(getenv("NYA_TEST_VERIFY_SHARED")) {CHECK(m->block_count>=2);m->layers[1].kv_source=0;m->layers[1].cache_offset=m->layers[0].cache_offset;}
    size_t v=m->tokenizer.vocabulary_size;CHECK(total<=m->context_length && v && v<=UINT32_MAX && v<=(SIZE_MAX/sizeof(float)-2)/(count+2));
    reference=malloc((count+2)*v*sizeof(float));actual=malloc((count*v+2)*sizeof(float));last=malloc(v*sizeof(float));tokens=malloc(total*sizeof(*tokens));
    CHECK(reference && actual && last && tokens);
    for(size_t i=0;i<total;++i) tokens[i]=(uint32_t)((20+i%128)%v);
    int test_logits=getenv("NYA_TEST_VERIFY_LOGITS")!=NULL;
    int test_two=getenv("NYA_TEST_VERIFY_INTERLEAVED_TWO")!=NULL;
    int test_interleaved=test_two || getenv("NYA_TEST_VERIFY_INTERLEAVED")!=NULL;
    const char *interleaved_name=test_two?"NYA_CUDA_INTERLEAVED_TWO":"NYA_CUDA_INTERLEAVED_REUSE";
    if(test_interleaved)CHECK(!interleaved_setting(interleaved_name,"0"));
    if(test_split)CHECK(!interleaved_setting("NYA_CUDA_WIDE_BATCH_SPLIT","0"));
    nya_compute_stats reference_stats;
    if(test_logits) CHECK(!batch_logits_setting("0"));
    s=nya_llm_session_create(m,total,error,sizeof(error));CHECK(s);
    nya_llm_session_stats(s,&reference_stats);
    if(prefix) CHECK(!nya_llm_session_run(s,tokens,prefix,0));
    for(size_t i=0;i<=count;++i) {CHECK(!nya_llm_session_run(s,tokens+prefix+i,1,1));memcpy(reference+i*v,nya_llm_session_logits(s),v*sizeof(float));}
    uint32_t replacement=(tokens[total-1]+1)%(uint32_t)v;
    nya_llm_session_reset(s);
    CHECK(!nya_llm_session_run(s,tokens,total-1,1));
    CHECK(!nya_llm_session_run(s,&replacement,1,1));
    memcpy(reference+(count+1)*v,nya_llm_session_logits(s),v*sizeof(float));
    nya_llm_session_free(s);s=NULL;
    if(test_logits) CHECK(!batch_logits_setting("1"));
    if(test_interleaved)CHECK(!interleaved_setting(interleaved_name,"1"));
    if(test_split)CHECK(!interleaved_setting("NYA_CUDA_WIDE_BATCH_SPLIT","1"));
    if(getenv("NYA_TEST_VERIFY_FAILURE")) CHECK(!inject_failure());
    s=nya_llm_session_create(m,total,error,sizeof(error));CHECK(s);
    if(getenv("NYA_TEST_VERIFY_RESIDENT")) CHECK(!strcmp(nya_llm_session_execution(s),"resident"));
    if(test_interleaved) {
        nya_compute_stats candidate_stats;nya_llm_session_stats(s,&candidate_stats);
        size_t columns=m->embedding_length>m->maximum_hidden_length?m->embedding_length:m->maximum_hidden_length;
        if(m->maximum_query_length>columns)columns=m->maximum_query_length;
        size_t width=test_two?2:4;
        size_t extra=candidate_stats.prefill_batch>=width?(width*columns+3)*sizeof(float):0;
        const char *four=getenv("NYA_CUDA_INTERLEAVED_REUSE");
        if(test_two && four && !strcmp(four,"1") && candidate_stats.prefill_batch>=4)extra=0;
        CHECK(candidate_stats.prefill_batch==reference_stats.prefill_batch);
        CHECK(candidate_stats.scratch_bytes==reference_stats.scratch_bytes+extra);
        CHECK(candidate_stats.kv_bytes==reference_stats.kv_bytes);
        CHECK(candidate_stats.weights_bytes==reference_stats.weights_bytes);
    }
    if(test_split) {
        nya_compute_stats candidate_stats;nya_llm_session_stats(s,&candidate_stats);
        size_t partials=0;
        for(size_t i=0;i<m->block_count;++i) {
            const nya_llm_layer *layer=&m->layers[i];
            if((layer->head_dimension==256 || layer->head_dimension==512) &&
                (!layer->sliding_window || layer->sliding_window>256)) {
                size_t rows=layer->head_count*(layer->head_dimension+2)*4;
                if(rows>partials)partials=rows;
            }
        }
        size_t factor=candidate_stats.prefill_batch>=4?4:candidate_stats.prefill_batch>=2?2:1;
        CHECK(candidate_stats.prefill_batch==reference_stats.prefill_batch);
        CHECK(candidate_stats.scratch_bytes==reference_stats.scratch_bytes+(factor-1)*partials*sizeof(float));
        CHECK(candidate_stats.kv_bytes==reference_stats.kv_bytes);
        CHECK(candidate_stats.weights_bytes==reference_stats.weights_bytes);
    }
    if(test_logits) {
        nya_compute_stats candidate_stats;nya_llm_session_stats(s,&candidate_stats);
        CHECK(candidate_stats.prefill_batch==reference_stats.prefill_batch);
        size_t rows=candidate_stats.prefill_batch<4?candidate_stats.prefill_batch:4;
        CHECK(rows && candidate_stats.scratch_bytes==reference_stats.scratch_bytes+(rows-1)*v*sizeof(float));
        CHECK(candidate_stats.kv_bytes==reference_stats.kv_bytes);
        CHECK(candidate_stats.weights_bytes==reference_stats.weights_bytes);
    }
    if(getenv("NYA_TEST_VERIFY_FAILURE")) {
        CHECK(!nya_llm_session_run(s,tokens,prefix,0));
        CHECK(nya_llm_session_verify(s,tokens+prefix,count,actual+1,count*v)<0);
        CHECK(!nya_llm_session_logits(s));
        CHECK(nya_llm_session_rewind(s,0)<0);
        CHECK(nya_llm_session_verify(s,tokens+prefix,1,actual+1,v)<0);
        nya_llm_session_reset(s);
        CHECK(!strcmp(nya_llm_session_execution(s),"cpu-reference"));
    }
    for(unsigned reset=0;reset<2;++reset) {
        nya_llm_session_reset(s);
        if(prefix) CHECK(!nya_llm_session_run(s,tokens,prefix,0));
        if(prefix) memcpy(last,nya_llm_session_logits(s),v*sizeof(float));
        actual[0]=12345;actual[count*v+1]=12345;
        CHECK(nya_llm_session_verify(s,tokens+prefix,count,actual+1,count*v-1)<0);
        CHECK(nya_llm_session_verify(s,tokens+prefix,0,actual+1,count*v)<0);
        CHECK(nya_llm_session_verify(s,tokens+prefix,SIZE_MAX,actual+1,SIZE_MAX)<0);
        CHECK(nya_llm_session_verify(s,NULL,count,actual+1,count*v)<0);
        CHECK(nya_llm_session_verify(s,tokens+prefix,count,NULL,count*v)<0);
        uint32_t saved=tokens[prefix+count-1];tokens[prefix+count-1]=(uint32_t)v;
        CHECK(nya_llm_session_verify(s,tokens+prefix,count,actual+1,count*v)<0);tokens[prefix+count-1]=saved;
        if(prefix) CHECK(!memcmp(last,nya_llm_session_logits(s),v*sizeof(float)));
        nya_compute_stats before,after;nya_llm_session_stats(s,&before);
        CHECK(!nya_llm_session_verify(s,tokens+prefix,count,actual+1,count*v));
        nya_llm_session_stats(s,&after);
        if(!strcmp(nya_llm_session_execution(s),"resident")) {
            CHECK(after.downloads-before.downloads==count);
            if(getenv("NYA_TEST_VERIFY_SERIAL"))CHECK(after.prefill_batch==1);
            else CHECK(after.prefill_batch>1);
        }
        CHECK(actual[0]==12345 && actual[count*v+1]==12345);
        double maximum=0;
        for(size_t i=0;i<count*v;++i) {
            float a=actual[i+1],b=reference[i];
            if(a==-INFINITY && b==-INFINITY) continue;
            CHECK(isfinite(a) && isfinite(b));
            double delta=fabs((double)a-b)/(1+fabs(b));if(delta>maximum)maximum=delta;
        }
        printf("reset=%u prefix=%zu tokens=%zu max_scaled=%.9g execution=%s\n",reset,prefix,count,maximum,nya_llm_session_execution(s));fflush(stdout);
        CHECK(maximum<=0.0005);
        CHECK(!memcmp(nya_llm_session_logits(s),actual+1+(count-1)*v,v*sizeof(float)));
        /* Rejected last proposal: truncate, forbid stale suffix reads/logits,
           then regenerate it and compare against the independent serial run. */
        CHECK(nya_llm_session_rewind(s,SIZE_MAX)<0);
        CHECK(!memcmp(nya_llm_session_logits(s),actual+1+(count-1)*v,v*sizeof(float)));
        CHECK(!nya_llm_session_rewind(s,prefix+count));
        CHECK(nya_llm_session_logits(s)); /* no-op retains the current logits */
        CHECK(!nya_llm_session_rewind(s,prefix+count-1));
        CHECK(!nya_llm_session_logits(s));
        CHECK(nya_llm_session_rewind(s,prefix+count)<0); /* cannot resurrect suffix */
        size_t width=m->layers[0].head_dimension*m->layers[0].kv_head_count;
        CHECK(nya_llm_session_read_kv(s,0,prefix+count-1,1,actual+1,last,width)<0);
        if(getenv("NYA_TEST_VERIFY_RING") && !strcmp(nya_llm_session_execution(s),"resident")) {
            CHECK(nya_llm_session_rewind(s,1)<0); /* evicted prefix */
        }
        CHECK(!nya_llm_session_verify(s,tokens+prefix+count-1,1,actual+1,v));
        for(size_t i=0;i<v;++i) {
            float a=actual[i+1],b=reference[(count-1)*v+i];
            CHECK((a==-INFINITY && b==-INFINITY) || (isfinite(a) && isfinite(b) && fabs((double)a-b)<=0.0005*(1+fabs(b))));
        }
        CHECK(!nya_llm_session_verify(s,tokens+prefix+count,1,actual+1,v));
        for(size_t i=0;i<v;++i) {
            float a=actual[i+1],b=reference[count*v+i];
            CHECK((a==-INFINITY && b==-INFINITY) || (isfinite(a) && isfinite(b) && fabs((double)a-b)<=0.0005*(1+fabs(b))));
        }
        CHECK(!nya_llm_session_rewind(s,total-1));
        CHECK(!nya_llm_session_verify(s,&replacement,1,actual+1,v));
        for(size_t i=0;i<v;++i) {
            float a=actual[i+1],b=reference[(count+1)*v+i];
            CHECK((a==-INFINITY && b==-INFINITY) || (isfinite(a) && isfinite(b) && fabs((double)a-b)<=0.0005*(1+fabs(b))));
        }
        CHECK(!nya_llm_session_rewind(s,0));
        CHECK(!nya_llm_session_logits(s));
        CHECK(nya_llm_session_rewind(s,1)<0);
    }
    result=0;
done:
    if(result) fprintf(stderr,"%s\n",error);
    nya_llm_session_free(s);nya_llm_free(m);free(reference);free(actual);free(last);free(tokens);return result;
}
