#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "llm_internal.h"
#include "pretraining.h"
#include "file.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int mode(const char *s)
{
#ifdef _WIN32
    return _putenv_s("NYA_KV_TYPE",s);
#else
    return setenv("NYA_KV_TYPE",s,1);
#endif
}
int main(int argc,char **argv)
{
    const char *path=argc>1?argv[1]:"turboquant-fixture.gguf";
    char error[512]={0};int result=1;FILE *f=NULL;
    nya_train_decoder *d=NULL;nya_llm_context *m=NULL;
    nya_llm_session *ref=NULL,*s=NULL;
    float *expected=NULL,*f32=NULL;uint64_t bytes=0;
    if (argc==1) {
        nya_train_decoder_config cfg;nya_train_decoder_defaults(&cfg);
        cfg.embedding_length=128;cfg.feed_forward_length=256;cfg.head_count=2;
        cfg.kv_head_count=1;cfg.block_count=2;cfg.context_length=64;
        d=nya_train_decoder_create(&cfg,error,sizeof(error));if (!d) goto done;
        f=fopen(path,"wb");if (!f || nya_train_decoder_export(d,f,error,sizeof(error))) goto done;
        fclose(f);f=NULL;
    }
    f=fopen(path,"rb");if (!f || nya_file_regular_size(f,&bytes)) goto done;
    fclose(f);f=NULL;
    if (nya_llm_load(path,bytes,&m,error,sizeof(error))) goto done;
    nya_compute_free(m->compute);m->compute=nya_compute_create_for("cpu");
    if (!m->compute) goto done;
    size_t n=m->tokenizer.vocabulary_size;
    expected=malloc(n*sizeof(float));f32=malloc(n*sizeof(float));if (!expected || !f32) goto done;
    uint32_t tokens[12];for (size_t i=0;i<12;++i) tokens[i]=(uint32_t)((17+i*13)%n);
    for (unsigned scenario=0;scenario<(argc==1?3U:1U);++scenario) {
    if (argc==1) {
        m->sliding_rope_base=m->rope_frequency_base;
        for (size_t i=0;i<m->block_count;++i) m->layers[i].sliding_window=scenario==1?5U:0U;
        if (scenario==2) {m->layers[1].kv_source=0;m->layers[1].cache_offset=m->layers[0].cache_offset;}
    }
    if (mode("f32")) goto done;
    ref=nya_llm_session_create(m,12,error,sizeof(error));
    if (!ref || nya_llm_session_run(ref,tokens,12,1)) goto done;
    memcpy(f32,nya_llm_session_logits(ref),n*sizeof(float));
    nya_llm_session_free(ref);ref=NULL;
    for (unsigned bits=3;bits<=4;++bits) {
        const char *setting=bits==3?"turboquant3":"turboquant4";
        if (mode(setting)) goto done;
        s=nya_llm_session_create(m,12,error,sizeof(error));if (!s) goto done;
        if (strcmp(nya_llm_session_kv_type(s),setting) || strcmp(nya_llm_session_execution(s),"cpu-turboquant-reference")) goto done;
        if (nya_llm_session_run(s,tokens,12,0)) goto done;
        memcpy(expected,nya_llm_session_logits(s),n*sizeof(float));
        double squared=0,maximum=0;
        for (size_t i=0;i<n;++i) {
            if (expected[i]==f32[i]) continue; /* identical masked -inf logits */
            if (!isfinite(expected[i]) || !isfinite(f32[i])) goto done;
            double delta=(double)expected[i]-f32[i];squared+=delta*delta;maximum=fmax(maximum,fabs(delta));
        }
        nya_compute_stats stats;nya_llm_session_stats(s,&stats);
        printf("scenario=%u %s logit_rmse=%.9g max_abs=%.9g host_cache_including_transforms=%zu\n",scenario,setting,sqrt(squared/(double)n),maximum,stats.kv_bytes);
        /* Quality diagnostic, NOT F32-equivalence acceptance. Integration must
         * reproduce precisely across reset, batched input and single steps. */
        nya_llm_session_reset(s);
        if (nya_llm_session_logits(s)) goto done;
        for (size_t i=0;i<12;++i) if (nya_llm_session_run(s,tokens+i,1,1)) goto done;
        if (memcmp(expected,nya_llm_session_logits(s),n*sizeof(float))) goto done;
        size_t width=m->layers[0].head_dimension*m->layers[0].kv_head_count;
        float *kv=malloc(2*width*sizeof(float));if (!kv) goto done;
        int bad=nya_llm_session_read_kv(s,0,11,1,kv,kv+width,width);
        for (size_t i=0;i<2*width;++i) if (!isfinite(kv[i])) bad=1;
        if (!nya_llm_session_read_kv(s,0,12,1,kv,kv+width,width)) bad=1;
        free(kv);if (bad) goto done;
        nya_llm_session_free(s);s=NULL;
    }
    }
    if (mode("invalid")) goto done;
    s=nya_llm_session_create(m,12,error,sizeof(error));if (s) goto done;
    result=0;
done:
    if (result) fprintf(stderr,"TurboQuant inference failed: %s\n",error);
    if (f) fclose(f);
    nya_llm_session_free(s);nya_llm_session_free(ref);nya_llm_free(m);
    nya_train_decoder_free(d);free(expected);free(f32);
    if (argc==1) remove(path);
    mode("f32");return result;
}
