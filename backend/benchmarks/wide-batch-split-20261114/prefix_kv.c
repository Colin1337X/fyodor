/* Diagnostic only: compare batched and fully serial prefixes in one plan. */
#include "llm_internal.h"
#include "file.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);goto done;}} while(0)
static double difference(const float *a,const float *b,size_t count,double *absolute)
{
    double maximum=0;*absolute=0;
    for(size_t i=0;i<count;++i) {
        if(a[i]==-INFINITY && b[i]==-INFINITY)continue;
        if(!isfinite(a[i]) || !isfinite(b[i]))return INFINITY;
        double delta=fabs((double)a[i]-b[i]);
        if(delta>*absolute)*absolute=delta;
        delta/=1+fabs((double)b[i]);if(delta>maximum)maximum=delta;
    }
    return maximum;
}
int main(int argc,char **argv)
{
    if(argc!=2 && argc!=3)return 2;
    size_t prefix=8192;
    if(argc==3) {
        if(!argv[2][0])return 2;
        for(const char *p=argv[2];*p;++p)if(*p<'0' || *p>'9')return 2;
        char *end=NULL;errno=0;
        unsigned long long parsed=strtoull(argv[2],&end,10);
        if(errno || *end || parsed<3 || parsed>262142)return 2;
        prefix=(size_t)parsed;
    }
    int result=1;char error[512]={0};uint64_t bytes=0;
    nya_llm_context *m=NULL;nya_llm_session *s=NULL;
    uint32_t *tokens=NULL;float *snapshot=NULL,*scratch=NULL,*logits=NULL;size_t *offsets=NULL;
    FILE *file=nya_file_open_read(argv[1]);if(!file)return 2;
    int bad=nya_file_regular_size(file,&bytes);fclose(file);if(bad)return 2;
    CHECK(!nya_llm_load(argv[1],bytes,&m,error,sizeof(error)));
    const size_t positions[]={0,prefix/2,prefix-1};
    CHECK(m->context_length>=prefix+2 && m->block_count<SIZE_MAX/sizeof(size_t)-1);
    size_t vocabulary=m->tokenizer.vocabulary_size,total=0,maximum_width=0;
    CHECK(vocabulary && vocabulary<=UINT32_MAX && vocabulary<=SIZE_MAX/sizeof(float));
    offsets=calloc(m->block_count+1,sizeof(*offsets));CHECK(offsets);
    for(size_t layer=0;layer<m->block_count;++layer) {
        const nya_llm_layer *w=&m->layers[layer];
        CHECK(w->head_dimension && w->kv_head_count<=SIZE_MAX/w->head_dimension);
        size_t width=w->head_dimension*w->kv_head_count;
        CHECK(width && width<=(SIZE_MAX-total)/6);
        offsets[layer]=total;total+=6*width;
        if(width>maximum_width)maximum_width=width;
    }
    CHECK(total<=SIZE_MAX/sizeof(float));
    snapshot=malloc(total*sizeof(float));scratch=malloc(2*maximum_width*sizeof(float));
    logits=malloc(vocabulary*sizeof(float));tokens=malloc(prefix*sizeof(*tokens));
    CHECK(snapshot && scratch && logits && tokens);
    for(size_t i=0;i<prefix;++i)tokens[i]=(uint32_t)((20+i%128)%vocabulary);
    s=nya_llm_session_create(m,prefix+2,error,sizeof(error));CHECK(s);
    CHECK(!strcmp(nya_llm_session_execution(s),"resident"));
    CHECK(!nya_llm_session_run(s,tokens,prefix,0));
    memcpy(logits,nya_llm_session_logits(s),vocabulary*sizeof(float));
    for(size_t layer=0;layer<m->block_count;++layer) {
        size_t width=m->layers[layer].head_dimension*m->layers[layer].kv_head_count;
        for(size_t sample=0;sample<3;++sample) {
            float *keys=snapshot+offsets[layer]+sample*2*width;
            CHECK(!nya_llm_session_read_kv(s,layer,positions[sample],1,keys,keys+width,width));
        }
    }
    puts("Batched prefix snapshots complete");fflush(stdout);
    nya_llm_session_reset(s);CHECK(!nya_llm_session_run(s,tokens,prefix,1));
    double absolute=0,scaled=difference(logits,nya_llm_session_logits(s),vocabulary,&absolute);
    printf("{\"kind\":\"logits\",\"max_scaled\":%.12g,\"max_absolute\":%.12g,\"within_0_0005\":%s}\n",scaled,absolute,scaled<=.0005?"true":"false");
    CHECK(isfinite(scaled));
    for(size_t layer=0;layer<m->block_count;++layer) {
        size_t width=m->layers[layer].head_dimension*m->layers[layer].kv_head_count;
        for(size_t sample=0;sample<3;++sample) {
            CHECK(!nya_llm_session_read_kv(s,layer,positions[sample],1,scratch,scratch+width,width));
            float *keys=snapshot+offsets[layer]+sample*2*width;
            double key_abs=0,value_abs=0;
            double key_error=difference(keys,scratch,width,&key_abs);
            double value_error=difference(keys+width,scratch+width,width,&value_abs);
            CHECK(isfinite(key_error) && isfinite(value_error));
            printf("{\"kind\":\"kv\",\"layer\":%zu,\"position\":%zu,\"key_scaled\":%.12g,\"value_scaled\":%.12g,\"key_absolute\":%.12g,\"value_absolute\":%.12g,\"exact\":%s}\n",layer,positions[sample],key_error,value_error,key_abs,value_abs,memcmp(keys,scratch,2*width*sizeof(float))?"false":"true");
        }
    }
    result=0;
done:
    if(result)fprintf(stderr,"%s\n",error);
    nya_llm_session_free(s);nya_llm_free(m);
    free(offsets);free(snapshot);free(scratch);free(logits);free(tokens);return result;
}
