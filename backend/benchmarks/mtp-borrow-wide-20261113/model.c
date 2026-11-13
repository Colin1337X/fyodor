#include "llm_internal.h"
#include "file.h"
#include "train_clock.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);goto done;} } while(0)
int main(int argc,char **argv)
{
    if(argc<2||argc>3)return 2;
    size_t prefix=argc==3?(size_t)strtoul(argv[2],NULL,10):2049;
    if(prefix<258||prefix>8193)return 2;
    int result=1;char error[512]={0};uint64_t bytes=0;
    nya_llm_context *model=NULL;nya_compute_plan *plan=NULL;
    float *keys=NULL,*values=NULL,*query=NULL,*reference=NULL,*actual=NULL,*logits=NULL,*hidden=NULL,*after=NULL;
    double *scores=NULL;uint32_t *tokens=NULL;
    FILE *file=nya_file_open_read(argv[1]);if(!file)return 2;
    int bad=nya_file_regular_size(file,&bytes);fclose(file);if(bad)return 2;
    CHECK(!nya_llm_load(argv[1],bytes,&model,error,sizeof(error)));
    CHECK(!strcmp(nya_compute_name(model->compute),"cuda"));
    size_t layers[2]={SIZE_MAX,SIZE_MAX};
    for(size_t i=0;i<model->block_count;++i)if(model->layers[i].kv_source==i){
        const nya_llm_layer *w=&model->layers[i];
        if(w->head_dimension==256 && w->sliding_window)layers[0]=i;
        if(w->head_dimension==512 && !w->sliding_window)layers[1]=i;
    }
    CHECK(layers[0]!=SIZE_MAX && layers[1]!=SIZE_MAX);
    tokens=malloc(prefix*sizeof(*tokens));logits=malloc(model->tokenizer.vocabulary_size*sizeof(float));
    hidden=malloc(model->embedding_length*sizeof(float));after=malloc(model->embedding_length*sizeof(float));
    CHECK(tokens&&logits&&hidden&&after);
    for(size_t i=0;i<prefix;++i)tokens[i]=(uint32_t)((20+i)%model->tokenizer.vocabulary_size);
    plan=nya_compute_plan_create(model->compute,model,prefix+4);CHECK(plan && nya_compute_plan_can_borrow(plan));
    CHECK(!nya_compute_plan_prefill(plan,tokens,prefix,0,logits));
    CHECK(!nya_compute_plan_hidden(plan,hidden,model->embedding_length));
    size_t frozen=prefix-1;
    for(size_t kind=0;kind<2;++kind){
        const nya_llm_layer *w=&model->layers[layers[kind]];
        size_t width=w->head_dimension,heads=w->head_count,kw=width*w->kv_head_count,qw=heads*width;
        keys=malloc(frozen*kw*sizeof(float));values=malloc(frozen*kw*sizeof(float));scores=malloc(frozen*sizeof(double));
        query=malloc(qw*sizeof(float));reference=malloc(qw*sizeof(float));actual=malloc(qw*sizeof(float));
        CHECK(keys&&values&&scores&&query&&reference&&actual);
        for(size_t i=0;i<qw;++i)query[i]=(float)((int)(i%17)-8)/50.0f;
        for(size_t narrow=0;narrow<2;++narrow){
            size_t window=narrow?257:w->sliding_window,span=window&&window<frozen?window:frozen;
            CHECK(!nya_compute_plan_read_kv(plan,layers[kind],frozen-span,span,keys,values,span*kw));
            CHECK(!_putenv_s("NYA_CUDA_MTP_WIDE_VALUE","0"));
            CHECK(!nya_compute_plan_borrow_attention(plan,layers[kind],frozen,query,heads,window,reference,qw));
            CHECK(!_putenv_s("NYA_CUDA_MTP_WIDE_VALUE","1"));
            nya_compute_stats before,stats;nya_compute_plan_stats(plan,&before);
            CHECK(!nya_compute_plan_borrow_attention(plan,layers[kind],frozen,query,heads,window,actual,qw));
            nya_compute_plan_stats(plan,&stats);
            CHECK(stats.uploads==before.uploads+1 && stats.downloads==before.downloads+1 && stats.kv_bytes==before.kv_bytes && stats.scratch_bytes==before.scratch_bytes);
            CHECK(!memcmp(reference,actual,qw*sizeof(float)));
            double max_error=0;
            for(size_t h=0;h<heads;++h){
                size_t kh=h/(heads/w->kv_head_count);double maximum=-INFINITY,den=0;
                for(size_t t=0;t<span;++t){double dot=0;for(size_t j=0;j<width;++j)dot+=(double)query[h*width+j]*keys[t*kw+kh*width+j];scores[t]=dot;if(dot>maximum)maximum=dot;}
                for(size_t t=0;t<span;++t){scores[t]=exp(scores[t]-maximum);den+=scores[t];}
                for(size_t j=0;j<width;++j){double want=0;for(size_t t=0;t<span;++t)want+=scores[t]/den*values[t*kw+kh*width+j];
                    double e=fabs(actual[h*width+j]-want)/(1+fabs(want));CHECK(isfinite(actual[h*width+j])&&e<=.0005);if(e>max_error)max_error=e;}
            }
            printf("oracle width=%zu prefix=%zu window=%zu exact=1 scaled_error=%.9g\n",width,frozen,window,max_error);fflush(stdout);
            for(int order=0;order<2;++order)for(int step=0;step<2;++step){
                int mode=order?1-step:step;CHECK(!_putenv_s("NYA_CUDA_MTP_WIDE_VALUE",mode?"1":"0"));
                for(int warm=0;warm<3;++warm)CHECK(!nya_compute_plan_borrow_attention(plan,layers[kind],frozen,query,heads,window,actual,qw));
                for(int repeat=0;repeat<10;++repeat){double start=tr_seconds();CHECK(!nya_compute_plan_borrow_attention(plan,layers[kind],frozen,query,heads,window,actual,qw));
                    double ms=(tr_seconds()-start)*1000;CHECK(!memcmp(reference,actual,qw*sizeof(float)));
                    printf("timing width=%zu prefix=%zu window=%zu order=%d mode=%d repeat=%d ms=%.9f\n",width,frozen,window,order,mode,repeat,ms);}
                fflush(stdout);
            }
        }
        free(keys);keys=NULL;free(values);values=NULL;free(scores);scores=NULL;free(query);query=NULL;free(reference);reference=NULL;free(actual);actual=NULL;
    }
    CHECK(!nya_compute_plan_hidden(plan,after,model->embedding_length));CHECK(!memcmp(hidden,after,model->embedding_length*sizeof(float)));
    CHECK(!nya_compute_plan_rewind(plan,frozen));CHECK(nya_compute_plan_hidden(plan,after,model->embedding_length)<0);
    result=0;
done:
    if(result)fprintf(stderr,"%s\n",error);
    nya_compute_plan_free(plan);nya_llm_free(model);free(tokens);free(logits);free(hidden);free(after);
    free(keys);free(values);free(scores);free(query);free(reference);free(actual);return result;
}
