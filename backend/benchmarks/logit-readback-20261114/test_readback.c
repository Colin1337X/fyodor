#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CUDA_SUCCESS 0
typedef struct { struct {size_t vocabulary_size, suppressed_count; const unsigned *suppressed_tokens;} tokenizer; } nya_llm_context;
typedef struct { int (*stream_sync)(void *); int (*download)(float *, uintptr_t, size_t); void *stream; } nya_cuda_context;
typedef struct { nya_llm_context *model; nya_cuda_context *cuda; size_t logits_batch; uintptr_t logits;
    struct {size_t synchronizations, downloads;} stats; } nya_cuda_plan;
static unsigned sync_calls, copy_calls;
static int sync_error, copy_error;
static size_t copied;
static int sync_stub(void *stream) {(void)stream; ++sync_calls; return sync_error;}
static int copy_stub(float *out, uintptr_t source, size_t bytes)
{ ++copy_calls; copied=bytes; if(copy_error) return copy_error; memcpy(out,(const void *)source,bytes); return 0; }
#include "readback.inc"
#define CHECK(x) do {if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;}} while(0)
int main(void)
{
    unsigned suppressed[]={1,3};
    nya_llm_context model={{5,2,suppressed}};
    nya_cuda_context cuda={sync_stub,copy_stub,NULL};
    float source[20], output[22];
    for(size_t i=0;i<20;++i)source[i]=(float)i;
    nya_cuda_plan plan={&model,&cuda,4,(uintptr_t)source,{0,0}};
    for(size_t rows=1;rows<=4;++rows) {
        for(size_t i=0;i<22;++i)output[i]=12345;
        sync_calls=copy_calls=0;
        CHECK(!plan_read_logits_bank(&plan,output+1,rows));
        CHECK(sync_calls==1 && copy_calls==1 && copied==rows*5*sizeof(float));
        CHECK(output[0]==12345 && output[rows*5+1]==12345);
        for(size_t i=0;i<rows*5;++i)
            CHECK(output[i+1]==(i%5==1 || i%5==3 ? -INFINITY : source[i]));
    }
    /* Validate every row before suppression, including suppressed entries. */
    for(size_t i=0;i<20;++i) {
        float saved=source[i]; source[i]=NAN;
        CHECK(plan_read_logits_bank(&plan,output+1,4)<0);
        source[i]=INFINITY; CHECK(plan_read_logits_bank(&plan,output+1,4)<0);
        source[i]=-INFINITY; CHECK(plan_read_logits_bank(&plan,output+1,4)<0);
        source[i]=saved;
    }
    sync_calls=copy_calls=0;
    CHECK(plan_read_logits_bank(&plan,NULL,4)<0);
    CHECK(plan_read_logits_bank(&plan,output,0)<0);
    CHECK(plan_read_logits_bank(&plan,output,5)<0);
    model.tokenizer.vocabulary_size=SIZE_MAX;
    CHECK(plan_read_logits_bank(&plan,output,4)<0);
    model.tokenizer.vocabulary_size=SIZE_MAX/sizeof(float);
    CHECK(plan_read_logits_bank(&plan,output,4)<0);
    model.tokenizer.vocabulary_size=0;
    CHECK(plan_read_logits_bank(&plan,output,4)<0);
    CHECK(!sync_calls && !copy_calls);
    model.tokenizer.vocabulary_size=5;
    sync_error=1; CHECK(plan_read_logits_bank(&plan,output,4)<0);
    CHECK(sync_calls==1 && !copy_calls);
    sync_error=0; copy_error=1;
    CHECK(plan_read_logits_bank(&plan,output,4)<0);
    CHECK(sync_calls==2 && copy_calls==1);
    puts("readback guards, all-row validation, suppression and failure handling passed");
    return 0;
}
