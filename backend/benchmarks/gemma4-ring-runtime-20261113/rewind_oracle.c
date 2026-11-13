#include "llm_internal.h"
#include "file.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Explicitly prove dispatch/fallback in addition to comparing every logit. */
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    char *end;
    unsigned long parsed = strtoul(argv[1], &end, 10);
    if (*end || parsed < 3 || parsed > 8194) return 2;
    size_t count = (size_t)parsed;
    const char *path = "E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf";
    FILE *file = nya_file_open_read(path);
    uint64_t bytes = 0;
    if (!file) return 2;
    int failed = nya_file_regular_size(file, &bytes);
    fclose(file);
    if (failed) return 2;
    char error[512] = {0};
    nya_llm_context *model = NULL;
    if (nya_llm_load(path, bytes, &model, error, sizeof(error))) { puts(error); return 2; }
    size_t vocabulary = model->tokenizer.vocabulary_size;
    float *reference = malloc(vocabulary * sizeof(*reference));
    uint32_t *tokens = malloc(count * sizeof(*tokens));
    if (!reference || !tokens) { free(reference); free(tokens); nya_llm_free(model); return 2; }
    for (size_t i = 0; i < count; ++i) tokens[i] = 20 + (uint32_t)(i % 128);
    float *outputs=malloc(4*vocabulary*sizeof(float));
    if(!outputs) return 2;
    for(unsigned mode=0;mode<2 && !failed;++mode) {
        _putenv_s("NYA_CUDA_RING_KV",mode?"1":"0");
        nya_compute_plan *plan=nya_compute_plan_create(model->compute,model,count);
        if(!plan) {failed=1;break;}
        if(nya_compute_plan_prefill(plan,tokens,count-2,0,reference)) failed=1;
        /* Rewind within retention, then prove rejected deep rewind changes no state. */
        size_t positions[]={count-3,count-2,0,1};
        for(size_t i=0;i<4 && !failed;++i) {
            if(i==1 && mode && nya_compute_plan_token(plan,tokens[1024],1024,1,reference)==0) failed=1;
            if(nya_compute_plan_token(plan,tokens[positions[i]],positions[i],1,reference)) failed=1;
            if(!mode) memcpy(outputs+i*vocabulary,reference,vocabulary*sizeof(float));
            else if(memcmp(outputs+i*vocabulary,reference,vocabulary*sizeof(float))) failed=1;
            printf("mode=%u rewind_or_reset=%zu exact=%d\n",mode,positions[i],!failed);fflush(stdout);
        }
        nya_compute_plan_free(plan);
    }
    free(outputs);free(tokens);free(reference);nya_llm_free(model);return failed;
}
