/* CPU-only comparison utility. Select one independent tokenizer at compile time. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef REFERENCE_TOKENIZER
#include "llama.h"
#else
#include "llm_internal.h"
#include <sys/stat.h>
#endif

int main(int argc,char **argv)
{
    if (argc!=3) return 2;
    FILE *input=fopen(argv[2],"rb");
    if (!input) return 2;
    char prompt[40000]; size_t bytes=fread(prompt,1,sizeof(prompt)-1,input);
    int failed=ferror(input) || !feof(input); fclose(input);
    if (failed || memchr(prompt,0,bytes)) return 2;
    prompt[bytes]=0;
    int result=1;
#ifdef REFERENCE_TOKENIZER
    struct llama_model_params parameters=llama_model_default_params();
    parameters.vocab_only=true; parameters.n_gpu_layers=0;
    struct llama_model *model=llama_model_load_from_file(argv[1],parameters);
    if (!model) return 1;
    const struct llama_vocab *vocab=llama_model_get_vocab(model);
    int32_t count=llama_tokenize(vocab,prompt,(int32_t)bytes,NULL,0,true,true);
    llama_token *tokens=NULL;
    if (count<0 && count>-100000) {
        count=-count; tokens=malloc((size_t)count*sizeof(*tokens));
        if (tokens && llama_tokenize(vocab,prompt,(int32_t)bytes,tokens,count,true,true)==count) {
            for (int32_t i=0;i<count;++i) printf("%s%d",i?" ":"",tokens[i]);
            putchar('\n'); result=0;
        }
    }
    free(tokens); llama_model_free(model);
#else
    struct _stat64 information;
    if (_stat64(argv[1],&information) || information.st_size<=0) return 1;
    nya_llm_context *context=NULL; char error[512]={0};
    uint32_t *tokens=NULL; size_t count=0;
    if (!nya_llm_load(argv[1],(uint64_t)information.st_size,&context,error,sizeof(error)) &&
        !nya_llm_tokenize(context,prompt,&tokens,&count,error,sizeof(error))) {
        for (size_t i=0;i<count;++i) printf("%s%u",i?" ":"",tokens[i]);
        putchar('\n'); result=0;
    }
    if (result) fprintf(stderr,"%s\n",error);
    free(tokens); nya_llm_free(context);
#endif
    return result;
}
