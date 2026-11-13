#include "model.h"
#include "generation.h"
#include "train_clock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv)
{
    nya_model_registry *registry=calloc(1,sizeof(*registry));
    if(!registry)return 2;
    nya_model_registry_init(registry);
    const nya_model *target=NULL,*draft=NULL;
    char error[512]={0},prompt[40000];
    int result=1;
    if(nya_model_load(registry,"E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",&target) ||
       nya_model_load(registry,"E:/models/mtp-gemma-4-12B-it.gguf",&draft))goto done;
    strcpy(prompt,"<|turn>user\n");
    if(argc>1 && !strcmp(argv[1],"long"))for(int i=0;i<4090;++i)strcat(prompt,"Hello. ");
    strcat(prompt,"Write a numbered list of twenty concise facts about the Moon.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    nya_generation_request request={0};
    request.prompt=prompt;request.max_tokens=32;request.max_output_bytes=8192;request.top_p=1;request.seed=42;
    nya_generation_response reference={0};
    for(int round=0;round<3;++round)for(int step=0;step<4;++step){
        int mode=round==1?3-step:step;
        const char *names[]={"ordinary","mtp-serial2","mtp-batch2","mtp-joint3"};
        request.draft_model=mode?(nya_model*)draft:NULL;
        request.speculative_tokens=mode?(mode==3?4u:2u):0u;
        if(_putenv_s("NYA_SPECULATIVE_BATCH",mode>=2?"1":"0"))goto responses;
        nya_generation_response out={0};error[0]=0;
        fprintf(stderr,"CASE round=%d mode=%s\n",round,names[mode]);fflush(stderr);
        double start=tr_seconds();
        int rc=nya_generation_run((nya_model*)target,&request,&out,error,sizeof(error));
        double seconds=tr_seconds()-start;
        int parity=!rc && (!reference.text || (reference.generated_tokens==out.generated_tokens &&
            reference.stop_reason==out.stop_reason && reference.text_length==out.text_length &&
            !memcmp(reference.text,out.text,out.text_length)));
        printf("{\"round\":%d,\"mode\":\"%s\",\"seconds\":%.9f,\"prompt\":%zu,\"generated\":%zu,\"draft\":%zu,\"accepted\":%zu,\"target_steps\":%zu,\"parity\":%s}\n",
            round,names[mode],seconds,out.prompt_tokens,out.generated_tokens,out.draft_tokens,out.accepted_draft_tokens,out.target_steps,parity?"true":"false");fflush(stdout);
        if(!parity){fprintf(stderr,"FAIL %s\n",error);nya_generation_response_free(&out);goto responses;}
        if(!reference.text){reference=out;printf("TEXT %s\n",out.text);fflush(stdout);}
        else nya_generation_response_free(&out);
    }
    result=0;
responses:
    nya_generation_response_free(&reference);
done:
    nya_model_registry_shutdown(registry);free(registry);return result;
}
