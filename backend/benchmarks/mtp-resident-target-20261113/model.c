#include "model.h"
#include "generation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv){nya_model_registry *r=calloc(1,sizeof(*r));if(!r)return 2;nya_model_registry_init(r);const nya_model *target=NULL,*draft=NULL;char error[512]={0};int result=1;
 if(nya_model_load(r,"E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",&target)||nya_model_load(r,"E:/models/mtp-gemma-4-12B-it.gguf",&draft))goto done;
 nya_generation_request req={0};req.prompt="<|turn>user\nWrite a short greeting.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";req.max_tokens=12;req.max_output_bytes=4096;req.top_p=1;req.seed=42;
 char long_prompt[4096];
 if(argc>1) {strcpy(long_prompt,"<|turn>user\n");for(int i=0;i<(!strcmp(argv[1],"window")?520:128);++i)strcat(long_prompt,"Hello. ");strcat(long_prompt,"Write a short greeting.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");req.prompt=long_prompt;}
 nya_generation_response baseline={0},actual={0};int a=nya_generation_run((nya_model*)target,&req,&baseline,error,sizeof(error));printf("prompt_tokens=%zu\n",baseline.prompt_tokens);printf("baseline result=%d error=%s tokens=%zu text=%s\n",a,error,baseline.generated_tokens,baseline.text?baseline.text:"");fflush(stdout);
 req.draft_model=(nya_model*)draft;req.speculative_tokens=2;error[0]=0;int b=nya_generation_run((nya_model*)target,&req,&actual,error,sizeof(error));
 printf("mtp result=%d error=%s generated=%zu draft=%zu accepted=%zu target_steps=%zu text=%s\n",b,error,actual.generated_tokens,actual.draft_tokens,actual.accepted_draft_tokens,actual.target_steps,actual.text?actual.text:"");
 result=a||b||baseline.generated_tokens!=actual.generated_tokens||baseline.text_length!=actual.text_length||(baseline.text_length&&memcmp(baseline.text,actual.text,baseline.text_length));printf("greedy_parity=%s\n",result?"failed":"passed");nya_generation_response_free(&baseline);nya_generation_response_free(&actual);
 done:nya_model_registry_shutdown(r);free(r);return result;}
