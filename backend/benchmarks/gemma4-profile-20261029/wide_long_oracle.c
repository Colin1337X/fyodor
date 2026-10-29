#include "llm_internal.h"
#include "file.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
int main(void){const char *path="E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf";FILE *f=nya_file_open_read(path);uint64_t bytes=0;if(!f)return 2;int rc=nya_file_regular_size(f,&bytes);fclose(f);if(rc)return 2;nya_llm_context *m=NULL;char error[512]={0};if(nya_llm_load(path,bytes,&m,error,sizeof(error))){puts(error);return 2;}
 size_t vocab=m->tokenizer.vocabulary_size;float *reference=malloc(vocab*sizeof(float));uint32_t tokens[8194];for(unsigned i=0;i<8194;++i)tokens[i]=20+i%128;int result=0;
 const size_t sizes[]={4097,8193};for(unsigned test=0;test<2&&!result;++test){size_t count=sizes[test];
 for(unsigned mode=0;mode<2;++mode){_putenv_s("NYA_CUDA_WIDE_SPLIT",mode?"1":"0");nya_llm_session *s=nya_llm_session_create(m,count,error,sizeof(error));if(!s){puts(error);result=1;break;}if(strcmp(nya_llm_session_execution(s),"resident")||nya_llm_session_run(s,tokens,count-2,0)||nya_llm_session_run(s,tokens+count-2,1,1)||nya_llm_session_run(s,tokens+count-1,1,1)){puts("resident inference failed");result=1;nya_llm_session_free(s);break;}
 const float *actual=nya_llm_session_logits(s);if(!mode)memcpy(reference,actual,vocab*sizeof(float));else{double max_abs=0,max_scaled=0,squared=0;for(size_t j=0;j<vocab;++j){if(actual[j]==reference[j])continue;double diff=fabs((double)actual[j]-reference[j]),scaled=diff/(1+fabs(reference[j]));if(!isfinite(actual[j])||!isfinite(reference[j]))result=1;if(diff>max_abs)max_abs=diff;if(scaled>max_scaled)max_scaled=scaled;squared+=diff*diff;}printf("tokens=%zu max_abs=%.9g max_scaled=%.9g rms=%.9g\n",count,max_abs,max_scaled,sqrt(squared/vocab));fflush(stdout);if(max_scaled>0.001)result=1;}
 nya_llm_session_free(s);}}
 free(reference);nya_llm_free(m);return result;}
