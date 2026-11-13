#include "llm_internal.h"
#include "file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
 if(argc!=2) return 2;
 FILE *f=nya_file_open_read(argv[1]);uint64_t bytes=0;if(!f)return 2;
 int failed=nya_file_regular_size(f,&bytes);fclose(f);if(failed)return 2;
 nya_llm_context *m=NULL;char error[512]={0};
 if(nya_llm_load(argv[1],bytes,&m,error,sizeof(error))){puts(error);return 2;}
 size_t vocab=m->tokenizer.vocabulary_size;float *reference=malloc(8*vocab*sizeof(float));if(!reference)return 2;
 const char *starts[]={"off","0","10","16","17","33","bad","184467440737095516160",""};
 for(size_t mode=0;mode<sizeof(starts)/sizeof(starts[0]) && !failed;++mode) {
  _putenv_s("NYA_CUDA_PROFILE",mode?"1":"0");_putenv_s("NYA_CUDA_PROFILE_FROM",starts[mode]);
  fprintf(stderr,"CASE %s\n",starts[mode]);
  nya_llm_session *s=nya_llm_session_create(m,32,error,sizeof(error));
  if(!s || strcmp(nya_llm_session_execution(s),"resident")) {puts(error);return 2;}
  size_t index=0;
  for(unsigned pass=0;pass<2 && !failed;++pass) {
   uint32_t tokens[19];for(size_t i=0;i<19;++i)tokens[i]=(uint32_t)((i*7+pass+1)%vocab);
   nya_llm_session_reset(s);
   for(size_t step=0;step<4;++step) {
    if(nya_llm_session_run(s,tokens+(step?15+step:0),step?1:16,step!=0)){failed=1;break;}
    const float *actual=nya_llm_session_logits(s);
    if(!mode)memcpy(reference+index*vocab,actual,vocab*sizeof(float));
    else if(memcmp(reference+index*vocab,actual,vocab*sizeof(float)))failed=1;
    ++index;
   }
  }
  nya_llm_session_free(s);printf("scope=%s exact=%d\n",starts[mode],!failed);
 }
 free(reference);nya_llm_free(m);return failed;
}
