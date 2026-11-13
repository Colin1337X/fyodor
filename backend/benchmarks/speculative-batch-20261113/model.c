#include "llm_internal.h"
#include "file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
static double now(void){LARGE_INTEGER t,f;QueryPerformanceCounter(&t);QueryPerformanceFrequency(&f);return (double)t.QuadPart/(double)f.QuadPart;}
int main(int argc,char **argv){
 if(argc!=2)return 2;
 FILE *f=nya_file_open_read(argv[1]);uint64_t bytes=0;if(!f)return 2;
 int rc=nya_file_regular_size(f,&bytes);fclose(f);if(rc)return 2;
 nya_llm_context *m=NULL;char error[512]={0};if(nya_llm_load(argv[1],bytes,&m,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 2;}
 for(int run=0;run<8;++run){
  int sampling=run%2;
  nya_generation_request r={0};r.prompt="The capital of France is";r.max_tokens=32;r.max_output_bytes=4096;r.top_p=1;r.top_k=sampling?40:0;r.temperature=sampling?0.7f:0;r.seed=123;r.speculative_tokens=4;
  nya_generation_response a={0},b={0};
  double serial=0,batch=0;
  for(int phase=0;phase<2;++phase){
   int enabled=phase^(run/2%2);_putenv_s("NYA_SPECULATIVE_BATCH",enabled?"1":"0");
   double start=now();rc=nya_llm_generate_draft(m,m,&r,enabled?&b:&a,error,sizeof(error));
   if(enabled)batch=now()-start;else serial=now()-start;if(rc)goto fail;
  }
  printf("run=%d sampling=%d serial_seconds=%.6f batch_seconds=%.6f tokens=%zu accepted=%zu serial_steps=%zu batch_steps=%zu equal=%d\n",run,sampling,serial,batch,b.generated_tokens,b.accepted_draft_tokens,a.target_steps,b.target_steps,!strcmp(a.text,b.text));fflush(stdout);
  rc=strcmp(a.text,b.text)||a.generated_tokens!=b.generated_tokens||a.stop_reason!=b.stop_reason;
  nya_generation_response_free(&a);nya_generation_response_free(&b);if(rc)goto fail;
 }
 nya_llm_free(m);return 0;
fail:fprintf(stderr,"%s\n",error);nya_llm_free(m);return 1;
}
