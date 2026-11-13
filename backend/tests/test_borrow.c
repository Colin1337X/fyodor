#include "llm_internal.h"
#include "file.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);goto done;}}while(0)
int main(int argc,char **argv){
 if(argc!=2)return 2;
 FILE *file=nya_file_open_read(argv[1]);uint64_t bytes=0;if(!file)return 2;
 int bad=nya_file_regular_size(file,&bytes);fclose(file);if(bad)return 2;
 nya_llm_context *m=NULL;nya_llm_session *s=NULL;nya_compute_plan *p=NULL;char error[512]={0};int result=1;
 float *cpu=NULL,*gpu=NULL,*keys=NULL,*values=NULL,*query=NULL,*output=NULL,*logits=NULL;uint32_t tokens[16];
 CHECK(!nya_llm_load(argv[1],bytes,&m,error,sizeof(error)));
 for(size_t i=0;i<m->block_count;++i)m->layers[i].sliding_window=3;
 size_t d=m->embedding_length,v=m->tokenizer.vocabulary_size,heads=m->layers[0].head_count,hd=m->layers[0].head_dimension;
 size_t kw=hd*m->layers[0].kv_head_count,qw=heads*hd;
 cpu=malloc(d*sizeof(float));gpu=malloc(d*sizeof(float));keys=malloc(16*kw*sizeof(float));values=malloc(16*kw*sizeof(float));
 query=malloc(qw*sizeof(float));output=malloc(qw*sizeof(float));logits=malloc(v*sizeof(float));CHECK(cpu&&gpu&&keys&&values&&query&&output&&logits);
 for(size_t i=0;i<16;++i)tokens[i]=(uint32_t)((20+i)%v);
 nya_llm_context cpu_model=*m;cpu_model.compute=NULL;
 s=nya_llm_session_create(&cpu_model,16,error,sizeof(error));CHECK(s);
 p=nya_compute_plan_create(m->compute,m,16);CHECK(p && nya_compute_plan_can_borrow(p));
 CHECK(nya_compute_plan_hidden(p,gpu,d)<0);
 CHECK(!nya_llm_session_run(s,tokens,12,1));CHECK(!nya_llm_session_hidden(s,cpu,d));
 CHECK(!nya_compute_plan_prefill(p,tokens,12,0,logits));CHECK(!nya_compute_plan_hidden(p,gpu,d));
 for(size_t i=0;i<d;++i)CHECK(isfinite(gpu[i]) && fabs(gpu[i]-cpu[i])<=.0005*(1+fabs(cpu[i])));
 CHECK(!nya_llm_session_read_kv(s,0,0,12,keys,values,12*kw));
 for(size_t i=0;i<qw;++i)query[i]=(float)((int)(i%13)-6)/20;
 CHECK(nya_compute_plan_borrow_attention(p,0,13,query,heads,3,output,qw)<0);
 CHECK(nya_compute_plan_borrow_attention(p,0,12,query,heads,3,output,qw-1)<0);
 if(!getenv("NYA_TEST_BORROW_LINEAR")) CHECK(nya_compute_plan_borrow_attention(p,0,12,query,heads,0,output,qw)<0); /* evicted */
 nya_compute_stats before,after;nya_compute_plan_stats(p,&before);
 for(size_t prefix=11;prefix<=12;++prefix){
  size_t span=getenv("NYA_TEST_BORROW_LINEAR")?prefix:3;
  CHECK(!nya_compute_plan_borrow_attention(p,0,prefix,query,heads,span==prefix?0:3,output,qw));
  for(size_t h=0;h<heads;++h){
   double score[16],maximum=-INFINITY,den=0;size_t kh=h/(heads/m->layers[0].kv_head_count);
   for(size_t t=0;t<span;++t){score[t]=0;for(size_t j=0;j<hd;++j)score[t]+=(double)query[h*hd+j]*keys[(prefix-span+t)*kw+kh*hd+j];if(score[t]>maximum)maximum=score[t];}
   for(size_t t=0;t<span;++t){score[t]=exp(score[t]-maximum);den+=score[t];}
   for(size_t j=0;j<hd;++j){double want=0;for(size_t t=0;t<span;++t)want+=score[t]/den*values[(prefix-span+t)*kw+kh*hd+j];CHECK(isfinite(output[h*hd+j]) && fabs(output[h*hd+j]-want)<=.0005*(1+fabs(want)));}
  }
 }
 nya_compute_plan_stats(p,&after);CHECK(after.downloads==before.downloads+2 && after.uploads==before.uploads+2 && before.kv_bytes==after.kv_bytes && before.scratch_bytes==after.scratch_bytes);
 CHECK(!nya_compute_plan_hidden(p,gpu,d)); /* borrowed attention did not destroy raw hidden */
 for(size_t i=0;i<d;++i)CHECK(fabs(gpu[i]-cpu[i])<=.0005*(1+fabs(cpu[i])));
 CHECK(!nya_compute_plan_rewind(p,11));CHECK(nya_compute_plan_hidden(p,gpu,d)<0);
 CHECK(!nya_compute_plan_token(p,tokens[11],11,1,logits));CHECK(!nya_compute_plan_hidden(p,gpu,d));
 for(size_t i=0;i<d;++i)CHECK(fabs(gpu[i]-cpu[i])<=.0005*(1+fabs(cpu[i])));
 CHECK(!nya_compute_plan_rewind(p,11));CHECK(!nya_compute_plan_token(p,tokens[11],11,1,logits));CHECK(!nya_compute_plan_hidden(p,gpu,d));
 for(size_t i=0;i<d;++i)CHECK(fabs(gpu[i]-cpu[i])<=.0005*(1+fabs(cpu[i])));
 CHECK(!nya_compute_plan_rewind(p,0));CHECK(nya_compute_plan_hidden(p,gpu,d)<0);
 puts("borrowed KV, hidden preservation, eviction and rewind checks passed");result=0;
done:if(result)fprintf(stderr,"%s\n",error);nya_compute_plan_free(p);nya_llm_session_free(s);nya_llm_free(m);free(cpu);free(gpu);free(keys);free(values);free(query);free(output);free(logits);return result;
}
