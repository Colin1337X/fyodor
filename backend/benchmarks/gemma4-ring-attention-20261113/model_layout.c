#include "llm_internal.h"
#include "file.h"
#include <stdio.h>
#include <inttypes.h>
int main(void) {
 const char *path="E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf";
 FILE *f=nya_file_open_read(path);uint64_t bytes=0;if(!f)return 2;
 int rc=nya_file_regular_size(f,&bytes);fclose(f);if(rc)return 2;
 nya_llm_context *m=NULL;char error[512]={0};
 if(nya_llm_load(path,bytes,&m,error,sizeof(error))){puts(error);return 2;}
 size_t capacity=8194,batch=512;uint64_t ring=0;
 printf("{\"capacity\":%zu,\"maximum_batch\":%zu,\"layers\":[",capacity,batch);
 for(size_t i=0;i<m->block_count;++i){const nya_llm_layer *w=&m->layers[i];
  size_t slots=capacity;if(w->sliding_window && w->sliding_window+batch-1<slots)slots=w->sliding_window+batch-1;
  uint64_t amount=(uint64_t)slots*w->head_dimension*w->kv_head_count*8;
  if(w->kv_source==i)ring+=amount;
  printf("%s{\"layer\":%zu,\"kv_source\":%zu,\"head_width\":%zu,\"kv_heads\":%zu,\"window\":%u,\"proposed_slots\":%zu}",i?",":"",i,(size_t)w->kv_source,(size_t)w->head_dimension,(size_t)w->kv_head_count,w->sliding_window,slots);
 }
 printf("],\"linear_kv_bytes\":%" PRIu64 ",\"proposed_ring_kv_bytes\":%" PRIu64 ",\"runtime_integrated\":false}\n",(uint64_t)m->cache_width*capacity*8,ring);
 nya_llm_free(m);return 0;
}
