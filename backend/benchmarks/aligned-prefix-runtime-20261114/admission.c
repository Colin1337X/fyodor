/* Real-model plan admission, without inference or performance claims. */
#include "llm_internal.h"
#include "file.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv)
{
    if(argc!=3)return 2;
    char *end;errno=0;unsigned long parsed=strtoul(argv[2],&end,10);
    if(errno || !parsed || *end || parsed>262144)return 2;
    uint64_t bytes=0;char error[512]={0};int result=1;
    nya_llm_context *model=NULL;nya_llm_session *session=NULL;
    FILE *file=nya_file_open_read(argv[1]);if(!file)return 2;
    int bad=nya_file_regular_size(file,&bytes);fclose(file);
    if(bad || nya_llm_load(argv[1],bytes,&model,error,sizeof(error)))goto done;
    session=nya_llm_session_create(model,(size_t)parsed,error,sizeof(error));
    if(!session)goto done;
    nya_compute_stats stats={0};nya_llm_session_stats(session,&stats);
    printf("{\"capacity\":%lu,\"resident\":%s,\"batch\":%zu,\"weights\":%zu,\"kv\":%zu,\"scratch\":%zu}\n",
        parsed,!strcmp(nya_llm_session_execution(session),"resident")?"true":"false",
        stats.prefill_batch,stats.weights_bytes,stats.kv_bytes,stats.scratch_bytes);
    result=0;
done:
    if(result)fprintf(stderr,"Admission probe failed: %s\n",error);
    nya_llm_session_free(session);nya_llm_free(model);return result;
}
