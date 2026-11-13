/* Diagnostic only: report actual admission, without executing inference. */
#include "llm_internal.h"
#include "file.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(int argc,char **argv)
{
    if(argc!=2)return 2;
    uint64_t bytes=0;char error[512]={0};int result=1;
    nya_llm_context *model=NULL;nya_llm_session *session=NULL;
    FILE *file=nya_file_open_read(argv[1]);if(!file)return 2;
    int bad=nya_file_regular_size(file,&bytes);fclose(file);if(bad)return 2;
    if(nya_llm_load(argv[1],bytes,&model,error,sizeof(error)))goto done;
    const size_t prefixes[]={1024,4096,8192};
    for(size_t i=0;i<sizeof(prefixes)/sizeof(prefixes[0]);++i) {
        session=nya_llm_session_create(model,prefixes[i]+2,error,sizeof(error));
        if(!session || strcmp(nya_llm_session_execution(session),"resident"))goto done;
        nya_compute_stats stats={0};nya_llm_session_stats(session,&stats);
        printf("{\"prefix\":%zu,\"prefill_batch\":%zu,\"weights_bytes\":%zu,\"kv_bytes\":%zu,\"scratch_bytes\":%zu,\"external_matmul_bytes\":%zu}\n",
            prefixes[i],stats.prefill_batch,stats.weights_bytes,stats.kv_bytes,
            stats.scratch_bytes,stats.external_matmul_bytes);
        fflush(stdout);nya_llm_session_free(session);session=NULL;
    }
    result=0;
done:
    if(result)fprintf(stderr,"Plan diagnostic failed: %s\n",error);
    nya_llm_session_free(session);nya_llm_free(model);return result;
}
