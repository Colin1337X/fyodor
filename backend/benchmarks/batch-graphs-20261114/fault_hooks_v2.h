/* Single-threaded test process accounting; never included in production. */
static void *nya_test_graphs[64], *nya_test_execs[64];
static unsigned nya_test_created[2], nya_test_freed[2], nya_test_ownership_errors;
static void nya_test_resource_report(void)
{
    unsigned live=0;
    for(unsigned i=0;i<64;++i)live+=(unsigned)(nya_test_graphs[i]!=NULL)+(unsigned)(nya_test_execs[i]!=NULL);
    fprintf(stderr,"GRAPH resources: graphs=%u/%u execs=%u/%u live=%u errors=%u\n",
        nya_test_created[0],nya_test_freed[0],nya_test_created[1],nya_test_freed[1],live,nya_test_ownership_errors);
}
static void nya_test_resource_add(void *handle,int executable)
{
    static int registered;
    if(!registered){if(atexit(nya_test_resource_report))++nya_test_ownership_errors;registered=1;}
    void **slots=executable?nya_test_execs:nya_test_graphs;
    if(!handle){++nya_test_ownership_errors;return;}
    for(unsigned i=0;i<64;++i)if(slots[i]==handle){++nya_test_ownership_errors;return;}
    for(unsigned i=0;i<64;++i)if(!slots[i]){slots[i]=handle;++nya_test_created[executable];return;}
    ++nya_test_ownership_errors;
}
static void nya_test_resource_remove(void *handle,int executable,CUresult status)
{
    void **slots=executable?nya_test_execs:nya_test_graphs;
    if(status!=CUDA_SUCCESS){++nya_test_ownership_errors;return;}
    for(unsigned i=0;i<64;++i)if(slots[i]==handle && handle){slots[i]=NULL;++nya_test_freed[executable];return;}
    ++nya_test_ownership_errors;
}
/* Test-only C driver wrappers. Included only in a separate generated build.
   Set NYA_TEST_GRAPH_FAULT immediately before the operation being tested. */
static int nya_test_graph_fault(const char *site)
{
    const char *requested=getenv("NYA_TEST_GRAPH_FAULT");
    if(!requested || strcmp(requested,site))return 0;
    fprintf(stderr,"INJECTED graph failure: %s\n",site);
    return 1;
}
static CUresult nya_test_capture_begin(nya_cuda_context *c,CUstream stream,CUstreamCaptureMode mode)
{
    if(nya_test_graph_fault("begin"))return CUDA_ERROR_UNKNOWN;
    return c->capture_begin(stream,mode);
}
static CUresult nya_test_capture_end(nya_cuda_context *c,CUstream stream,CUgraph *graph)
{
    /* End the real capture first: exercise ownership of a returned graph even
       when the caller receives failure. Never leave the stream capturing. */
    CUresult status=c->capture_end(stream,graph);
    if(*graph)nya_test_resource_add(*graph,0);
    if(status==CUDA_SUCCESS && nya_test_graph_fault("end"))return CUDA_ERROR_UNKNOWN;
    return status;
}
static CUresult nya_test_graph_instantiate(nya_cuda_context *c,CUgraphExec *exec,CUgraph graph,unsigned long long flags)
{
    if(nya_test_graph_fault("instantiate")){*exec=NULL;return CUDA_ERROR_UNKNOWN;}
    CUresult status=c->graph_instantiate(exec,graph,flags);
    if(*exec)nya_test_resource_add(*exec,1);
    return status;
}
static CUresult nya_test_graph_launch(nya_cuda_context *c,CUgraphExec exec,CUstream stream)
{
    if(nya_test_graph_fault("launch"))return CUDA_ERROR_UNKNOWN;
    return c->graph_launch(exec,stream);
}

static CUresult nya_test_graph_free(nya_cuda_context *c,CUgraph graph)
{
    CUresult status=c->graph_free(graph);nya_test_resource_remove(graph,0,status);return status;
}
static CUresult nya_test_graph_exec_free(nya_cuda_context *c,CUgraphExec exec)
{
    CUresult status=c->graph_exec_free(exec);nya_test_resource_remove(exec,1,status);return status;
}
