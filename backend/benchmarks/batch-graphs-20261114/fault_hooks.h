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
    if(status==CUDA_SUCCESS && nya_test_graph_fault("end"))return CUDA_ERROR_UNKNOWN;
    return status;
}
static CUresult nya_test_graph_instantiate(nya_cuda_context *c,CUgraphExec *exec,CUgraph graph,unsigned long long flags)
{
    if(nya_test_graph_fault("instantiate")){*exec=NULL;return CUDA_ERROR_UNKNOWN;}
    return c->graph_instantiate(exec,graph,flags);
}
static CUresult nya_test_graph_launch(nya_cuda_context *c,CUgraphExec exec,CUstream stream)
{
    if(nya_test_graph_fault("launch"))return CUDA_ERROR_UNKNOWN;
    return c->graph_launch(exec,stream);
}
