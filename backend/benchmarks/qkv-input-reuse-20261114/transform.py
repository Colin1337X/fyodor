"""Isolated C-host QKV input reuse; no changes to device arithmetic."""
def transform(source):
    def once(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('Unexpected anchor: '+old)
        source=source.replace(old,new)
    once('    int reuse_interleaved_input;', '    int qkv_input_reuse;\n    unsigned long long qkv_reuse_hits;\n    int reuse_interleaved_input;')
    once('        p->interleaved_four_enabled=interleaved && batch>=4;',
         '        const char *qkv_reuse=getenv("NYA_CUDA_QKV_INPUT_REUSE");\n        p->qkv_input_reuse=qkv_reuse && !strcmp(qkv_reuse,"1");\n        p->interleaved_four_enabled=interleaved && batch>=4;')
    once('    if(p->batched_logits_calls && mtp_trace',
         '    if(p->qkv_reuse_hits && mtp_trace && !strcmp(mtp_trace,"1"))\n        fprintf(stderr,"CUDA QKV input reuse capture/direct hits: %llu\\n",p->qkv_reuse_hits);\n    if(p->batched_logits_calls && mtp_trace')
    helper = """/* Local Q/K/V sequence only. Normalizing q does not touch norm or
   interleaved_input. Clear the shared reuse flag before propagating failure. */
static int plan_matvec_norm_reuse(nya_cuda_plan *p, CUdeviceptr out,
    const nya_llm_tensor *t, size_t rows, size_t columns, int *ready)
{
    int eligible=p->qkv_input_reuse && (p->batch==2 || p->batch==4) &&
        plan_interleaved_kernel(p,t->type,columns) &&
        !(p->blas_storage && nya_blas_bytes(p->cuda,rows,columns,p->batch));
    p->reuse_interleaved_input=eligible && *ready;
    if(p->reuse_interleaved_input)++p->qkv_reuse_hits;
    int result=plan_matvec(p,out,t,p->norm,rows,columns);
    p->reuse_interleaved_input=0;
    if(!result && eligible)*ready=1;
    return result;
}
"""
    once('static int plan_gate_up(',helper+'static int plan_gate_up(')
    once('        int rope_pair =', '        int qkv_norm_ready=0;\n        int rope_pair =')
    once('else RUN(plan_matvec(p, p->q, w->query, p->norm, qdim, d));',
         'else RUN(plan_matvec_norm_reuse(p,p->q,w->query,qdim,d,&qkv_norm_ready));')
    once('RUN(plan_matvec(p, key, w->key, p->norm, kvdim, d));',
         'RUN(plan_matvec_norm_reuse(p,key,w->key,kvdim,d,&qkv_norm_ready));')
    once('RUN(plan_matvec(p, value, w->value, p->norm, kvdim, d));',
         'RUN(plan_matvec_norm_reuse(p,value,w->value,kvdim,d,&qkv_norm_ready));')
    return source
