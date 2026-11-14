"""Isolated batch-four gate/up integration; host dispatch remains C."""
def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected anchor: ' + old)
    return source.replace(old, new)

def provider(source):
    source = replace_once(source, 'pair_rows, pair_rows_q6;',
                          'pair_rows, pair_rows_q6, batch_gate;')
    anchor = '                    c->interleaved_integer_q4=candidate;'
    return replace_once(source, anchor, anchor + '''
                    const char *batch_gate=getenv("NYA_CUDA_Q4_BATCH_GATE");
                    if(batch_gate && !strcmp(batch_gate,"1")) {
                        CUfunction fused=NULL;
                        if(c->function_get(&fused,c->module,"nya_q4_batch_gate")==CUDA_SUCCESS)
                            c->batch_gate=fused;
                    }''')

def resident(source):
    source = replace_once(source, '    unsigned long long q6_pair_rows_calls;',
        '    unsigned long long batch_gate_calls;\n    unsigned long long q6_pair_rows_calls;')
    source = replace_once(source, '    if(p->q6_pair_rows_calls && mtp_trace', '''    if(p->batch_gate_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA Q4 batch gate capture/direct calls: %llu\\n",p->batch_gate_calls);
    if(p->q6_pair_rows_calls && mtp_trace''')
    helper = '''static int plan_batch_gate_eligible(const nya_cuda_plan *p,
    const nya_llm_tensor *gate, const nya_llm_tensor *up,
    size_t rows, size_t columns)
{
    return p->cuda->batch_gate && !p->cuda->reference && p->fuse_gate_up &&
        p->model->is_gemma && p->batch==4 && rows==15360 && columns==3840 &&
        gate && up && gate->type==12 && up->type==12 &&
        gate->data_size==rows*(columns/256)*144 &&
        up->data_size==gate->data_size && p->interleaved_integer_enabled &&
        plan_interleaved_kernel(p,12,columns) &&
        !(p->blas_storage && nya_blas_bytes(p->cuda,rows,columns,p->batch));
}
static int plan_batch_gate(nya_cuda_plan *p, const nya_llm_tensor *gate,
    const nya_llm_tensor *up, size_t rows, size_t columns)
{
    CUdeviceptr g=plan_tensor(p,gate), u=plan_tensor(p,up);
    unsigned long long cols=columns, stride=columns/256*144;
    unsigned r=(unsigned)rows, batch=4;
    void *convert[]={&p->norm,&p->interleaved_input,&cols};
    if(plan_launch(p,p->cuda->interleave_four,(columns+255)/256,convert))return -1;
    void *args[]={&g,&u,&p->interleaved_input,&p->gate,&cols,&stride,&r,&batch};
    p->matrix_profile=plan_matrix_slot(p,rows,columns,12,5);
    int result=plan_grid(p,p->cuda->batch_gate,(rows+7)/8,1,256,1,args);
    p->matrix_profile=0;
    if(!result)++p->batch_gate_calls;
    return result;
}
'''
    source = replace_once(source, 'static int plan_gate_up(', helper + 'static int plan_gate_up(')
    source = replace_once(source, '        if (gemma_gate && gemma_up) {', '''        if (count==4 && plan_batch_gate_eligible(p,w->feed_forward_gate,w->feed_forward_up,h,d)) {
            RUN(plan_batch_gate(p,w->feed_forward_gate,w->feed_forward_up,h,d));
        } else if (gemma_gate && gemma_up) {''')
    return source
