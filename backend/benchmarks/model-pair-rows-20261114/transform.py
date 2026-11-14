"""Isolated opt-in dispatch for measured native Q4 paired-row shapes."""
def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected anchor: ' + old)
    return source.replace(old, new)


def provider(source):
    source = replace_once(source,
        'interleaved_reuse[31], interleaved_integer_q4;',
        'interleaved_reuse[31], interleaved_integer_q4, pair_rows;')
    anchor = '                    c->interleaved_integer_q4=candidate;'
    return replace_once(source, anchor, anchor + '''
                    const char *pair=getenv("NYA_CUDA_PAIR_ROWS");
                    if(pair && !strcmp(pair,"1")) {
                        CUfunction paired=NULL;
                        if(c->function_get(&paired,c->module,"nya_q4_pair_rows")==CUDA_SUCCESS)
                            c->pair_rows=paired;
                    }''')


def resident(source):
    source = replace_once(source, '    int qkv_input_reuse;',
        '    unsigned long long pair_rows_calls;\n    int qkv_input_reuse;')
    source = replace_once(source, '    if(p->qkv_reuse_hits && mtp_trace',
        '''    if(p->pair_rows_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA paired rows capture/direct calls: %llu\\n",p->pair_rows_calls);
    if(p->qkv_reuse_hits && mtp_trace''')
    helper = '''static int plan_pair_rows(const nya_cuda_plan *p, unsigned type,
    size_t rows, size_t columns, unsigned long long stride)
{
    return p->cuda->pair_rows && p->batch==4 &&
        p->interleaved_integer_enabled && type==12 &&
        ((columns==3840 && rows==15360) ||
         (columns==8192 && rows==3840) ||
         (columns==15360 && rows==3840)) &&
        stride==(unsigned long long)(columns/256)*144;
}
'''
    source = replace_once(source, 'static int plan_matvec_impl(', helper + 'static int plan_matvec_impl(')
    source = replace_once(source,
        '            int result=plan_grid(p,interleaved,(r+7)/8,1,256,1,gemm_args);',
        '''            int paired=plan_pair_rows(p,type,rows,columns,stride);
            unsigned tile=paired?16U:8U;
            int result=plan_grid(p,paired?p->cuda->pair_rows:interleaved,
                (r+tile-1)/tile,1,256,1,gemm_args);
            if(!result && paired)++p->pair_rows_calls;''')
    return source
