"""Isolated Q6 batch-four dispatch on top of the validated Q4 candidate."""
def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected anchor: ' + old)
    return source.replace(old, new)


def provider(source):
    source = replace_once(source, 'interleaved_integer_q4, pair_rows;',
                          'interleaved_integer_q4, pair_rows, pair_rows_q6;')
    anchor = '                    c->interleaved_integer_q4=candidate;'
    return replace_once(source, anchor, anchor + '''
                    const char *q6pair=getenv("NYA_CUDA_Q6_PAIR_ROWS");
                    if(q6pair && !strcmp(q6pair,"1")) {
                        CUfunction paired_q6=NULL;
                        if(c->function_get(&paired_q6,c->module,"nya_q6_pair_rows_four")==CUDA_SUCCESS)
                            c->pair_rows_q6=paired_q6;
                    }''')


def resident(source):
    source = replace_once(source, '    unsigned long long pair_rows_calls;',
        '    unsigned long long q6_pair_rows_calls;\n    unsigned long long pair_rows_calls;')
    source = replace_once(source, '    if(p->pair_rows_calls && mtp_trace',
        '''    if(p->q6_pair_rows_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA Q6 paired rows capture/direct calls: %llu\\n",p->q6_pair_rows_calls);
    if(p->pair_rows_calls && mtp_trace''')
    helper = '''static int plan_q6_pair_rows(const nya_cuda_plan *p, unsigned type,
    size_t rows, size_t columns, unsigned long long stride)
{
    return p->cuda->pair_rows_q6 && p->batch==4 &&
        p->interleaved_integer_enabled && type==14 &&
        ((columns==3840 && rows==2048) ||
         (columns==8192 && rows==3840) ||
         (columns==15360 && rows==3840)) &&
        stride==(unsigned long long)(columns/256)*210;
}
'''
    source = replace_once(source, 'static int plan_matvec_impl(', helper + 'static int plan_matvec_impl(')
    source = replace_once(source, '''            unsigned tile=paired?16U:8U;
            int result=plan_grid(p,paired?p->cuda->pair_rows:interleaved,''',
        '''            int paired_q6=plan_q6_pair_rows(p,type,rows,columns,stride);
            unsigned tile=(paired || paired_q6)?16U:8U;
            CUfunction selected=paired_q6?p->cuda->pair_rows_q6:
                (paired?p->cuda->pair_rows:interleaved);
            int result=plan_grid(p,selected,''')
    source = replace_once(source, '            if(!result && paired)++p->pair_rows_calls;',
        '''            if(!result && paired)++p->pair_rows_calls;
            if(!result && paired_q6)++p->q6_pair_rows_calls;''')
    return source
