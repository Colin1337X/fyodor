"""Isolated shape-specific Q4 four-row dispatch; portable host code stays C."""
def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('Unexpected anchor: '+old)
    return source.replace(old,new)

def provider(source):
    source=replace_once(source,'pair_rows, pair_rows_q6;','pair_rows, pair_rows_q6, quad_rows;')
    anchor='                    c->interleaved_integer_q4=candidate;'
    return replace_once(source,anchor,anchor+'''
                    const char *quad=getenv("NYA_CUDA_Q4_QUAD_ROWS");
                    if(quad && !strcmp(quad,"1")) {
                        CUfunction four=NULL;
                        if(c->function_get(&four,c->module,"nya_q4_quad_rows")==CUDA_SUCCESS)
                            c->quad_rows=four;
                    }''')

def resident(source):
    source=replace_once(source,'    unsigned long long q6_pair_rows_calls;','    unsigned long long quad_rows_calls;\n    unsigned long long q6_pair_rows_calls;')
    source=replace_once(source,'    if(p->q6_pair_rows_calls && mtp_trace','''    if(p->quad_rows_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA Q4 four rows capture/direct calls: %llu\\n",p->quad_rows_calls);
    if(p->q6_pair_rows_calls && mtp_trace''')
    helper='''static int plan_q4_quad_rows(const nya_cuda_plan *p, unsigned type,
    size_t rows, size_t columns, unsigned long long stride)
{
    return p->cuda->quad_rows && p->batch==4 &&
        p->interleaved_integer_enabled && type==12 &&
        columns==3840 && rows==15360 &&
        stride==(unsigned long long)(columns/256)*144;
}
'''
    source=replace_once(source,'static int plan_matvec_impl(',helper+'static int plan_matvec_impl(')
    source=replace_once(source,'''            unsigned tile=(paired || paired_q6)?16U:8U;
            CUfunction selected=paired_q6?p->cuda->pair_rows_q6:
                (paired?p->cuda->pair_rows:interleaved);''','''            int quad=plan_q4_quad_rows(p,type,rows,columns,stride);
            unsigned tile=quad?32U:((paired || paired_q6)?16U:8U);
            CUfunction selected=quad?p->cuda->quad_rows:(paired_q6?p->cuda->pair_rows_q6:
                (paired?p->cuda->pair_rows:interleaved));''')
    source=replace_once(source,'            if(!result && paired)++p->pair_rows_calls;','''            if(!result && paired && !quad)++p->pair_rows_calls;
            if(!result && quad)++p->quad_rows_calls;''')
    return source
