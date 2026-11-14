"""Isolated C host dispatch for validated cached-query attention shapes."""
def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected anchor: ' + old)
    return source.replace(old, new)

def provider(source):
    source = replace_once(source, '    CUfunction wide_batch_split[2];',
                          '    CUfunction wide_batch_split[2], cached_query[2];')
    anchor = '            if(c->function_get(&candidate,c->module,names[i])==CUDA_SUCCESS)c->wide_batch_split[i]=candidate;'
    return replace_once(source, anchor, anchor + '''
            const char *cached=getenv("NYA_CUDA_CACHED_QUERY");
            if(cached && !strcmp(cached,"1")) {
                static const char *optional[]={"nya_attention_cached_query_split","nya_attention_cached_query_ring_split"};
                CUfunction cached_function=NULL;
                if(c->function_get(&cached_function,c->module,optional[i])==CUDA_SUCCESS)
                    c->cached_query[i]=cached_function;
            }''')

def resident(source):
    source = replace_once(source, '    unsigned long long wide_batch_split_calls;',
        '    unsigned long long cached_query_calls;\n    unsigned long long wide_batch_split_calls;')
    source = replace_once(source, '    if(p->wide_batch_split_calls && mtp_trace', '''    if(p->cached_query_calls && mtp_trace && !strcmp(mtp_trace,"1"))
        fprintf(stderr,"CUDA cached query capture/direct calls: %llu\\n",p->cached_query_calls);
    if(p->wide_batch_split_calls && mtp_trace''')
    helper = '''static int plan_cached_query_eligible(const nya_cuda_plan *p, int batch_split,
    size_t count, unsigned width, unsigned heads, unsigned kvheads,
    unsigned window, unsigned ring_slots)
{
    return batch_split && !p->cuda->reference && p->model->is_gemma &&
        p->cuda->cached_query[ring_slots?1:0] && (count==2 || count==4) && heads==16 &&
        ((width==256 && kvheads==8 && window==1024) ||
         (width==512 && kvheads==1 && window==0));
}
'''
    source = replace_once(source, 'static int plan_gate_up(', helper + 'static int plan_gate_up(')
    anchor = '            CUfunction kernel=batch_split?p->cuda->wide_batch_split[ring_slots?1:0]:p->wide_value[(ring_slots?2:0)+(split?1:0)];'
    source = replace_once(source, anchor, anchor + '''
            int cached_query=plan_cached_query_eligible(p,batch_split,count,width,nh,nk,window,ring_slots);
            if(cached_query) kernel=p->cuda->cached_query[ring_slots?1:0];''')
    source = replace_once(source, '            if(batch_split) ++p->wide_batch_split_calls;',
        '            if(cached_query) ++p->cached_query_calls;\n            if(batch_split) ++p->wide_batch_split_calls;')
    anchor = '        for(unsigned i=0;i<2;++i) if(f==p->cuda->wide_batch_split[i]) timing->kind=4U;'
    return replace_once(source, anchor, anchor + '\n        for(unsigned i=0;i<2;++i) if(f==p->cuda->cached_query[i]) timing->kind=4U;')
