"""Isolated native batched-graph host transformation; production files untouched."""


def transform(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Expected one host anchor: '+old[:90])
        source = source.replace(old, new)

    replace('typedef struct nya_cuda_plan {', '''typedef struct nya_cuda_batch_graph_counts {
    unsigned long long values[9];
} nya_cuda_batch_graph_counts;

typedef struct nya_cuda_plan {''')
    replace('    CUgraphExec executable[2];', '''    CUgraphExec batch_executable[6];
    nya_cuda_batch_graph_counts batch_counts[6];
    CUfunction batch_request, graph_gather;
    int use_batch_graphs;
    unsigned long long batch_graph_captures, batch_graph_replays;
    CUgraphExec executable[2];''')
    fields = ['stats.kernel_launches','stats.external_matmul_calls','stats.cutlass_matmul_calls',
              'interleaved_calls','interleaved_two_calls','interleaved_integer_calls',
              'batched_wide_calls','wide_batch_split_calls','fdpp_batch_calls']
    snapshot, restore = [], []
    for i, field in enumerate(fields):
        before, after = ('#ifdef NYA_ALIGNED_PREFIX\n','\n#endif') if i == 8 else ('','')
        snapshot.append(before+f'    result.values[{i}]=p->{field};'+after)
        restore.append(before+f'    p->{field}+=counts->values[{i}];'+after)
    helpers = '''/* Logical dispatch counters must also advance when host lowering is skipped.
   Transfers and batched-logit counts remain on the uncaptured readback path. */
static nya_cuda_batch_graph_counts plan_batch_graph_snapshot(const nya_cuda_plan *p)
{
    nya_cuda_batch_graph_counts result={{0}};
SNAPSHOT
    return result;
}
static void plan_batch_graph_count_replay(nya_cuda_plan *p,
    const nya_cuda_batch_graph_counts *counts)
{
RESTORE
}

'''.replace('SNAPSHOT','\n'.join(snapshot)).replace('RESTORE','\n'.join(restore))
    replace('static void plan_profile_flush(nya_cuda_plan *p)', helpers+'static void plan_profile_flush(nya_cuda_plan *p)')
    replace('        p->cuda->deallocate(p->storage);', '''        for(size_t i=0;i<6;++i)
            if(p->batch_executable[i]) p->cuda->graph_exec_free(p->batch_executable[i]);
        p->cuda->deallocate(p->storage);''')
    replace('    if (p->timings) fprintf(stderr,', '''    if(p->use_batch_graphs) fprintf(stderr,
        "CUDA batched graph captures: %llu replays: %llu\\n",
        p->batch_graph_captures,p->batch_graph_replays);
    if (p->timings) fprintf(stderr,''')
    # Scratch must cover batched K/V projections in linear as well as ring mode.
    replace('    if(ring_kv && (maximum_kv>SIZE_MAX/batch || nya_cuda_kv_layout_build(m,capacity,batch,NULL,&kv_count))) return NULL;',
            '''    const char *batch_graphs=getenv("NYA_CUDA_BATCH_GRAPHS");
    int request_batch=batch_graphs && !strcmp(batch_graphs,"1");
    if((ring_kv || request_batch) && maximum_kv>SIZE_MAX/batch) return NULL;
    if(ring_kv && nya_cuda_kv_layout_build(m,capacity,batch,NULL,&kv_count)) return NULL;''')
    replace('maximum_kv*(ring_kv?batch:1), maximum_kv*(ring_kv?batch:1), 2, partial_total,',
            'maximum_kv*((ring_kv || request_batch)?batch:1), maximum_kv*((ring_kv || request_batch)?batch:1), request_batch?5:2, partial_total,')
    replace('            counts[11]=counts[12]=maximum_kv*batch;\n        }',
            '        }\n        if(ring_kv || request_batch) counts[11]=counts[12]=maximum_kv*batch;')
    replace('    const char *graphs = getenv("NYA_CUDA_GRAPHS");',
            '    p->use_batch_graphs=request_batch;\n    const char *graphs = getenv("NYA_CUDA_GRAPHS");')
    replace('    if (c->pop(&previous) != CUDA_SUCCESS) { p->failed = 1; cuda_plan_free(p); return NULL; }', '''    if(p->use_batch_graphs) {
        if(c->function_get(&p->batch_request,c->module,"nya_decode_batch_request")!=CUDA_SUCCESS ||
           c->function_get(&p->graph_gather,c->module,"nya_gather_graph_batch")!=CUDA_SUCCESS) goto failure;
        fprintf(stderr,"CUDA batched graphs: enabled\\n");
    }
    if (c->pop(&previous) != CUDA_SUCCESS) { p->failed = 1; cuda_plan_free(p); return NULL; }''')
    replace('    p->active_request = 0;\n    if (p->use_graphs && count == 1 && logits) {', '''    int batch_graph=p->use_batch_graphs && p->use_graphs && logits==2 &&
        (count==2 || count==4) && count<=p->logits_batch;
    /* graph_index still means SINGLE-token split attention. Never repurpose it.
       Batch dispatch depends on starting position at both 256 and 8192. */
    unsigned batch_index=(count==4?3U:0U)+(position>=8192?2U:position>=256?1U:0U);
    nya_cuda_batch_graph_counts batch_start={{0}};
    p->active_request = 0;
    if(batch_graph) {
        struct { unsigned values[4]; } packet={{0}};
        memcpy(packet.values,tokens,count*sizeof(*tokens));
        unsigned pos=(unsigned)position;
        void *args[]={&p->request,&packet,&pos};
        if(plan_grid(p,p->batch_request,1,1,1,1,args)) goto failure;
        p->active_request=p->request;
        if(p->batch_executable[batch_index]) {
            if(c->graph_launch(p->batch_executable[batch_index],c->stream)!=CUDA_SUCCESS) goto failure;
            plan_batch_graph_count_replay(p,&p->batch_counts[batch_index]);
            ++p->stats.graph_replays; ++p->batch_graph_replays;
            goto read_batch_logits;
        }
        batch_start=plan_batch_graph_snapshot(p);
        if(c->capture_begin(c->stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)!=CUDA_SUCCESS) goto failure;
        capturing=1;
    }
    if (p->use_graphs && count == 1 && logits) {''')
    replace('''        void *gather[] = {&embedding,&p->x,&n,&row_bytes,&type,&packet,&batch,&scale};
        RUN(plan_launch(p,p->gather_batch,(d*count+255)/256,gather));''', '''        if(batch_graph) {
            void *gather[]={&embedding,&p->x,&n,&row_bytes,&type,&batch,&scale,&p->active_request};
            RUN(plan_launch(p,p->graph_gather,(d*count+255)/256,gather));
        } else {
            void *gather[] = {&embedding,&p->x,&n,&row_bytes,&type,&packet,&batch,&scale};
            RUN(plan_launch(p,p->gather_batch,(d*count+255)/256,gather));
        }''')
    replace('            RUN(plan_read_logits_bank(p,output,count));', '''            if(capturing) {
                CUresult ended=c->capture_end(c->stream,&graph); capturing=0;
                if(ended!=CUDA_SUCCESS) goto failure;
                if(c->graph_instantiate(&p->batch_executable[batch_index],graph,0)!=CUDA_SUCCESS) goto failure;
                nya_cuda_batch_graph_counts after=plan_batch_graph_snapshot(p);
                for(size_t i=0;i<9;++i)
                    p->batch_counts[batch_index].values[i]=after.values[i]-batch_start.values[i];
                ++p->stats.graph_captures; ++p->batch_graph_captures;
                c->graph_free(graph); graph=NULL;
                if(c->graph_launch(p->batch_executable[batch_index],c->stream)!=CUDA_SUCCESS) goto failure;
                ++p->stats.graph_replays; ++p->batch_graph_replays;
            }
read_batch_logits:
            RUN(plan_read_logits_bank(p,output,count));''')
    return source
