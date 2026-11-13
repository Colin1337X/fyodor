"""Experimental host edits for a fixed Gemma diagnostic, not runtime defaults."""
def transform(source):
 def replace(old,new,count=1):
  nonlocal source
  assert source.count(old)==count,old
  source=source.replace(old,new)
 replace('    CUfunction fdpp_partial, fdpp_combine;',
         '    CUfunction fdpp_partial, fdpp_combine, fdpp_batch_partial, fdpp_batch_combine;\n    unsigned long long fdpp_batch_calls;')
 replace('batch_partial_count?(batch>=4?4:batch>=2?2:0):0',
         'batch_partial_count?batch:0',2)
 anchor='    const char *mtp_trace=getenv("NYA_MTP_TRACE");'
 replace(anchor,anchor+'\n    fprintf(stderr,"CUDA aligned FDPP batch calls: %llu\\n",p->fdpp_batch_calls);')
 anchor='    if (c->function_get(&p->fdpp_partial,c->module,fdpp_partial_name) != CUDA_SUCCESS ||'
 replace(anchor,'''    if (c->function_get(&p->fdpp_batch_partial,c->module,"nya_fdpp_partial_batch") != CUDA_SUCCESS ||
        c->function_get(&p->fdpp_batch_combine,c->module,"nya_fdpp_combine_batch") != CUDA_SUCCESS) goto failure;
'''+anchor)
 replace('int wide_batch=p->use_wide_batch && (count==2 || count==4) && position>=256;',
         'int wide_batch=p->use_wide_batch && count>1 && position>=256;')
 replace('''int batch_split=wide_batch && count<=p->wide_partial_queries && (!window || window>256) &&
                (count==2 || (count==4 && width==512 && !window && position>=8192));''',
         'int batch_split=wide_batch && count<=p->wide_partial_queries && (!window || window>256);')
 anchor='        if(p->use_wide_value && ((count==1 && graph_index) || wide_batch) && (width==256 || width==512)) {'
 replace(anchor,'''        if(count>1 && position<256 && count<=256-position && width==256 && p->use_fdpp==2) {
            unsigned partitions=4;
            void *partial[]={&p->q,&keys,&values,&p->attention_parts,&width,&nh,&nk,
                &window,&end,&scale,&p->active_request,&partitions};
            RUN(plan_grid(p,p->fdpp_batch_partial,heads,count*partitions,256,1,partial));
            void *combine[]={&p->q,&keys,&values,&p->norm,&p->attention_parts,&width,&nh,&nk,
                &window,&end,&scale,&p->active_request,&partitions,&p->scores,&capacity};
            RUN(plan_grid(p,p->fdpp_batch_combine,heads,count,256,1,combine));
            ++p->fdpp_batch_calls;
        } else
'''+anchor)
 anchor='''        size_t chunk = count - i < p->maximum_batch ? count - i : p->maximum_batch;
        if (cuda_plan_batch(p, tokens + i, chunk, position + i, i + chunk == count, output)) return -1;'''
 replace(anchor,'''        size_t chunk = count - i < p->maximum_batch ? count - i : p->maximum_batch;
        if(position+i<256 && chunk>256-position-i)chunk=256-position-i;
        if (cuda_plan_batch(p, tokens + i, chunk, position + i, i + chunk == count, output)) return -1;''')
 return source
