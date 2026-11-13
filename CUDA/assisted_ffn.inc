/* Native C host lowering for an optional assisted feed-forward chain. Existing
   native CUDA kernels retain intermediates on device; no optcpp dependency. */
static int assisted_ffn_layout(size_t rows,size_t columns,unsigned type,size_t bytes,size_t *stride)
{
    size_t block=1,block_bytes=4;
    switch(type) {
        case 0:break;
        case 1:case 30:block_bytes=2;break;
        case 2:block=32;block_bytes=18;break;
        case 8:block=32;block_bytes=34;break;
        case 12:block=256;block_bytes=144;break;
        case 14:block=256;block_bytes=210;break;
        default:return -1;
    }
    if(!rows || !columns || columns%block || columns/block>SIZE_MAX/block_bytes) return -1;
    *stride=columns/block*block_bytes;
    return rows>SIZE_MAX/(*stride) || rows*(*stride)!=bytes?-1:0;
}
static size_t assisted_ffn_weight(nya_cuda_context *c,const void *source,size_t rows,size_t columns,unsigned type)
{
    size_t i=0;
    for(;i<c->weight_count;++i) {
        const nya_cuda_weight *w=&c->weights[i];
        if(w->source==source && w->rows==rows && w->columns==columns && w->type==type) break;
    }
    return i;
}
static int assisted_ffn_dot(nya_cuda_context *c,CUdeviceptr matrix,CUdeviceptr input,
    CUdeviceptr output,size_t rows,size_t columns,size_t row_bytes,unsigned type)
{
    unsigned long long n=columns,stride=row_bytes,count=rows;
    void *args[]={&matrix,&input,&output,&n,&stride,&count};
    return c->launch(c->dot[type],(unsigned)((rows+7)/8),1,1,256,1,1,0,c->stream,args,NULL)==CUDA_SUCCESS?0:-1;
}
static int cuda_assisted_ffn(void *opaque,const nya_compute_ffn *f)
{
    nya_cuda_context *c=opaque;
    const char *enabled=getenv("NYA_CUDA_ASSISTED_FFN");
    if(!nya_cuda_active(c) || c->reference || !enabled || strcmp(enabled,"1")) return -1;
    size_t d=f->width,h=f->hidden;
    if(d>(size_t)c->grid_limit || h>(size_t)c->grid_limit || d>SIZE_MAX/sizeof(float) ||
       h>(SIZE_MAX/sizeof(float)-d)/2) return -1;
    const void *sources[]={f->gate,f->up,f->down};
    size_t rows[]={h,h,d},columns[]={d,d,h},bytes[]={f->gate_bytes,f->up_bytes,f->down_bytes},strides[3];
    unsigned types[]={f->gate_type,f->up_type,f->down_type};
    size_t extra=0,missing=0;
    for(size_t i=0;i<3;++i) {
        if(assisted_ffn_layout(rows[i],columns[i],types[i],bytes[i],&strides[i])) return -1;
        if(assisted_ffn_weight(c,sources[i],rows[i],columns[i],types[i])==c->weight_count) {
            if(bytes[i]>SIZE_MAX-extra) return -1;
            extra+=bytes[i];++missing;
        }
    }
    size_t input_bytes=d*sizeof(float),output_bytes=(2*h+d)*sizeof(float);
    size_t reserved=c->cached_bytes;
    size_t additions[]={extra,input_bytes>c->input_bytes?input_bytes:c->input_bytes,
        output_bytes>c->output_bytes?output_bytes:c->output_bytes,c->blas_bytes};
    for(size_t i=0;i<4;++i) {
        if(reserved>c->cache_limit || additions[i]>c->cache_limit-reserved) return -1;
        reserved+=additions[i];
    }
    if(missing>NYA_CUDA_WEIGHT_LIMIT-c->weight_count) return -1;
    CUcontext previous;int result=-1;CUdeviceptr matrices[3];
    if(c->push(c->context)!=CUDA_SUCCESS) {c->failed=1;return -1;}
    if(!c->assisted_vector && c->function_get(&c->assisted_vector,c->module,"nya_vector")!=CUDA_SUCCESS) goto done;
    for(size_t i=0;i<3;++i) {
        size_t index=assisted_ffn_weight(c,sources[i],rows[i],columns[i],types[i]);
        if(index==c->weight_count) {
            nya_cuda_weight *w=&c->weights[index];
            if(c->allocate(&w->memory,bytes[i])!=CUDA_SUCCESS) goto done;
            /* Own each allocation before any upload can fail. */
            w->source=sources[i];w->rows=rows[i];w->columns=columns[i];w->type=types[i];w->bytes=bytes[i];
            ++c->weight_count;c->cached_bytes+=bytes[i];
            if(c->upload(w->memory,sources[i],bytes[i])!=CUDA_SUCCESS) goto done;
        }
        matrices[i]=c->weights[index].memory;
    }
    if(nya_cuda_grow(c,&c->input,&c->input_bytes,input_bytes) ||
       nya_cuda_grow(c,&c->output,&c->output_bytes,output_bytes) ||
       c->upload(c->input,f->input,input_bytes)!=CUDA_SUCCESS || c->stream_sync(NULL)!=CUDA_SUCCESS) goto done;
    CUdeviceptr gate=c->output,up=gate+h*sizeof(float),output=up+h*sizeof(float);
    if(assisted_ffn_dot(c,matrices[0],c->input,gate,h,d,strides[0],types[0]) ||
       assisted_ffn_dot(c,matrices[1],c->input,up,h,d,strides[1],types[1])) goto done;
    unsigned long long length=h;unsigned kind=f->gelu?3U:2U;float scale=1;
    void *args[]={&gate,&gate,&up,&length,&kind,&scale};
    if(c->launch(c->assisted_vector,(unsigned)((h+255)/256),1,1,256,1,1,0,c->stream,args,NULL)!=CUDA_SUCCESS ||
       assisted_ffn_dot(c,matrices[2],gate,output,d,h,strides[2],types[2]) ||
       c->stream_sync(c->stream)!=CUDA_SUCCESS || c->download(f->output,output,input_bytes)!=CUDA_SUCCESS) goto done;
    ++c->assisted_ffn_calls;result=0;
done:
    if(result) c->stream_sync(c->stream);
    if(c->pop(&previous)!=CUDA_SUCCESS) result=-1;
    if(result) c->failed=1;
    return result;
}
