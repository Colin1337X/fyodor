#include "training_device_backend.h"
#include <stdlib.h>
#include <string.h>

struct nya_train_device { const nya_train_device_interface *api; void *context; };
#ifdef NYA_ENABLE_CUDA
const nya_train_device_interface *nya_cuda_training_device(void);
#endif
nya_train_device *nya_train_device_create(const char *backend, size_t capacity)
{
    const nya_train_device_interface *api = NULL;
    if (!backend || !capacity) return NULL;
#ifdef NYA_ENABLE_CUDA
    if (!strcmp(backend,"cuda")) api = nya_cuda_training_device();
#endif
    if (!api) return NULL;
    nya_train_device *d = calloc(1,sizeof(*d));
    if (!d) return NULL;
    d->api = api; d->context = api->create(capacity);
    if (!d->context) { free(d); return NULL; }
    return d;
}
void nya_train_device_free(nya_train_device *d)
{ if (d) { d->api->destroy(d->context); free(d); } }
const char *nya_train_device_error(const nya_train_device *d)
{ return d ? d->api->error(d->context) : "training device unavailable"; }
void nya_train_device_get_stats(const nya_train_device *d, nya_train_device_stats *s)
{ if (s) { memset(s,0,sizeof(*s)); if (d) d->api->stats(d->context,s); } }
nya_train_buffer nya_train_device_alloc(nya_train_device *d, size_t bytes)
{ return d ? d->api->allocate(d->context,bytes) : 0; }
int nya_train_device_write(nya_train_device *d, nya_train_buffer b, size_t offset, const void *p, size_t bytes)
{ return d ? d->api->transfer(d->context,b,offset,(void *)p,bytes,1) : -1; }
int nya_train_device_read(nya_train_device *d, nya_train_buffer b, size_t offset, void *p, size_t bytes)
{ return d ? d->api->transfer(d->context,b,offset,p,bytes,0) : -1; }
int nya_train_device_zero(nya_train_device *d, nya_train_buffer b)
{ return d ? d->api->zero(d->context,b) : -1; }
int nya_train_device_finish(nya_train_device *d)
{ return d ? d->api->finish(d->context) : -1; }
nya_train_indices nya_train_device_indices(nya_train_device *d, const uint32_t *ids, size_t count, size_t rows)
{
    if (!d) return 0;
    nya_train_index_pack pack;
    if (nya_train_index_pack_create(&pack,ids,count,rows)) return d->api->indices(d->context,NULL);
    nya_train_indices result=d->api->indices(d->context,&pack);
    nya_train_index_pack_free(&pack);
    return result;
}
int nya_train_device_embedding(nya_train_device *d, nya_train_buffer output, nya_train_buffer table,
    unsigned type, size_t columns, nya_train_indices indices)
{ return d ? d->api->embedding(d->context,output,table,type,columns,indices,0) : -1; }
int nya_train_device_embedding_backward(nya_train_device *d, nya_train_buffer dtable,
    nya_train_buffer dy, size_t columns, nya_train_indices indices)
{ return d ? d->api->embedding(d->context,dtable,dy,0,columns,indices,1) : -1; }
nya_train_scope nya_train_device_scratch_begin(nya_train_device *d)
{ return d ? d->api->scratch_begin(d->context) : 0; }
int nya_train_device_scratch_end(nya_train_device *d, nya_train_scope scope)
{ return d ? d->api->scratch_end(d->context,scope) : -1; }
int nya_train_device_check_finite(nya_train_device *d, nya_train_buffer status, nya_train_buffer data, size_t count, uint32_t tag)
{ return d ? d->api->check_finite(d->context,status,data,count,tag) : -1; }
int nya_train_device_unary(nya_train_device *d, nya_train_buffer y, nya_train_buffer x,
    size_t count, nya_train_unary_op operation, double scalar)
{ return d ? d->api->unary(d->context,y,x,0,count,(unsigned)operation,scalar,0) : -1; }
int nya_train_device_unary_backward(nya_train_device *d, nya_train_buffer dx, nya_train_buffer x,
    nya_train_buffer dy, size_t count, nya_train_unary_op operation, double scalar)
{ return d ? d->api->unary(d->context,dx,x,dy,count,(unsigned)operation,scalar,1) : -1; }
int nya_train_device_binary(nya_train_device *d, nya_train_buffer y, nya_train_view a, nya_train_view b,
    nya_train_binary_op operation)
{ return d ? d->api->binary(d->context,y,0,a,b,0,(unsigned)operation,0) : -1; }
int nya_train_device_binary_backward(nya_train_device *d, nya_train_buffer da, nya_train_buffer db,
    nya_train_view a, nya_train_view b, nya_train_buffer dy, nya_train_binary_op operation)
{ return d ? d->api->binary(d->context,da,db,a,b,dy,(unsigned)operation,1) : -1; }
int nya_train_device_linear(nya_train_device *d, nya_train_buffer y, nya_train_buffer w,
    unsigned type, size_t o, size_t i, nya_train_buffer x, size_t n)
{ return d ? d->api->matrix(d->context,y,w,x,type,o,i,n,0) : -1; }
int nya_train_device_rms_norm(nya_train_device *d, nya_train_buffer y, nya_train_buffer inverse,
    nya_train_view x, nya_train_buffer weight, float epsilon)
{ return d ? d->api->rms_norm(d->context,y,0,inverse,x,weight,0,epsilon,0) : -1; }
int nya_train_device_rms_norm_backward(nya_train_device *d, nya_train_buffer dx, nya_train_buffer dweight,
    nya_train_buffer inverse, nya_train_view x, nya_train_buffer weight, nya_train_buffer dy)
{ return d ? d->api->rms_norm(d->context,dx,dweight,inverse,x,weight,dy,0,1) : -1; }
int nya_train_device_linear_dx(nya_train_device *d, nya_train_buffer dx, nya_train_buffer w,
    unsigned type, size_t o, size_t i, nya_train_buffer dy, size_t n)
{ return d ? d->api->matrix(d->context,dx,w,dy,type,o,i,n,1) : -1; }
int nya_train_device_linear_dw(nya_train_device *d, nya_train_buffer dw, nya_train_buffer x,
    nya_train_buffer dy, size_t o, size_t i, size_t n)
{ return d ? d->api->matrix(d->context,dw,x,dy,0,o,i,n,2) : -1; }

int nya_train_device_rope(nya_train_device *d, nya_train_buffer y, nya_train_view x,
    size_t heads, size_t dimension, nya_train_buffer frequencies, int split_half)
{ return d ? d->api->rope(d->context,y,x,heads,dimension,frequencies,split_half,0) : -1; }
int nya_train_device_rope_backward(nya_train_device *d, nya_train_buffer dx, nya_train_view dy,
    size_t heads, size_t dimension, nya_train_buffer frequencies, int split_half)
{ return d ? d->api->rope(d->context,dx,dy,heads,dimension,frequencies,split_half,1) : -1; }

size_t nya_train_attention_workspace_bytes(size_t tokens, size_t heads)
{
    size_t tile=tokens<16?tokens:16;
    if (!tokens || !heads || heads>SIZE_MAX/tokens || tokens*heads>SIZE_MAX/tile/16) return 0;
    return tokens*heads*tile*16;
}
size_t nya_train_loss_state_bytes(size_t rows)
{ return rows && rows <= (SIZE_MAX-8)/32 ? rows*32+8 : 0; }
int nya_train_device_loss(nya_train_device *d, nya_train_buffer y, nya_train_buffer state,
    nya_train_view logits, nya_train_buffer labels, nya_train_buffer mask, nya_train_loss_op operation)
{ return d ? d->api->loss(d->context,y,state,logits,labels,mask,0,(unsigned)operation,0) : -1; }
int nya_train_device_loss_backward(nya_train_device *d, nya_train_buffer dx, nya_train_buffer state,
    nya_train_view logits, nya_train_buffer labels, nya_train_buffer mask, nya_train_buffer dy)
{ return d ? d->api->loss(d->context,dx,state,logits,labels,mask,dy,0,1) : -1; }
int nya_train_device_dpo(nya_train_device *d, nya_train_buffer y, nya_train_buffer state,
    nya_train_buffer chosen, nya_train_buffer rejected, double rc, double rr, float beta)
{ return d ? d->api->dpo(d->context,y,0,state,chosen,rejected,0,rc,rr,beta,0) : -1; }
int nya_train_device_dpo_backward(nya_train_device *d, nya_train_buffer dc, nya_train_buffer dr,
    nya_train_buffer state, nya_train_buffer dy)
{ return d ? d->api->dpo(d->context,dc,dr,state,0,0,dy,0,0,0,1) : -1; }
int nya_train_device_attention(nya_train_device *d, nya_train_buffer y,
    nya_train_buffer state, nya_train_attention_desc descriptor)
{ return d ? d->api->attention(d->context,y,0,0,state,0,0,descriptor,0) : -1; }
int nya_train_device_attention_backward(nya_train_device *d, nya_train_buffer dq,
    nya_train_buffer dk, nya_train_buffer dv, nya_train_buffer state, nya_train_buffer dy,
    nya_train_buffer workspace, nya_train_attention_desc descriptor)
{ return d ? d->api->attention(d->context,dq,dk,dv,state,dy,workspace,descriptor,1) : -1; }
