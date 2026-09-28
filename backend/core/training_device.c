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
nya_train_scope nya_train_device_scratch_begin(nya_train_device *d)
{ return d ? d->api->scratch_begin(d->context) : 0; }
int nya_train_device_scratch_end(nya_train_device *d, nya_train_scope scope)
{ return d ? d->api->scratch_end(d->context,scope) : -1; }
int nya_train_device_linear(nya_train_device *d, nya_train_buffer y, nya_train_buffer w,
    unsigned type, size_t o, size_t i, nya_train_buffer x, size_t n)
{ return d ? d->api->matrix(d->context,y,w,x,type,o,i,n,0) : -1; }
int nya_train_device_linear_dx(nya_train_device *d, nya_train_buffer dx, nya_train_buffer w,
    unsigned type, size_t o, size_t i, nya_train_buffer dy, size_t n)
{ return d ? d->api->matrix(d->context,dx,w,dy,type,o,i,n,1) : -1; }
int nya_train_device_linear_dw(nya_train_device *d, nya_train_buffer dw, nya_train_buffer x,
    nya_train_buffer dy, size_t o, size_t i, size_t n)
{ return d ? d->api->matrix(d->context,dw,x,dy,0,o,i,n,2) : -1; }
