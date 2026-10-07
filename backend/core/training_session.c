#include "training_session.h"
#include "training_parameter.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct nya_train_session {
    nya_train_device *device;
    nya_train_parameter **parameters;
    nya_train_adamw_tensor *tensors;
    size_t count, elements;
    nya_train_optimizer_plan plan;
    nya_train_buffer step, status, norm;
    nya_train_adamw optimizer;
    nya_train_adamw_config pending_settings;
    int leased, pending;
    char error[160];
};

static int session_error(nya_train_session *s, const char *message)
{ if (s) snprintf(s->error,sizeof(s->error),"%s",message); return -1; }
static int session_valid(const nya_train_session *s)
{ return s && s->device && s->leased; }
static int settings_valid(nya_train_adamw_config c)
{
    return isfinite(c.learning_rate) && c.learning_rate>0 && isfinite(c.beta1) && c.beta1>=0 && c.beta1<1 &&
        isfinite(c.beta2) && c.beta2>=0 && c.beta2<1 && isfinite(c.epsilon) && c.epsilon>0 &&
        isfinite(c.weight_decay) && c.weight_decay>=0 && isfinite(c.max_grad_norm) && c.max_grad_norm>=0;
}
static nya_train_adamw_config settings_of(nya_train_adamw o)
{ return (nya_train_adamw_config){o.learning_rate,o.beta1,o.beta2,o.epsilon,o.weight_decay,o.max_grad_norm}; }
static void set_settings(nya_train_adamw *o, nya_train_adamw_config c)
{
    o->learning_rate=c.learning_rate; o->beta1=c.beta1; o->beta2=c.beta2;
    o->epsilon=c.epsilon; o->weight_decay=c.weight_decay; o->max_grad_norm=c.max_grad_norm;
}
static void session_unlease(nya_train_session *s)
{
    if (s->leased) for (size_t i=0;i<s->count;++i) {
        s->parameters[i]->resident_owner=NULL;
        nya_train_parameter_free(s->parameters[i]);
    }
    s->leased=0;
}
void nya_train_session_free(nya_train_session *s)
{
    if (!s) return;
    nya_train_device_free(s->device);
    session_unlease(s);
    free(s->tensors); free(s->parameters); free(s);
}
const char *nya_train_session_error(const nya_train_session *s)
{ return s?s->error:"missing resident training session"; }
nya_train_device *nya_train_session_device(nya_train_session *s)
{ return session_valid(s)?s->device:NULL; }
nya_train_buffer nya_train_session_status(nya_train_session *s)
{ return session_valid(s)?s->status:0; }
int nya_train_session_tensor(nya_train_session *s,size_t index,nya_train_adamw_tensor *tensor)
{
    if (!session_valid(s) || !tensor || index>=s->count) return session_error(s,"invalid session tensor index");
    *tensor=s->tensors[index]; return 0;
}

nya_train_session *nya_train_session_create(const char *backend,size_t capacity,
    const nya_train_adamw *optimizer,nya_train_parameter *const *parameters,size_t count,char *error,size_t error_capacity)
{
    const char *failure="invalid resident session settings or parameter list";
    nya_train_session *s=NULL;
    size_t total=0;
    if (!backend || !capacity || !optimizer || !settings_valid(settings_of(*optimizer)) || !parameters || !count ||
        count>SIZE_MAX/sizeof(nya_train_adamw_tensor) || count>SIZE_MAX/sizeof(*parameters)) goto fail;
    for (size_t i=0;i<count;++i) {
        nya_train_parameter *p=parameters[i];
        if (!p || p->resident_owner || p->cpu_leaves || p->references==SIZE_MAX || p->count>SIZE_MAX-total) goto fail;
        for (size_t j=0;j<i;++j) if (parameters[j]==p) goto fail;
        total+=p->count;
        float *arrays[]={p->data,p->gradient,p->moment,p->variance};
        for (size_t a=0;a<4;++a) for (size_t k=0;k<p->count;++k)
            if (!isfinite(arrays[a][k]) || (a==3 && arrays[a][k]<0)) goto fail;
    }
    if (total>SIZE_MAX/(4*sizeof(float))) goto fail;
    failure="resident session host allocation failed";
    s=calloc(1,sizeof(*s)); if (!s) goto fail;
    s->parameters=malloc(count*sizeof(*s->parameters)); s->tensors=calloc(count,sizeof(*s->tensors));
    if (!s->parameters || !s->tensors) goto fail;
    memcpy(s->parameters,parameters,count*sizeof(*parameters)); s->count=count; s->elements=total; s->optimizer=*optimizer;
    failure="resident training device unavailable";
    s->device=nya_train_device_create(backend,capacity); if (!s->device) goto fail;
    for (size_t i=0;i<count;++i) {
        nya_train_parameter *p=parameters[i]; size_t bytes=p->count*sizeof(float);
        nya_train_adamw_tensor *t=s->tensors+i;
        *t=(nya_train_adamw_tensor){nya_train_device_alloc(s->device,bytes),nya_train_device_alloc(s->device,bytes),
            nya_train_device_alloc(s->device,bytes),nya_train_device_alloc(s->device,bytes),p->count};
        nya_train_buffer ids[]={t->values,t->gradient,t->moment,t->variance};
        float *arrays[]={p->data,p->gradient,p->moment,p->variance};
        for (size_t a=0;a<4;++a) if (!ids[a] || nya_train_device_write(s->device,ids[a],0,arrays[a],bytes)) goto device_fail;
    }
    s->step=nya_train_device_alloc(s->device,8); s->status=nya_train_device_alloc(s->device,4); s->norm=nya_train_device_alloc(s->device,8);
    s->plan=nya_train_device_adamw_plan(s->device,s->tensors,count);
    if (!s->step || !s->status || !s->norm || !s->plan ||
        nya_train_device_write(s->device,s->step,0,&optimizer->step,8) || nya_train_device_finish(s->device)) goto device_fail;
    for (size_t i=0;i<count;++i) { ++parameters[i]->references; parameters[i]->resident_owner=s; }
    s->leased=1;
    if (error && error_capacity) error[0]='\0';
    return s;
device_fail:
    failure=nya_train_device_error(s->device);
fail:
    if (error && error_capacity) snprintf(error,error_capacity,"%s",failure);
    nya_train_session_free(s); return NULL;
}

int nya_train_session_zero_grad(nya_train_session *s)
{
    if (!session_valid(s) || s->pending) return session_error(s,"observe the pending update before resetting gradients");
    for (size_t i=0;i<s->count;++i) if (nya_train_device_zero(s->device,s->tensors[i].gradient))
        return session_error(s,nya_train_device_error(s->device));
    if (nya_train_device_zero(s->device,s->status)) return session_error(s,nya_train_device_error(s->device));
    s->error[0]='\0'; return 0;
}
int nya_train_session_step(nya_train_session *s,nya_train_adamw_config settings)
{
    if (!session_valid(s) || s->pending || !settings_valid(settings))
        return session_error(s,"invalid settings or update still awaiting observation");
    if (nya_train_device_adamw(s->device,s->plan,settings,s->step,s->status,s->norm,1))
        return session_error(s,nya_train_device_error(s->device));
    s->pending_settings=settings; s->pending=1; return 0;
}
int nya_train_session_observe(nya_train_session *s,nya_train_session_metrics *metrics)
{
    if (!session_valid(s) || !metrics) return session_error(s,"missing session or metrics destination");
    nya_train_session_metrics m={0};
    if (nya_train_device_read(s->device,s->status,0,&m.status,4) || nya_train_device_read(s->device,s->step,0,&m.step,8) ||
        nya_train_device_read(s->device,s->norm,0,&m.gradient_norm,8)) return session_error(s,nya_train_device_error(s->device));
    uint64_t expected=s->optimizer.step;
    if (s->pending && !m.status) {
        if (expected==UINT64_MAX) return session_error(s,"resident optimizer counter overflow");
        ++expected;
    }
    if (m.step!=expected || !isfinite(m.gradient_norm) || m.gradient_norm<0)
        return session_error(s,"inconsistent resident optimizer state");
    if (s->pending && !m.status) { set_settings(&s->optimizer,s->pending_settings); s->optimizer.step=m.step; }
    s->pending=0; *metrics=m;
    if (m.status) return session_error(s,"resident numerical failure; correct inputs and reset gradients");
    s->error[0]='\0'; return 0;
}

/* Independent CPU-shaped staging lets the existing checkpoint serializer keep
   its validation/format contract. No original parameter is temporarily unlocked
   or partially refreshed, even when a later transfer or value check fails. */
typedef struct session_snapshot {
    nya_train_parameter *storage, **parameters;
    float *arrays;
} session_snapshot;
static void snapshot_free(session_snapshot *snapshot)
{ free(snapshot->storage); free(snapshot->parameters); free(snapshot->arrays); }
static int snapshot_read(nya_train_session *s,session_snapshot *snapshot)
{
    nya_train_session_metrics metrics;
    if (nya_train_session_observe(s,&metrics)) return -1;
    snapshot->storage=calloc(s->count,sizeof(*snapshot->storage));
    snapshot->parameters=malloc(s->count*sizeof(*snapshot->parameters));
    snapshot->arrays=malloc(s->elements*4*sizeof(float));
    if (!snapshot->storage || !snapshot->parameters || !snapshot->arrays) return session_error(s,"resident snapshot staging allocation failed");
    size_t offset=0;
    for (size_t i=0;i<s->count;++i) {
        nya_train_parameter *p=snapshot->storage+i, *source=s->parameters[i];
        p->rows=source->rows; p->columns=source->columns; p->count=source->count;
        p->data=snapshot->arrays+offset; p->gradient=p->data+p->count; p->moment=p->gradient+p->count; p->variance=p->moment+p->count;
        snapshot->parameters[i]=p; offset+=4*p->count;
        float *arrays[]={p->data,p->gradient,p->moment,p->variance};
        nya_train_adamw_tensor *t=s->tensors+i;
        nya_train_buffer ids[]={t->values,t->gradient,t->moment,t->variance};
        for (size_t a=0;a<4;++a) {
            if (nya_train_device_read(s->device,ids[a],0,arrays[a],p->count*sizeof(float)))
                return session_error(s,nya_train_device_error(s->device));
            for (size_t k=0;k<p->count;++k) if (!isfinite(arrays[a][k]) || (a==3 && arrays[a][k]<0))
                return session_error(s,"resident snapshot contains invalid parameter state");
        }
    }
    return 0;
}
int nya_train_session_checkpoint_write(nya_train_session *s,FILE *file)
{
    session_snapshot snapshot={0}; int result=-1;
    if (!session_valid(s) || !file) return session_error(s,"missing session or checkpoint stream");
    if (!snapshot_read(s,&snapshot)) {
        result=nya_train_checkpoint_write(file,&s->optimizer,snapshot.parameters,s->count);
        if (result) session_error(s,"resident checkpoint write failed; discard incomplete stream");
    }
    snapshot_free(&snapshot); return result;
}
int nya_train_session_detach(nya_train_session *s,nya_train_adamw *optimizer)
{
    session_snapshot snapshot={0};
    if (!session_valid(s) || !optimizer) return session_error(s,"missing session or optimizer destination");
    if (snapshot_read(s,&snapshot)) { snapshot_free(&snapshot); return -1; }
    for (size_t i=0;i<s->count;++i) {
        nya_train_parameter *to=s->parameters[i], *from=snapshot.parameters[i]; size_t bytes=to->count*sizeof(float);
        memcpy(to->data,from->data,bytes); memcpy(to->gradient,from->gradient,bytes);
        memcpy(to->moment,from->moment,bytes); memcpy(to->variance,from->variance,bytes);
    }
    *optimizer=s->optimizer; snapshot_free(&snapshot);
    nya_train_device_free(s->device); s->device=NULL;
    session_unlease(s); s->error[0]='\0'; return 0;
}
