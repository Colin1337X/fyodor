#include "training_device.h"
#include "training.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,nya_train_device_error(d)); return 1; } } while (0)
static int close_value(double expected,float actual)
{
    if (isfinite(actual) && fabs(expected-actual)<=1e-6*(1+fabs(expected))) return 1;
    fprintf(stderr,"expected %.12g actual %.9g\n",expected,(double)actual); return 0;
}
static int attention_case(nya_train_device *d,size_t n,size_t heads,size_t kv,size_t dim,size_t window,int grouped,unsigned optional)
{
    if (getenv("NYA_TRAIN_TEST_TRACE")) { fprintf(stderr,"case n=%zu heads=%zu kv=%zu dim=%zu window=%zu groups=%d optional=%u\n",n,heads,kv,dim,window,grouped,optional); fflush(stderr); }
    size_t nq=n*heads*dim,nk=n*kv*dim,counts[3]={nq,nk,nk};
    float *values[3],*seed=malloc(nq*4),*actual=malloc((nq+1)*4),*expected=malloc(nq*4);
    uint32_t *groups=malloc(n*4); CHECK(seed && actual && expected && groups);
    nya_train_parameter *parameters[3];
    for (size_t a=0;a<3;++a) {
        values[a]=malloc(counts[a]*4); CHECK(values[a]);
        for (size_t j=0;j<counts[a];++j) values[a][j]=(float)((int)((j*7+a*3)%37)-18)/16;
        parameters[a]=nya_train_parameter_create(n,counts[a]/n,values[a]); CHECK(parameters[a]);
        for (size_t j=0;j<counts[a];++j) actual[j]=0.125f;
        nya_train_graph *initial=nya_train_graph_create(64*1024*1024); CHECK(initial);
        CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(nya_train_leaf(initial,parameters[a]),1,counts[a]),
            nya_train_input(initial,1,counts[a],actual))));
        nya_train_graph_free(initial);
    }
    for (size_t j=0;j<nq;++j) seed[j]=(float)((int)(j%19)-9)/16;
    for (size_t j=0;j<n;++j) groups[j]=grouped==2?(j%3?UINT32_MAX:0):(j%5?(uint32_t)(j/3+1):0);
    float scale=0.7f;
    nya_train_graph *graph=nya_train_graph_create(64*1024*1024); CHECK(graph);
    nya_train_tensor *out=nya_train_attention(nya_train_leaf(graph,parameters[0]),nya_train_leaf(graph,parameters[1]),
        nya_train_leaf(graph,parameters[2]),heads,kv,dim,scale,window,grouped?groups:NULL); CHECK(out);
    memcpy(expected,nya_train_data(out),nq*4);
    CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(out,1,nq),nya_train_input(graph,1,nq,seed))));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer inputs[3],gradients[3];
    for (size_t a=0;a<3;++a) {
        inputs[a]=nya_train_device_alloc(d,counts[a]*4); gradients[a]=nya_train_device_alloc(d,(counts[a]+1)*4);
        CHECK(inputs[a] && gradients[a] && !nya_train_device_write(d,inputs[a],0,values[a],counts[a]*4));
        for (size_t j=0;j<counts[a];++j) actual[j]=0.125f;
        actual[counts[a]]=12345;
        CHECK(!nya_train_device_write(d,gradients[a],0,actual,(counts[a]+1)*4));
    }
    size_t work_bytes=nya_train_attention_workspace_bytes(n,heads); CHECK(work_bytes);
    nya_train_buffer y=nya_train_device_alloc(d,(nq+1)*4),dy=nya_train_device_alloc(d,nq*4),state=nya_train_device_alloc(d,n*heads*16+8);
    nya_train_buffer workspace=nya_train_device_alloc(d,work_bytes+8),bg=nya_train_device_alloc(d,n*4);
    CHECK(y && dy && state && workspace && bg && !nya_train_device_write(d,dy,0,seed,nq*4) && !nya_train_device_write(d,bg,0,groups,n*4));
    float sentinel=12345; double guard=98765;
    CHECK(!nya_train_device_write(d,y,nq*4,&sentinel,4) && !nya_train_device_write(d,state,n*heads*16,&guard,8) &&
        !nya_train_device_write(d,workspace,work_bytes,&guard,8));
    nya_train_attention_desc desc={{inputs[0],n,heads*dim},{inputs[1],n,kv*dim},{inputs[2],n,kv*dim},heads,kv,dim,window,scale,grouped?bg:0};
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_attention(d,y,state,desc));
    for (unsigned repeat=0;repeat<2;++repeat) CHECK(!nya_train_device_attention_backward(d,
        optional&1?gradients[0]:0,optional&2?gradients[1]:0,optional&4?gradients[2]:0,state,dy,workspace,desc));
    nya_train_device_get_stats(d,&after);
    size_t tile_count=n/16+(n%16!=0),kernels=1+2*tile_count*(1u+((optional&1)?1u:0u)+((optional&6)?1u:0u));
    CHECK(after.kernel_launches==before.kernel_launches+kernels && after.uploads==before.uploads && after.downloads==before.downloads &&
        after.used_bytes==before.used_bytes && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,y,0,actual,(nq+1)*4) && actual[nq]==sentinel);
    for (size_t j=0;j<nq;++j) if (!close_value(expected[j],actual[j])) {
        fprintf(stderr,"forward n=%zu heads=%zu kv=%zu dim=%zu window=%zu groups=%d at=%zu\n",n,heads,kv,dim,window,grouped,j); return 1;
    }
    /* Replay CPU backward to preserve accumulation into parameter gradients. */
    nya_train_graph_free(graph); graph=nya_train_graph_create(64*1024*1024); CHECK(graph);
    out=nya_train_attention(nya_train_leaf(graph,parameters[0]),nya_train_leaf(graph,parameters[1]),nya_train_leaf(graph,parameters[2]),heads,kv,dim,scale,window,grouped?groups:NULL); CHECK(out);
    CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(out,1,nq),nya_train_input(graph,1,nq,seed))));
    for (size_t a=0;a<3;++a) {
        CHECK(!nya_train_device_read(d,gradients[a],0,actual,(counts[a]+1)*4) && actual[counts[a]]==sentinel);
        for (size_t j=0;j<counts[a];++j) {
            double reference=(optional&(1u<<a))?nya_train_parameter_gradient(parameters[a])[j]:0.125;
            if (!close_value(reference,actual[j])) {
                fprintf(stderr,"gradient=%zu n=%zu heads=%zu kv=%zu dim=%zu window=%zu groups=%d at=%zu\n",a,n,heads,kv,dim,window,grouped,j); return 1;
            }
        }
    }
    CHECK(!nya_train_device_read(d,state,n*heads*16,&guard,8) && guard==98765);
    CHECK(!nya_train_device_read(d,workspace,work_bytes,&guard,8) && guard==98765);
    nya_train_graph_free(graph);
    for (size_t a=0;a<3;++a) { free(values[a]); nya_train_parameter_free(parameters[a]); }
    free(groups); free(seed); free(actual); free(expected);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static double objective(double *q,double *k,double *v,const float *seed,int grouped)
{
    double loss=0; const unsigned groups[]={0,1,1};
    for (size_t row=0;row<3;++row) for (size_t head=0;head<2;++head) {
        double p[3]={0},maximum=-INFINITY,mass=0;
        for (size_t col=0;col<3;++col) {
            int visible=col<=row;
            if (grouped) visible=(row<2 || col>=row-1) && (col<=row || (groups[row] && groups[row]==groups[col]));
            if (!visible) { p[col]=-INFINITY; continue; }
            for (size_t j=0;j<3;++j) p[col]+=q[(row*2+head)*3+j]*k[col*3+j];
            p[col]*=(double)0.7f; if (p[col]>maximum) maximum=p[col];
        }
        for (size_t col=0;col<3;++col) { p[col]=exp(p[col]-maximum); mass+=p[col]; }
        for (size_t col=0;col<3;++col) for (size_t j=0;j<3;++j) loss+=seed[(row*2+head)*3+j]*(p[col]/mass)*v[col*3+j];
    }
    return loss;
}
static int differences(nya_train_device *d,int grouped)
{
    double q[18],k[9],v[9],*arrays[]={q,k,v}; float input[18],seed[18],actual[18];
    size_t counts[]={18,9,9}; uint32_t groups[]={0,1,1};
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer buffers[3],grads[3];
    for (size_t a=0;a<3;++a) {
        for (size_t j=0;j<counts[a];++j) input[j]=(float)(arrays[a][j]=((double)((j*5+a*7)%13)-6)/8);
        buffers[a]=nya_train_device_alloc(d,counts[a]*4); grads[a]=nya_train_device_alloc(d,counts[a]*4);
        CHECK(buffers[a] && grads[a] && !nya_train_device_write(d,buffers[a],0,input,counts[a]*4));
    }
    for (size_t j=0;j<18;++j) seed[j]=(float)((int)(j%7)-3)/8;
    nya_train_buffer by=nya_train_device_alloc(d,72),dy=nya_train_device_alloc(d,72),state=nya_train_device_alloc(d,96);
    nya_train_buffer work=nya_train_device_alloc(d,nya_train_attention_workspace_bytes(3,2)),bg=nya_train_device_alloc(d,12);
    CHECK(by && dy && state && work && bg && !nya_train_device_write(d,dy,0,seed,72) && !nya_train_device_write(d,bg,0,groups,12));
    nya_train_attention_desc desc={{buffers[0],3,6},{buffers[1],3,3},{buffers[2],3,3},2,1,3,grouped?2u:0u,0.7f,grouped?bg:0};
    CHECK(!nya_train_device_attention(d,by,state,desc) && !nya_train_device_attention_backward(d,grads[0],grads[1],grads[2],state,dy,work,desc));
    for (size_t a=0;a<3;++a) {
        CHECK(!nya_train_device_read(d,grads[a],0,actual,counts[a]*4));
        for (size_t j=0;j<counts[a];++j) {
            double old=arrays[a][j],h=1e-5; arrays[a][j]=old+h; double plus=objective(q,k,v,seed,grouped);
            arrays[a][j]=old-h; double minus=objective(q,k,v,seed,grouped); arrays[a][j]=old;
            CHECK(close_value((plus-minus)/(2*h),actual[j]));
        }
    }
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int descriptors(nya_train_device *d)
{
    if (getenv("NYA_TRAIN_TEST_TRACE")) { fprintf(stderr,"descriptors\n"); fflush(stderr); }
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer stale=nya_train_device_alloc(d,256); CHECK(stale && !nya_train_device_scratch_end(d,scope));
    scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer b[11]; for (size_t j=0;j<11;++j) { b[j]=nya_train_device_alloc(d,j==10?1:256); CHECK(b[j]); }
    uint32_t id=0; nya_train_indices map=nya_train_device_indices(d,&id,1,1); CHECK(map);
    nya_train_device *other=nya_train_device_create("cuda",1024); CHECK(other);
    nya_train_buffer foreign=nya_train_device_alloc(other,256); CHECK(foreign);
    nya_train_attention_desc desc={{b[0],2,4},{b[1],2,2},{b[2],2,2},2,1,2,2,0.7f,b[9]};
    float sentinels[64]; for (size_t j=0;j<64;++j) sentinels[j]=42;
    CHECK(!nya_train_device_write(d,b[3],0,sentinels,256));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    const nya_train_buffer invalid[]={0,stale,foreign,b[10],map};
    for (size_t j=0;j<5;++j) {
        nya_train_attention_desc bad=desc;
        CHECK(nya_train_device_attention(d,invalid[j],b[4],desc));
        CHECK(nya_train_device_attention(d,b[3],invalid[j],desc));
        bad.q.buffer=invalid[j]; CHECK(nya_train_device_attention(d,b[3],b[4],bad));
        bad=desc; bad.k.buffer=invalid[j]; CHECK(nya_train_device_attention(d,b[3],b[4],bad));
        bad=desc; bad.v.buffer=invalid[j]; CHECK(nya_train_device_attention(d,b[3],b[4],bad));
        if (invalid[j]) { bad=desc; bad.groups=invalid[j]; CHECK(nya_train_device_attention(d,b[3],b[4],bad)); }
        CHECK(nya_train_device_attention_backward(d,invalid[j],0,0,b[4],b[5],b[6],desc));
        CHECK(nya_train_device_attention_backward(d,0,invalid[j],0,b[4],b[5],b[6],desc));
        CHECK(nya_train_device_attention_backward(d,0,0,invalid[j],b[4],b[5],b[6],desc));
        CHECK(nya_train_device_attention_backward(d,b[3],0,0,invalid[j],b[5],b[6],desc));
        CHECK(nya_train_device_attention_backward(d,b[3],0,0,b[4],invalid[j],b[6],desc));
        CHECK(nya_train_device_attention_backward(d,b[3],0,0,b[4],b[5],invalid[j],desc));
    }
    for (size_t j=0;j<15;++j) {
        nya_train_attention_desc bad=desc;
        switch(j) {
            case 0:bad.q.rows=0;break; case 1:bad.q.rows=SIZE_MAX;break;
            case 2:bad.q.columns=SIZE_MAX;break; case 3:bad.k.rows=3;break;
            case 4:bad.v.rows=3;break; case 5:bad.k.columns=4;break;
            case 6:bad.v.columns=4;break; case 7:bad.heads=0;break;
            case 8:bad.kv_heads=0;break; case 9:bad.dimension=0;break;
            case 10:bad.heads=3;bad.kv_heads=2;break; case 11:bad.heads=SIZE_MAX;break;
            case 12:bad.scale=0;break; case 13:bad.scale=INFINITY;break; default:bad.scale=NAN;break;
        }
        CHECK(nya_train_device_attention(d,b[3],b[4],bad));
        CHECK(nya_train_device_attention_backward(d,b[3],b[7],b[8],b[4],b[5],b[6],bad));
    }
    const nya_train_buffer inputs[]={b[0],b[1],b[2],b[4],b[5],b[6],b[9]};
    for (size_t j=0;j<7;++j) {
        CHECK(nya_train_device_attention_backward(d,inputs[j],0,0,b[4],b[5],b[6],desc));
        CHECK(nya_train_device_attention_backward(d,0,inputs[j],0,b[4],b[5],b[6],desc));
        CHECK(nya_train_device_attention_backward(d,0,0,inputs[j],b[4],b[5],b[6],desc));
        if (j<4 || j==6) CHECK(nya_train_device_attention(d,inputs[j],b[4],desc));
        if (j<3 || j==6) CHECK(nya_train_device_attention(d,b[3],inputs[j],desc));
        if (j!=5) CHECK(nya_train_device_attention_backward(d,b[3],0,0,b[4],b[5],inputs[j],desc));
    }
    CHECK(nya_train_device_attention_backward(d,b[3],b[3],0,b[4],b[5],b[6],desc));
    CHECK(nya_train_device_attention_backward(d,b[3],0,b[3],b[4],b[5],b[6],desc));
    CHECK(nya_train_device_attention_backward(d,0,b[3],b[3],b[4],b[5],b[6],desc));
    CHECK(nya_train_device_attention_backward(d,b[3],0,0,b[4],b[4],b[6],desc));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && after.kernel_launches==before.kernel_launches && after.uploads==before.uploads &&
        after.downloads==before.downloads && after.synchronizations==before.synchronizations && after.used_bytes==before.used_bytes);
    CHECK(!nya_train_device_read(d,b[3],0,sentinels,256));
    for (size_t j=0;j<64;++j) CHECK(sentinels[j]==42);
    desc.k.buffer=desc.q.buffer; desc.v.buffer=desc.q.buffer;
    CHECK(!nya_train_device_attention(d,b[3],b[4],desc));
    CHECK(!nya_train_device_attention_backward(d,b[7],b[8],b[5],b[4],b[3],b[6],desc));
    nya_train_device_free(other); CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int scope_reuse(nya_train_device *d)
{
    if (getenv("NYA_TRAIN_TEST_TRACE")) { fprintf(stderr,"scope_reuse\n"); fflush(stderr); }
    float ones[68],actual[34],expected[34]={0}; for (size_t j=0;j<68;++j) ones[j]=1;
    nya_train_buffer q=nya_train_device_alloc(d,272),k=nya_train_device_alloc(d,136),v=nya_train_device_alloc(d,136);
    nya_train_buffer dy=nya_train_device_alloc(d,272),dq=nya_train_device_alloc(d,272),dk=nya_train_device_alloc(d,136),dv=nya_train_device_alloc(d,136);
    CHECK(q && k && v && dy && dq && dk && dv && !nya_train_device_write(d,v,0,ones,136) && !nya_train_device_write(d,dy,0,ones,272));
    nya_train_attention_desc desc={{q,17,4},{k,17,2},{v,17,2},2,1,2,0,0.7f,0};
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    for (unsigned repeat=0;repeat<100;++repeat) {
        nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
        nya_train_buffer y=nya_train_device_alloc(d,272),state=nya_train_device_alloc(d,544),work=nya_train_device_alloc(d,nya_train_attention_workspace_bytes(17,2));
        CHECK(y && state && work && !nya_train_device_attention(d,y,state,desc));
        CHECK(!nya_train_device_attention_backward(d,dq,dk,dv,state,dy,work,desc));
        CHECK(!nya_train_device_scratch_end(d,scope));
        for (size_t row=0;row<17;++row) for (size_t head=0;head<2;++head)
            for (size_t col=0;col<=row;++col) for (size_t j=0;j<2;++j) expected[col*2+j]+=(float)(1.0/(double)(row+1));
    }
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+1000 && after.scratch_resets==before.scratch_resets+100 &&
        after.used_bytes==before.used_bytes && after.buffers==before.buffers && after.uploads==before.uploads &&
        after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,dv,0,actual,136));
    CHECK(!memcmp(actual,expected,sizeof(actual)));
    return 0;
}
static int numerical_status(nya_train_device *d)
{
    if (getenv("NYA_TRAIN_TEST_TRACE")) { fprintf(stderr,"numerical_status\n"); fflush(stderr); }
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer q=nya_train_device_alloc(d,8),k=nya_train_device_alloc(d,8),v=nya_train_device_alloc(d,8);
    nya_train_buffer y=nya_train_device_alloc(d,8),state=nya_train_device_alloc(d,16),status=nya_train_device_alloc(d,4);
    float query[2]={FLT_MAX,0},key[2]={FLT_MAX,0},value[2]={1,1},actual[2]={0};
    CHECK(q && k && v && y && state && status && !nya_train_device_write(d,v,0,value,8));
    nya_train_attention_desc desc={{q,1,2},{k,1,2},{v,1,2},1,1,2,0,1,0};
    for (int mode=0;mode<3;++mode) {
        if (mode) { query[0]=mode==1?FLT_MAX:-FLT_MAX; query[1]=mode==1?1e30f:-1e30f; key[0]=key[1]=1; }
        nya_train_graph *g=nya_train_graph_create(65536); CHECK(g);
        CHECK(!nya_train_attention(nya_train_input(g,1,2,query),nya_train_input(g,1,2,key),nya_train_input(g,1,2,value),1,1,2,1,0,NULL));
        nya_train_graph_free(g);
        CHECK(!nya_train_device_write(d,q,0,query,8) && !nya_train_device_write(d,k,0,key,8) && !nya_train_device_zero(d,status));
        CHECK(!nya_train_device_attention(d,y,state,desc) && !nya_train_device_check_finite(d,status,y,2,31));
        uint32_t tag=0; CHECK(!nya_train_device_read(d,status,0,&tag,4) && tag==31);
        nya_train_device_stats stats; nya_train_device_get_stats(d,&stats); CHECK(!stats.failed);
    }
    /* A one-token softmax must not conceal score overflow as probability one.
       Correct the input and recover the same device. */
    query[0]=FLT_MAX; query[1]=0;
    CHECK(!nya_train_device_write(d,q,0,query,8) && !nya_train_device_zero(d,status));
    CHECK(!nya_train_device_attention(d,y,state,desc) && !nya_train_device_check_finite(d,status,y,2,31));
    uint32_t tag=99;
    CHECK(!nya_train_device_read(d,status,0,&tag,4) && !tag && !nya_train_device_read(d,y,0,actual,8) && actual[0]==1 && actual[1]==1);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}

int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    CHECK(!nya_train_attention_workspace_bytes(0,1) && !nya_train_attention_workspace_bytes(1,0));
    CHECK(!nya_train_attention_workspace_bytes(SIZE_MAX,1) && !nya_train_attention_workspace_bytes(16,SIZE_MAX));
    CHECK(nya_train_attention_workspace_bytes(1,1)==16 && nya_train_attention_workspace_bytes(17,3)==17*3*16*16);
    CHECK(nya_train_device_attention(NULL,0,0,(nya_train_attention_desc){0}));
    CHECK(nya_train_device_attention_backward(NULL,0,0,0,0,0,0,(nya_train_attention_desc){0}));
    if (argc==1) return 0;
    d=nya_train_device_create("cuda",32*1024*1024); if (!d) return 77;
    if (!strcmp(argv[1],"--failure")) {
        nya_train_buffer b[10]; for (size_t j=0;j<10;++j) { b[j]=nya_train_device_alloc(d,16384); CHECK(b[j]); }
        nya_train_attention_desc desc={{b[0],17,4},{b[1],17,2},{b[2],17,2},2,1,2,0,0.7f,0};
        int result=nya_train_device_attention(d,b[3],b[8],desc);
        if (!result) result=nya_train_device_attention_backward(d,b[5],b[6],b[7],b[8],b[4],b[9],desc);
        CHECK(result); nya_train_device_stats stats; nya_train_device_get_stats(d,&stats); CHECK(stats.failed);
        CHECK(stats.kernel_launches==strtoull(getenv("NYA_TRAIN_DEVICE_FAIL_LAUNCH_AFTER"),NULL,10));
        float value=123; CHECK(nya_train_device_read(d,b[5],0,&value,4) && value==123 && nya_train_device_finish(d));
        CHECK(nya_train_device_attention(d,b[3],b[8],desc)); nya_train_device_free(d); return 0;
    }
    const size_t rows[]={1,3,15,16,17,33,65,257},dims[]={1,2,7,32,64,257};
    for (size_t r=0;r<8;++r) for (size_t j=0;j<6;++j) {
        CHECK(!attention_case(d,rows[r],4,2,dims[j],0,0,7));
        CHECK(!attention_case(d,rows[r],3,1,dims[j],5,1,7));
    }
    for (unsigned mask=1;mask<7;++mask) CHECK(!attention_case(d,17,2,2,5,5,0,mask));
    CHECK(!differences(d,0) && !differences(d,1) && !descriptors(d) && !numerical_status(d) && !scope_reuse(d));
    CHECK(!attention_case(d,33,4,1,7,2,2,7));
    CHECK(!attention_case(d,17,2,1,5,0,2,7));
    CHECK(!attention_case(d,17,2,1,5,1,0,7));
    CHECK(!attention_case(d,1025,2,1,2,0,0,7));
    nya_train_device_free(d); puts("resident attention CPU autograd comparisons passed"); return 0;
}
