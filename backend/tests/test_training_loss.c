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
static int loss_case(nya_train_device *d,size_t rows,size_t cols,int mode,int masked,int extreme)
{
    size_t n=rows*cols;
    float *x=malloc(n*4),*actual=malloc((n+1)*4);
    uint32_t *labels=malloc(rows*4); unsigned char *mask=malloc(rows);
    CHECK(x && actual && labels && mask);
    for (size_t i=0;i<n;++i) x[i]=(float)((int)(i%137)-68)/16;
    if (extreme) for (size_t i=0;i<n;++i) x[i]=extreme==1?FLT_MAX:(i%cols? -1000:1000);
    for (size_t r=0;r<rows;++r) {
        mask[r]=(unsigned char)(masked && r%3==1 ? 0 : r%2?255:2);
        labels[r]=mask[r]?(uint32_t)((r*17)%cols):UINT32_MAX;
    }
    nya_train_parameter *p=nya_train_parameter_create(rows,cols,x);
    nya_train_graph *g=nya_train_graph_create(128*1024*1024); CHECK(p && g);
    nya_train_tensor *leaf=nya_train_leaf(g,p);
    nya_train_tensor *loss=mode?nya_train_logprob(leaf,labels,masked?mask:NULL,rows):
        nya_train_cross_entropy(leaf,labels,masked?mask:NULL,rows);
    CHECK(loss && !nya_train_backward(nya_train_scale(loss,-0.75f)));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer bx=nya_train_device_alloc(d,n*4),bl=nya_train_device_alloc(d,rows*4),bm=nya_train_device_alloc(d,rows);
    size_t state_bytes=nya_train_loss_state_bytes(rows);
    nya_train_buffer y=nya_train_device_alloc(d,8),state=nya_train_device_alloc(d,state_bytes+8);
    nya_train_buffer dx=nya_train_device_alloc(d,(n+1)*4),dy=nya_train_device_alloc(d,4);
    CHECK(bx && bl && bm && y && state && dx && dy);
    float seed=-0.75f,sentinel=12345; double state_sentinel=12345678;
    for (size_t i=0;i<n;++i) actual[i]=0.125f;
    actual[n]=sentinel;
    CHECK(!nya_train_device_write(d,bx,0,x,n*4) && !nya_train_device_write(d,bl,0,labels,rows*4) &&
        !nya_train_device_write(d,bm,0,mask,rows) && !nya_train_device_write(d,dy,0,&seed,4) &&
        !nya_train_device_write(d,dx,0,actual,(n+1)*4) && !nya_train_device_write(d,y,4,&sentinel,4) &&
        !nya_train_device_write(d,state,state_bytes,&state_sentinel,8));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_loss(d,y,state,(nya_train_view){bx,rows,cols},bl,masked?bm:0,(nya_train_loss_op)mode));
    for (int repeat=0;repeat<2;++repeat)
        CHECK(!nya_train_device_loss_backward(d,dx,state,(nya_train_view){bx,rows,cols},bl,masked?bm:0,dy));
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+4 && after.used_bytes==before.used_bytes &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,y,0,actual,8) && actual[1]==sentinel && close_value(*nya_train_data(loss),actual[0]));
    CHECK(!nya_train_device_read(d,state,state_bytes,&state_sentinel,8) && state_sentinel==12345678);
    CHECK(!nya_train_device_read(d,dx,0,actual,(n+1)*4) && actual[n]==sentinel);
    for (size_t i=0;i<n;++i) {
        float expected=0.125f; expected+=nya_train_parameter_gradient(p)[i]; expected+=nya_train_parameter_gradient(p)[i];
        if (!close_value(expected,actual[i])) { fprintf(stderr,"rows=%zu cols=%zu mode=%d masked=%d index=%zu\n",rows,cols,mode,masked,i); return 1; }
    }
    nya_train_graph_free(g); nya_train_parameter_free(p); free(x); free(actual); free(labels); free(mask);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static double objective(const double *x,int mode)
{
    double total=0;
    for (size_t row=0;row<3;row+=2) {
        double maximum=x[row*5],mass=0;
        for (size_t j=1;j<5;++j) maximum=fmax(maximum,x[row*5+j]);
        for (size_t j=0;j<5;++j) mass+=exp(x[row*5+j]-maximum);
        total+=(x[row*5+row]-maximum)-log(mass);
    }
    return total*(mode?1.0:-0.5);
}
static int differences(nya_train_device *d,int mode)
{
    double x[15]; float xf[15],actual[15],seed=1;
    for (size_t i=0;i<15;++i) xf[i]=(float)(x[i]=((double)i-7)/8);
    uint32_t labels[3]={0,UINT32_MAX,2}; unsigned char mask[3]={1,0,255};
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer bx=nya_train_device_alloc(d,sizeof(xf)),bl=nya_train_device_alloc(d,sizeof(labels)),bm=nya_train_device_alloc(d,sizeof(mask));
    nya_train_buffer y=nya_train_device_alloc(d,4),s=nya_train_device_alloc(d,nya_train_loss_state_bytes(3));
    nya_train_buffer dx=nya_train_device_alloc(d,sizeof(actual)),dy=nya_train_device_alloc(d,4);
    CHECK(bx && bl && bm && y && s && dx && dy);
    CHECK(!nya_train_device_write(d,bx,0,xf,sizeof(xf)) && !nya_train_device_write(d,bl,0,labels,sizeof(labels)) &&
        !nya_train_device_write(d,bm,0,mask,sizeof(mask)) && !nya_train_device_write(d,dy,0,&seed,4));
    CHECK(!nya_train_device_loss(d,y,s,(nya_train_view){bx,3,5},bl,bm,(nya_train_loss_op)mode));
    CHECK(!nya_train_device_loss_backward(d,dx,s,(nya_train_view){bx,3,5},bl,bm,dy) && !nya_train_device_read(d,dx,0,actual,sizeof(actual)));
    for (size_t i=0;i<15;++i) {
        double old=x[i],h=1e-5; x[i]=old+h; double plus=objective(x,mode);
        x[i]=old-h; double minus=objective(x,mode); x[i]=old;
        CHECK(close_value((plus-minus)/(2*h),actual[i]));
    }
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int dpo_case(nya_train_device *d,float chosen,float rejected,double rc,double rr,float beta,int outputs)
{
    nya_train_parameter *pc=nya_train_parameter_create(1,1,&chosen),*pr=nya_train_parameter_create(1,1,&rejected);
    nya_train_graph *g=nya_train_graph_create(65536); CHECK(pc && pr && g);
    nya_train_tensor *loss=nya_train_dpo(nya_train_leaf(g,pc),nya_train_leaf(g,pr),rc,rr,beta);
    CHECK(loss && !nya_train_backward(nya_train_scale(loss,-0.75f)));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer c=nya_train_device_alloc(d,4),r=nya_train_device_alloc(d,4),y=nya_train_device_alloc(d,8),s=nya_train_device_alloc(d,16);
    nya_train_buffer dc=nya_train_device_alloc(d,8),dr=nya_train_device_alloc(d,8),dy=nya_train_device_alloc(d,4);
    CHECK(c && r && y && s && dc && dr && dy);
    float seed=-0.75f,initial[2]={0.125f,12345}; double sentinel=12345678;
    CHECK(!nya_train_device_write(d,c,0,&chosen,4) && !nya_train_device_write(d,r,0,&rejected,4) &&
        !nya_train_device_write(d,dy,0,&seed,4) && !nya_train_device_write(d,dc,0,initial,8) &&
        !nya_train_device_write(d,dr,0,initial,8) && !nya_train_device_write(d,y,4,initial+1,4) &&
        !nya_train_device_write(d,s,8,&sentinel,8));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_dpo(d,y,s,c,r,rc,rr,beta));
    for (int k=0;k<2;++k) CHECK(!nya_train_device_dpo_backward(d,(outputs&1) || outputs==4?dc:0,outputs==4?dc:outputs&2?dr:0,s,dy));
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+3 && after.uploads==before.uploads &&
        after.downloads==before.downloads && after.synchronizations==before.synchronizations && after.used_bytes==before.used_bytes);
    float actual[2]; CHECK(!nya_train_device_read(d,y,0,actual,8) && actual[1]==12345 && close_value(*nya_train_data(loss),actual[0]));
    for (int j=0;j<2;++j) {
        CHECK(!nya_train_device_read(d,j?dr:dc,0,actual,8) && actual[1]==12345);
        float expected=0.125f;
        for (int k=0;k<2;++k) {
            if (!j && ((outputs&1) || outputs==4)) expected+=*nya_train_parameter_gradient(pc);
            if ((j && (outputs&2)) || (!j && outputs==4)) expected+=*nya_train_parameter_gradient(pr);
        }
        CHECK(close_value(expected,actual[0]));
        if (outputs==4) CHECK(!memcmp(&expected,&actual[0],4));
    }
    CHECK(!nya_train_device_read(d,s,8,&sentinel,8) && sentinel==12345678);
    nya_train_graph_free(g); nya_train_parameter_free(pc); nya_train_parameter_free(pr);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
/* Complete chosen/rejected logprob -> DPO -> both logit gradients; no scalar
   readback between kernels. Compare CPU autograd and independent differences. */
static int pair(nya_train_device *d)
{
    float c[15],r[15],actual[15],seed=1; double xd[15];
    for (size_t i=0;i<15;++i) { c[i]=(float)((int)i-7)/8; r[i]=-c[i]/2; xd[i]=c[i]; }
    uint32_t labels[3]={0,UINT32_MAX,2}; unsigned char mask[3]={1,0,255};
    nya_train_parameter *pc=nya_train_parameter_create(3,5,c),*pr=nya_train_parameter_create(3,5,r);
    nya_train_graph *g=nya_train_graph_create(65536); CHECK(pc && pr && g);
    nya_train_tensor *lc=nya_train_logprob(nya_train_leaf(g,pc),labels,mask,3),*lr=nya_train_logprob(nya_train_leaf(g,pr),labels,mask,3);
    nya_train_tensor *loss=nya_train_dpo(lc,lr,-3.7,-2.3,0.5f); CHECK(loss && !nya_train_backward(loss));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer bc=nya_train_device_alloc(d,60),br=nya_train_device_alloc(d,60),bl=nya_train_device_alloc(d,12),bm=nya_train_device_alloc(d,3);
    nya_train_buffer yc=nya_train_device_alloc(d,4),yr=nya_train_device_alloc(d,4),sc=nya_train_device_alloc(d,104),sr=nya_train_device_alloc(d,104);
    nya_train_buffer y=nya_train_device_alloc(d,4),s=nya_train_device_alloc(d,8),dc=nya_train_device_alloc(d,60),dr=nya_train_device_alloc(d,60);
    nya_train_buffer gc=nya_train_device_alloc(d,4),gr=nya_train_device_alloc(d,4),dy=nya_train_device_alloc(d,4);
    CHECK(bc && br && bl && bm && yc && yr && sc && sr && y && s && dc && dr && gc && gr && dy);
    CHECK(!nya_train_device_write(d,bc,0,c,60) && !nya_train_device_write(d,br,0,r,60) &&
        !nya_train_device_write(d,bl,0,labels,12) && !nya_train_device_write(d,bm,0,mask,3) && !nya_train_device_write(d,dy,0,&seed,4));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_loss(d,yc,sc,(nya_train_view){bc,3,5},bl,bm,NYA_TRAIN_LOSS_LOGPROB));
    CHECK(!nya_train_device_loss(d,yr,sr,(nya_train_view){br,3,5},bl,bm,NYA_TRAIN_LOSS_LOGPROB));
    CHECK(!nya_train_device_dpo(d,y,s,yc,yr,-3.7,-2.3,0.5f) && !nya_train_device_dpo_backward(d,gc,gr,s,dy));
    CHECK(!nya_train_device_loss_backward(d,dc,sc,(nya_train_view){bc,3,5},bl,bm,gc));
    CHECK(!nya_train_device_loss_backward(d,dr,sr,(nya_train_view){br,3,5},bl,bm,gr));
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+8 && after.uploads==before.uploads &&
        after.downloads==before.downloads && after.synchronizations==before.synchronizations && after.used_bytes==before.used_bytes);
    CHECK(!nya_train_device_read(d,y,0,actual,4) && close_value(*nya_train_data(loss),actual[0]));
    for (int which=0;which<2;++which) {
        CHECK(!nya_train_device_read(d,which?dr:dc,0,actual,60));
        for (size_t i=0;i<15;++i) CHECK(close_value(nya_train_parameter_gradient(which?pr:pc)[i],actual[i]));
    }
    CHECK(!nya_train_device_read(d,dc,0,actual,60));
    double rd[15]; for (size_t i=0;i<15;++i) rd[i]=r[i];
    double rejected=objective(rd,1);
    for (size_t i=0;i<15;++i) {
        double old=xd[i],h=1e-5; xd[i]=old+h; double mp=0.5*(objective(xd,1)-rejected+1.4);
        xd[i]=old-h; double mm=0.5*(objective(xd,1)-rejected+1.4); xd[i]=old;
        double plus=fmax(-mp,0)+log1p(exp(-fabs(mp))),minus=fmax(-mm,0)+log1p(exp(-fabs(mm)));
        CHECK(close_value((plus-minus)/(2*h),actual[i]));
    }
    nya_train_graph_free(g); nya_train_parameter_free(pc); nya_train_parameter_free(pr);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int descriptors(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer stale=nya_train_device_alloc(d,128); CHECK(stale && !nya_train_device_scratch_end(d,scope));
    scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer b[8]; for (size_t i=0;i<8;++i) { b[i]=nya_train_device_alloc(d,128); CHECK(b[i]); }
    nya_train_buffer tiny=nya_train_device_alloc(d,1); uint32_t id=0;
    nya_train_indices map=nya_train_device_indices(d,&id,1,1);
    nya_train_device *other=nya_train_device_create("cuda",1024); CHECK(other);
    nya_train_buffer foreign=nya_train_device_alloc(other,128); CHECK(tiny && map && foreign);
    const nya_train_buffer invalid[]={0,stale,foreign,tiny,map};
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    for (size_t k=0;k<5;++k) {
        nya_train_buffer z=invalid[k];
        CHECK(nya_train_device_loss(d,z,b[1],(nya_train_view){b[2],2,3},b[3],b[4],NYA_TRAIN_LOSS_CE));
        CHECK(nya_train_device_loss(d,b[0],z,(nya_train_view){b[2],2,3},b[3],b[4],NYA_TRAIN_LOSS_CE));
        CHECK(nya_train_device_loss(d,b[0],b[1],(nya_train_view){z,2,3},b[3],b[4],NYA_TRAIN_LOSS_CE));
        CHECK(nya_train_device_loss(d,b[0],b[1],(nya_train_view){b[2],2,3},z,b[4],NYA_TRAIN_LOSS_CE));
        if (z) CHECK(nya_train_device_loss(d,b[0],b[1],(nya_train_view){b[2],2,3},b[3],z,NYA_TRAIN_LOSS_CE));
        CHECK(nya_train_device_loss_backward(d,z,b[1],(nya_train_view){b[2],2,3},b[3],b[4],b[5]));
        CHECK(nya_train_device_loss_backward(d,b[0],z,(nya_train_view){b[2],2,3},b[3],b[4],b[5]));
        CHECK(nya_train_device_loss_backward(d,b[0],b[1],(nya_train_view){b[2],2,3},b[3],b[4],z));
        CHECK(nya_train_device_dpo(d,z,b[1],b[2],b[3],0,0,1));
        CHECK(nya_train_device_dpo(d,b[0],z,b[2],b[3],0,0,1));
        CHECK(nya_train_device_dpo(d,b[0],b[1],z,b[3],0,0,1));
        CHECK(nya_train_device_dpo(d,b[0],b[1],b[2],z,0,0,1));
        CHECK(nya_train_device_dpo_backward(d,z,0,b[1],b[2]));
        if (z) CHECK(nya_train_device_dpo_backward(d,0,z,b[1],b[2]));
        CHECK(nya_train_device_dpo_backward(d,b[0],b[3],z,b[2]));
        CHECK(nya_train_device_dpo_backward(d,b[0],b[3],b[1],z));
    }
    for (size_t k=1;k<6;++k) {
        CHECK(nya_train_device_loss_backward(d,b[k],b[1],(nya_train_view){b[2],2,3},b[3],b[4],b[5]));
        if (k<5) CHECK(nya_train_device_loss(d,b[k],b[1],(nya_train_view){b[2],2,3},b[3],b[4],NYA_TRAIN_LOSS_CE));
    }
    for (size_t k=2;k<6;++k)
        CHECK(nya_train_device_loss_backward(d,b[0],b[k],(nya_train_view){b[2],2,3},b[3],b[4],b[5]));
    const nya_train_view bad[]={{b[2],0,3},{b[2],2,0},{b[2],SIZE_MAX,2},{b[2],2,SIZE_MAX},{b[2],2,17},{b[2],4,3}};
    for (size_t k=0;k<6;++k) CHECK(nya_train_device_loss(d,b[0],b[1],bad[k],b[3],b[4],NYA_TRAIN_LOSS_CE));
    CHECK(nya_train_device_loss(d,b[0],b[1],(nya_train_view){b[2],2,3},b[3],0,(nya_train_loss_op)2));
    const double badref[]={NAN,INFINITY,-INFINITY};
    for (size_t k=0;k<3;++k) {
        CHECK(nya_train_device_dpo(d,b[0],b[1],b[2],b[3],badref[k],0,1));
        CHECK(nya_train_device_dpo(d,b[0],b[1],b[2],b[3],0,badref[k],1));
    }
    const float badbeta[]={0,-1,NAN,INFINITY};
    for (size_t k=0;k<4;++k) CHECK(nya_train_device_dpo(d,b[0],b[1],b[2],b[3],0,0,badbeta[k]));
    for (size_t k=1;k<4;++k) CHECK(nya_train_device_dpo(d,b[k],b[1],b[2],b[3],0,0,1));
    CHECK(nya_train_device_dpo(d,b[0],b[2],b[2],b[3],0,0,1));
    CHECK(nya_train_device_dpo(d,b[0],b[3],b[2],b[3],0,0,1));
    CHECK(nya_train_device_dpo_backward(d,0,0,b[1],b[2]));
    CHECK(nya_train_device_dpo_backward(d,b[1],0,b[1],b[2]));
    CHECK(nya_train_device_dpo_backward(d,0,b[2],b[1],b[2]));
    CHECK(nya_train_device_dpo_backward(d,b[0],0,b[2],b[2]));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && after.kernel_launches==before.kernel_launches && after.uploads==before.uploads &&
        after.downloads==before.downloads && after.synchronizations==before.synchronizations && after.used_bytes==before.used_bytes);
    CHECK(!nya_train_device_loss(d,b[0],b[1],(nya_train_view){b[2],2,3},b[3],0,NYA_TRAIN_LOSS_CE));
    CHECK(!nya_train_device_dpo(d,b[0],b[1],b[2],b[2],0,0,1));
    nya_train_device_free(other); CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int status_and_residency(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer x=nya_train_device_alloc(d,16),labels=nya_train_device_alloc(d,8),mask=nya_train_device_alloc(d,2);
    nya_train_buffer y=nya_train_device_alloc(d,4),s=nya_train_device_alloc(d,72),dx=nya_train_device_alloc(d,16);
    nya_train_buffer dy=nya_train_device_alloc(d,4),status=nya_train_device_alloc(d,4),dc=nya_train_device_alloc(d,4);
    CHECK(x && labels && mask && y && s && dx && dy && status && dc);
    float seed=1; CHECK(!nya_train_device_write(d,dy,0,&seed,4));
    for (int k=0;k<9;++k) {
        float values[4]={0,1,2,3}; uint32_t ids[2]={0,1}; unsigned char masks[2]={1,1};
        if (k==0) ids[1]=UINT32_MAX;
        if (k==1) masks[0]=masks[1]=0;
        if (k==2) values[0]=NAN;
        if (k==3) values[0]=INFINITY;
        if (k==4) values[0]=-INFINITY;
        if (k==5) { values[0]=-FLT_MAX; values[1]=FLT_MAX; values[2]=FLT_MAX; values[3]=-FLT_MAX; }
        if (k>=6) { masks[0]=0; ids[0]=UINT32_MAX; values[0]=k==6?NAN:k==7?INFINITY:-INFINITY; }
        CHECK(!nya_train_device_write(d,x,0,values,16) && !nya_train_device_write(d,labels,0,ids,8) &&
            !nya_train_device_write(d,mask,0,masks,2) && !nya_train_device_zero(d,status));
        CHECK(!nya_train_device_loss(d,y,s,(nya_train_view){x,2,2},labels,mask,NYA_TRAIN_LOSS_CE));
        CHECK(!nya_train_device_check_finite(d,status,y,1,17));
        uint32_t tag=0; CHECK(!nya_train_device_read(d,status,0,&tag,4) && tag==(k<6?17u:0u));
        if (k>=6) {
            CHECK(!nya_train_device_zero(d,dx) && !nya_train_device_loss_backward(d,dx,s,(nya_train_view){x,2,2},labels,mask,dy));
            CHECK(!nya_train_device_read(d,dx,0,values,16) && values[0]==0 && values[1]==0 && isfinite(values[2]) && isfinite(values[3]));
        }
        nya_train_device_stats stats; nya_train_device_get_stats(d,&stats); CHECK(!stats.failed);
    }
    for (int k=0;k<4;++k) {
        float value=k==0?NAN:k==1?INFINITY:k==2?-INFINITY:-FLT_MAX;
        CHECK(!nya_train_device_write(d,x,0,&value,4) && !nya_train_device_zero(d,status));
        CHECK(!nya_train_device_dpo(d,y,s,x,dy,k==3?DBL_MAX:0,k==3?-DBL_MAX:0,1));
        CHECK(!nya_train_device_check_finite(d,status,y,1,19));
        uint32_t tag=0; CHECK(!nya_train_device_read(d,status,0,&tag,4) && tag==19);
    }
    float value=-FLT_MAX; CHECK(!nya_train_device_write(d,x,0,&value,4) && !nya_train_device_zero(d,status));
    CHECK(!nya_train_device_dpo(d,y,s,x,dy,0,0,FLT_MAX) && !nya_train_device_check_finite(d,status,y,1,21));
    uint32_t tag=0; CHECK(!nya_train_device_read(d,status,0,&tag,4) && tag==21);
    CHECK(!nya_train_device_scratch_end(d,scope));
    /* Persistent source/gradient/status, disposable loss state, 100 reuses. */
    x=nya_train_device_alloc(d,16); labels=nya_train_device_alloc(d,8); dy=nya_train_device_alloc(d,4);
    dx=nya_train_device_alloc(d,16); status=nya_train_device_alloc(d,4);
    CHECK(x && labels && dy && dx && status && !nya_train_device_write(d,dy,0,&seed,4));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    for (int repeat=0;repeat<100;++repeat) {
        scope=nya_train_device_scratch_begin(d); CHECK(scope);
        y=nya_train_device_alloc(d,4); s=nya_train_device_alloc(d,72); CHECK(y && s);
        CHECK(!nya_train_device_loss(d,y,s,(nya_train_view){x,2,2},labels,0,NYA_TRAIN_LOSS_CE));
        CHECK(!nya_train_device_loss_backward(d,dx,s,(nya_train_view){x,2,2},labels,0,dy));
        CHECK(!nya_train_device_check_finite(d,status,y,1,31) && !nya_train_device_check_finite(d,status,dx,4,32));
        CHECK(!nya_train_device_scratch_end(d,scope));
    }
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+700 && after.used_bytes==before.used_bytes &&
        after.buffers==before.buffers && after.scratch_resets==before.scratch_resets+100 &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,status,0,&tag,4) && !tag);
    float actual[4]; CHECK(!nya_train_device_read(d,dx,0,actual,16));
    CHECK(actual[0]==-25 && actual[1]==25 && actual[2]==-25 && actual[3]==25);
    return 0;
}
int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    CHECK(nya_train_loss_state_bytes(1)==40 && nya_train_loss_state_bytes(65)==2088);
    CHECK(!nya_train_loss_state_bytes(0) && !nya_train_loss_state_bytes(SIZE_MAX) && !nya_train_loss_state_bytes((SIZE_MAX-8)/32+1));
    CHECK(nya_train_device_loss(NULL,0,0,(nya_train_view){0},0,0,NYA_TRAIN_LOSS_CE));
    CHECK(nya_train_device_loss_backward(NULL,0,0,(nya_train_view){0},0,0,0));
    CHECK(nya_train_device_dpo(NULL,0,0,0,0,0,0,1) && nya_train_device_dpo_backward(NULL,0,0,0,0));
    if (argc==1) return 0;
    d=nya_train_device_create("cuda",32*1024*1024); if (!d) return 77;
    if (!strcmp(argv[1],"--failure")) {
        nya_train_buffer b[9]; for (size_t i=0;i<9;++i) { b[i]=nya_train_device_alloc(d,64); CHECK(b[i]); }
        int result=nya_train_device_loss(d,b[0],b[1],(nya_train_view){b[2],1,2},b[3],0,NYA_TRAIN_LOSS_CE);
        if (!result) result=nya_train_device_loss_backward(d,b[4],b[1],(nya_train_view){b[2],1,2},b[3],0,b[5]);
        if (!result) result=nya_train_device_dpo(d,b[6],b[7],b[0],b[5],0,0,1);
        if (!result) result=nya_train_device_dpo_backward(d,b[4],b[8],b[7],b[5]);
        nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
        CHECK(result && stats.failed && stats.kernel_launches==strtoull(getenv("NYA_TRAIN_DEVICE_FAIL_LAUNCH_AFTER"),NULL,10));
        float sentinel=42; CHECK(nya_train_device_read(d,b[0],0,&sentinel,4) && sentinel==42 && nya_train_device_finish(d));
        nya_train_device_free(d); return 0;
    }
    const size_t rows[]={1,3,65},cols[]={1,2,7,255,256,257,1025,32000};
    for (int mode=0;mode<2;++mode) {
        for (size_t r=0;r<3;++r) for (size_t c=0;c<8;++c) for (int mask=0;mask<2;++mask)
            CHECK(!loss_case(d,rows[r],cols[c],mode,mask,0));
        CHECK(!loss_case(d,3,257,mode,1,1) && !loss_case(d,65,257,mode,1,2) && !differences(d,mode));
    }
    for (int outputs=1;outputs<=4;++outputs) {
        CHECK(!dpo_case(d,-4,-7,-5,-8,0.1f,outputs));
        CHECK(!dpo_case(d,1000,-1000,0,0,1,outputs));
        CHECK(!dpo_case(d,-1000,1000,0,0,1,outputs));
        CHECK(!dpo_case(d,-FLT_MAX,-FLT_MAX,-DBL_MAX,-DBL_MAX,FLT_TRUE_MIN,outputs));
        CHECK(!dpo_case(d,0,0,0,0,1e9f,outputs));
    }
    CHECK(!pair(d) && !descriptors(d) && !status_and_residency(d));
    nya_train_device_free(d);
    puts("resident losses: CPU autograd, differences, masking, wide vocabulary, extreme logits, DPO and resident composition passed");
    return 0;
}
