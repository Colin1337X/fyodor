#include "training_device.h"
#include "training.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,nya_train_device_error(d)); return 1; } } while (0)
static int near_value(double expected, double magnitude, float actual)
{
    if (isfinite(actual) && fabs(actual-expected)<=1e-6*(1+magnitude)) return 1;
    fprintf(stderr,"expected %.12g actual %.9g magnitude %.9g\n",expected,(double)actual,magnitude);
    return 0;
}
static double unary_value(unsigned op,double x,double scalar)
{
    if (op==NYA_TRAIN_UNARY_SCALE) return x*scalar;
    if (op==NYA_TRAIN_UNARY_SOFTCAP) return scalar*tanh(x/scalar);
    if (op==NYA_TRAIN_UNARY_GELU) return 0.5*x*(1+tanh(sqrt(2/3.14159265358979323846)*(x+0.044715*x*x*x)));
    return x*(x>=0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x)));
}
static nya_train_tensor *cpu_unary(nya_train_tensor *x,unsigned op,float scalar)
{
    if (op==NYA_TRAIN_UNARY_SCALE) return nya_train_scale(x,scalar);
    if (op==NYA_TRAIN_UNARY_SOFTCAP) return nya_train_softcap(x,scalar);
    if (op==NYA_TRAIN_UNARY_GELU) return nya_train_gelu(x);
    return nya_train_silu(x);
}
static int unary_case(nya_train_device *d,size_t n,unsigned op)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    float *x=malloc(n*4), *dy=malloc(n*4), *actual=malloc((n+1)*4), *base=malloc((n+1)*4);
    CHECK(x && dy && actual && base);
    float scalar=op==NYA_TRAIN_UNARY_SCALE ? -1.25f : op==NYA_TRAIN_UNARY_SOFTCAP ? 2.25f : 0;
    for (size_t i=0;i<n;++i) { x[i]=(float)((int)(i%137)-68)/9; dy[i]=(float)((int)(i%17)-8)/13; base[i]=0.125f; }
    base[n]=12345;
    nya_train_parameter *p=nya_train_parameter_create(1,n,x);
    nya_train_graph *g=nya_train_graph_create(1024*1024); CHECK(p && g);
    nya_train_tensor *y=cpu_unary(nya_train_leaf(g,p),op,scalar); CHECK(y);
    nya_train_tensor *loss=nya_train_linear(y,nya_train_input(g,1,n,dy)); CHECK(loss && !nya_train_backward(loss));
    nya_train_buffer bx=nya_train_device_alloc(d,n*4), bd=nya_train_device_alloc(d,n*4);
    nya_train_buffer by=nya_train_device_alloc(d,(n+1)*4), dx=nya_train_device_alloc(d,(n+1)*4);
    CHECK(bx && bd && by && dx);
    CHECK(!nya_train_device_write(d,bx,0,x,n*4) && !nya_train_device_write(d,bd,0,dy,n*4));
    CHECK(!nya_train_device_write(d,by,0,base,(n+1)*4) && !nya_train_device_write(d,dx,0,base,(n+1)*4));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_unary(d,by,bx,n,(nya_train_unary_op)op,scalar));
    CHECK(!nya_train_device_unary_backward(d,dx,bx,bd,n,(nya_train_unary_op)op,scalar));
    CHECK(!nya_train_device_unary_backward(d,dx,bx,bd,n,(nya_train_unary_op)op,scalar));
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+3 && after.used_bytes==before.used_bytes &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,by,0,actual,(n+1)*4) && actual[n]==base[n]);
    for (size_t i=0;i<n;++i) CHECK(near_value(nya_train_data(y)[i],fabs(nya_train_data(y)[i]),actual[i]));
    CHECK(!nya_train_device_read(d,dx,0,actual,(n+1)*4) && actual[n]==base[n]);
    for (size_t i=0;i<n;++i) {
        double expected=0.125+2*(double)nya_train_parameter_gradient(p)[i];
        CHECK(near_value(expected,fabs(expected),actual[i]));
        double h=0.0001, derivative=(unary_value(op,x[i]+h,scalar)-unary_value(op,x[i]-h,scalar))/(2*h);
        CHECK(near_value(0.125+2*dy[i]*derivative,0.125+2*fabs(dy[i]*derivative),actual[i]));
    }
    nya_train_graph_free(g); nya_train_parameter_free(p);
    free(x); free(dy); free(actual); free(base);
    CHECK(!nya_train_device_scratch_end(d,scope));
    return 0;
}
static int unary_extremes(nya_train_device *d)
{
    float values[]={-FLT_MAX,-1000,-100,-80,-20,-0.0f,0.0f,FLT_TRUE_MIN,-FLT_TRUE_MIN,20,80,100,1000,FLT_MAX};
    enum { N=sizeof(values)/sizeof(values[0]) };
    float dy[N],output[N],grad[N]; for (size_t i=0;i<N;++i) dy[i]=0.25f;
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer x=nya_train_device_alloc(d,sizeof(values)), y=nya_train_device_alloc(d,sizeof(values));
    nya_train_buffer dx=nya_train_device_alloc(d,sizeof(values)), bd=nya_train_device_alloc(d,sizeof(values));
    nya_train_buffer status=nya_train_device_alloc(d,4); CHECK(x && y && dx && bd && status);
    CHECK(!nya_train_device_write(d,x,0,values,sizeof(values)) && !nya_train_device_write(d,bd,0,dy,sizeof(dy)));
    for (unsigned op=0;op<4;++op) {
        double scalar=op==NYA_TRAIN_UNARY_SCALE ? 0.5 : op==NYA_TRAIN_UNARY_SOFTCAP ? 2.25 : 0;
        CHECK(!nya_train_device_zero(d,dx));
        CHECK(!nya_train_device_unary(d,y,x,N,(nya_train_unary_op)op,scalar));
        CHECK(!nya_train_device_unary_backward(d,dx,x,bd,N,(nya_train_unary_op)op,scalar));
        CHECK(!nya_train_device_check_finite(d,status,y,N,op+1) && !nya_train_device_check_finite(d,status,dx,N,op+1));
        CHECK(!nya_train_device_read(d,y,0,output,sizeof(output)) && !nya_train_device_read(d,dx,0,grad,sizeof(grad)));
        for (size_t i=0;i<N;++i) CHECK(near_value(unary_value(op,values[i],scalar),fabs(unary_value(op,values[i],scalar)),output[i]) && isfinite(grad[i]));
        CHECK(grad[0]==(op==NYA_TRAIN_UNARY_SCALE?0.125f:0.0f));
        CHECK(grad[N-1]==(op==NYA_TRAIN_UNARY_SCALE?0.125f:op==NYA_TRAIN_UNARY_SOFTCAP?0.0f:0.25f));
    }
    uint32_t failure=1; CHECK(!nya_train_device_read(d,status,0,&failure,4) && !failure);
    CHECK(!nya_train_device_scratch_end(d,scope));
    return 0;
}
static int binary_case(nya_train_device *d,size_t ar,size_t ac,size_t br,size_t bc,unsigned op)
{
    size_t rows=ar>br?ar:br, cols=ac>bc?ac:bc, na=ar*ac, nb=br*bc, n=rows*cols;
    float *a=malloc(na*4), *b=malloc(nb*4), *dy=malloc(n*4), *actual=malloc((n+1)*4);
    CHECK(a && b && dy && actual);
    for (size_t i=0;i<na;++i) a[i]=(float)((int)(i%13)-6)/8;
    for (size_t i=0;i<nb;++i) b[i]=(float)((int)(i%11)-5)/16;
    for (size_t i=0;i<n;++i) dy[i]=(float)((int)(i%7)-3)/8;
    nya_train_parameter *pa=nya_train_parameter_create(ar,ac,a), *pb=nya_train_parameter_create(br,bc,b);
    nya_train_graph *g=nya_train_graph_create(1024*1024); CHECK(pa && pb && g);
    nya_train_tensor *ta=nya_train_leaf(g,pa), *tb=nya_train_leaf(g,pb);
    nya_train_tensor *y=op==NYA_TRAIN_BINARY_ADD ? nya_train_add(ta,tb) : nya_train_mul(ta,tb); CHECK(y);
    nya_train_tensor *loss=nya_train_linear(nya_train_reshape(y,1,n),nya_train_input(g,1,n,dy));
    CHECK(loss && !nya_train_backward(loss));
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer ba=nya_train_device_alloc(d,na*4), bb=nya_train_device_alloc(d,nb*4), bd=nya_train_device_alloc(d,n*4);
    nya_train_buffer by=nya_train_device_alloc(d,(n+1)*4), da=nya_train_device_alloc(d,(na+1)*4), db=nya_train_device_alloc(d,(nb+1)*4);
    CHECK(ba && bb && bd && by && da && db);
    float guard=12345;
    CHECK(!nya_train_device_write(d,ba,0,a,na*4) && !nya_train_device_write(d,bb,0,b,nb*4) && !nya_train_device_write(d,bd,0,dy,n*4));
    CHECK(!nya_train_device_write(d,by,n*4,&guard,4) && !nya_train_device_write(d,da,na*4,&guard,4) && !nya_train_device_write(d,db,nb*4,&guard,4));
    nya_train_view av={ba,ar,ac}, bv={bb,br,bc};
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_binary(d,by,av,bv,(nya_train_binary_op)op));
    CHECK(!nya_train_device_binary_backward(d,da,db,av,bv,bd,(nya_train_binary_op)op));
    CHECK(!nya_train_device_binary_backward(d,da,0,av,bv,bd,(nya_train_binary_op)op));
    CHECK(!nya_train_device_binary_backward(d,0,db,av,bv,bd,(nya_train_binary_op)op));
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+4 && after.used_bytes==before.used_bytes &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,by,0,actual,(n+1)*4) && actual[n]==guard);
    for (size_t i=0;i<n;++i) CHECK(near_value(nya_train_data(y)[i],fabs(nya_train_data(y)[i]),actual[i]));
    CHECK(!nya_train_device_read(d,da,0,actual,(na+1)*4) && actual[na]==guard);
    for (size_t i=0;i<na;++i) CHECK(actual[i]==2*nya_train_parameter_gradient(pa)[i]);
    CHECK(!nya_train_device_read(d,db,0,actual,(nb+1)*4) && actual[nb]==guard);
    for (size_t i=0;i<nb;++i) CHECK(actual[i]==2*nya_train_parameter_gradient(pb)[i]);
    nya_train_graph_free(g); nya_train_parameter_free(pa); nya_train_parameter_free(pb);
    CHECK(!nya_train_device_scratch_end(d,scope));
    free(a); free(b); free(dy); free(actual);
    return 0;
}
static int shared_gradients(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer a=nya_train_device_alloc(d,257*4), dy=nya_train_device_alloc(d,257*4), grad=nya_train_device_alloc(d,257*4);
    CHECK(a && dy && grad);
    float ones[257], large[257], actual[257];
    for (size_t i=0;i<257;++i) { ones[i]=1; large[i]=16777216; }
    CHECK(!nya_train_device_write(d,a,0,ones,sizeof(ones)) && !nya_train_device_write(d,dy,0,ones,sizeof(ones)));
    for (unsigned op=0;op<2;++op) {
        CHECK(!nya_train_device_write(d,grad,0,large,sizeof(large)));
        CHECK(!nya_train_device_binary_backward(d,grad,grad,(nya_train_view){a,1,257},(nya_train_view){a,1,257},dy,(nya_train_binary_op)op));
        CHECK(!nya_train_device_read(d,grad,0,actual,sizeof(actual)));
        /* Two additions of one each round away. Combining them into two before
           accumulation would incorrectly change this shared gradient. */
        for (size_t i=0;i<257;++i) CHECK(actual[i]==large[i]);
        CHECK(!nya_train_device_zero(d,grad));
        CHECK(!nya_train_device_binary_backward(d,grad,grad,(nya_train_view){a,1,257},(nya_train_view){a,1,257},dy,(nya_train_binary_op)op));
        CHECK(!nya_train_device_read(d,grad,0,actual,sizeof(actual)));
        for (size_t i=0;i<257;++i) CHECK(actual[i]==2);
    }
    CHECK(!nya_train_device_scratch_end(d,scope));
    return 0;
}
static int descriptors(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer a=nya_train_device_alloc(d,16), b=nya_train_device_alloc(d,16), dy=nya_train_device_alloc(d,16);
    nya_train_buffer y=nya_train_device_alloc(d,16), tiny=nya_train_device_alloc(d,3); CHECK(a && b && dy && y && tiny);
    nya_train_view av={a,2,2}, bv={b,2,2};
    nya_train_device *other=nya_train_device_create("cuda",1024); CHECK(other);
    nya_train_buffer foreign=nya_train_device_alloc(other,16); CHECK(foreign);
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(nya_train_device_unary(d,y,a,0,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary(d,y,a,SIZE_MAX,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary(d,y,a,5,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary(d,y,a,4,(nya_train_unary_op)99,1));
    CHECK(nya_train_device_unary(d,y,a,4,NYA_TRAIN_UNARY_SCALE,NAN));
    CHECK(nya_train_device_unary(d,y,a,4,NYA_TRAIN_UNARY_SOFTCAP,0));
    CHECK(nya_train_device_unary(d,y,a,4,NYA_TRAIN_UNARY_SOFTCAP,-1));
    CHECK(nya_train_device_unary(d,a,a,4,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary(d,tiny,a,1,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary(d,y,foreign,4,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary_backward(d,y,a,0,4,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary_backward(d,y,a,tiny,4,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary_backward(d,dy,a,dy,4,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_binary(d,y,(nya_train_view){a,0,2},bv,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary(d,y,(nya_train_view){a,SIZE_MAX,2},bv,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary(d,y,(nya_train_view){a,3,1},bv,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary(d,y,av,(nya_train_view){b,1,3},NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary(d,y,av,(nya_train_view){foreign,2,2},NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary(d,y,av,bv,(nya_train_binary_op)99));
    CHECK(nya_train_device_binary(d,a,av,bv,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary(d,tiny,av,bv,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(d,0,0,av,bv,dy,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(d,y,0,av,bv,0,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(d,y,0,av,bv,tiny,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(d,y,foreign,av,bv,dy,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(d,y,b,av,bv,dy,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(d,a,y,av,bv,dy,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(d,y,y,av,(nya_train_view){b,1,2},dy,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(d,y,dy,av,bv,dy,NYA_TRAIN_BINARY_ADD));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && after.kernel_launches==before.kernel_launches && after.used_bytes==before.used_bytes &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    float output[4]; CHECK(!nya_train_device_read(d,y,0,output,sizeof(output)));
    for (size_t i=0;i<4;++i) CHECK(output[i]==0);
    nya_train_device_free(other);
    CHECK(!nya_train_device_scratch_end(d,scope));
    CHECK(nya_train_device_unary(d,y,a,4,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_binary(d,y,av,bv,NYA_TRAIN_BINARY_ADD));
    return 0;
}
static int gated_branch(nya_train_device *d)
{
    enum { N=3, I=5, H=7, X=N*I, A=N*H, W=I*H };
    float values[4][W], seed[X], expected[X], actual[W];
    const size_t rows[]={N,H,H,I}, cols[]={I,I,I,H}, counts[]={X,W,W,W};
    nya_train_parameter *parameters[4]; nya_train_buffer p[4],grad[4];
    for (size_t j=0;j<4;++j) {
        for (size_t k=0;k<counts[j];++k) values[j][k]=(float)((int)((k+3*j)%19)-9)/16;
        parameters[j]=nya_train_parameter_create(rows[j],cols[j],values[j]); CHECK(parameters[j]);
        p[j]=nya_train_device_alloc(d,counts[j]*4); grad[j]=nya_train_device_alloc(d,counts[j]*4); CHECK(p[j] && grad[j]);
        CHECK(!nya_train_device_write(d,p[j],0,values[j],counts[j]*4));
    }
    for (size_t k=0;k<X;++k) seed[k]=(float)((int)(k%7)-3)/8;
    for (unsigned pass=0;pass<2;++pass) {
        nya_train_graph *g=nya_train_graph_create(1024*1024); CHECK(g);
        nya_train_tensor *x=nya_train_leaf(g,parameters[0]);
        nya_train_tensor *a=nya_train_linear(x,nya_train_leaf(g,parameters[1]));
        nya_train_tensor *b=nya_train_linear(x,nya_train_leaf(g,parameters[2]));
        nya_train_tensor *mixed=nya_train_mul(nya_train_silu(a),b);
        nya_train_tensor *z=nya_train_linear(mixed,nya_train_leaf(g,parameters[3]));
        nya_train_tensor *out=nya_train_scale(nya_train_add(z,x),0.75f); CHECK(out);
        memcpy(expected,nya_train_data(out),sizeof(expected));
        CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(out,1,X),nya_train_input(g,1,X,seed))));
        nya_train_graph_free(g);
    }
    nya_train_buffer dy=nya_train_device_alloc(d,sizeof(seed)), output=nya_train_device_alloc(d,sizeof(expected));
    nya_train_buffer status=nya_train_device_alloc(d,4); CHECK(dy && output && status);
    CHECK(!nya_train_device_write(d,dy,0,seed,sizeof(seed)));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    for (unsigned pass=0;pass<2;++pass) {
        nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
        nya_train_buffer a=nya_train_device_alloc(d,A*4), b=nya_train_device_alloc(d,A*4), s=nya_train_device_alloc(d,A*4), m=nya_train_device_alloc(d,A*4);
        nya_train_buffer z=nya_train_device_alloc(d,X*4), r=nya_train_device_alloc(d,X*4), dr=nya_train_device_alloc(d,X*4), dz=nya_train_device_alloc(d,X*4);
        nya_train_buffer dm=nya_train_device_alloc(d,A*4), ds=nya_train_device_alloc(d,A*4), db=nya_train_device_alloc(d,A*4), da=nya_train_device_alloc(d,A*4);
        CHECK(a && b && s && m && z && r && dr && dz && dm && ds && db && da);
        CHECK(!nya_train_device_linear(d,a,p[1],0,H,I,p[0],N) && !nya_train_device_linear(d,b,p[2],0,H,I,p[0],N));
        CHECK(!nya_train_device_unary(d,s,a,A,NYA_TRAIN_UNARY_SILU,0));
        CHECK(!nya_train_device_binary(d,m,(nya_train_view){s,N,H},(nya_train_view){b,N,H},NYA_TRAIN_BINARY_MUL));
        CHECK(!nya_train_device_linear(d,z,p[3],0,I,H,m,N));
        CHECK(!nya_train_device_binary(d,r,(nya_train_view){z,N,I},(nya_train_view){p[0],N,I},NYA_TRAIN_BINARY_ADD));
        CHECK(!nya_train_device_unary(d,output,r,X,NYA_TRAIN_UNARY_SCALE,0.75));
        CHECK(!nya_train_device_unary_backward(d,dr,r,dy,X,NYA_TRAIN_UNARY_SCALE,0.75));
        CHECK(!nya_train_device_binary_backward(d,dz,grad[0],(nya_train_view){z,N,I},(nya_train_view){p[0],N,I},dr,NYA_TRAIN_BINARY_ADD));
        CHECK(!nya_train_device_linear_dx(d,dm,p[3],0,I,H,dz,N) && !nya_train_device_linear_dw(d,grad[3],m,dz,I,H,N));
        CHECK(!nya_train_device_binary_backward(d,ds,db,(nya_train_view){s,N,H},(nya_train_view){b,N,H},dm,NYA_TRAIN_BINARY_MUL));
        CHECK(!nya_train_device_unary_backward(d,da,a,ds,A,NYA_TRAIN_UNARY_SILU,0));
        CHECK(!nya_train_device_linear_dx(d,grad[0],p[2],0,H,I,db,N) && !nya_train_device_linear_dx(d,grad[0],p[1],0,H,I,da,N));
        CHECK(!nya_train_device_linear_dw(d,grad[2],p[0],db,H,I,N) && !nya_train_device_linear_dw(d,grad[1],p[0],da,H,I,N));
        CHECK(!nya_train_device_check_finite(d,status,output,X,1));
        for (size_t k=0;k<4;++k) CHECK(!nya_train_device_check_finite(d,status,grad[k],counts[k],2+(uint32_t)k));
        CHECK(!nya_train_device_scratch_end(d,scope));
    }
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && after.used_bytes==before.used_bytes && after.kernel_launches==before.kernel_launches+68 &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,output,0,actual,sizeof(expected)));
    for (size_t k=0;k<X;++k) CHECK(near_value(expected[k],fabs(expected[k]),actual[k]));
    for (size_t j=0;j<4;++j) {
        CHECK(!nya_train_device_read(d,grad[j],0,actual,counts[j]*4));
        for (size_t k=0;k<counts[j];++k) CHECK(near_value(nya_train_parameter_gradient(parameters[j])[k],fabs(nya_train_parameter_gradient(parameters[j])[k]),actual[k]));
        nya_train_parameter_free(parameters[j]);
    }
    uint32_t failure=1; CHECK(!nya_train_device_read(d,status,0,&failure,4) && !failure);
    printf("gated branch: two microbatches, 68 launches, zero in-graph transfers/fences, full output/gradient parity passed\n");
    return 0;
}
int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    nya_train_view nil={0};
    CHECK(nya_train_device_unary(NULL,0,0,1,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_unary_backward(NULL,0,0,0,1,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_binary(NULL,0,nil,nil,NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_binary_backward(NULL,0,0,nil,nil,0,NYA_TRAIN_BINARY_ADD));
    if (argc==1) return 0;
    d=nya_train_device_create("cuda",4*1024*1024);
    if (!d) return 77;
    if (!strcmp(argv[1],"--failure")) {
        nya_train_buffer a=nya_train_device_alloc(d,4), b=nya_train_device_alloc(d,4);
        nya_train_buffer dy=nya_train_device_alloc(d,4), out=nya_train_device_alloc(d,4);
        CHECK(a && b && dy && out);
        CHECK(nya_train_device_binary_backward(d,out,0,(nya_train_view){a,1,1},(nya_train_view){b,1,1},dy,NYA_TRAIN_BINARY_MUL));
        CHECK(nya_train_device_unary(d,out,a,1,NYA_TRAIN_UNARY_SILU,0));
        nya_train_device_stats stats; nya_train_device_get_stats(d,&stats); CHECK(stats.failed && stats.kernel_launches==4);
        float value=123; CHECK(nya_train_device_read(d,out,0,&value,4) && value==123 && nya_train_device_finish(d));
        nya_train_device_free(d); return 0;
    }
    const size_t sizes[]={1,31,32,33,255,256,257,1025};
    for (unsigned op=0;op<4;++op) for (size_t k=0;k<sizeof(sizes)/sizeof(sizes[0]);++k) CHECK(!unary_case(d,sizes[k],op));
    CHECK(!unary_extremes(d));
    const size_t shapes[][4]={{3,5,3,5},{1,5,3,5},{3,5,1,5},{3,1,3,5},{3,5,3,1},
        {1,1,3,5},{3,5,1,1},{3,1,1,5},{1,5,3,1},{1,33,257,33},{257,33,1,33},
        {257,1,1,33},{1,1,257,33},{1,1,1,1}};
    for (unsigned op=0;op<2;++op) for (size_t k=0;k<sizeof(shapes)/sizeof(shapes[0]);++k)
        CHECK(!binary_case(d,shapes[k][0],shapes[k][1],shapes[k][2],shapes[k][3],op));
    CHECK(!shared_gradients(d));
    CHECK(!descriptors(d) && !gated_branch(d));
    nya_train_device_free(d);
    printf("resident elementwise: CPU autograd, finite differences, broadcasts, shared gradients and tails passed\n");
    return 0;
}
