#include "training_device.h"
#include "training_indices.h"
#include "training.h"
#include "llm_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,nya_train_device_error(d)); return 1; } } while (0)
static int packing(void)
{
    nya_train_device *d=NULL;
    nya_train_index_pack pack;
    uint32_t ids[4097]={0}, random=42;
    CHECK(nya_train_index_pack_create(NULL,ids,1,1));
    CHECK(nya_train_index_pack_create(&pack,NULL,1,1) && !pack.data);
    CHECK(nya_train_index_pack_create(&pack,ids,0,1) && !pack.data);
    CHECK(nya_train_index_pack_create(&pack,ids,1,0) && !pack.data);
    CHECK(nya_train_index_pack_create(&pack,ids,SIZE_MAX,1) && !pack.data);
    ids[0]=UINT32_MAX; CHECK(nya_train_index_pack_create(&pack,ids,1,UINT32_MAX) && !pack.data);
    for (size_t trial=0;trial<80;++trial) {
        size_t count=trial==79?4097:1+trial*7, rows=trial%3?97u:UINT32_MAX;
        for (size_t k=0;k<count;++k) { random=random*1664525u+1013904223u; ids[k]=random%(uint32_t)rows; }
        if (trial%4==0) for (size_t k=0;k<count;++k) ids[k]=3;
        CHECK(!nya_train_index_pack_create(&pack,ids,count,rows));
        CHECK(pack.rows==rows && pack.count==count && pack.groups && pack.groups<=count);
        CHECK(pack.bytes==(2*count+2*pack.groups+1)*4 && !memcmp(pack.data,ids,count*4));
        uint32_t *keys=pack.data+count, *offsets=keys+pack.groups, *positions=offsets+pack.groups+1;
        unsigned char seen[4097]={0};
        CHECK(offsets[0]==0 && offsets[pack.groups]==count);
        for (size_t g=0;g<pack.groups;++g) {
            CHECK((!g || keys[g]>keys[g-1]) && offsets[g]<offsets[g+1]);
            for (size_t k=offsets[g];k<offsets[g+1];++k) {
                CHECK(positions[k]<count && !seen[positions[k]] && ids[positions[k]]==keys[g]);
                CHECK(k==offsets[g] || positions[k]>positions[k-1]);
                seen[positions[k]]=1;
            }
        }
        for (size_t k=0;k<count;++k) CHECK(seen[k]);
        nya_train_index_pack_free(&pack); CHECK(!pack.data && !pack.bytes && !pack.count);
        nya_train_index_pack_free(&pack);
    }
    nya_train_index_pack_free(NULL);
    return 0;
}
static int gather(nya_train_device *d,unsigned type,size_t columns,size_t count)
{
    const size_t rows=7;
    size_t block=type==2 || type==8?32:type==12 || type==14?256:1;
    size_t unit=type==0?4:type==1 || type==30?2:type==2?18:type==8?34:type==12?144:210;
    size_t bytes=rows*(columns/block)*unit, n=count*columns, table_count=rows*columns;
    unsigned char *raw=malloc(bytes); uint32_t *ids=malloc(count*4);
    float *table=malloc(table_count*4), *seed=malloc(n*4), *actual=malloc((n>table_count?n+1:table_count+1)*4);
    CHECK(raw && ids && table && seed && actual);
    for (size_t k=0;k<bytes;++k) raw[k]=(unsigned char)((k*37+13)%251);
    if (block==1) for (size_t k=0;k<table_count;++k) {
        float value=(float)((int)(k%71)-35)/64;
        if (!type) memcpy(raw+k*4,&value,4);
        else {
            uint32_t bits; memcpy(&bits,&value,4);
            unsigned v=type==30?bits>>16:0x3000u+(unsigned)(k%1024)+(k%2?0x8000u:0);
            raw[k*2]=(unsigned char)v; raw[k*2+1]=(unsigned char)(v>>8);
        }
    } else for (size_t k=0;k<bytes;k+=unit) {
        size_t offset=type==14?208:0; raw[k+offset]=0; raw[k+offset+1]=0x28;
        if (type==12) { raw[k+2]=0; raw[k+3]=0x24; }
    }
    nya_llm_tensor mapped={0}; mapped.data=raw; mapped.type=type;
    for (size_t k=0;k<table_count;++k) table[k]=nya_llm_tensor_value(&mapped,k);
    for (size_t k=0;k<count;++k) ids[k]=(uint32_t)(k%5?6-k%3:0);
    for (size_t k=0;k<n;++k) seed[k]=(float)((int)(k%17)-8)/16;
    nya_train_parameter *p=nya_train_parameter_create(rows,columns,table); CHECK(p);
    for (unsigned pass=0;pass<2;++pass) {
        nya_train_graph *g=nya_train_graph_create(16*1024*1024); CHECK(g);
        nya_train_tensor *y=nya_train_embedding(nya_train_leaf(g,p),ids,count); CHECK(y);
        for (size_t k=0;k<n;++k) CHECK(nya_train_data(y)[k]==table[(size_t)ids[k/columns]*columns+k%columns]);
        CHECK(!nya_train_backward(nya_train_linear(nya_train_reshape(y,1,n),nya_train_input(g,1,n,seed))));
        nya_train_graph_free(g);
    }
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_indices map=nya_train_device_indices(d,ids,count,rows); CHECK(map);
    nya_train_buffer bt=nya_train_device_alloc(d,bytes), by=nya_train_device_alloc(d,(n+1)*4);
    nya_train_buffer dy=nya_train_device_alloc(d,n*4), dt=nya_train_device_alloc(d,(table_count+1)*4); CHECK(bt && by && dy && dt);
    CHECK(!nya_train_device_write(d,bt,0,raw,bytes) && !nya_train_device_write(d,dy,0,seed,n*4));
    float guard=12345;
    CHECK(!nya_train_device_write(d,by,n*4,&guard,4) && !nya_train_device_write(d,dt,table_count*4,&guard,4));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_embedding(d,by,bt,type,columns,map));
    CHECK(!nya_train_device_embedding_backward(d,dt,dy,columns,map));
    CHECK(!nya_train_device_embedding_backward(d,dt,dy,columns,map));
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+3 && after.used_bytes==before.used_bytes &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,by,0,actual,(n+1)*4) && actual[n]==guard);
    for (size_t k=0;k<n;++k) CHECK(actual[k]==table[(size_t)ids[k/columns]*columns+k%columns]);
    CHECK(!nya_train_device_read(d,dt,0,actual,(table_count+1)*4) && actual[table_count]==guard);
    CHECK(!memcmp(actual,nya_train_parameter_gradient(p),table_count*4));
    /* Changing the source host IDs cannot mutate an already bound map. */
    memset(ids,0,count*4);
    CHECK(!nya_train_device_embedding(d,by,bt,type,columns,map));
    CHECK(!nya_train_device_read(d,by,0,actual,n*4));
    for (size_t k=0;k<n;++k) {
        size_t token=k/columns, row=token%5?6-token%3:0;
        CHECK(actual[k]==table[row*columns+k%columns]);
    }
    CHECK(!nya_train_device_scratch_end(d,scope));
    CHECK(nya_train_device_embedding(d,by,bt,type,columns,map));
    nya_train_parameter_free(p); free(raw); free(ids); free(table); free(seed); free(actual);
    return 0;
}
static int lifetime(nya_train_device *d)
{
    uint32_t ids[]={2,1,2}; float values[]={1,2,3}, actual[3];
    nya_train_buffer table=nya_train_device_alloc(d,12), dt=nya_train_device_alloc(d,12);
    nya_train_indices map=nya_train_device_indices(d,ids,3,3); CHECK(table && dt && map);
    CHECK(!nya_train_device_write(d,table,0,values,12));
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    for (unsigned pass=0;pass<100;++pass) {
        nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
        nya_train_buffer y=nya_train_device_alloc(d,12); CHECK(y);
        CHECK(!nya_train_device_embedding(d,y,table,0,1,map));
        CHECK(!nya_train_device_embedding_backward(d,dt,y,1,map));
        CHECK(!nya_train_device_scratch_end(d,scope));
    }
    nya_train_device_get_stats(d,&after);
    CHECK(after.kernel_launches==before.kernel_launches+300 && after.used_bytes==before.used_bytes && after.buffers==before.buffers &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,dt,0,actual,12) && actual[0]==0 && actual[1]==200 && actual[2]==600);
    /* Sequential additions must not be replaced by a sum of contributions. */
    uint32_t repeated[]={0,0,0,0}; float ones[]={1,1,1,1}, base=16777216, value;
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_indices repeats=nya_train_device_indices(d,repeated,4,1); CHECK(repeats);
    nya_train_buffer dy=nya_train_device_alloc(d,16), grad=nya_train_device_alloc(d,4); CHECK(dy && grad);
    CHECK(!nya_train_device_write(d,dy,0,ones,16) && !nya_train_device_write(d,grad,0,&base,4));
    CHECK(!nya_train_device_embedding_backward(d,grad,dy,1,repeats));
    CHECK(!nya_train_device_read(d,grad,0,&value,4) && value==base);
    CHECK(!nya_train_device_scratch_end(d,scope));
    CHECK(nya_train_device_embedding_backward(d,dt,table,1,repeats));
    return 0;
}
static int matrix_order(nya_train_device *d)
{
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_buffer a=nya_train_device_alloc(d,16), b=nya_train_device_alloc(d,16), out=nya_train_device_alloc(d,4);
    CHECK(a && b && out);
    float ones[4]={1,1,1,1}, initial=16777216, value;
    CHECK(!nya_train_device_write(d,a,0,ones,16) && !nya_train_device_write(d,b,0,ones,16));
    CHECK(!nya_train_device_write(d,out,0,&initial,4));
    CHECK(!nya_train_device_linear_dx(d,out,a,0,4,1,b,1));
    CHECK(!nya_train_device_read(d,out,0,&value,4) && value==initial);
    CHECK(!nya_train_device_linear_dw(d,out,a,b,1,1,4));
    CHECK(!nya_train_device_read(d,out,0,&value,4) && value==initial);
    float left[4]={0x1.000002p0f,0,0,0}, right[4]={0x1.fffffcp-1f,0,0,0};
    initial=-1;
    CHECK(!nya_train_device_write(d,a,0,left,16) && !nya_train_device_write(d,b,0,right,16));
    CHECK(!nya_train_device_write(d,out,0,&initial,4));
    CHECK(!nya_train_device_linear_dx(d,out,a,0,4,1,b,1));
    CHECK(!nya_train_device_read(d,out,0,&value,4) && value==0);
    CHECK(!nya_train_device_write(d,out,0,&initial,4));
    CHECK(!nya_train_device_linear_dw(d,out,a,b,1,1,4));
    CHECK(!nya_train_device_read(d,out,0,&value,4) && value==0);
    CHECK(!nya_train_device_scratch_end(d,scope)); return 0;
}
static int descriptors(nya_train_device *d)
{
    uint32_t ids[]={2,1,2}, bad[]={0,3,0}; float host[8]={42,42,42,42,42,42,42,42};
    nya_train_scope scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_indices stale=nya_train_device_indices(d,ids,3,3); CHECK(stale && !nya_train_device_scratch_end(d,scope));
    scope=nya_train_device_scratch_begin(d); CHECK(scope);
    nya_train_indices map=nya_train_device_indices(d,ids,3,3); CHECK(map);
    nya_train_buffer x=nya_train_device_alloc(d,32), y=nya_train_device_alloc(d,32), inv=nya_train_device_alloc(d,24), tiny=nya_train_device_alloc(d,1);
    CHECK(x && y && inv && tiny && !nya_train_device_write(d,y,0,host,32));
    nya_train_device *other=nya_train_device_create("cuda",1024); CHECK(other);
    nya_train_indices foreign=nya_train_device_indices(other,ids,3,3); CHECK(foreign);
    nya_train_device_stats before,after; nya_train_device_get_stats(d,&before);
    CHECK(!nya_train_device_indices(d,NULL,3,3) && !nya_train_device_indices(d,bad,3,3));
    CHECK(!nya_train_device_indices(d,ids,0,3) && !nya_train_device_indices(d,ids,3,0));
    CHECK(!nya_train_device_indices(d,ids,SIZE_MAX,3));
    CHECK(nya_train_device_write(d,map,0,host,4) && nya_train_device_read(d,map,0,host,4) && nya_train_device_zero(d,map));
    CHECK(nya_train_device_check_finite(d,map,x,1,1));
    CHECK(nya_train_device_unary(d,map,x,1,NYA_TRAIN_UNARY_SCALE,1));
    CHECK(nya_train_device_binary(d,map,(nya_train_view){x,1,1},(nya_train_view){x,1,1},NYA_TRAIN_BINARY_ADD));
    CHECK(nya_train_device_rms_norm(d,y,map,(nya_train_view){x,1,1},0,1e-5f));
    CHECK(nya_train_device_linear(d,map,x,0,1,1,x,1));
    CHECK(nya_train_device_unary(d,y,map,1,NYA_TRAIN_UNARY_SCALE,1));
    const nya_train_indices invalid[]={0,stale,foreign,x};
    for (size_t k=0;k<4;++k) {
        CHECK(nya_train_device_embedding(d,y,x,0,1,invalid[k]));
        CHECK(nya_train_device_embedding_backward(d,y,x,1,invalid[k]));
    }
    CHECK(nya_train_device_embedding(d,map,x,0,1,map) && nya_train_device_embedding(d,y,map,0,1,map));
    CHECK(nya_train_device_embedding(d,x,x,0,1,map));
    CHECK(nya_train_device_embedding_backward(d,x,x,1,map));
    CHECK(nya_train_device_embedding(d,y,x,99,1,map));
    CHECK(nya_train_device_embedding(d,y,x,2,33,map));
    CHECK(nya_train_device_embedding(d,y,x,0,0,map));
    CHECK(nya_train_device_embedding(d,y,x,0,SIZE_MAX,map));
    CHECK(nya_train_device_embedding_backward(d,y,x,SIZE_MAX,map));
    CHECK(nya_train_device_embedding(d,y,x,0,3,map));
    CHECK(nya_train_device_embedding(d,y,tiny,0,1,map));
    CHECK(nya_train_device_embedding(d,tiny,x,0,1,map));
    CHECK(nya_train_device_embedding_backward(d,tiny,x,1,map));
    CHECK(nya_train_device_embedding_backward(d,y,tiny,1,map));
    nya_train_device_get_stats(d,&after);
    CHECK(!after.failed && after.used_bytes==before.used_bytes && after.kernel_launches==before.kernel_launches &&
        after.uploads==before.uploads && after.downloads==before.downloads && after.synchronizations==before.synchronizations);
    CHECK(!nya_train_device_read(d,y,0,host,32));
    for (size_t k=0;k<8;++k) CHECK(host[k]==42);
    nya_train_device_free(other); CHECK(!nya_train_device_scratch_end(d,scope));
    /* Recoverable arena-budget failure must not publish a partial map. */
    other=nya_train_device_create("cuda",16); CHECK(other);
    CHECK(!nya_train_device_indices(other,ids,3,3));
    nya_train_device_get_stats(other,&after); CHECK(!after.failed && !after.used_bytes && !after.buffers);
    nya_train_device_free(other); return 0;
}
int main(int argc,char **argv)
{
    nya_train_device *d=NULL;
    CHECK(!packing());
    CHECK(!nya_train_device_indices(NULL,NULL,0,0));
    CHECK(nya_train_device_embedding(NULL,0,0,0,1,0));
    CHECK(nya_train_device_embedding_backward(NULL,0,0,1,0));
    if (argc==1) return 0;
    d=nya_train_device_create("cuda",16*1024*1024); if (!d) return 77;
    if (!strcmp(argv[1],"--failure")) {
        nya_train_buffer t=nya_train_device_alloc(d,4), y=nya_train_device_alloc(d,4), dy=nya_train_device_alloc(d,4), dt=nya_train_device_alloc(d,4);
        CHECK(t && y && dy && dt);
        uint32_t id=0; nya_train_indices map=nya_train_device_indices(d,&id,1,1);
        int result=map?nya_train_device_embedding(d,y,t,0,1,map):-1;
        if (!result) result=nya_train_device_embedding_backward(d,dt,dy,1,map);
        CHECK(result);
        nya_train_device_stats stats; nya_train_device_get_stats(d,&stats);
        CHECK(stats.failed && stats.kernel_launches==strtoull(getenv("NYA_TRAIN_DEVICE_FAIL_LAUNCH_AFTER"),NULL,10));
        float value=42; CHECK(nya_train_device_read(d,dt,0,&value,4) && value==42 && nya_train_device_finish(d));
        CHECK(!nya_train_device_indices(d,&id,1,1)); nya_train_device_free(d); return 0;
    }
    const unsigned types[]={0,1,30,2,8,12,14};
    const size_t counts[]={1,7,33,257}, widths[]={1,31,32,33,255,256,257,513};
    for (size_t kind=0;kind<7;++kind) for (size_t k=0;k<4;++k) {
        if (kind<3) for (size_t j=0;j<8;++j) CHECK(!gather(d,types[kind],widths[j],counts[k]));
        else CHECK(!gather(d,types[kind],k%2?512:256,counts[k]));
    }
    CHECK(!lifetime(d) && !matrix_order(d) && !descriptors(d));
    nya_train_device_free(d);
    puts("resident embeddings: seven storage types, CPU autograd, ordered repeats, opaque maps and scratch reuse passed");
    return 0;
}
