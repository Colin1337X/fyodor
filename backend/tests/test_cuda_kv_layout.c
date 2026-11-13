#include "../../CUDA/kv_layout.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do {if(!(x)) {fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;}} while(0)
int main(void)
{
    nya_llm_layer layers[3];memset(layers,0,sizeof(layers));
    nya_llm_context m;memset(&m,0,sizeof(m));m.layers=layers;m.block_count=3;
    for(unsigned i=0;i<3;++i) {layers[i].kv_source=i;layers[i].head_dimension=64;layers[i].kv_head_count=2;layers[i].sliding_window=4;}
    nya_cuda_kv_layout layout[3];size_t count=0;
    CHECK(!nya_cuda_kv_layout_build(&m,100,3,layout,&count));
    CHECK(count==3*6*128 && layout[1].offset==6*128 && layout[2].slots==6);
    layers[1].kv_source=0;layers[1].sliding_window=9;
    CHECK(!nya_cuda_kv_layout_build(&m,100,3,layout,&count));
    CHECK(count==(11+6)*128 && layout[0].slots==11 && layout[1].offset==0 && layout[2].offset==11*128);
    layers[1].sliding_window=0;
    CHECK(!nya_cuda_kv_layout_build(&m,100,3,layout,&count));
    CHECK(layout[0].slots==100 && layout[1].slots==100 && count==106*128);
    layers[1].head_dimension=128;CHECK(nya_cuda_kv_layout_build(&m,100,3,layout,&count));
    layers[1].head_dimension=64;layers[0].kv_source=1;CHECK(nya_cuda_kv_layout_build(&m,100,3,layout,&count));
    layers[0].kv_source=0;m.block_count=1;
    CHECK(nya_cuda_kv_layout_build(&m,0,1,layout,&count));
    CHECK(nya_cuda_kv_layout_build(&m,10,0,layout,&count));
    CHECK(nya_cuda_kv_layout_build(&m,10,11,layout,&count));
    layers[0].sliding_window=0;CHECK(nya_cuda_kv_layout_build(&m,SIZE_MAX,1,layout,&count));
    for(size_t cap=1;cap<32;++cap) for(size_t batch=1;batch<=cap;++batch) for(unsigned window=1;window<32;++window) {
        layers[0].sliding_window=window;
        CHECK(!nya_cuda_kv_layout_build(&m,cap,batch,layout,&count));
        size_t slots=layout[0].slots;
        CHECK(slots<=cap && slots>=batch);
        for(size_t pos=0;pos+batch<=cap;++pos) {
            size_t first=pos>=window?pos-window+1:0;
            /* Simultaneous batch writes must not alias any query's live prefix. */
            for(size_t a=first;a<pos+batch;++a) for(size_t b=a+1;b<pos+batch;++b) CHECK(a%slots!=b%slots);
        }
    }
    CHECK(nya_cuda_kv_retained(2049,1535,514));
    CHECK(!nya_cuda_kv_retained(2049,1535,513));
    CHECK(nya_cuda_kv_retained(3,1535,0));
    puts("CUDA host KV layout: shared consumers, overflow, batch retention and eviction passed");return 0;
}
