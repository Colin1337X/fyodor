/* Native C host layout for optional bounded CUDA KV storage. */
#ifndef NYA_CUDA_KV_LAYOUT_H
#define NYA_CUDA_KV_LAYOUT_H
#include "../backend/core/llm_internal.h"
typedef struct nya_cuda_kv_layout { size_t offset, slots; } nya_cuda_kv_layout;
/* Offsets/count are floats per K or V. Every shared consumer contributes its
   retention window; any full-attention consumer keeps the complete prefix. */
static int nya_cuda_kv_layout_build(const nya_llm_context *m, size_t capacity,
    size_t batch, nya_cuda_kv_layout *layout, size_t *count)
{
    size_t total=0;
    if(!capacity || !batch || batch>capacity) return -1;
    for(size_t i=0;i<m->block_count;++i) {
        const nya_llm_layer *w=&m->layers[i];
        if(w->kv_source>i || m->layers[w->kv_source].kv_source!=w->kv_source) return -1;
        if(w->kv_source!=i) {
            const nya_llm_layer *owner=&m->layers[w->kv_source];
            if(owner->head_dimension!=w->head_dimension || owner->kv_head_count!=w->kv_head_count) return -1;
            if(layout) layout[i]=layout[w->kv_source];
            continue;
        }
        size_t window=0,slots=capacity;
        int full=0;
        for(size_t j=i;j<m->block_count;++j) if(m->layers[j].kv_source==i) {
            size_t needed=m->layers[j].sliding_window;
            if(!needed) full=1;
            if(needed>window) window=needed;
        }
        if(!full && window<capacity && batch-1<capacity-window) slots=window+batch-1;
        if(!w->head_dimension || !w->kv_head_count || w->kv_head_count>SIZE_MAX/w->head_dimension) return -1;
        size_t width=(size_t)w->head_dimension*w->kv_head_count;
        if(slots>SIZE_MAX/width || total>SIZE_MAX-slots*width) return -1;
        if(layout) layout[i]=(nya_cuda_kv_layout){total,slots};
        total+=slots*width;
    }
    *count=total;return 0;
}
/* A rewind cannot recreate evicted rows. High-water is reset only at position 0. */
static int nya_cuda_kv_retained(size_t high_water,size_t slots,size_t position)
{ return high_water<=slots || position>=high_water-slots; }
#endif
