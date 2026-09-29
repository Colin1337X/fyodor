#include "training_indices.h"
#include <stdlib.h>
#include <string.h>

typedef struct nya_train_index_pair { uint32_t row, position; } nya_train_index_pair;
static int nya_train_index_compare(const void *left, const void *right)
{
    const nya_train_index_pair *a=left, *b=right;
    if (a->row != b->row) return a->row < b->row ? -1 : 1;
    return a->position < b->position ? -1 : a->position != b->position;
}
int nya_train_index_pack_create(nya_train_index_pack *pack, const uint32_t *ids, size_t count, size_t rows)
{
    if (!pack) return -1;
    memset(pack,0,sizeof(*pack));
    if (!ids || !count || !rows || rows > UINT32_MAX || count > UINT32_MAX ||
        count > (SIZE_MAX/sizeof(uint32_t)-1)/4 || count > SIZE_MAX/sizeof(nya_train_index_pair)) return -1;
    for (size_t i=0;i<count;++i) if (ids[i]>=rows) return -1;
    nya_train_index_pair *pairs=malloc(count*sizeof(*pairs));
    if (!pairs) return -1;
    for (size_t i=0;i<count;++i) pairs[i]=(nya_train_index_pair){ids[i],(uint32_t)i};
    qsort(pairs,count,sizeof(*pairs),nya_train_index_compare);
    size_t groups=1;
    for (size_t i=1;i<count;++i) if (pairs[i].row!=pairs[i-1].row) ++groups;
    size_t bytes=(2*count+2*groups+1)*sizeof(uint32_t);
    uint32_t *data=malloc(bytes);
    if (!data) { free(pairs); return -1; }
    memcpy(data,ids,count*sizeof(*data));
    uint32_t *keys=data+count, *offsets=keys+groups, *positions=offsets+groups+1;
    size_t group=0;
    for (size_t i=0;i<count;++i) {
        if (!i || pairs[i].row!=pairs[i-1].row) {
            keys[group]=pairs[i].row; offsets[group]=(uint32_t)i; ++group;
        }
        positions[i]=pairs[i].position;
    }
    offsets[groups]=(uint32_t)count;
    free(pairs);
    *pack=(nya_train_index_pack){data,bytes,count,rows,groups};
    return 0;
}
void nya_train_index_pack_free(nya_train_index_pack *pack)
{
    if (pack) { free(pack->data); memset(pack,0,sizeof(*pack)); }
}
