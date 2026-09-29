#ifndef NYA_TRAINING_INDICES_H
#define NYA_TRAINING_INDICES_H
#include <stddef.h>
#include <stdint.h>

/* Host staging only. Create requires a fresh/freed pack. Packed U32 sections: ids[count], keys[groups],
   offsets[groups+1], positions[count]. Positions within a group ascend. */
typedef struct nya_train_index_pack {
    uint32_t *data;
    size_t bytes, count, rows, groups;
} nya_train_index_pack;
int nya_train_index_pack_create(nya_train_index_pack *pack, const uint32_t *ids, size_t count, size_t rows);
void nya_train_index_pack_free(nya_train_index_pack *pack);
#endif
