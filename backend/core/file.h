#ifndef NYA_FILE_INTERNAL_H
#define NYA_FILE_INTERNAL_H

#include <stdint.h>
#include <stdio.h>

/* Model paths at the API boundary are UTF-8 on every platform. */
FILE *nya_file_open_read(const char *path);
int nya_file_regular_size(FILE *file, uint64_t *size);

#endif
