#ifndef NYA_FILE_INTERNAL_H
#define NYA_FILE_INTERNAL_H

#include <stdint.h>
#include <stdio.h>

/* Model paths at the API boundary are UTF-8 on every platform. */
FILE *nya_file_open_read(const char *path);
/* UTF-8 paths; exclusive binary creation never truncates an existing file. */
FILE *nya_file_create_exclusive(const char *path);
int nya_file_remove(const char *path);
int nya_file_regular_size(FILE *file, uint64_t *size);

#endif
