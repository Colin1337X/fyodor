#ifndef FYODOR_CLIENT_H
#define FYODOR_CLIENT_H
#include <stdio.h>
/* Shared native command dispatcher. No subprocesses, engine or HTTP needed for
 * local resource operations. Exit: 0 success, 1 operation/I/O, 2 usage/input,
 * 3 missing resource, 4 revision conflict, 5 busy, 6 denied. Streams remain caller-owned. */
int fyodor_command_run(int argc, char **argv, FILE *input, FILE *output, FILE *error);
#endif
