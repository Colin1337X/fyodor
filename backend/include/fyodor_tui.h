#ifndef FYODOR_TUI_H
#define FYODOR_TUI_H
#include <stddef.h>
/* In-place shell-free tokenization. Quotes group text, backslashes are literal.
 * Returns argument count or -1 for unmatched quotes/argument overflow. */
int fyodor_tui_split(char *line,char **argv,size_t capacity);
int fyodor_tui_run(int argc,char **argv);
#endif
