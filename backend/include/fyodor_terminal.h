#ifndef FYODOR_TERMINAL_H
#define FYODOR_TERMINAL_H
#include <stdio.h>
/* One foreground terminal per process. No application-service dependencies. */
enum { FYODOR_KEY_END=-1, FYODOR_KEY_IDLE=-2, FYODOR_KEY_UP=256,
       FYODOR_KEY_DOWN, FYODOR_KEY_LEFT, FYODOR_KEY_RIGHT,
       FYODOR_KEY_HOME, FYODOR_KEY_LAST, FYODOR_KEY_PAGE_UP, FYODOR_KEY_PAGE_DOWN };
int fyodor_terminal_begin(void);
void fyodor_terminal_end(void);
int fyodor_terminal_key(void);
/* Reader returns a byte or END/IDLE. Initialize pending to FYODOR_KEY_IDLE. */
int fyodor_terminal_decode_key(int (*reader)(void *),void *context,int *pending);
void fyodor_terminal_size(unsigned *columns, unsigned *rows);
/* Escapes controls and non-ASCII bytes into printable ASCII, including bidi
 * and terminal escape sequences. Width therefore equals rendered byte count. */
void fyodor_terminal_text(FILE *out, const char *text, size_t limit);
/* Tail clipping for command entry, with an ellipsis when clipped. */
void fyodor_terminal_text_tail(FILE *out, const char *text, size_t limit);
#endif
