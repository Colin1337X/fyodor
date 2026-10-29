#ifndef NYA_TUI_H
#define NYA_TUI_H
#include "format.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#define TUI_HISTORY 192
#define TUI_WIDTH 240
#define TUI_HEIGHT 80
typedef struct tui_sample {
    uint64_t step;
    double loss, rate, validation, gradient, used, peak, capacity;
    unsigned optional;
    char compute[24];
} tui_sample;
typedef struct tui_state {
    char model_path[1024], metrics_path[1024], status[256], command[1100];
    nya_format_info model;
    uint64_t model_bytes;
    int model_valid, view, help, editing, paused, quit, overflow;
    size_t command_length, samples, malformed;
    tui_sample history[TUI_HISTORY];
    unsigned compiled;
} tui_state;
typedef struct tui_cell { uint32_t code; unsigned char style; } tui_cell;
typedef struct tui_frame { int width,height; tui_cell cells[TUI_WIDTH*TUI_HEIGHT]; } tui_frame;
enum { TUI_UP=0x110001,TUI_DOWN,TUI_LEFT,TUI_RIGHT,TUI_ESCAPE,TUI_ENTER,TUI_BACKSPACE,TUI_TAB };
enum { TUI_NONE,TUI_REFRESH,TUI_INSPECT,TUI_METRICS };
int fyodor_monitor_run(int argc,char **argv);
void tui_init(tui_state *state);
int tui_key(tui_state *state,uint32_t key);
int tui_inspect(tui_state *state);
int tui_metrics(tui_state *state);
void tui_render(const tui_state *state,tui_frame *frame,int width,int height);
void tui_write(FILE *file,const tui_frame *frame,int ansi,int color);
/* Returns UTF-8 byte count, or zero for an invalid scalar value. */
size_t tui_utf8(uint32_t code,char out[4]);
#endif
