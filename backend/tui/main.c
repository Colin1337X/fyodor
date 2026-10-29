#include "tui.h"
#include "compute.h"
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#ifdef _WIN32
#include <windows.h>
static HANDLE input,output;static DWORD saved_input,saved_output;static UINT saved_cp;static int active;
static void restore(void) {if(active) {fputs("\033[0m\033[?25h\033[?1049l",stdout);fflush(stdout);SetConsoleMode(input,saved_input);SetConsoleMode(output,saved_output);SetConsoleOutputCP(saved_cp);active=0;}}
static int terminal(void) {
    input=GetStdHandle(STD_INPUT_HANDLE);output=GetStdHandle(STD_OUTPUT_HANDLE);
    if(!GetConsoleMode(input,&saved_input) || !GetConsoleMode(output,&saved_output)) return -1;
    if(!SetConsoleMode(output,saved_output|ENABLE_VIRTUAL_TERMINAL_PROCESSING)) return -1;
    if(!SetConsoleMode(input,ENABLE_WINDOW_INPUT|ENABLE_EXTENDED_FLAGS)) {SetConsoleMode(output,saved_output);return -1;}
    saved_cp=GetConsoleOutputCP();SetConsoleOutputCP(CP_UTF8);active=1;atexit(restore);return 0;
}
static void dimensions(int *w,int *h) {CONSOLE_SCREEN_BUFFER_INFO info;if(GetConsoleScreenBufferInfo(output,&info)) {*w=info.srWindow.Right-info.srWindow.Left+1;*h=info.srWindow.Bottom-info.srWindow.Top+1;}}
static uint32_t event(void) {
    static uint32_t high=0;INPUT_RECORD r;DWORD n;
    if(WaitForSingleObject(input,100)!=WAIT_OBJECT_0 || !ReadConsoleInputW(input,&r,1,&n) || !n || r.EventType!=KEY_EVENT || !r.Event.KeyEvent.bKeyDown) return 0;
    KEY_EVENT_RECORD k=r.Event.KeyEvent;
    switch(k.wVirtualKeyCode) {
        case VK_UP:return TUI_UP;case VK_DOWN:return TUI_DOWN;case VK_LEFT:return TUI_LEFT;case VK_RIGHT:return TUI_RIGHT;
        case VK_ESCAPE:return TUI_ESCAPE;case VK_RETURN:return TUI_ENTER;case VK_BACK:return TUI_BACKSPACE;case VK_TAB:return TUI_TAB;default:break;
    }
    uint32_t c=k.uChar.UnicodeChar;
    if(c>=0xd800 && c<=0xdbff) {high=c;return 0;}
    if(c>=0xdc00 && c<=0xdfff) {uint32_t result=high?0x10000+((high-0xd800)<<10)+(c-0xdc00):0;high=0;return result;}
    high=0;return c;
}
#else
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/ioctl.h>
static struct termios saved;static int active;
static void restore(void) {if(active) {fputs("\033[0m\033[?25h\033[?1049l",stdout);fflush(stdout);tcsetattr(STDIN_FILENO,TCSANOW,&saved);active=0;}}
static int terminal(void) {if(!isatty(0) || !isatty(1) || tcgetattr(0,&saved)) return -1;struct termios raw=saved;raw.c_lflag&=(tcflag_t)~(ICANON|ECHO|ISIG);raw.c_iflag&=(tcflag_t)~(IXON|ICRNL);raw.c_cc[VMIN]=0;raw.c_cc[VTIME]=0;if(tcsetattr(0,TCSANOW,&raw)) return -1;active=1;atexit(restore);return 0;}
static void dimensions(int *w,int *h) {struct winsize size;if(!ioctl(1,TIOCGWINSZ,&size) && size.ws_col && size.ws_row) {*w=size.ws_col;*h=size.ws_row;}}
static int byte(void) {fd_set set;FD_ZERO(&set);FD_SET(0,&set);struct timeval timeout={0,100000};unsigned char c;if(select(1,&set,NULL,NULL,&timeout)>0 && read(0,&c,1)==1) return c;return -1;}
static uint32_t event(void) {
    int c=byte();if(c<0) return 0;
    if(c==27) {int next=byte();if(next=='[' || next=='O') {next=byte();if(next=='A') return TUI_UP;if(next=='B') return TUI_DOWN;if(next=='C') return TUI_RIGHT;if(next=='D') return TUI_LEFT;}return TUI_ESCAPE;}
    if(c==13 || c==10) return TUI_ENTER;
    if(c==127 || c==8) return TUI_BACKSPACE;
    if(c==9) return TUI_TAB;
    if(c<128) return (uint32_t)c;
    unsigned need;uint32_t value,minimum;
    if(c>=0xc2 && c<=0xdf) {need=1;value=(uint32_t)c&31;minimum=128;}
    else if(c>=0xe0 && c<=0xef) {need=2;value=(uint32_t)c&15;minimum=2048;}
    else if(c>=0xf0 && c<=0xf4) {need=3;value=(uint32_t)c&7;minimum=65536;}else return 0;
    while(need--) {c=byte();if(c<0 || (c&0xc0)!=0x80) return 0;value=(value<<6)|((uint32_t)c&63);}
    return value>=minimum && value<=0x10ffff && !(value>=0xd800 && value<=0xdfff)?value:0;
}
#endif
static volatile sig_atomic_t stopping;
static void stop(int signum) {(void)signum;stopping=1;}
int fyodor_monitor_run(int argc,char **argv)
{
    tui_state state;tui_init(&state);int snapshot=0,width=100,height=28;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--help")) {puts("fyodor-tui [--model PATH] [--metrics CSV] [--snapshot]\nKeys: 1-3 views, : commands, ? help, Space pause, q quit");return 0;}
        if(!strcmp(argv[i],"--snapshot")) {snapshot=1;continue;}
        char *target=NULL;size_t capacity=0;
        if(!strcmp(argv[i],"--model")) {target=state.model_path;capacity=sizeof(state.model_path);state.view=1;}
        else if(!strcmp(argv[i],"--metrics")) {target=state.metrics_path;capacity=sizeof(state.metrics_path);state.view=2;}
        if(!target || ++i==argc || strlen(argv[i])>=capacity || !*argv[i]) {fputs("Invalid arguments; use --help\n",stderr);return 2;}
        memcpy(target,argv[i],strlen(argv[i])+1);
    }
    state.compiled=(nya_compute_cuda_compiled()?1U:0U)|(nya_compute_vulkan_compiled()?2U:0U)|(nya_compute_backend_compiled("rocm")?4U:0U)|(nya_compute_backend_compiled("mlx")?8U:0U);
    if(*state.model_path) tui_inspect(&state);
    if(*state.metrics_path) tui_metrics(&state);
    tui_frame *frame=malloc(sizeof(*frame));if(!frame) return 1;
    if(snapshot) {tui_render(&state,frame,width,height);tui_write(stdout,frame,0,0);free(frame);return 0;}
    if(terminal()) {free(frame);fputs("Interactive terminal unavailable; use --snapshot for plain output.\n",stderr);return 2;}
    signal(SIGINT,stop);signal(SIGTERM,stop);fputs("\033[?1049h\033[?25l",stdout);
    unsigned ticks=0;int color=getenv("NO_COLOR")==NULL; int dirty=1,old_width=0,old_height=0;
    while(!state.quit && !stopping) {
        dimensions(&width,&height);
        if(dirty || width!=old_width || height!=old_height) {tui_render(&state,frame,width,height);tui_write(stdout,frame,1,color);old_width=width;old_height=height;dirty=0;}
        uint32_t key=event();int action=key?tui_key(&state,key):TUI_NONE;if(key) dirty=1;
        if(action==TUI_INSPECT || (action==TUI_REFRESH && *state.model_path)) tui_inspect(&state);
        if(action==TUI_METRICS || action==TUI_REFRESH || ++ticks>=10) {if(*state.metrics_path && !state.paused) {tui_metrics(&state);dirty=1;} ticks=0;}
    }
    restore();free(frame);return 0;
}
