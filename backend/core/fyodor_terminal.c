#include "fyodor_terminal.h"
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static HANDLE input_handle, output_handle;
static DWORD input_mode, output_mode;
#else
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <poll.h>
static struct termios saved_mode;
#endif
static int active;
static volatile sig_atomic_t interrupted;
static void (*saved_int)(int), (*saved_term)(int);
static void on_signal(int number) { (void)number; interrupted=1; }
void fyodor_terminal_end(void)
{
    if(!active) return;
    fputs("\033[0m\033[?25h\033[?1049l",stdout); (void)fflush(stdout);
#ifdef _WIN32
    (void)SetConsoleMode(input_handle,input_mode);
    (void)SetConsoleMode(output_handle,output_mode);
#else
    (void)tcsetattr(STDIN_FILENO,TCSANOW,&saved_mode);
#endif
    (void)signal(SIGINT,saved_int); (void)signal(SIGTERM,saved_term);
    active=0;
}
int fyodor_terminal_begin(void)
{
    if(active) return 1;
    const char *term=getenv("TERM");
    if(term && strcmp(term,"dumb")==0) return 0;
#ifdef _WIN32
    input_handle=GetStdHandle(STD_INPUT_HANDLE); output_handle=GetStdHandle(STD_OUTPUT_HANDLE);
    if(!GetConsoleMode(input_handle,&input_mode)||!GetConsoleMode(output_handle,&output_mode)) return 0;
    if(!SetConsoleMode(output_handle,output_mode|ENABLE_VIRTUAL_TERMINAL_PROCESSING)) return 0;
    if(!SetConsoleMode(input_handle,(input_mode|ENABLE_WINDOW_INPUT)&
        ~(DWORD)(ENABLE_LINE_INPUT|ENABLE_ECHO_INPUT|ENABLE_PROCESSED_INPUT))) {
        (void)SetConsoleMode(output_handle,output_mode); return 0;
    }
#else
    if(!isatty(STDIN_FILENO)||!isatty(STDOUT_FILENO)||tcgetattr(STDIN_FILENO,&saved_mode)!=0) return 0;
    struct termios raw=saved_mode;
    raw.c_lflag &= (tcflag_t)~(ICANON|ECHO|ISIG);
    raw.c_iflag &= (tcflag_t)~(IXON|ICRNL);
    raw.c_cc[VMIN]=0; raw.c_cc[VTIME]=1;
    if(tcsetattr(STDIN_FILENO,TCSANOW,&raw)!=0) return 0;
#endif
    active=1; interrupted=0;
    saved_int=signal(SIGINT,on_signal); saved_term=signal(SIGTERM,on_signal);
    if(atexit(fyodor_terminal_end)!=0) { fyodor_terminal_end(); return 0; }
    fputs("\033[?1049h\033[?25l",stdout); (void)fflush(stdout); return 1;
}
void fyodor_terminal_size(unsigned *columns,unsigned *rows)
{
    *columns=80; *rows=24;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    /* CONOUT$ follows the active alternate buffer; the saved stdout handle
     * may still refer to the original screen after DECSET 1049. */
    HANDLE screen=CreateFileW(L"CONOUT$",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,OPEN_EXISTING,0,NULL);
    if(screen!=INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(screen,&info)) {
        *columns=(unsigned)(info.srWindow.Right-info.srWindow.Left+1);
        *rows=(unsigned)(info.srWindow.Bottom-info.srWindow.Top+1);
    }
    if(screen!=INVALID_HANDLE_VALUE) (void)CloseHandle(screen);
#else
    struct winsize size;
    if(ioctl(STDOUT_FILENO,TIOCGWINSZ,&size)==0) { *columns=size.ws_col; *rows=size.ws_row; }
#endif
    if(*columns<1) *columns=1;
    if(*columns>300) *columns=300;
    if(*rows<1) *rows=1;
    if(*rows>100) *rows=100;
}
int fyodor_terminal_key(void)
{
    if(interrupted) return FYODOR_KEY_END;
#ifdef _WIN32
    static unsigned char pending[4]; static int at,count;
    if(at<count) return pending[at++];
    DWORD wait=WaitForSingleObject(input_handle,100);
    if(wait==WAIT_TIMEOUT) return FYODOR_KEY_IDLE;
    if(wait!=WAIT_OBJECT_0) return FYODOR_KEY_END;
    INPUT_RECORD record; DWORD got=0;
    if(!ReadConsoleInputW(input_handle,&record,1,&got)||got==0) return FYODOR_KEY_END;
    if(record.EventType!=KEY_EVENT||!record.Event.KeyEvent.bKeyDown) return FYODOR_KEY_IDLE;
    KEY_EVENT_RECORD key=record.Event.KeyEvent;
    if(key.wVirtualKeyCode==VK_UP) return FYODOR_KEY_UP;
    if(key.wVirtualKeyCode==VK_DOWN) return FYODOR_KEY_DOWN;
    if(key.wVirtualKeyCode==VK_LEFT) return FYODOR_KEY_LEFT;
    if(key.wVirtualKeyCode==VK_RIGHT) return FYODOR_KEY_RIGHT;
    if(key.uChar.UnicodeChar==0) return FYODOR_KEY_IDLE;
    count=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,&key.uChar.UnicodeChar,1,(char *)pending,4,NULL,NULL);
    at=0;
    if(count<=0) return FYODOR_KEY_IDLE;
    return pending[at++];
#else
    struct pollfd fd={STDIN_FILENO,POLLIN,0};
    int ready=poll(&fd,1,100);
    if(ready<0) return interrupted?FYODOR_KEY_END:FYODOR_KEY_IDLE;
    if(ready==0) return FYODOR_KEY_IDLE;
    unsigned char byte;
    if(read(STDIN_FILENO,&byte,1)!=1) return FYODOR_KEY_END;
    return byte;
#endif
}
void fyodor_terminal_text(FILE *out,const char *text,size_t limit)
{
    const unsigned char *p=(const unsigned char *)text;
    size_t written=0;
    while(*p && written<limit) {
        unsigned char c=*p++;
        if(c>=32&&c<127) { fputc(c,out); ++written; }
        else { if(limit-written<4) break; fprintf(out,"\\x%02x",(unsigned)c); written+=4; }
    }
}
