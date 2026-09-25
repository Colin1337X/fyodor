/* Private runner control: a parent owns the write end of stdin's pipe.
   Poll only between complete optimizer updates; never do model I/O from a
   signal handler or interrupt a backward/AdamW operation. */
#ifndef NYA_TRAIN_CONTROL_H
#define NYA_TRAIN_CONTROL_H
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
static int train_control_valid(void)
{
    intptr_t handle = _get_osfhandle(0);
    return handle != -1 && GetFileType((HANDLE)handle) == FILE_TYPE_PIPE;
}
static int train_control_stop(void)
{
    HANDLE pipe = (HANDLE)_get_osfhandle(0);
    DWORD available = 0, count = 0;
    char bytes[32];
    /* A lost controller also requests a stop. No operation here can wait for
       input: ReadFile is issued only for bytes already present in the pipe. */
    if (!PeekNamedPipe(pipe,NULL,0,NULL,&available,NULL)) return 1;
    if (!available) return 0;
    if (available > sizeof(bytes)) available = sizeof(bytes);
    if (!ReadFile(pipe,bytes,available,&count,NULL)) return 1;
    for (DWORD i = 0; i < count; ++i) if (bytes[i] == 'S') return 1;
    return 0;
}
#else
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
static int train_control_valid(void)
{
    struct stat info;
    return fstat(STDIN_FILENO,&info) == 0 && S_ISFIFO(info.st_mode);
}
static int train_control_stop(void)
{
    struct pollfd input = {STDIN_FILENO,POLLIN,0};
    int ready = poll(&input,1,0);
    if (ready < 0) return errno != EINTR;
    if (!ready) return 0;
    if (input.revents & (POLLERR|POLLNVAL)) return 1;
    char bytes[32];
    ssize_t count = read(STDIN_FILENO,bytes,sizeof(bytes));
    if (count <= 0) return count == 0 || errno != EINTR;
    for (ssize_t i = 0; i < count; ++i) if (bytes[i] == 'S') return 1;
    return 0;
}
#endif

static int train_control_init(void)
{
    if (!train_control_valid()) return 0;
#ifndef _WIN32
    /* A controller can disappear with both control and log pipes. Losing the
       progress reader must not terminate the process before its final save. */
    if (signal(SIGPIPE,SIG_IGN) == SIG_ERR) return 0;
#endif
    return 1;
}
#endif
