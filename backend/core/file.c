#include "file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#ifdef _WIN32
static wchar_t *file_wide_path(const char *path)
{
    if (path == NULL) { errno = EINVAL; return NULL; }
    int count = MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,NULL,0);
    if (count <= 0 || (size_t)count > SIZE_MAX/sizeof(wchar_t)) { errno = EINVAL; return NULL; }
    wchar_t *wide = malloc((size_t)count*sizeof(*wide));
    if (wide == NULL) { errno = ENOMEM; return NULL; }
    if (MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,wide,count) != count) {
        free(wide); errno = EINVAL; return NULL;
    }
    return wide;
}
#endif

/* Windows narrow fopen interprets its argument through the active code page,
 * whereas JSON and the GGUF mapper use UTF-8. Convert explicitly so inspection
 * and generation open the same path for Korean, emoji, and other Unicode names.
 * POSIX filenames already carry byte strings and need no conversion. */
FILE *nya_file_open_read(const char *path)
{
    if (path == NULL) { errno = EINVAL; return NULL; }
#ifdef _WIN32
    {
        wchar_t *wide = file_wide_path(path);
        if (wide == NULL) return NULL;
        FILE *file = _wfopen(wide, L"rb");
        free(wide);
        return file;
    }
#else
    return fopen(path, "rb");
#endif
}

FILE *nya_file_create_exclusive(const char *path)
{
    if (path == NULL) { errno = EINVAL; return NULL; }
#ifdef _WIN32
    wchar_t *wide = file_wide_path(path);
    if (wide == NULL) return NULL;
    int fd = _wopen(wide,_O_WRONLY|_O_CREAT|_O_EXCL|_O_BINARY|_O_NOINHERIT,_S_IREAD|_S_IWRITE);
    if (fd < 0) { free(wide); return NULL; }
    FILE *file = _fdopen(fd,"wb");
    if (file == NULL) { _close(fd); _wremove(wide); }
    free(wide);
#else
    int fd = open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if (fd < 0) return NULL;
    FILE *file = fdopen(fd,"wb");
    if (file == NULL) { close(fd); remove(path); }
#endif
    return file;
}

int nya_file_remove(const char *path)
{
    if (path == NULL) { errno = EINVAL; return -1; }
#ifdef _WIN32
    wchar_t *wide = file_wide_path(path);
    if (wide == NULL) return -1;
    int result = _wremove(wide);
    free(wide);
    return result;
#else
    return remove(path);
#endif
}

/* Query the opened descriptor instead of looking the path up a second time.
 * This keeps file identity stable during inspection and uses 64-bit sizes. */
int nya_file_regular_size(FILE *file, uint64_t *size)
{
    if (file == NULL || size == NULL) return -1;
#ifdef _WIN32
    {
        struct _stat64 information;
        if (_fstat64(_fileno(file), &information) != 0 ||
            (information.st_mode & _S_IFREG) == 0 || information.st_size < 0) return -1;
        *size = (uint64_t)information.st_size;
    }
#else
    {
        struct stat information;
        if (fstat(fileno(file), &information) != 0 ||
            !S_ISREG(information.st_mode) || information.st_size < 0) return -1;
        *size = (uint64_t)information.st_size;
    }
#endif
    return 0;
}
