#include "fyodor_client.h"
#ifdef FYODOR_TUI_MAIN
#include "fyodor_tui.h"
#endif
#include <stdint.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
int wmain(int argc,wchar_t **wide)
{
    if(argc<0||(size_t)argc>SIZE_MAX/sizeof(char *)-1) return 2;
    char **argv=calloc((size_t)argc+1,sizeof(*argv));
    if(argv==NULL) return 1;
    int result=1;
    for(int i=0;i<argc;++i) {
        int bytes=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wide[i],-1,NULL,0,NULL,NULL);
        if(bytes<=0) goto cleanup;
        argv[i]=malloc((size_t)bytes);
        if(argv[i]==NULL||WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wide[i],-1,argv[i],bytes,NULL,NULL)!=bytes) goto cleanup;
    }
    if(_setmode(_fileno(stdin),_O_BINARY)==-1||_setmode(_fileno(stdout),_O_BINARY)==-1) goto cleanup;
#ifdef FYODOR_TUI_MAIN
    result=fyodor_tui_run(argc,argv);
#else
    result=fyodor_command_run(argc,argv,stdin,stdout,stderr);
#endif
cleanup:
    for(int i=0;i<argc;++i) free(argv[i]);
    free(argv); return result;
}
#else
int main(int argc,char **argv) {
#ifdef FYODOR_TUI_MAIN
    return fyodor_tui_run(argc,argv);
#else
    return fyodor_command_run(argc,argv,stdin,stdout,stderr);
#endif
}
#endif
