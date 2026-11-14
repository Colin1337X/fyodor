#include "fyodor_tui.h"
#include "fyodor_terminal.h"
#include "fyodor_theme.h"
#include <stdio.h>
#include <string.h>
#define CHECK(c) do { if(!(c)) { fprintf(stderr,"line %d: %s\n",__LINE__,#c); return 1; } } while(0)
typedef struct { const unsigned char *bytes; size_t at; } key_stream;
static int next_key(void *context)
{
    key_stream *s=context;
    return s->bytes[s->at] ? s->bytes[s->at++] : FYODOR_KEY_IDLE;
}
int main(void)
{
    key_stream stream={(const unsigned char *)"\033[A\033[1;5B\033OC\033[D\033[3~x\033:y",0};
    int pending=FYODOR_KEY_IDLE;
    const int expected[]={FYODOR_KEY_UP,FYODOR_KEY_DOWN,FYODOR_KEY_RIGHT,
        FYODOR_KEY_LEFT,FYODOR_KEY_IDLE,'x',27,':','y',FYODOR_KEY_IDLE};
    for(size_t i=0;i<sizeof(expected)/sizeof(expected[0]);++i)
        CHECK(fyodor_terminal_decode_key(next_key,&stream,&pending)==expected[i]);
    key_stream pages={(const unsigned char *)"\033[H\033OF\033[1~\033[4~\033[7~\033[8~\033[5~\033[6;2~\033[999999~",0};
    const int page_keys[]={FYODOR_KEY_HOME,FYODOR_KEY_LAST,FYODOR_KEY_HOME,FYODOR_KEY_LAST,
        FYODOR_KEY_HOME,FYODOR_KEY_LAST,FYODOR_KEY_PAGE_UP,FYODOR_KEY_PAGE_DOWN,FYODOR_KEY_IDLE};
    for(size_t i=0;i<sizeof(page_keys)/sizeof(page_keys[0]);++i)
        CHECK(fyodor_terminal_decode_key(next_key,&pages,&pending)==page_keys[i]);
    CHECK(fyodor_theme_count()==10);
    CHECK(fyodor_theme_find("bad")==NULL && fyodor_theme_at(10)==NULL);
    for(size_t i=0;i<fyodor_theme_count();++i) {
        const fyodor_theme *theme=fyodor_theme_at(i); CHECK(fyodor_theme_find(theme->id)==theme);
        for(int mode=0;mode<=3;++mode) {
            char sgr[80];
            CHECK(fyodor_theme_sgr(theme,FYODOR_TEXT,FYODOR_BACKGROUND,(fyodor_color_mode)mode,sgr,sizeof(sgr))>0);
            CHECK(strncmp(sgr,"\033[0",3)==0);
            if(mode==3) CHECK(strstr(sgr,"38;2;") && strstr(sgr,"48;2;"));
            if(mode==2) CHECK(strstr(sgr,"38;5;") && strstr(sgr,"48;5;"));
            if(mode==0) CHECK(strcmp(sgr,"\033[0m")==0);
            strcpy(sgr,"unchanged"); CHECK(fyodor_theme_sgr(theme,FYODOR_TEXT,FYODOR_BACKGROUND,(fyodor_color_mode)mode,sgr,2)==0); CHECK(strcmp(sgr,"unchanged")==0);
        }
    }
    char *args[5];
    char line[]=" context  append 'two words' \"C:\\Users\\name\" '' ";
    CHECK(fyodor_tui_split(line,args,5)==5);
    CHECK(strcmp(args[0],"context")==0 && strcmp(args[2],"two words")==0);
    CHECK(strcmp(args[3],"C:\\Users\\name")==0 && strcmp(args[4],"")==0);
    char bad[]="a 'unfinished"; CHECK(fyodor_tui_split(bad,args,5)==-1);
    char overflow[]="a b c d e f"; CHECK(fyodor_tui_split(overflow,args,5)==-1);
    char concatenate[]="a\"b c\"d"; CHECK(fyodor_tui_split(concatenate,args,5)==1); CHECK(strcmp(args[0],"ab cd")==0);
    char empty[]=" \t"; CHECK(fyodor_tui_split(empty,args,0)==0);
    FILE *file=tmpfile(); CHECK(file!=NULL);
    fyodor_terminal_text(file,"A\033]52;secret\007\n\xc2\x9b",100);
    CHECK(fflush(file)==0 && fseek(file,0,SEEK_SET)==0);
    char text[100]={0}; CHECK(fread(text,1,sizeof(text)-1,file)>0);
    CHECK(strcmp(text,"A\\x1b]52;secret\\x07\\x0a\\xc2\\x9b")==0);
    fclose(file);
    file=tmpfile(); CHECK(file!=NULL); fyodor_terminal_text(file,"ab\033c",5);
    CHECK(fflush(file)==0 && fseek(file,0,SEEK_SET)==0);
    memset(text,0,sizeof(text)); CHECK(fread(text,1,sizeof(text)-1,file)==2); CHECK(strcmp(text,"ab")==0); fclose(file);
    return 0;
}
