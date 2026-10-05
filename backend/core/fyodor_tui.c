#include "fyodor_tui.h"
#include "fyodor_terminal.h"
#include "fyodor_client.h"
#include "fyodor_store.h"
#include "fyodor_theme.h"
#include <stdlib.h>
#include <string.h>

#define LINE_CAPACITY 8192
#define LOG_CAPACITY 65536
typedef struct {
    fyodor_store *store;
    char *path,*name_space;
    fyodor_resource_summary *items;
    size_t count,selected,scroll;
    char after[FYODOR_RESOURCE_URI_CAPACITY];
    char log[LOG_CAPACITY];
    int detail;
    const fyodor_theme *theme;
    fyodor_color_mode colors;
} tui_state;
static void style(const tui_state *state,fyodor_theme_role foreground,fyodor_theme_role background)
{
    char sgr[80];
    if(fyodor_theme_sgr(state->theme,foreground,background,state->colors,sgr,sizeof(sgr))) fputs(sgr,stdout);
}
int fyodor_tui_split(char *line,char **argv,size_t capacity)
{
    char *read=line,*write=line; size_t count=0;
    while(*read) {
        while(*read==' '||*read=='\t') ++read;
        if(!*read) break;
        if(count>=capacity) return -1;
        argv[count++]=write; char quote=0;
        while(*read) {
            char c=*read++;
            if(quote) { if(c==quote) quote=0; else *write++=c; }
            else if(c=='\''||c=='"') quote=c;
            else if(c==' '||c=='\t') break;
            else *write++=c;
        }
        if(quote) return -1;
        *write++=0;
    }
    return (int)count;
}
static void page(tui_state *state,int next)
{
    char cursor[FYODOR_RESOURCE_URI_CAPACITY]={0};
    if(next && state->count) (void)fyodor_resource_format(&state->items[state->count-1].ref,cursor,sizeof(cursor));
    fyodor_resource_summary *items=NULL; size_t count=0;
    fyodor_store_result result=fyodor_store_list(state->store,state->name_space,next?cursor:NULL,20,&items,&count);
    if(result!=FYODOR_STORE_OK) { (void)snprintf(state->log,sizeof(state->log),"List failed (%d)",(int)result); return; }
    if(next && count==0) { fyodor_resource_summaries_free(items,count); strcpy(state->log,"End of resources. Press r for the first page."); return; }
    fyodor_resource_summaries_free(state->items,state->count);
    state->items=items; state->count=count; state->selected=0; state->detail=0; state->scroll=0;
    strcpy(state->after,cursor); strcpy(state->log,"Local administrator workspace. Commands use the shared native service.");
}
static void inspect(tui_state *state)
{
    if(state->count==0) return;
    fyodor_resource_record record={0};
    fyodor_store_result result=fyodor_store_get(state->store,state->name_space,&state->items[state->selected].ref,0,&record);
    if(result!=FYODOR_STORE_OK) { (void)snprintf(state->log,sizeof(state->log),"Read failed (%d)",(int)result); return; }
    char uri[FYODOR_RESOURCE_URI_CAPACITY]; (void)fyodor_resource_format(&record.ref,uri,sizeof(uri));
    int length=snprintf(state->log,sizeof(state->log),"%s\n%s\nRevision %llu\n\n%s",record.title,uri,(unsigned long long)record.revision,record.content);
    if(length<0) strcpy(state->log,"Unable to format resource.");
    else if((size_t)length>=sizeof(state->log)) {
        const char marker[]="\n[Preview truncated at 64 KiB; use resource export for full content.]";
        memcpy(state->log+sizeof(state->log)-sizeof(marker),marker,sizeof(marker));
    }
    state->detail=1; state->scroll=0; fyodor_resource_record_free(&record);
}
static void command(tui_state *state,char *line)
{
    char *args[96]={"fyodor","--store",state->path,"--namespace",state->name_space};
    int count=fyodor_tui_split(line,args+5,91);
    if(count<1) { strcpy(state->log,"Command syntax error: use matching quotes and at most 91 arguments."); return; }
    /* Do not let an import consume the terminal command stream. */
    if(count>=2 && strcmp(args[5],"resource")==0 && strcmp(args[6],"import")==0) {
        strcpy(state->log,"Import packages through the CLI. TUI commands do not consume stdin packages."); return;
    }
    FILE *empty=tmpfile(),*capture=tmpfile();
    if(!empty||!capture) { if(empty) fclose(empty); if(capture) fclose(capture); strcpy(state->log,"Cannot create command capture."); return; }
    int code=fyodor_command_run(count+5,args,empty,capture,capture);
    if(fflush(capture)!=0||fseek(capture,0,SEEK_SET)!=0) strcpy(state->log,"Cannot read command output.");
    else {
        int prefix=snprintf(state->log,sizeof(state->log),"Command exit %d\n",code);
        size_t offset=(size_t)prefix;
        size_t got=fread(state->log+offset,1,sizeof(state->log)-offset-1,capture);
        state->log[offset+got]=0;
        if(fgetc(capture)!=EOF) {
            const char marker[]="\n[Output preview truncated at 64 KiB.]";
            memcpy(state->log+sizeof(state->log)-sizeof(marker),marker,sizeof(marker));
        }
    }
    fclose(empty); fclose(capture); state->detail=1; state->scroll=0;
}
static const char help[]="Browse: j/k move, Enter opens, n next page, r first page, b back, q quit.\n"
    "Details: j/k scroll. : opens command entry; Escape cancels.\n"
    "Commands omit fyodor --store PATH --namespace NAME. Example: resource list\n"
    "Quotes group arguments; backslashes are literal. No shell expansion.\n"
    "Local administrator mode. Scoped context commands enforce supplied principal grants.\n"
    "ASCII-safe display escapes non-ASCII bytes. Plain mode: list, next, open N, back, :COMMAND, quit.\n"
    "Generation is synchronous; streaming and cancellation are not yet integrated.";
static void safe_lines(const char *text,size_t skip,unsigned rows,unsigned columns)
{
    const char *p=text;
    while(skip && *p) { if(*p++=='\n') --skip; }
    for(unsigned row=0;row<rows && *p;++row) {
        char line[4096]; size_t length=0;
        while(*p && *p!='\n') { if(length<sizeof(line)-1) line[length++]=*p; ++p; }
        if(*p=='\n') ++p;
        line[length]=0; fyodor_terminal_text(stdout,line,columns-1); fputs("\r\n",stdout);
    }
}
static void render(tui_state *state,const char *entry,int editing,unsigned columns,unsigned rows)
{
    style(state,FYODOR_TEXT,FYODOR_BACKGROUND);
    fputs("\033[H\033[2J",stdout);
    if(columns<45||rows<8) { fyodor_terminal_text(stdout,"Resize terminal (45x8 minimum); q quits.",columns-1); (void)fflush(stdout); return; }
    fyodor_terminal_text(stdout,"Fyodor | Resources | local administrator",columns-1); fputs("\r\n",stdout);
    style(state,FYODOR_TEXT_MUTED,FYODOR_BACKGROUND);
    fyodor_terminal_text(stdout,"j/k move Enter open n next r first b back : command t theme ? help q quit",columns-1); fputs("\r\n",stdout);
    style(state,FYODOR_TEXT,FYODOR_BACKGROUND);
    if(state->detail) safe_lines(state->log,state->scroll,rows-4,columns);
    else {
        size_t start=state->selected>rows-5?state->selected-(rows-5):0;
        for(size_t i=start;i<state->count && i-start<rows-4;++i) {
            style(state,FYODOR_TEXT,i==state->selected?FYODOR_SELECTION:FYODOR_BACKGROUND);
            fprintf(stdout,"%c %2u ",i==state->selected?'>':' ',(unsigned)(i+1));
            fyodor_terminal_text(stdout,state->items[i].title,columns-7); fputs("\r\n",stdout);
        }
        style(state,FYODOR_TEXT,FYODOR_BACKGROUND);
        if(!state->count) fputs("No resources. Import a package with the native CLI.\r\n",stdout);
    }
    fprintf(stdout,"\033[%u;1H",rows-1);
    style(state,FYODOR_TEXT,FYODOR_SURFACE_RAISED);
    if(editing) { fputc(':',stdout); fyodor_terminal_text(stdout,entry,columns-2); }
    else fyodor_terminal_text(stdout,state->detail?"b back | j/k scroll":state->log,columns-1);
    (void)fflush(stdout);
}
static int plain(tui_state *state)
{
    char line[LINE_CAPACITY]; fputs("Fyodor plain terminal | local administrator\nType help for commands.\n",stdout);
    while(fgets(line,sizeof(line),stdin)) {
        size_t length=strlen(line);
        if(length && line[length-1]!='\n' && !feof(stdin)) {
            int c; while((c=fgetc(stdin))!=EOF && c!='\n') { }
            fputs("Command too long; discarded.\n",stdout); continue;
        }
        line[strcspn(line,"\r\n")]=0;
        if(strcmp(line,"quit")==0||strcmp(line,"q")==0) break;
        if(strcmp(line,"help")==0) { safe_lines(help,0,100,300); }
        else if(line[0]==':') { command(state,line+1); safe_lines(state->log,0,65536,300); }
        else if(strncmp(line,"open ",5)==0) {
            char *end=NULL; unsigned long index=strtoul(line+5,&end,10);
            if(*end || index==0 || index>state->count) fputs("Invalid resource number.\n",stdout);
            else { state->selected=(size_t)index-1; inspect(state); safe_lines(state->log,0,65536,300); }
        } else if(strcmp(line,"list")==0||strcmp(line,"next")==0||strcmp(line,"back")==0) {
            if(strcmp(line,"back")!=0) page(state,strcmp(line,"next")==0);
            for(size_t i=0;i<state->count;++i) { fprintf(stdout,"%u ",(unsigned)(i+1)); fyodor_terminal_text(stdout,state->items[i].title,4096); fputc('\n',stdout); }
        } else if(*line) fputs("Unknown command. Type help.\n",stdout);
        if(fflush(stdout)!=0) return 1;
    }
    return ferror(stdin)?1:0;
}
int fyodor_tui_run(int argc,char **argv)
{
    if(argc==2 && strcmp(argv[1],"--licenses")==0) { fputs(fyodor_theme_license(),stdout); return 0; }
    if(argc==2 && strcmp(argv[1],"--themes")==0) {
        for(size_t i=0;i<fyodor_theme_count();++i) fprintf(stdout,"%s\n",fyodor_theme_at(i)->id);
        return 0;
    }
    const char *theme_id=getenv("FYODOR_THEME"),*color_id="auto";
    int force_plain=0;
    if(argc<5||strcmp(argv[1],"--store")!=0||strcmp(argv[3],"--namespace")!=0) goto usage;
    for(int i=5;i<argc;++i) {
        if(strcmp(argv[i],"--plain")==0) force_plain=1;
        else if(strcmp(argv[i],"--theme")==0 && i+1<argc) theme_id=argv[++i];
        else if(strcmp(argv[i],"--color")==0 && i+1<argc) color_id=argv[++i];
        else goto usage;
    }
    const fyodor_theme *theme=fyodor_theme_find(theme_id?theme_id:"fyodor");
    if(!theme) goto usage;
    fyodor_color_mode colors=FYODOR_COLOR_16;
    if(strcmp(color_id,"auto")==0) {
        const char *term=getenv("TERM"),*capability=getenv("COLORTERM");
        if(getenv("NO_COLOR")) colors=FYODOR_COLOR_NONE;
        else if(capability && (strcmp(capability,"truecolor")==0||strcmp(capability,"24bit")==0)) colors=FYODOR_COLOR_TRUE;
        else if(term && strstr(term,"256color")) colors=FYODOR_COLOR_256;
    } else if(strcmp(color_id,"truecolor")==0) colors=FYODOR_COLOR_TRUE;
    else if(strcmp(color_id,"256")==0) colors=FYODOR_COLOR_256;
    else if(strcmp(color_id,"16")==0) colors=FYODOR_COLOR_16;
    else if(strcmp(color_id,"none")==0) colors=FYODOR_COLOR_NONE;
    else goto usage;
    tui_state *state=calloc(1,sizeof(*state)); if(!state) return 1;
    state->theme=theme; state->colors=colors;
    state->path=argv[2]; state->name_space=argv[4];
    fyodor_store_result result=fyodor_store_open(state->path,&state->store);
    if(result!=FYODOR_STORE_OK) { fprintf(stderr,"Cannot open workspace (%d).\n",(int)result); free(state); return 1; }
    page(state,0); int code=0;
    if(force_plain||!fyodor_terminal_begin()) code=plain(state);
    else {
        char entry[LINE_CAPACITY]={0}; size_t length=0; int editing=0,dirty=1,overflow=0; unsigned previous_columns=0,previous_rows=0;
        for(;;) {
            unsigned columns,rows; fyodor_terminal_size(&columns,&rows);
            if(dirty||columns!=previous_columns||rows!=previous_rows) { render(state,entry,editing,columns,rows); dirty=0; previous_columns=columns; previous_rows=rows; }
            int key=fyodor_terminal_key();
            if(key==FYODOR_KEY_END||key==3||key==4) break;
            if(key==FYODOR_KEY_IDLE) continue;
            dirty=1;
            if(editing) {
                if(key==27) editing=0;
                else if(key==13||key==10) { if(overflow) { strcpy(state->log,"Command too long; discarded."); state->detail=1; state->scroll=0; } else command(state,entry); editing=0; }
                else if(key==8||key==127) { if(length) { do { --length; } while(length && ((unsigned char)entry[length]&0xc0)==0x80); entry[length]=0; } }
                else if(key>=32&&key<=255) { if(length<sizeof(entry)-1) { entry[length++]=(char)key; entry[length]=0; } else overflow=1; }
                continue;
            }
            if(key=='q') break;
            if(key==':') { editing=1; length=0; entry[0]=0; overflow=0; }
            else if(key=='?') { strcpy(state->log,help); state->detail=1; state->scroll=0; }
            else if(key=='b') state->detail=0;
            else if(key=='r') page(state,0);
            else if(key=='n') page(state,1);
            else if(key=='t') {
                size_t i=0; while(fyodor_theme_at(i)!=state->theme) ++i;
                state->theme=fyodor_theme_at((i+1)%fyodor_theme_count());
                (void)snprintf(state->log,sizeof(state->log),"Theme: %s",state->theme->name);
                state->detail=0;
            }
            else if(key=='j'||key==FYODOR_KEY_DOWN) {
                if(state->detail) { size_t lines=0; for(const char *p=state->log;*p;++p) if(*p=='\n') ++lines; if(state->scroll<lines) ++state->scroll; }
                else if(state->selected+1<state->count) ++state->selected;
            } else if(key=='k'||key==FYODOR_KEY_UP) { if(state->detail) { if(state->scroll) --state->scroll; } else if(state->selected) --state->selected; }
            else if(key==13||key==10) inspect(state);
        }
        fyodor_terminal_end();
    }
    fyodor_resource_summaries_free(state->items,state->count); fyodor_store_close(state->store); free(state); return code;
usage:
    fputs("Usage: fyodor-tui --store PATH --namespace NAME [--plain] [--theme ID] [--color auto|truecolor|256|16|none]\n       fyodor-tui --themes\n",stderr); return 2;
}
