#include "tui.h"
#include "file.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void tui_init(tui_state *s)
{
    memset(s,0,sizeof(*s));
    snprintf(s->status,sizeof(s->status),"Ready. Press : for commands or ? for help.");
}
static int path_copy(char *out,size_t capacity,const char *text,tui_state *s)
{
    size_t n=strlen(text);
    if (!n || n>=capacity) {snprintf(s->status,sizeof(s->status),"Path must contain 1..%zu UTF-8 bytes",capacity-1);return -1;}
    memcpy(out,text,n+1);return 0;
}
static int command(tui_state *s)
{
    char *c=s->command;
    if (!strcmp(c,"quit") || !strcmp(c,"q")) s->quit=1;
    else if (!strcmp(c,"help")) s->help=1;
    else if (!strcmp(c,"overview")) s->view=0;
    else if (!strcmp(c,"model")) s->view=1;
    else if (!strcmp(c,"training")) s->view=2;
    else if (!strcmp(c,"pause")) s->paused=!s->paused;
    else if (!strcmp(c,"refresh")) return TUI_REFRESH;
    else if (!strncmp(c,"open ",5)) {
        if (!path_copy(s->model_path,sizeof(s->model_path),c+5,s)) {s->view=1;return TUI_INSPECT;}
    } else if (!strncmp(c,"metrics ",8)) {
        if (!path_copy(s->metrics_path,sizeof(s->metrics_path),c+8,s)) {s->view=2;s->paused=0;return TUI_METRICS;}
    } else snprintf(s->status,sizeof(s->status),"Unknown command. Press ? for available commands.");
    return TUI_NONE;
}
size_t tui_utf8(uint32_t c,char out[4])
{
    if (c>=0xd800 && c<=0xdfff) return 0;
    if (c<128) {out[0]=(char)c;return 1;}
    if (c<2048) {out[0]=(char)(0xc0|(c>>6));out[1]=(char)(0x80|(c&63));return 2;}
    if (c<65536) {out[0]=(char)(0xe0|(c>>12));out[1]=(char)(0x80|((c>>6)&63));out[2]=(char)(0x80|(c&63));return 3;}
    if (c<=0x10ffff) {out[0]=(char)(0xf0|(c>>18));out[1]=(char)(0x80|((c>>12)&63));out[2]=(char)(0x80|((c>>6)&63));out[3]=(char)(0x80|(c&63));return 4;}
    return 0;
}
int tui_key(tui_state *s,uint32_t key)
{
    if (key==3) {s->quit=1;return TUI_NONE;}
    if (s->editing) {
        if (key==TUI_ESCAPE) {s->editing=0;s->overflow=0;return TUI_NONE;}
        if (key==TUI_ENTER) {
            if (s->overflow) {snprintf(s->status,sizeof(s->status),"Command too long; edit or press Escape.");return TUI_NONE;}
            s->editing=0;return command(s);
        }
        if (key==TUI_BACKSPACE) {
            if (s->command_length) {
                do {--s->command_length;} while (s->command_length && ((unsigned char)s->command[s->command_length]&0xc0)==0x80);
                s->command[s->command_length]=0;
            }
            s->overflow=0;return TUI_NONE;
        }
        if (key>=32 && key<0x110000 && key!=127) {
            char bytes[4];size_t n=tui_utf8(key,bytes);
            if (s->command_length+n<sizeof(s->command)) {
                memcpy(s->command+s->command_length,bytes,n);s->command_length+=n;s->command[s->command_length]=0;
            } else s->overflow=1;
        }
        return TUI_NONE;
    }
    if (key==TUI_ESCAPE) s->help=0;
    else if (key=='?') s->help=!s->help;
    else if (key=='q') s->quit=1;
    else if (s->help) return TUI_NONE;
    else if (key==':' || key=='/') {s->editing=1;s->command_length=0;s->command[0]=0;s->overflow=0;}

    else if (key>='1' && key<='3') s->view=(int)(key-'1');
    else if (key==TUI_TAB || key==TUI_RIGHT || key=='l' || key==TUI_DOWN || key=='j') s->view=(s->view+1)%3;
    else if (key==TUI_LEFT || key=='h' || key==TUI_UP || key=='k') s->view=(s->view+2)%3;
    else if (key==' ') s->paused=!s->paused;
    else if (key=='r') return TUI_REFRESH;
    return TUI_NONE;
}
int tui_inspect(tui_state *s)
{
    s->model_valid=0;s->model_bytes=0;memset(&s->model,0,sizeof(s->model));
    FILE *f=nya_file_open_read(s->model_path);
    if (!f) {snprintf(s->status,sizeof(s->status),"Cannot open model file");return -1;}
    int failed=nya_file_regular_size(f,&s->model_bytes);fclose(f);
    if (failed || nya_format_inspect(s->model_path,s->model_bytes,&s->model)!=NYA_FORMAT_OK) {
        snprintf(s->status,sizeof(s->status),"Model container is invalid, unsupported, or exceeds inspection limits");return -1;
    }
    s->model_valid=1;snprintf(s->status,sizeof(s->status),"Container inspected. No inference context or GPU allocation created.");return 0;
}
static size_t fields(char *line,char **parts,size_t limit)
{
    size_t n=0;char *p=line;
    for (;;) {
        if (n==limit) return 0;
        parts[n++]=p;char *comma=strchr(p,',');if (!comma) break;*comma=0;p=comma+1;
    }
    return n;
}
static int index_of(char **header,size_t n,const char *name)
{
    int result=-1;
    for (size_t i=0;i<n;++i) if (!strcmp(header[i],name)) {if (result>=0) return -2;result=(int)i;}
    return result;
}
static int number(const char *text,double *out)
{
    if (!*text) return -1;
    char *end;errno=0;double n=strtod(text,&end);
    if (errno || *end || !isfinite(n)) return -1;
    *out=n;return 0;
}
static int parse_sample(char *line,const int *indexes,size_t count,tui_sample *sample)
{
    char *parts[64];if (fields(line,parts,64)!=count) return -1;
    memset(sample,0,sizeof(*sample));
    const char *step=parts[indexes[0]];char *end;
    if (*step<'0' || *step>'9') return -1;
    errno=0;unsigned long long value=strtoull(step,&end,10);
    if (errno || *end || value>UINT64_MAX) return -1;
    sample->step=(uint64_t)value;
    if (number(parts[indexes[1]],&sample->loss) || number(parts[indexes[2]],&sample->rate) || sample->rate<0) return -1;
    double *optional[]={&sample->validation,&sample->gradient,&sample->used,&sample->peak,&sample->capacity};
    for (unsigned i=0;i<5;++i) if (indexes[i+3]>=0 && *parts[indexes[i+3]]) {
        if (number(parts[indexes[i+3]],optional[i]) || (i>0 && *optional[i]<0)) return -1;
        sample->optional|=1U<<i;
    }
    if (indexes[8]>=0) snprintf(sample->compute,sizeof(sample->compute),"%s",parts[indexes[8]]);
    return 0;
}
int tui_metrics(tui_state *s)
{
    if (!*s->metrics_path || s->paused) return 0;
    FILE *f=nya_file_open_read(s->metrics_path);uint64_t size=0;
    if (!f) {s->samples=0;snprintf(s->status,sizeof(s->status),"Waiting for metrics file; r retries immediately");return -1;}
    char header[4096],*names[64];
    if (nya_file_regular_size(f,&size) || !fgets(header,sizeof(header),f) || !strchr(header,'\n')) {
        fclose(f);snprintf(s->status,sizeof(s->status),"Waiting for a complete CSV header");s->samples=0;return -1;
    }
    header[strcspn(header,"\r\n")]=0;size_t n=fields(header,names,64);
    const char *wanted[]={"step","loss","tokens_per_second","validation_loss","gradient_norm","device_used_bytes","device_peak_bytes","device_capacity_bytes","compute"};
    int indexes[9];
    for (unsigned i=0;i<9;++i) {
        indexes[i]=index_of(names,n,wanted[i]);
        if (indexes[i]==-2 || (i<3 && indexes[i]<0)) {fclose(f);s->samples=0;snprintf(s->status,sizeof(s->status),"Metrics CSV schema is missing or duplicates required columns");return -1;}
    }
    /* A bounded tail keeps refresh cost independent of total training duration.
       Negative offset from end is representable even on 32-bit-long platforms. */
    if (size>65536 && fseek(f,-65536L,SEEK_END)) {fclose(f);return -1;}
    char line[4096];
    if (size>65536) {int c;do {c=fgetc(f);} while(c!=EOF && c!='\n');}
    s->samples=0;s->malformed=0;
    while (fgets(line,sizeof(line),f)) {
        if (!strchr(line,'\n')) {
            if (!feof(f)) {int c;do {c=fgetc(f);} while(c!=EOF && c!='\n');++s->malformed;}
            continue; /* an unfinished writer row is not a metric */
        }
        line[strcspn(line,"\r\n")]=0;tui_sample sample;
        if (parse_sample(line,indexes,n,&sample)) {++s->malformed;continue;}
        if (s->samples==TUI_HISTORY) {memmove(s->history,s->history+1,(TUI_HISTORY-1)*sizeof(sample));--s->samples;}
        s->history[s->samples++]=sample;
    }
    int failed=ferror(f);fclose(f);
    snprintf(s->status,sizeof(s->status),failed?"Metrics read failed":s->malformed?"Metrics updated; malformed rows skipped (see training panel)":"Metrics refreshed from completed CSV rows. Space freezes this view.");
    return failed?-1:0;
}
