#include "../tui/tui.h"
#include <float.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do {if(!(x)) {fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;}} while(0)
static int send(tui_state *s,const char *command)
{
    tui_key(s,':');while(*command) tui_key(s,(unsigned char)*command++);return tui_key(s,TUI_ENTER);
}
static int write_csv(const char *path,const char *contents)
{
    FILE *f=fopen(path,"wb");if(!f) return -1;int failed=fputs(contents,f)<0;return fclose(f) || failed?-1:0;
}
int main(int argc,char **argv)
{
    CHECK(argc==2);tui_state s;tui_init(&s);
    CHECK(s.view==0);tui_key(&s,TUI_LEFT);CHECK(s.view==2);tui_key(&s,TUI_TAB);CHECK(s.view==0);
    tui_key(&s,'?');tui_key(&s,'2');CHECK(s.help && s.view==0);tui_key(&s,TUI_ESCAPE);tui_key(&s,'2');CHECK(!s.help && s.view==1);
    CHECK(send(&s,"open model with spaces.gguf")==TUI_INSPECT);CHECK(!strcmp(s.model_path,"model with spaces.gguf"));
    CHECK(send(&s,"metrics metrics with spaces.csv")==TUI_METRICS);CHECK(s.view==2 && !s.paused);
    tui_key(&s,':');tui_key(&s,0x1f600);CHECK(s.command_length==4);tui_key(&s,TUI_BACKSPACE);CHECK(!s.command_length);
    for(int i=0;i<1200;++i) {tui_key(&s,'x');} CHECK(s.overflow);tui_key(&s,TUI_ENTER);CHECK(s.editing);tui_key(&s,TUI_ESCAPE);CHECK(!s.editing);
    snprintf(s.metrics_path,sizeof(s.metrics_path),"%s",argv[1]);
    CHECK(!write_csv(argv[1],"step,loss,tokens_per_second,validation_loss,compute\n1,4,12,,cpu\n2,3,13,2.5,cuda\n3,nan,14,,cpu\n4,2,"));
    CHECK(!tui_metrics(&s));CHECK(s.samples==2 && s.malformed==1 && s.history[1].optional==1 && s.history[1].validation==2.5);
    CHECK(!tui_metrics(&s) && s.samples==2);
    s.paused=1;CHECK(!write_csv(argv[1],"step,loss,tokens_per_second\n7,2,20\n"));CHECK(!tui_metrics(&s) && s.history[1].step==2);
    s.paused=0;CHECK(!tui_metrics(&s) && s.samples==1 && s.history[0].step==7);
    CHECK(!write_csv(argv[1],"step,loss,loss,tokens_per_second\n"));CHECK(tui_metrics(&s)<0 && !s.samples);
    FILE *csv=fopen(argv[1],"wb");CHECK(csv);fputs("step,loss,tokens_per_second\n",csv);
    for(int i=0;i<10000;++i) fprintf(csv,"%d,%.3f,20\n",i,1.0+(double)i/10000.0);
    CHECK(!fclose(csv));CHECK(!tui_metrics(&s) && s.samples==TUI_HISTORY && s.history[TUI_HISTORY-1].step==9999);
    tui_frame *f=malloc(sizeof(*f));CHECK(f);
    for(int view=0;view<3;++view) for(int w=0;w<=250;w+=7) for(int h=0;h<=90;h+=3) {
        s.view=view;s.history[0].loss=-DBL_MAX;s.history[1].loss=DBL_MAX;
        tui_render(&s,f,w,h);CHECK(f->width>=1 && f->width<=TUI_WIDTH && f->height>=1 && f->height<=TUI_HEIGHT);
        for(int i=0;i<f->width*f->height;++i) CHECK(f->cells[i].code==0 || f->cells[i].code>=32);
    }
    s.view=1;strcpy(s.model_path,"bad\033[2J\n\xc2\x9b\xe2\x80\xae\xff name");tui_render(&s,f,80,24);
    FILE *plain=tmpfile();CHECK(plain);tui_write(plain,f,0,0);rewind(plain);int c;while((c=fgetc(plain))!=EOF) CHECK(c==10 || c>=32);fclose(plain);
    s.help=1;tui_key(&s,'q');CHECK(s.quit);
    free(f);CHECK(!remove(argv[1]));puts("TUI navigation, UTF-8 editing, CSV tail/restart/pause, clipping and control sanitization passed");return 0;
}
