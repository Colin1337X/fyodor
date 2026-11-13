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
    CHECK(s.theme==fyodor_theme_at(0));
    CHECK(!tui_theme(&s,"catppuccin-mocha"));
    const fyodor_theme *chosen=s.theme;
    CHECK(tui_theme(&s,"missing-theme")<0 && s.theme==chosen);
    tui_key(&s,'t');CHECK(s.theme!=chosen);
    CHECK(send(&s,"theme catppuccin-latte")==TUI_NONE);
    CHECK(s.theme==fyodor_theme_find("catppuccin-latte"));
    fyodor_color_mode colors=FYODOR_COLOR_NONE;
    CHECK(!fyodor_theme_parse_color("truecolor",&colors) && colors==FYODOR_COLOR_TRUE);
    CHECK(!fyodor_theme_parse_color("256",&colors) && colors==FYODOR_COLOR_256);
    CHECK(!fyodor_theme_parse_color("16",&colors) && colors==FYODOR_COLOR_16);
    CHECK(!fyodor_theme_parse_color("none",&colors) && colors==FYODOR_COLOR_NONE);
    CHECK(fyodor_theme_parse_color("invalid",&colors)<0 && colors==FYODOR_COLOR_NONE);
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
    CHECK(send(&s,"chart gradient")==TUI_NONE && s.chart==2 && s.view==2);
    CHECK(send(&s,"chart invalid")==TUI_NONE && s.chart==2);
    tui_key(&s,'g');CHECK(s.chart==3);tui_key(&s,'g');CHECK(s.chart==0);
    s.samples=3;memset(s.history,0,3*sizeof(*s.history));
    s.history[0].optional=2;s.history[0].gradient=1;
    s.history[2].optional=2;s.history[2].gradient=3;
    s.chart=2;tui_render(&s,f,80,30);
    int points=0;for(int y=14;y<24;++y) for(int x=4;x<76;++x)
        if(f->cells[y*80+x].code=='*') {++points;CHECK(x==4 || x==6);}
    CHECK(points==2);
    s.chart=3;tui_render(&s,f,80,30);
    points=0;for(int y=14;y<24;++y) for(int x=4;x<76;++x) points+=f->cells[y*80+x].code=='*';
    CHECK(!points);
    s.chart=0;
    tui_key(&s,':');
    for(int i=0;i<70;++i) tui_key(&s,0xac00);
    tui_key(&s,'X');tui_render(&s,f,48,16);
    CHECK(f->cells[14*48+2].code=='<');
    int found_end=0;
    for(int i=3;i<47;++i) if(f->cells[14*48+i].code=='X') found_end=1;
    CHECK(found_end);tui_key(&s,TUI_ESCAPE);
    for(int view=0;view<3;++view) for(int w=0;w<=250;w+=7) for(int h=0;h<=90;h+=3) {
        s.view=view;s.history[0].loss=-DBL_MAX;s.history[1].loss=DBL_MAX;
        tui_render(&s,f,w,h);CHECK(f->width>=1 && f->width<=TUI_WIDTH && f->height>=1 && f->height<=TUI_HEIGHT);
        for(int i=0;i<f->width*f->height;++i) CHECK(f->cells[i].code==0 || f->cells[i].code>=32);
    }
    s.view=1;strcpy(s.model_path,"bad\033[2J\n\xc2\x9b\xe2\x80\xae\xff name");tui_render(&s,f,80,24);
    FILE *plain=tmpfile();CHECK(plain);tui_write(plain,f,0,0);rewind(plain);int c;while((c=fgetc(plain))!=EOF) CHECK(c==10 || c>=32);fclose(plain);
    for(size_t theme=0;theme<fyodor_theme_count();++theme) {
        CHECK(!tui_theme(&s,fyodor_theme_at(theme)->id));tui_render(&s,f,80,24);
        for(int mode=FYODOR_COLOR_16;mode<=FYODOR_COLOR_TRUE;++mode) {
            FILE *styled=tmpfile();CHECK(styled);tui_write(styled,f,1,(fyodor_color_mode)mode);rewind(styled);
            char actual[128]={0},expected[80]={0};CHECK(fread(actual,1,sizeof(actual)-1,styled)>0);
            CHECK(fyodor_theme_sgr(s.theme,FYODOR_TEXT,FYODOR_BACKGROUND,(fyodor_color_mode)mode,expected,sizeof(expected)));
            CHECK(!memcmp(actual,"\033[H",3) && !strncmp(actual+3,expected,strlen(expected)));fclose(styled);
        }
    }
    s.help=1;tui_key(&s,'q');CHECK(s.quit);
    free(f);CHECK(!remove(argv[1]));puts("TUI navigation, UTF-8 editing, CSV tail/restart/pause, clipping and control sanitization passed");return 0;
}
