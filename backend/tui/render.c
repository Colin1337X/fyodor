#include "tui.h"
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>

static void cell(tui_frame *f,int x,int y,uint32_t c,unsigned char style)
{
    if(x>=0 && x<f->width && y>=0 && y<f->height) f->cells[y*f->width+x]=(tui_cell){c,style};
}
/* Decode untrusted metadata without ever emitting terminal control sequences. */
static uint32_t scalar(const unsigned char **cursor)
{
    const unsigned char *p=*cursor;uint32_t c=*p++;unsigned need=0;uint32_t minimum=0;
    if(c>=0xc2 && c<=0xdf) {c&=31;need=1;minimum=128;}
    else if(c>=0xe0 && c<=0xef) {c&=15;need=2;minimum=2048;}
    else if(c>=0xf0 && c<=0xf4) {c&=7;need=3;minimum=65536;}
    else if(c>=128) c='?';
    for(unsigned i=0;i<need;++i) {
        if((*p&0xc0)!=0x80) {*cursor=p;return '?';}
        c=(c<<6)|(*p++&63);
    }
    *cursor=p;
    if(c<minimum || c>0x10ffff || (c>=0xd800 && c<=0xdfff)) return '?';
    if(c<32 || (c>=127 && c<=159) || (c>=0x200b && c<=0x200f) || (c>=0x2028 && c<=0x202e) || (c>=0x2060 && c<=0x206f)) return '?';
    return c;
}
static int wide(uint32_t c)
{
    return (c>=0x1100 && c<=0x115f) || (c>=0x2e80 && c<=0xa4cf) || (c>=0xac00 && c<=0xd7a3) || (c>=0xf900 && c<=0xfaff) || (c>=0xfe10 && c<=0xfe6f) || (c>=0xff01 && c<=0xff60) || (c>=0x1f300 && c<=0x1faff) || (c>=0x20000 && c<=0x3ffff);
}
static void text(tui_frame *f,int x,int y,int limit,unsigned char style,const char *s)
{
    const unsigned char *p=(const unsigned char *)s;int end=x+limit;
    if(end>f->width) end=f->width;
    while(*p && x<end) {
        uint32_t c=scalar(&p);
        /* Combining sequences are rendered visibly, avoiding ambiguous cell ownership. */
        if((c>=0x300 && c<=0x36f) || (c>=0xfe00 && c<=0xfe0f)) c='?';
        int w=wide(c)?2:1;if(x+w>end) break;
        cell(f,x++,y,c,style);if(w==2) cell(f,x++,y,0,style);
    }
}
static void line(tui_frame *f,int x,int y,int limit,unsigned char style,const char *format,...)
{
    char buffer[2048];va_list args;va_start(args,format);vsnprintf(buffer,sizeof(buffer),format,args);va_end(args);text(f,x,y,limit,style,buffer);
}
static void box(tui_frame *f,int x,int y,int w,int h,const char *title)
{
    if(w<2 || h<2) return;
    for(int i=0;i<w;++i) {cell(f,x+i,y,'-',1);cell(f,x+i,y+h-1,'-',1);}
    for(int i=0;i<h;++i) {cell(f,x,y+i,'|',1);cell(f,x+w-1,y+i,'|',1);}
    cell(f,x,y,'+',1);cell(f,x+w-1,y,'+',1);cell(f,x,y+h-1,'+',1);cell(f,x+w-1,y+h-1,'+',1);
    text(f,x+2,y,w-4,2,title);
}
static void training(const tui_state *s,tui_frame *f,int x,int y,int w,int h)
{
    box(f,x,y,w,h,s->paused?" TRAINING / PAUSED ":" TRAINING / CSV ");
    text(f,x+2,y+1,w-4,3,*s->metrics_path?s->metrics_path:"Use :metrics PATH to follow fyodor-train --metrics output");
    if(!s->samples) {text(f,x+2,y+3,w-4,0,"Waiting for completed training steps.");return;}
    const tui_sample *last=&s->history[s->samples-1];
    line(f,x+2,y+3,w-4,2,"Step %" PRIu64 "    Loss %.6g    %.2f tokens/s",last->step,last->loss,last->rate);
    line(f,x+2,y+4,w-4,0,"Compute: %s   History: %zu   Rejected rows: %zu",*last->compute?last->compute:"unspecified",s->samples,s->malformed);
    if(h>6 && (last->optional&1U)) line(f,x+2,y+5,w-4,0,"Validation loss %.6g",last->validation);
    if(h>7 && (last->optional&2U)) line(f,x+2,y+6,w-4,0,"Gradient norm %.6g",last->gradient);
    if(h>8 && (last->optional&4U)) line(f,x+2,y+7,w-4,0,"Device used %.1f MiB",last->used/1048576.0);
    if(h<14) return;
    int gx=x+3,gy=y+10,gw=w-6,gh=h-12;
    size_t start=s->samples>(size_t)gw?s->samples-(size_t)gw:0;
    double low=s->history[start].loss,high=low;
    for(size_t i=start;i<s->samples;++i) {double v=s->history[i].loss;if(v<low) low=v;if(v>high) high=v;}
    line(f,x+2,y+9,w-4,3,"Loss / latest steps   %.5g .. %.5g",low,high);
    /* Scale before subtraction to avoid overflow for finite extreme input. */
    double scale=fmax(fabs(low),fabs(high));if(scale<1) scale=1;
    double range=high/scale-low/scale;
    for(size_t i=start;i<s->samples;++i) {
        double fraction=range>0?(s->history[i].loss/scale-low/scale)/range:0.5;
        int row=(int)(fraction*(double)(gh-1)+0.5);
        cell(f,gx+(int)(i-start),gy+gh-1-row,'*',2);
    }
}
void tui_render(const tui_state *s,tui_frame *f,int width,int height)
{
    f->width=width<1?1:width>TUI_WIDTH?TUI_WIDTH:width;
    f->height=height<1?1:height>TUI_HEIGHT?TUI_HEIGHT:height;
    for(int i=0;i<f->width*f->height;++i) f->cells[i]=(tui_cell){' ',0};
    int w=f->width,h=f->height;
    if(w<48 || h<16) {text(f,0,0,w,2,"FYODOR / enlarge terminal to 48 x 16");text(f,0,2,w,0,"q quit   ? help");return;}
    text(f,2,0,w-4,2,"FYODOR   /   native workbench");
    const char *tabs[]={"1 Overview","2 Model","3 Training"};
    for(int i=0;i<3;++i) text(f,2+i*15,2,14,(unsigned char)(s->view==i?2:3),tabs[i]);
    if(s->help) {
        box(f,1,4,w-2,h-8," KEYS & COMMANDS ");
        const char *help[]={"1 / 2 / 3, Tab, arrows, h j k l   Switch view","? help   Esc back   q / Ctrl-C quit","Space freeze metrics   r refresh   : command","open PATH       Inspect a model container","metrics PATH    Follow training CSV (no quotes needed)","overview / model / training / pause / refresh / quit","Inspection does not load a model or start inference."};
        for(int i=0;i<7 && i<h-10;++i) text(f,3,6+i,w-6,0,help[i]);
    } else if(s->view==2) training(s,f,1,4,w-2,h-8);
    else if(s->view==1) {
        box(f,1,4,w-2,h-8," MODEL / CONTAINER INSPECTION ");
        text(f,3,6,w-6,3,*s->model_path?s->model_path:"Use :open PATH or --model PATH");
        if(s->model_valid) {
            line(f,3,8,w-6,2,"%s / %s",nya_model_format_name(s->model.format),s->model.architecture);
            line(f,3,9,w-6,0,"%.2f GiB   %" PRIu64 " tensors   %" PRIu64 " metadata entries",(double)s->model_bytes/1073741824.0,s->model.tensor_count,s->model.metadata_count);
            text(f,3,10,w-6,0,s->model.producer);
        } else text(f,3,8,w-6,0,"No validated model container.");
    } else {
        box(f,1,4,w-2,h-8," WORKSPACE ");
        text(f,3,6,w-6,2,"Inspect models. Follow real training metrics.");
        text(f,3,8,w-6,0,"Model     :open PATH");text(f,3,9,w-6,0,"Training  :metrics PATH");
        line(f,3,11,w-6,3,"Compiled: CPU%s%s%s%s",s->compiled&1U?" CUDA":"",s->compiled&2U?" Vulkan":"",s->compiled&4U?" ROCm":"",s->compiled&8U?" MLX":"");
        if(h>19) text(f,3,13,w-6,3,"Compiled support does not imply an available device.");
    }
    text(f,2,h-3,w-4,3,s->status);
    if(s->editing) {text(f,1,h-2,1,2,":");text(f,2,h-2,w-3,0,s->command);}
    else text(f,2,h-1,w-4,1,"1-3 views  : command  ? help  Space pause  r refresh  q quit");
}
void tui_write(FILE *file,const tui_frame *f,int ansi,int color)
{
    static const char *styles[]={"\033[0m","\033[36m","\033[1;32m","\033[90m"};unsigned current=99;
    if(ansi) fputs("\033[H",file);
    for(int y=0;y<f->height;++y) {
        for(int x=0;x<f->width;++x) {
            tui_cell c=f->cells[y*f->width+x];if(!c.code) continue;
            if(ansi && color && c.style!=current) {fputs(styles[c.style<4?c.style:0],file);current=c.style;}
            char bytes[4];size_t n=tui_utf8(c.code,bytes);fwrite(bytes,1,n,file);
        }
        if(y+1<f->height) fputs(ansi?"\r\n":"\n",file);
    }
    if(ansi) fputs("\033[0m",file);else fputc('\n',file);
    fflush(file);
}
