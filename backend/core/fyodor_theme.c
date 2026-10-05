#include "fyodor_theme.h"
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "fyodor_themes.inc"
const char *fyodor_theme_license(void) { return theme_license; }
size_t fyodor_theme_count(void) { return sizeof(builtin_themes)/sizeof(builtin_themes[0]); }
const fyodor_theme *fyodor_theme_at(size_t index) { return index<fyodor_theme_count()?&builtin_themes[index]:NULL; }
const fyodor_theme *fyodor_theme_find(const char *id)
{
    if(!id) return NULL;
    for(size_t i=0;i<fyodor_theme_count();++i) if(strcmp(id,builtin_themes[i].id)==0) return &builtin_themes[i];
    return NULL;
}
static unsigned distance(uint32_t a,uint32_t b)
{
    int r=(int)((a>>16)&255)-(int)((b>>16)&255);
    int g=(int)((a>>8)&255)-(int)((b>>8)&255);
    int blue=(int)(a&255)-(int)(b&255);
    return (unsigned)(r*r+g*g+blue*blue);
}
static unsigned nearest(uint32_t color,int extended)
{
    static const uint32_t ansi[]={0x000000,0x800000,0x008000,0x808000,0x000080,0x800080,0x008080,0xc0c0c0,
        0x808080,0xff0000,0x00ff00,0xffff00,0x0000ff,0xff00ff,0x00ffff,0xffffff};
    static const unsigned levels[]={0,95,135,175,215,255};
    unsigned best=0,error=UINT_MAX;
    for(unsigned i=extended?16u:0u;i<(extended?256u:16u);++i) {
        uint32_t value;
        if(i<16) value=ansi[i];
        else if(i<232) { unsigned n=i-16; value=(levels[n/36]<<16)|(levels[(n/6)%6]<<8)|levels[n%6]; }
        else { unsigned n=8+10*(i-232); value=(n<<16)|(n<<8)|n; }
        unsigned d=distance(color,value);
        if(d<error) { error=d; best=i; }
    }
    return best;
}
size_t fyodor_theme_sgr(const fyodor_theme *theme,fyodor_theme_role fg,fyodor_theme_role bg,
                       fyodor_color_mode mode,char *output,size_t capacity)
{
    if(!theme||!output||capacity==0||(unsigned)fg>=FYODOR_THEME_ROLE_COUNT||(unsigned)bg>=FYODOR_THEME_ROLE_COUNT||
       (unsigned)mode>FYODOR_COLOR_TRUE) return 0;
    char text[80]; int length;
    uint32_t f=theme->colors[fg],b=theme->colors[bg];
    if(mode==FYODOR_COLOR_NONE) length=snprintf(text,sizeof(text),"\033[0m%s",bg==FYODOR_SELECTION?"\033[7m":"");
    else if(mode==FYODOR_COLOR_TRUE) length=snprintf(text,sizeof(text),"\033[0;38;2;%u;%u;%u;48;2;%u;%u;%um",
        (unsigned)(f>>16),(unsigned)((f>>8)&255),(unsigned)(f&255),(unsigned)(b>>16),(unsigned)((b>>8)&255),(unsigned)(b&255));
    else {
        unsigned fi=nearest(f,mode==FYODOR_COLOR_256),bi=nearest(b,mode==FYODOR_COLOR_256);
        if(fi==bi) fi=distance(b,0)>distance(b,0xffffff)?(mode==FYODOR_COLOR_256?16u:0u):(mode==FYODOR_COLOR_256?231u:15u);
        if(mode==FYODOR_COLOR_256) length=snprintf(text,sizeof(text),"\033[0;38;5;%u;48;5;%um",fi,bi);
        else length=snprintf(text,sizeof(text),"\033[0;%u;%um",fi<8?30+fi:90+fi-8,bi<8?40+bi:100+bi-8);
    }
    if(length<0||(size_t)length>=capacity) return 0;
    memcpy(output,text,(size_t)length+1); return (size_t)length;
}
