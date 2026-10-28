#include "../../../CUDA/q4_byte_pack.h"
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"failure line %d\n",__LINE__); return 1; } } while(0)
int main(void) {
    unsigned char source[144]={0}, output[162];
    CHECK(nya_q4_byte_size(144)==160);
    CHECK(!nya_q4_byte_size(0) && !nya_q4_byte_size(143) && !nya_q4_byte_size(145));
    CHECK(!nya_q4_byte_size(SIZE_MAX-SIZE_MAX%144));
    memset(output,0xA5,sizeof(output));
    CHECK(nya_q4_byte_pack(output+1,159,source,144)==-1);
    CHECK(nya_q4_byte_pack(NULL,160,source,144)==-1);
    CHECK(nya_q4_byte_pack(output+1,160,NULL,144)==-1);
    for (size_t i=0;i<sizeof(output);++i) CHECK(output[i]==0xA5);
    unsigned checked=0;
    for (unsigned group=0;group<8;++group) for (unsigned scale=0;scale<64;++scale) for (unsigned minimum=0;minimum<64;++minimum) {
        memset(source,0,sizeof(source)); unsigned char *s=source+4;
        if (group<4) { s[group]=(unsigned char)scale; s[group+4]=(unsigned char)minimum; }
        else { s[group+4]=(unsigned char)((scale&15)|((minimum&15)<<4)); s[group-4]=(unsigned char)((scale>>4)<<6); s[group]=(unsigned char)((minimum>>4)<<6); }
        for (unsigned i=0;i<4;++i) source[i]=(unsigned char)(group*31+scale+minimum+i);
        for (unsigned i=16;i<144;++i) source[i]=(unsigned char)(i*13+scale+minimum);
        memset(output,0xA5,sizeof(output));
        CHECK(!nya_q4_byte_pack(output+1,160,source,144));
        unsigned char *out=output+1;
        CHECK(output[0]==0xA5 && output[161]==0xA5);
        CHECK(!memcmp(out,source,4) && !memcmp(out+32,source+16,128));
        for (unsigned j=0;j<8;++j) { CHECK(out[4+j]==(j==group?scale:0)); CHECK(out[12+j]==(j==group?minimum:0)); }
        for (unsigned j=20;j<32;++j) CHECK(out[j]==0);
        ++checked;
    }
    unsigned char two[288], packed[322];
    memcpy(two,source,144); memcpy(two+144,source,144);
    memset(packed,0xA5,sizeof(packed));
    CHECK(nya_q4_byte_pack(packed+1,319,two,sizeof(two))==-1);
    for (unsigned i=0;i<sizeof(packed);++i) CHECK(packed[i]==0xA5);
    CHECK(!nya_q4_byte_pack(packed+1,320,two,sizeof(two)));
    CHECK(packed[0]==0xA5 && packed[321]==0xA5);
    CHECK(!memcmp(packed+1,output+1,160) && !memcmp(packed+161,output+1,160));
    puts("32768 independent metadata encodings, payload/half preservation, guards and size rejection passed");
    return checked==32768?0:1;
}
