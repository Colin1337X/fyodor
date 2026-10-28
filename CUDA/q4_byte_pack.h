/* Original Fyodor C-only lossless Q4_K metadata expansion. */
#ifndef NYA_Q4_BYTE_PACK_H
#define NYA_Q4_BYTE_PACK_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
static size_t nya_q4_byte_size(size_t bytes)
{
    if (!bytes || bytes % 144 || bytes / 144 > SIZE_MAX / 160) return 0;
    return bytes / 144 * 160;
}
static int nya_q4_byte_pack(void *destination, size_t capacity, const void *source, size_t bytes)
{
    size_t required = nya_q4_byte_size(bytes);
    if (!destination || !source || !required || capacity < required) return -1;
    unsigned char *out = destination;
    const unsigned char *in = source;
    for (size_t offset=0; offset<bytes; offset+=144, out+=160) {
        const unsigned char *b=in+offset, *s=b+4;
        memcpy(out,b,4);
        for (unsigned j=0;j<8;++j) {
            out[4+j]=(unsigned char)(j<4 ? s[j]&63U : (s[j+4]&15U)|((s[j-4]>>6)<<4));
            out[12+j]=(unsigned char)(j<4 ? s[j+4]&63U : ((unsigned)s[j+4]>>4)|(((unsigned)s[j]>>6)<<4));
        }
        memset(out+20,0,12); memcpy(out+32,b+16,128);
    }
    return 0;
}
#endif
