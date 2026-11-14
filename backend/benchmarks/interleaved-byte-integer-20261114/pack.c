#include "q4_byte_pack.h"
__declspec(dllexport) size_t nya_test_pack_size(size_t bytes)
{ return nya_q4_byte_size(bytes); }
__declspec(dllexport) int nya_test_pack(void *out,size_t capacity,const void *in,size_t bytes)
{ return nya_q4_byte_pack(out,capacity,in,bytes); }
