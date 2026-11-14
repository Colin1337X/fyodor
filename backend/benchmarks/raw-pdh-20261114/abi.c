#include <windows.h>
#include <pdh.h>
#include <stddef.h>
#include <stdio.h>
_Static_assert(sizeof(PDH_RAW_COUNTER)==40,"raw ABI");
_Static_assert(sizeof(PDH_RAW_COUNTER_ITEM_W)==48,"item ABI");
_Static_assert(offsetof(PDH_RAW_COUNTER,FirstValue)==16,"first ABI");
int main(void) { puts("PDH raw ABI: size 40, item 48, first offset 16"); return 0; }
