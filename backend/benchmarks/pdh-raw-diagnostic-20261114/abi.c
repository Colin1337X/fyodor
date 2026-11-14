#include <windows.h>
#include <pdh.h>
#include <stddef.h>
#include <stdio.h>
int main(void) {
 printf("%zu %zu %zu %zu %zu %zu %zu\n", sizeof(PDH_RAW_COUNTER),
 offsetof(PDH_RAW_COUNTER, TimeStamp), offsetof(PDH_RAW_COUNTER, FirstValue),
 offsetof(PDH_RAW_COUNTER, SecondValue), offsetof(PDH_RAW_COUNTER, MultiCount),
 sizeof(PDH_RAW_COUNTER_ITEM_W), offsetof(PDH_RAW_COUNTER_ITEM_W, RawValue));
 return 0;
}
