/* Original C diagnostic: query this process only; never change reservations. */
#define COBJMACROS
#include <windows.h>
#include <dxgi1_4.h>
#include <stdio.h>

int nya_diag_memory(const char *phase)
{
    IDXGIFactory1 *factory=NULL;
    HRESULT status=CreateDXGIFactory1(&IID_IDXGIFactory1,(void **)&factory);
    if(FAILED(status))return -1;
    int bad=0;unsigned adapters=0;
    for(UINT ordinal=0;;++ordinal) {
        IDXGIAdapter1 *adapter=NULL;IDXGIAdapter3 *budget=NULL;
        status=IDXGIFactory1_EnumAdapters1(factory,ordinal,&adapter);
        if(status==DXGI_ERROR_NOT_FOUND)break;
        if(FAILED(status)){bad=1;break;}
        DXGI_ADAPTER_DESC1 desc;
        status=IDXGIAdapter1_GetDesc1(adapter,&desc);
        if(SUCCEEDED(status))status=IDXGIAdapter1_QueryInterface(adapter,&IID_IDXGIAdapter3,(void **)&budget);
        if(FAILED(status)){bad=1;IDXGIAdapter1_Release(adapter);continue;}
        ++adapters;
        for(unsigned segment=0;segment<2;++segment) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info={0};
            status=IDXGIAdapter3_QueryVideoMemoryInfo(budget,0,
                segment?DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL:DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info);
            if(FAILED(status)){bad=1;continue;}
            fprintf(stderr,"GPU budget: phase=%s pid=%lu vendor=%u device=%u luid=0x%08lx_0x%08lx segment=%s budget=%llu usage=%llu available_reservation=%llu reservation=%llu\n",
                phase,(unsigned long)GetCurrentProcessId(),desc.VendorId,desc.DeviceId,
                (unsigned long)desc.AdapterLuid.HighPart,(unsigned long)desc.AdapterLuid.LowPart,
                segment?"nonlocal":"local",(unsigned long long)info.Budget,
                (unsigned long long)info.CurrentUsage,(unsigned long long)info.AvailableForReservation,
                (unsigned long long)info.CurrentReservation);
        }
        IDXGIAdapter3_Release(budget);IDXGIAdapter1_Release(adapter);
    }
    IDXGIFactory1_Release(factory);
    return bad || !adapters ? -1 : 0;
}
#ifdef NYA_GPU_BUDGET_MAIN
int main(void){return nya_diag_memory("standalone")?1:0;}
#endif
