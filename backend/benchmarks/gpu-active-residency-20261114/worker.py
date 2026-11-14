"""Actively touch the retained working set; no attribution or timing policy."""
import ctypes as C
import json
import sys
import time

P=C.c_void_p;Q=C.c_uint64;Z=C.c_size_t;I=C.c_int
cu=C.WinDLL('nvcuda.dll')
signatures={'cuInit':[C.c_uint], 'cuDeviceGet':[C.POINTER(I),I],
 'cuDevicePrimaryCtxRetain':[C.POINTER(P),I], 'cuDevicePrimaryCtxRelease_v2':[I],
 'cuCtxSetCurrent':[P], 'cuCtxSynchronize':[],
 'cuMemAlloc_v2':[C.POINTER(Q),Z], 'cuMemFree_v2':[Q],
 'cuMemAllocHost_v2':[C.POINTER(P),Z], 'cuMemFreeHost':[P],
 'cuMemcpyHtoD_v2':[Q,P,Z], 'cuMemcpyDtoH_v2':[P,Q,Z],
 'cuMemcpyDtoD_v2':[Q,Q,Z], 'cuMemGetInfo_v2':[C.POINTER(Z),C.POINTER(Z)],
 'cuMemsetD8_v2':[Q,C.c_ubyte,Z]}
for name,args in signatures.items():
    f=getattr(cu,name);f.argtypes=args;f.restype=I

def call(name,*args):
    status=getattr(cu,name)(*args)
    if status:raise RuntimeError(f'{name}: CUDA status {status}')

size=8*1024*1024
pageable=C.create_string_buffer(size)
pinned=P();device=I();context=P();allocations=[]
call('cuInit',0);call('cuDeviceGet',C.byref(device),0)
call('cuDevicePrimaryCtxRetain',C.byref(context),device)
try:
    call('cuCtxSetCurrent',context)
    for _ in range(2):
        address=Q();call('cuMemAlloc_v2',C.byref(address),size);allocations.append(address)
    # Keep 13 GiB initialized and allocated throughout the transfer controls.
    # Refuse oversubscription; leave at least 1 GiB plus the pinned-copy buffers.
    resident_bytes=13*1024*1024*1024
    free=Z();total=Z();call('cuMemGetInfo_v2',C.byref(free),C.byref(total))
    if free.value < resident_bytes + 1024*1024*1024:
        raise RuntimeError('Insufficient free device memory for residency control and headroom')
    chunk=256*1024*1024
    for _ in range(resident_bytes//chunk):
        address=Q();call('cuMemAlloc_v2',C.byref(address),chunk);allocations.append(address)
        call('cuMemsetD8_v2',address,0,chunk)
    call('cuCtxSynchronize')
    print(json.dumps(dict(resident_bytes=resident_bytes,free_before=free.value,total=total.value)),file=sys.stderr,flush=True)
    call('cuMemAllocHost_v2',C.byref(pinned),size)
    C.memset(pinned,0,size)
    call('cuMemcpyHtoD_v2',allocations[0],pageable,size)
    call('cuCtxSynchronize')
    def fill_working_set():
        for address in allocations[2:]:
            call('cuMemsetD8_v2',address,0,chunk)
    def copy_working_set():
        for index in range(2,len(allocations),2):
            call('cuMemcpyDtoD_v2',allocations[index+1],allocations[index],chunk)
    # Fill writes every retained byte. Copy reads one half and writes the other,
    # touching the same complete address range without allocating during a phase.
    phases=[('idle-before',None,0,0),
        ('resident-fill',fill_working_set,resident_bytes,resident_bytes),
        ('idle-after-fill',None,0,0),
        ('resident-copy',copy_working_set,resident_bytes//2,resident_bytes),
        ('idle-after-copy',None,0,0),
        ('pageable-download',lambda:call('cuMemcpyDtoH_v2',pageable,allocations[0],size),size,size),
        ('idle-after',None,0,0)]
    for name,operation,operation_bytes,touched_bytes in phases:
        call('cuCtxSynchronize')
        start=time.perf_counter();iterations=0
        while time.perf_counter()-start<5:
            if operation is None:time.sleep(.1)
            else:operation();call('cuCtxSynchronize');iterations+=1
        end=time.perf_counter()
        print(json.dumps(dict(phase=name,start=start,end=end,iterations=iterations,
            bytes_per_operation=operation_bytes,touched_address_bytes_per_operation=touched_bytes)),flush=True)
finally:
    if pinned:call('cuMemFreeHost',pinned)
    for address in reversed(allocations):call('cuMemFree_v2',address)
    call('cuDevicePrimaryCtxRelease_v2',device)
