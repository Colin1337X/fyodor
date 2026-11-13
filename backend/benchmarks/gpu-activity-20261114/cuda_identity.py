"""Read CUDA adapter identity without creating a context or launching work."""
import ctypes as c
import datetime,json,os,sys
from pathlib import Path

def identities():
    if os.name!='nt':raise RuntimeError('Windows LUID diagnostic only')
    driver=c.WinDLL('nvcuda.dll')
    signatures={
        'cuInit':[c.c_uint],
        'cuDriverGetVersion':[c.POINTER(c.c_int)],
        'cuDeviceGetCount':[c.POINTER(c.c_int)],
        'cuDeviceGet':[c.POINTER(c.c_int),c.c_int],
        'cuDeviceGetName':[c.POINTER(c.c_char),c.c_int,c.c_int],
        'cuDeviceGetLuid':[c.POINTER(c.c_char),c.POINTER(c.c_uint),c.c_int],
        'cuDeviceGetPCIBusId':[c.POINTER(c.c_char),c.c_int,c.c_int],
    }
    for name,args in signatures.items():
        fn=getattr(driver,name);fn.argtypes=args;fn.restype=c.c_int
    def call(name,*args):
        status=getattr(driver,name)(*args)
        if status:raise RuntimeError(f'{name}: CUDA status {status}')
    call('cuInit',0)
    version=c.c_int();call('cuDriverGetVersion',c.byref(version))
    count=c.c_int();call('cuDeviceGetCount',c.byref(count))
    devices=[]
    for ordinal in range(count.value):
        device=c.c_int();call('cuDeviceGet',c.byref(device),ordinal)
        name=c.create_string_buffer(256);pci=c.create_string_buffer(64)
        luid=c.create_string_buffer(8);mask=c.c_uint()
        call('cuDeviceGetName',name,len(name),device)
        call('cuDeviceGetPCIBusId',pci,len(pci),device)
        call('cuDeviceGetLuid',luid,c.byref(mask),device)
        low=int.from_bytes(luid.raw[:4],'little');high=int.from_bytes(luid.raw[4:],'little')
        devices.append({'ordinal':ordinal,'name':name.value.decode('ascii'),
                        'pci_bus_id':pci.value.decode('ascii'),
                        'luid_bytes_hex':luid.raw.hex(),'device_node_mask':mask.value,
                        'counter_luid':f'luid_0x{high:08x}_0x{low:08x}'})
    return {'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'accepted_benchmark':False,'driver_version':version.value,'devices':devices,
            'note':'Adapter identity only; no context creation, allocation or kernel execution. Match LUID within the same boot/session; retain node masks and do not assume an ordinal maps to every physical engine on multi-node adapters.'}

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: cuda_identity.py NEW_OUTPUT.json')
    path=Path(sys.argv[1])
    if path.exists():raise SystemExit('Refusing to overwrite evidence')
    result=identities()
    with path.open('x',encoding='utf-8') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result,indent=2))
