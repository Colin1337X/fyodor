"""Windows PDH wildcard-array diagnostic; no benchmark acceptance policy."""
import ctypes as C
class ValueUnion(C.Union):
 _fields_=[('double',C.c_double),('large',C.c_longlong),('pointer',C.c_void_p)]
class Value(C.Structure):
 _fields_=[('status',C.c_uint32),('value',ValueUnion)]
class Item(C.Structure):
 _fields_=[('name',C.c_wchar_p),('formatted',Value)]
class Counter:
 def __init__(self):
  self.lib=C.WinDLL('pdh.dll');self.query=C.c_void_p();self.counter=C.c_void_p()
  signatures={
   'PdhOpenQueryW':[C.c_wchar_p,C.c_size_t,C.POINTER(C.c_void_p)],
   'PdhAddEnglishCounterW':[C.c_void_p,C.c_wchar_p,C.c_size_t,C.POINTER(C.c_void_p)],
   'PdhCollectQueryData':[C.c_void_p],
   'PdhGetFormattedCounterArrayW':[C.c_void_p,C.c_uint32,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32),C.c_void_p],
   'PdhCloseQuery':[C.c_void_p]}
  for name,args in signatures.items():
   fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=C.c_uint32
  self.check(self.lib.PdhOpenQueryW(None,0,C.byref(self.query)))
  try:
   self.check(self.lib.PdhAddEnglishCounterW(self.query,r'\GPU Engine(*)\Utilization Percentage',0,C.byref(self.counter)))
   self.check(self.lib.PdhCollectQueryData(self.query))
  except BaseException:self.close();raise
 @staticmethod
 def check(status):
  if status:raise RuntimeError(f'PDH status 0x{status:08x}')
 def close(self):
  if self.query:
   status=self.lib.PdhCloseQuery(self.query);self.query=C.c_void_p();self.check(status)
 def collect(self):
  self.check(self.lib.PdhCollectQueryData(self.query))
  for attempt in range(4):
   size=C.c_uint32();count=C.c_uint32()
   status=self.lib.PdhGetFormattedCounterArrayW(self.counter,0x8200,C.byref(size),C.byref(count),None)
   if status!=0x800007d2:self.check(status)
   if not size.value:raise RuntimeError('Empty PDH array sizing result')
   data=C.create_string_buffer(size.value)
   status=self.lib.PdhGetFormattedCounterArrayW(self.counter,0x8200,C.byref(size),C.byref(count),data)
   if status==0x800007d2:continue
   self.check(status)
   if count.value*C.sizeof(Item)>len(data):raise RuntimeError('PDH array count exceeds allocation')
   items=C.cast(data,C.POINTER(Item))
   return [{'instance':items[i].name.lower(),'status':items[i].formatted.status,
            'engine_percent':items[i].formatted.value.double if items[i].formatted.status in (0,1) else None}
           for i in range(count.value)]
  raise RuntimeError('PDH instance array repeatedly changed size')
