from common import *
getattr_cuda=bind(cu,'cuFuncGetAttribute',[C.POINTER(I),I,P]);result={}
try:
 for name,f in funcs.items():
  attrs={}
  for label,index in [('registers',4),('local_bytes',3),('shared_bytes',1),('max_threads',0)]:
   value=I();check(getattr_cuda(C.byref(value),index,f));attrs[label]=value.value
  result[name]=attrs
finally:check(unload(module));check(release(device))
Path(__file__).with_name('resources.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
