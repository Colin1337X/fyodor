"""Conservative desktop GPU gates for external-reference diagnostic timing."""
import functools,json,os,subprocess,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET

active_pid=None
directory=None
record={}
desktop=set()
desktop_window=False

def snapshot():
    xml=subprocess.check_output(['nvidia-smi','-q','-x'],timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    tree=ET.fromstring(xml)
    processes=[{'pid':p.findtext('pid'),'type':p.findtext('type'),'name':p.findtext('process_name')}
               for p in tree.findall('.//process_info')]
    gpu=tree.find('gpu')
    power=(gpu.findtext('gpu_power_readings/instant_power_draw') or
           gpu.findtext('gpu_power_readings/power_draw') or gpu.findtext('power_readings/power_draw'))
    return {'time':time.time(),'utilization':int(gpu.findtext('utilization/gpu_util').split()[0]),
            'memory':gpu.findtext('fb_memory_usage/used'),'temperature':gpu.findtext('temperature/gpu_temp'),
            'power':power,'processes':processes}

def competitors(state):
    allowed={str(os.getpid()),str(active_pid)}
    return [p for p in state['processes'] if p['pid'] not in allowed and
            (p['type']=='C' or (p['pid'],p['name']) not in desktop or 'NVIDIA Overlay' in p['name'])]

def save():
    (directory/'environment.json').write_text(json.dumps(record,indent=2)+'\n')

def set_active_pid(pid):
    global active_pid
    active_pid=pid
    state={'owner_pid':os.getpid(),'active_pid':active_pid,'desktop':list(desktop)}
    temporary=directory/'allowed.tmp'
    temporary.write_text(json.dumps(state))
    temporary.replace(directory/'allowed.json')

def configure(path,window=False):
    global directory,record,desktop,desktop_window
    directory=path
    desktop_window=window
    state=snapshot()
    desktop={(p['pid'],p['name']) for p in state['processes'] if p['type']=='C+G' and 'NVIDIA Overlay' not in p['name']}
    record={'initial':state,'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True),
            'method':'Three consecutive <=30% and <=15W desktop-idle samples; reject unknown GPU identities, overlay and other compute; DWM/development UI remains',
            'monitoring':'Separate process; no shared Python GIL with reference training','activities':[]}
    if desktop_window:
        record['method']='Ten consecutive desktop samples: mean utilization <=30%, every sample <=45% and <=15W; reject unknown GPU identities, overlay and other compute; DWM/development UI remains'
    set_active_pid(None)
    save()

def idle():
    readings=[]; consecutive=0
    for _ in range(30):
        state=snapshot(); readings.append(state)
        if competitors(state):
            record['failure']={'reason':'competing process','readings':readings};save()
            raise RuntimeError('Competing GPU process')
        consecutive=consecutive+1 if state['utilization']<=30 and float(state['power'].split()[0])<=15 else 0
        if desktop_window:
            recent=readings[-10:]
            if len(recent)==10 and sum(s['utilization'] for s in recent)<=300 and all(
                    0<=s['utilization']<=45 and float(s['power'].split()[0])<=15 for s in recent):return readings
        elif consecutive==3:return readings
        time.sleep(1)
    record['failure']={'reason':'idle gate','readings':readings};save()
    raise RuntimeError('GPU did not settle')

def capture(label):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args,**kwargs):
            row={'label':label,'before':idle(),'samples':[]}
            record['activities'].append(row);save()
            index=len(record['activities'])
            output=directory/f'monitor-{index}.json';stop=directory/f'monitor-{index}.stop'
            ready=directory/f'monitor-{index}.ready'
            worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(directory),str(index)],creationflags=subprocess.CREATE_NO_WINDOW)
            deadline=time.monotonic()+15
            while not ready.exists():
                if worker.poll() is not None or time.monotonic()>deadline:
                    stop.touch();worker.wait(timeout=15)
                    raise RuntimeError('External monitor did not start')
                time.sleep(.02)
            try:result=function(*args,**kwargs)
            finally:
                stop.touch();worker.wait(timeout=15)
                captured=json.loads(output.read_text())
                row['samples']=captured['samples']
                if captured.get('error'):row['monitor_error']=captured['error']
                save()
            row['after']=idle()
            row['passed']=bool(row['samples']) and not row.get('monitor_error') and all(not s['competitors'] for s in row['samples'])
            save();assert row['passed'],'Environment monitoring rejected this activity'
            return result
        return wrapped
    return decorate

if __name__=='__main__':
    target=Path(sys.argv[1]);index=sys.argv[2];data={'samples':[]}
    stop=target/f'monitor-{index}.stop';ready=target/f'monitor-{index}.ready'
    try:
        while not stop.exists():
            state=snapshot();allowed=json.loads((target/'allowed.json').read_text())
            identities={tuple(p) for p in allowed['desktop']}
            owners={str(allowed['owner_pid']),str(allowed['active_pid'])}
            state['competitors']=[p for p in state['processes'] if p['pid'] not in owners and
                (p['type']=='C' or (p['pid'],p['name']) not in identities or 'NVIDIA Overlay' in p['name'])]
            data['samples'].append(state);ready.touch();time.sleep(.25)
    except Exception as error:data['error']=str(error)
    finally:(target/f'monitor-{index}.json').write_text(json.dumps(data,indent=2)+'\n')
