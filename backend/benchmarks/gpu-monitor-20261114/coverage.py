"""Conservative sampled GPU evidence checks; never benchmark acceptance."""
import math,re


def audit(rows, *, luid, start, end, lifetimes, max_gap=1.5):
    """lifetimes maps known child PIDs to QPC (start, end) intervals.

    A formatted counter describes the interval ending at collection, so an
    activity observation may overlap a child exit. No PID is globally allowed.
    Short-lived processes absent from PDH cannot be ruled out by this audit.
    """
    errors=[]; activity=[]
    def finite(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    if not all(finite(v) for v in (start,end,max_gap)) or end<=start or max_gap<=0:
        raise ValueError('Invalid measured interval or maximum gap')
    for pid,span in lifetimes.items():
        if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0 or len(span)!=2 or not all(finite(v) for v in span) or span[1]<=span[0]:
            raise ValueError('Invalid recorded process lifetime')
    if not rows or rows[0].get('kind')!='header' or rows[0].get('provider')!='PDH wildcard array' or rows[0].get('adapter_luid')!=luid.lower():
        errors.append('missing or mismatched PDH adapter header')
    if not rows or rows[-1].get('kind')!='end' or rows[-1].get('failed') is not False or rows[-1].get('requested_stop') is not True:
        errors.append('missing successful cooperative termination')
    if any(row.get('kind')=='error' for row in rows):errors.append('collector error')
    samples=[row for row in rows if row.get('kind')=='sample']
    times=[row.get('qpc_observed_seconds') for row in samples]
    if not times or not all(finite(t) for t in times):
        errors.append('missing or invalid sample timestamps')
    elif any(b<=a for a,b in zip(times,times[1:])):
        errors.append('nonmonotonic sample timestamps')
    else:
        if times[0]>start or times[-1]<end:errors.append('measured interval not bracketed')
        for previous,sample in zip(times,samples[1:]):
            now=sample['qpc_observed_seconds']
            if now<=start or previous>=end:continue
            if now-previous>max_gap:errors.append('sampling gap')
            counters=sample.get('target_counters')
            if not isinstance(counters,list) or not counters:
                errors.append('missing target counters');continue
            for counter in counters:
                value=counter.get('engine_percent');pid=counter.get('process_id')
                match=re.match(r'^pid_(\d+)_(luid_0x[0-9a-f]{8}_0x[0-9a-f]{8})_',counter.get('instance','').lower())
                if not match or match[2]!=luid.lower() or int(match[1])!=pid:
                    errors.append('counter identity mismatch')
                if counter.get('status') not in (0,1) or not finite(value) or not 0<=value<=100:
                    errors.append('invalid target counter');continue
                if value==0:continue
                span=lifetimes.get(pid)
                known=span is not None and previous<span[1] and now>span[0]
                activity.append({'process_id':pid,'interval':[previous,now],
                                 'engine_percent':value,'known_workload':known})
                if not known:errors.append('unattributed GPU activity')
    return {'accepted_benchmark':False,'sampled_coverage_clear':not errors,
            'errors':sorted(set(errors)),'positive_engines':activity,
            'limitations':['Sampling cannot exclude missing short-lived processes or PID reuse.',
                           'CPU activity, thermal state, workload equivalence and performance variance are not checked.']}
