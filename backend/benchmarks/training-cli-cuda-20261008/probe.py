"""Exercise public CPU/CUDA trajectories, telemetry and failed-evaluation saves."""
from pathlib import Path
import csv,json,math,struct,subprocess,tempfile
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
trainer=root/"build-cuda/fyodor-train.exe"
records=[]
with tempfile.TemporaryDirectory(prefix="fyodor-gpu-cli-") as folder:
    work=Path(folder);data=work/"corpus.txt";data.write_text("abc"*12,encoding="utf-8")
    def run(name,compute,steps,rate="0.015",extra=(),success=True):
        model=work/(name+".gguf");checkpoint=work/(name+".ckpt");metrics=work/(name+".csv")
        command=[str(trainer),"--compute",compute,"--device-memory-mib","32","--data",str(data),"--output",str(model),"--checkpoint",str(checkpoint),"--metrics",str(metrics),"--steps",str(steps),"--lr",rate,"--dimension","16","--ff","32","--layers","1","--heads","2","--kv-heads","1","--context","16",*extra]
        result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",timeout=120)
        record=dict(name=name,command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr);records.append(record)
        assert (result.returncode==0)==success,record
        if checkpoint.exists():record["checkpoint_step"]=struct.unpack_from("<Q",checkpoint.read_bytes(),32)[0]
        if success:
            rows=list(csv.DictReader(metrics.open(newline="")));record["metrics"]=rows
            for row in rows:
                assert row["compute"]==compute and row["timing_source"]==("cuda_events" if compute=="cuda" else "cpu_wall")
                assert all(math.isfinite(float(row[k])) and float(row[k])>=0 for k in ("forward_ms","backward_ms","optimizer_ms","step_ms"))
                if compute=="cuda":
                    assert 0<int(row["device_used_bytes"])<=int(row["device_peak_bytes"])<=int(row["device_capacity_bytes"])
                    assert math.isfinite(float(row["gradient_norm"])) and float(row["gradient_norm"])>=0
                else:assert row["gradient_norm"]==row["device_peak_bytes"]==""
        return checkpoint,model
    cpu,_=run("cpu","cpu",100);gpu,_=run("gpu","cuda",100)
    a=cpu.read_bytes()[16:];b=gpu.read_bytes()[16:];assert a[:48]==b[:48]
    count=struct.unpack_from("<Q",a,8)[0];offset=48;maximum=0
    for parameter in range(count):
        rows,columns=struct.unpack_from("<QQ",a,offset);assert a[offset:offset+16]==b[offset:offset+16];offset+=16
        for state in range(4):
            for index in range(rows*columns):
                expected=struct.unpack_from("<f",a,offset)[0];actual=struct.unpack_from("<f",b,offset)[0];offset+=4
                error=abs(expected-actual)/(1+abs(expected));assert math.isfinite(actual) and error<=1e-5,(parameter,state,index,error)
                maximum=max(maximum,error)
    assert offset+8==len(a)==len(b)
    first,_=run("first","cuda",40);resumed,_=run("resumed","cuda",60,extra=("--resume",str(first)))
    assert resumed.read_bytes()==gpu.read_bytes()
    # Evaluation fails only after a successful finite optimizer update. The
    # previous completed state must still be saved, with an overall failure code.
    for compute in ("cpu","cuda"):
        checkpoint,model=run("eval-failure-"+compute,compute,1,"1e30",("--eval-data",str(data),"--eval-every","1"),False)
        assert checkpoint.exists() and model.exists(),records[-1]
        assert "evaluation failed" in records[-1]["stderr"]
    # Explicit CUDA budget failure must not publish model/checkpoint outputs.
    checkpoint,model=run("budget","cuda",1,extra=("--device-memory-mib","1","--dimension","128","--ff","256","--heads","4"),success=False)
    assert not checkpoint.exists() and not model.exists()
record={"maximum_cpu_state_scaled_error":maximum,"exact_gpu_resume":True,"runs":records}
(out/"public-probe.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print("CPU/CUDA full-state parity, event telemetry, exact resume, evaluation-failure save and budget rejection passed.")
