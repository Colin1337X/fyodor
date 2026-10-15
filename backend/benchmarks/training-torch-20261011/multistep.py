"""Independent PyTorch tiny-decoder correctness oracle, before performance comparison."""
from pathlib import Path
import argparse,csv,json,math,struct,subprocess
import torch
import torch.nn.functional as F
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');args=parser.parse_args()
work=Path('D:/fyodor-validation/pytorch-multistep-oracle-'+args.device);work.mkdir(parents=True,exist_ok=False)
torch.set_default_device(args.device)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
torch.set_num_threads(6)
trainer=root/'build-cuda/fyodor-train.exe';corpus=work/'corpus.txt';corpus.write_text('abc'*20)
common=[str(trainer),'--data',str(corpus),'--dimension','32','--ff','64','--layers','1','--heads','2','--kv-heads','1','--context','8','--steps','20','--threads','6']
records=[]
for name,extra,input_text in [('initial',['--control-stdin','1'],'S'),('native',['--compute',args.device],None)]:
 command=common+['--output',str(work/(name+'.gguf')),'--checkpoint',str(work/(name+'.ckpt')),'--metrics',str(work/(name+'.csv'))]+extra
 p=subprocess.run(command,input=input_text,capture_output=True,text=True,timeout=60);records.append({'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0,records[-1]
def checkpoint(path):
 data=path.read_bytes();count,step=struct.unpack_from('<QQ',data,24);settings=struct.unpack_from('<6f',data,40);offset=64;states=[]
 for _ in range(count):
  rows,cols=struct.unpack_from('<QQ',data,offset);offset+=16;arrays=[]
  for _ in range(4):
   arrays.append(torch.tensor(struct.unpack_from('<'+str(rows*cols)+'f',data,offset)).reshape(rows,cols));offset+=rows*cols*4
  states.append(arrays)
 assert offset+8==len(data)
 return states,step,settings
initial,step,settings=checkpoint(work/'initial.ckpt');assert step==0
expected,step,_=checkpoint(work/'native.ckpt');assert step==20
params=[torch.nn.Parameter(state[0].clone()) for state in initial]
def norm(x,w):return (x.double()*torch.rsqrt(x.double().square().mean(-1,keepdim=True)+1e-5)*w.double()).float()
def linear(x,w):return F.linear(x,w)
def rope(x,heads):
 count=x.shape[0];x=x.reshape(count,heads,16);freq=torch.tensor([struct.unpack('<f',struct.pack('<f',10000**(-2*j/16)))[0] for j in range(8)],dtype=torch.float64)
 angles=torch.arange(count,dtype=torch.float64)[:,None,None]*freq[None,None,:]
 a=x[...,0::2].double();b=x[...,1::2].double()
 return torch.stack((a*angles.cos()-b*angles.sin(),a*angles.sin()+b*angles.cos()),dim=-1).float().reshape(count,-1)
tokens=torch.tensor([1]+[ord(c)+3 for c in ('abc'*20)])
lr,b1,b2,eps,decay,maxnorm=settings
optimizer=torch.optim.AdamW(params,lr=lr,betas=(b1,b2),eps=eps,weight_decay=decay,foreach=False)
losses=[]
for update in range(20):
 optimizer.zero_grad(set_to_none=True)
 offset=(update%((len(tokens)-2)//8+1))*8
 sequence=tokens[offset:offset+9];count=len(sequence)-1
 x=F.embedding(sequence[:-1],params[0]);n=norm(x,params[3])
 q=rope(linear(n,params[4]),2).reshape(count,2,16).transpose(0,1)
 k=rope(linear(n,params[5]),1).reshape(count,1,16).transpose(0,1).repeat_interleave(2,0)
 v=linear(n,params[6]).reshape(count,1,16).transpose(0,1).repeat_interleave(2,0)
 scores=(q@k.transpose(-2,-1))*.25;scores=scores.masked_fill(torch.ones(count,count,dtype=torch.bool).triu(1),float('-inf'))
 a=(scores.softmax(-1)@v).transpose(0,1).reshape(count,32)
 x=x+linear(a,params[7]);n=norm(x,params[8]);x=x+linear(F.silu(linear(n,params[9]))*linear(n,params[11]),params[10])
 logits=linear(norm(x,params[1]),params[2]);loss=F.cross_entropy(logits,sequence[1:]);loss.backward()
 losses.append(loss.item())
 raw=[p.grad.clone() for p in params]
 norm_value=torch.sqrt(sum(g.double().square().sum() for g in raw));clip=min(1.,maxnorm/norm_value.item())
 for p in params:p.grad.mul_(clip)
 optimizer.step()
errors={key:0. for key in ('parameter','gradient','moment','variance')}
for i,p in enumerate(params):
 actual=[p.detach(),raw[i],optimizer.state[p]['exp_avg'],optimizer.state[p]['exp_avg_sq']]
 for key,a,b in zip(errors,actual,expected[i]):errors[key]=max(errors[key],((a-b).abs()/(1+b.abs())).max().item())
with (work/'native.csv').open(newline='') as f:native_losses=[float(row['loss']) for row in csv.DictReader(f)]
assert len(native_losses)==len(losses)==20
maximum_loss_error=max(abs(a-b) for a,b in zip(losses,native_losses))
record={'torch_version':torch.__version__,'device':args.device,'runs':records,'torch_losses':losses,'native_losses':native_losses,'maximum_loss_absolute_error':maximum_loss_error,'state_maximum_scaled_errors':errors,'tolerance':1e-5,'passed':all(v<=1e-5 for v in errors.values()) and maximum_loss_error<=1e-5,'timing_accepted':False}
(out/('torch-multistep-oracle-'+args.device+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2));assert record['passed']
