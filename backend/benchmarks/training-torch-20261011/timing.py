"""Matched whole-update timing against an independent eager PyTorch decoder.

Fresh output directory required. Every pair checks losses and all four optimizer
arrays at the unchanged 1e-5 tolerance before its throughput is accepted.
"""
from pathlib import Path
import argparse, csv, hashlib, json, statistics, struct, subprocess, time
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--ff', type=int, default=176)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--kv-heads', type=int, default=2)
parser.add_argument('--context', type=int, default=32)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--repeats', type=int, default=4)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--diagnostic', action='store_true', help='Disable timing acceptance; allow single-update localization')
parser.add_argument('--optimizer', choices=('default','scalar'), default='default')
parser.add_argument('--reference-precision', choices=('float32','float64'), default='float32')
parser.add_argument('--finite-difference', action='store_true')
args = parser.parse_args()
assert args.reference_precision == 'float32' or args.diagnostic
assert not args.finite_difference or (args.diagnostic and args.reference_precision == 'float64' and args.steps == 1 and args.width == 256 and args.layers == 2)
assert args.steps >= 1 and args.repeats >= 1
assert args.diagnostic or (args.steps > args.warmup+1 >= 2 and args.repeats >= 2)
assert args.width % args.heads == 0 and args.heads % args.kv_heads == 0
root = Path(__file__).resolve().parents[3]
work = args.output.resolve(); work.mkdir(parents=True, exist_ok=False)
torch.set_default_device(args.device)
torch.set_num_threads(6)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
trainer = root / 'build-cuda/fyodor-train.exe'
corpus = work / 'corpus.txt'
data = ('The small cat sat beside the quiet library. ' * 100)
corpus.write_text(data, encoding='ascii')
common = [str(trainer), '--data', str(corpus), '--dimension', str(args.width),
          '--ff', str(args.ff), '--layers', str(args.layers), '--heads', str(args.heads),
          '--kv-heads', str(args.kv_heads), '--context', str(args.context),
          '--steps', str(args.steps), '--threads', '6', '--memory-mib', '4096']

def native(name, initial=False):
    command = common + ['--output', str(work / (name + '.gguf')),
                        '--checkpoint', str(work / (name + '.ckpt')),
                        '--metrics', str(work / (name + '.csv'))]
    command += ['--control-stdin', '1'] if initial else ['--compute', args.device]
    p = subprocess.run(command, input='S' if initial else None,
                       capture_output=True, text=True, timeout=1800)
    (work / (name + '.log')).write_text(p.stdout + p.stderr, encoding='utf-8')
    assert p.returncode == 0, p.stderr
    return command

def checkpoint(path):
    data = path.read_bytes(); count, step = struct.unpack_from('<QQ', data, 24)
    settings = struct.unpack_from('<6f', data, 40); offset = 64; states = []
    for _ in range(count):
        rows, cols = struct.unpack_from('<QQ', data, offset); offset += 16
        arrays = []
        for _ in range(4):
            arrays.append(torch.tensor(struct.unpack_from('<' + str(rows * cols) + 'f', data, offset)).reshape(rows, cols))
            offset += rows * cols * 4
        states.append(arrays)
    assert offset + 8 == len(data)
    return states, step, settings

native('initial', True)
initial, step, settings = checkpoint(work / 'initial.ckpt'); assert step == 0
tokens = torch.tensor([1] + [ord(c) + 3 for c in data])
hd = args.width // args.heads; context = args.context
freq = torch.tensor([struct.unpack('<f', struct.pack('<f', 10000 ** (-2*j/hd)))[0]
                     for j in range(hd // 2)], dtype=torch.float64)
angle = torch.arange(context, dtype=torch.float64)[:, None, None] * freq[None, None, :]
cos, sin = angle.cos(), angle.sin()
mask = torch.ones(context, context, dtype=torch.bool).triu(1)

def norm(x, w):
    return (x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-5) * w.double()).to(x.dtype)

def rope(x, heads):
    count = x.shape[0]; x = x.reshape(count, heads, hd)
    a, b = x[..., 0::2].double(), x[..., 1::2].double()
    return torch.stack((a*cos[:count]-b*sin[:count], a*sin[:count]+b*cos[:count]), -1).to(x.dtype).reshape(count,heads,hd)

def model_loss(params, sequence):
    count = len(sequence)-1
    x = F.embedding(sequence[:-1], params[0])
    for layer in range(args.layers):
        p = params[3+9*layer:12+9*layer]; n = norm(x,p[0])
        q = rope(F.linear(n,p[1]),args.heads).transpose(0,1)
        k = rope(F.linear(n,p[2]),args.kv_heads).transpose(0,1).repeat_interleave(args.heads//args.kv_heads,0)
        v = F.linear(n,p[3]).reshape(count,args.kv_heads,hd).transpose(0,1).repeat_interleave(args.heads//args.kv_heads,0)
        scores = ((q @ k.transpose(-2,-1)) * hd**-.5).masked_fill(mask[:count,:count],float('-inf'))
        attention = (scores.softmax(-1) @ v).transpose(0,1).reshape(count,args.width)
        x = x + F.linear(attention,p[4]); n = norm(x,p[5])
        x = x + F.linear(F.silu(F.linear(n,p[6])) * F.linear(n,p[8]),p[7])
    return F.cross_entropy(F.linear(norm(x,params[1]),params[2]),sequence[1:])

def reference():
    dtype = torch.float64 if args.reference_precision == 'float64' else torch.float32
    params = [torch.nn.Parameter(state[0].clone().to(dtype)) for state in initial]
    lr, b1, b2, eps, decay, maxnorm = settings
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(b1,b2), eps=eps,
                                 weight_decay=decay, fused=args.device == 'cuda' and args.optimizer == 'default',
                                 foreach=args.device == 'cpu' and args.optimizer == 'default')
    reference.finite_differences = []
    if args.finite_difference:
        sequence = tokens[:context+1]
        gradients = torch.autograd.grad(model_loss(params,sequence),params)
        for index,flat in ((10,149315),(9,108383),(7,44365)):
            p = params[index].flatten(); original = p[flat].item()
            derivative = gradients[index].flatten()[flat].item()
            for delta in (1e-3,1e-4):
                with torch.no_grad():
                    p[flat] = original+delta; plus = model_loss(params,sequence).item()
                    p[flat] = original-delta; minus = model_loss(params,sequence).item()
                    p[flat] = original
                numerical = (plus-minus)/(2*delta)
                reference.finite_differences.append({'parameter':index,'flat_index':flat,
                    'delta':delta,'autograd':derivative,'finite_difference':numerical,
                    'absolute_error':abs(numerical-derivative)})
    losses, elapsed = [], []
    for update in range(args.steps):
        if args.device == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        offset = (update % ((len(tokens)-2)//context+1))*context
        sequence = tokens[offset:offset+context+1]
        loss = model_loss(params,sequence)
        loss.backward(); losses.append(loss.item())
        # Only the final (excluded) update copies raw gradients for diagnostics.
        if update == args.steps-1: raw = [p.grad.clone() for p in params]
        total = torch.sqrt(sum(p.grad.double().square().sum() for p in params))
        clip = min(1.,maxnorm/total.item())
        for p in params: p.grad.mul_(clip)
        optimizer.step()
        if args.device == 'cuda': torch.cuda.synchronize()
        elapsed.append((time.perf_counter()-start)*1000)
    states = [[p.detach(),raw[i],optimizer.state[p]['exp_avg'],optimizer.state[p]['exp_avg_sq']]
              for i,p in enumerate(params)]
    return losses, elapsed, states

record = {'configuration':vars(args) | {'output':str(work)}, 'torch_version':torch.__version__,
          'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'trainer_sha256':hashlib.sha256(trainer.read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'reference':'eager '+args.reference_precision+' decoder; double RMS/RoPE/clip; TF32 off; AdamW '+args.optimizer,
          'timing':'whole update; startup/export excluded; first warmup and last diagnostic update excluded',
          'tolerance':1e-5,'timing_environment_verified':False,'pairs':[]}
for repeat in range(args.repeats):
    name = 'native-' + str(repeat)
    if repeat % 2: losses, elapsed, actual = reference()
    command = native(name)
    if not repeat % 2: losses, elapsed, actual = reference()
    expected, step, _ = checkpoint(work / (name+'.ckpt')); assert step == args.steps
    if args.diagnostic:
        torch.save([[x.detach().cpu() for x in state] for state in actual],work/('reference-'+str(repeat)+'.pt'))
    errors = {key:0. for key in ('parameter','gradient','moment','variance')}
    details = []
    for index,(a,b) in enumerate(zip(actual,expected)):
        for key,x,y in zip(errors,a,b):
            scaled = (x-y).abs()/(1+y.abs())
            flat = int(scaled.argmax().item())
            error = scaled.flatten()[flat].item()
            errors[key] = max(errors[key],error)
            details.append({'parameter':index,'array':key,'scaled_error':error,
                            'flat_index':flat,'torch':x.flatten()[flat].item(),'native':y.flatten()[flat].item()})
            if key == 'parameter':
                details[-1].update(initial=initial[index][0].flatten()[flat].item(),
                                   torch_raw_gradient=a[1].flatten()[flat].item(),
                                   native_raw_gradient=b[1].flatten()[flat].item())
    with (work/(name+'.csv')).open(newline='') as f: rows = list(csv.DictReader(f))
    assert len(rows) == len(losses) == args.steps
    loss_error = max(abs(float(row['loss'])-loss) for row,loss in zip(rows,losses))
    selected = slice(args.warmup,-1)
    native_ms = None if args.diagnostic else statistics.mean(float(row['step_ms']) for row in rows[selected])
    torch_ms = None if args.diagnostic else statistics.mean(elapsed[selected])
    passed = loss_error <= 1e-5 and all(x <= 1e-5 for x in errors.values())
    pair = {'repeat':repeat,'order':['torch','native'] if repeat%2 else ['native','torch'],
            'native_command':command,'torch_step_ms':elapsed,'torch_losses':losses,
            'native_mean_ms':native_ms,'torch_mean_ms':torch_ms,
            'maximum_loss_absolute_error':loss_error,'state_maximum_scaled_errors':errors,
            'state_details':details,'correctness_passed':passed,
            'finite_differences':reference.finite_differences,
            'throughput_percentage':torch_ms/native_ms*100 if passed and not args.diagnostic else None}
    if args.diagnostic and args.steps == 1:
        # Hold every raw gradient fixed to isolate optimizer arithmetic from
        # discrepancies produced by independent forward/backward reductions.
        control = [torch.nn.Parameter(state[0].clone()) for state in initial]
        lr,b1,b2,eps,decay,maxnorm = settings
        opt = torch.optim.AdamW(control,lr=lr,betas=(b1,b2),eps=eps,weight_decay=decay,
                               fused=args.device == 'cuda' and args.optimizer == 'default',
                               foreach=args.device == 'cpu' and args.optimizer == 'default')
        total = torch.sqrt(sum(state[1].double().square().sum() for state in expected))
        clip = min(1.,maxnorm/total.item())
        for p,state in zip(control,expected): p.grad = state[1].clone()*clip
        opt.step()
        pair['identical_native_gradient_optimizer_parameter_error'] = max(
            ((p.detach()-state[0]).abs()/(1+state[0].abs())).max().item()
            for p,state in zip(control,expected))
    record['pairs'].append(pair)
    (work/'results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(pair),flush=True)
    assert passed, 'Timing rejected: numerical comparison failed'
