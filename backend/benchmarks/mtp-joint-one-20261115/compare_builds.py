"""Verify an otherwise identical one/three-proposal comparison; never run GPU work."""
import hashlib,json
from pathlib import Path
OLD='if(joint_base) window = window > 4 ? 3 : window - 1;'
NEW='if(joint_base) window = 1; /* Diagnostic base plus one proposal. */'
BASE='mtp-joint-base-v2-20261114';CANDIDATE='mtp-joint-one-20261115'
BASE_WORK='joint-base-generation-v2-20261114';CANDIDATE_WORK='joint-one-generation-20261115'

def validate(base_core,candidate_core,base_model,candidate_model,base_commands,candidate_commands,base_archive,candidate_archive):
    if base_core.count(OLD)!=1 or base_core.replace(OLD,NEW)!=candidate_core:
        raise ValueError('Core differs beyond proposal-window assignment')
    if base_model.count('mtp-joint3')!=1 or base_model.replace('mtp-joint3','mtp-joint1')!=candidate_model:
        raise ValueError('Generation harness differs beyond joint label')
    if not base_archive or base_archive!=candidate_archive:
        raise ValueError('Provider archive mismatch')
    def normalized(commands,work,folder):
        return [[arg.replace('\\','/').replace(work,'WORK').replace(folder,'EXPERIMENT') for arg in command] for command in commands]
    if not base_commands or normalized(base_commands,BASE_WORK,BASE)!=normalized(candidate_commands,CANDIDATE_WORK,CANDIDATE):
        raise ValueError('Compiler/link commands differ')


def main():
    root=Path.cwd();out=Path(__file__).resolve().parent;paths=[out.parent/BASE/'generation-build.json',out/'generation-build.json'];proofs=[];hashes={}
    for path in paths:
        proof=json.loads(path.read_text());proofs.append(proof)
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        for n,h in proof['sha256'].items():
            source=root/n
            if hashlib.sha256(source.read_bytes()).hexdigest()!=h:raise ValueError('Changed build input: '+n)
            hashes[str(source.resolve())]=h
    archive=str((root/'build-cuda/libnya-engine.a').resolve())
    archives=[{str((root/n).resolve()):h for n,h in p['sha256'].items()}[archive] for p in proofs]
    validate((root/'build-cuda'/BASE_WORK/'core.c').read_text(),(root/'build-cuda'/CANDIDATE_WORK/'core.c').read_text(),
             (out.parent/BASE/'generation.c').read_text(),(out/'generation.c').read_text(),
             proofs[0]['commands'],proofs[1]['commands'],*archives)
    hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (out/'comparison-build-proof.json').open('x') as f:
        json.dump(dict(accepted_benchmark=False,passed=True,sha256=hashes,
                       limitation='Source/build comparability only; runtime environment, correctness, monitoring, order and variance remain separate checks.'),f,indent=2)
    print('Core differs by one scheduling assignment; harness differs by label; provider archive and normalized commands match')

if __name__=='__main__':main()
