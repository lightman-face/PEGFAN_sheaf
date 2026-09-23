"""Read-only Transformer diagnostics on the repaired Cornell replacement model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from augmentation_ops import transformer_diagnostics
from causal_ops import make_full_with_akx
from compare_pegfan import digest, write_json
from pegfan_baseline import prepare_pegfan_channels
from process import full_load_data


def main():
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    root=Path('results/causal_ablations/cornell_full_repaired')
    output=Path('results/architecture_augmentation/diagnostics/previous_full'); output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    for src in ('causal_ops.py','model.py','framelets_utils.py','process.py'):
        assert digest(src)==protocol['sources'][src],src
    for name,sha in protocol['data']['cornell'].items(): assert digest(Path('new_data/cornell')/name)==sha
    rs=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    device=torch.device('cuda:0')
    loaded=full_load_data('cornell',Path('splits/cornell_split_0.6_0.2_0.npz'),return_sparse=True)
    adj,_,_,x,y,_,_,_,nf,nc=loaded
    channels=[h.to(device) for h in prepare_pegfan_channels(x,loaded[1],loaded[2],'cornell')]
    x,y=x.to(device),y.to(device)
    summaries=[]
    for r in rs:
        if r['model']!='full_no_akx': continue
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model,_=make_full_with_akx(nf,64,nc,adj,channels[1:4],False,device)
        model.eval();split=r['split'];key=f'cornell_full_no_akx_split{split}_seed42'
        checkpoint=root/'checkpoints'/(key+'.pt');assert digest(checkpoint)==r['checkpoint_sha256']
        result=dict(dataset='cornell',model='previous_full',split=split,checkpoint_sha256=r['checkpoint_sha256'])
        errors=[];repeat_errors=[];changed=[]
        with torch.no_grad():
            for phase in ('initial','selected'):
                if phase=='selected':model.load_state_dict(torch.load(checkpoint,map_location=device))
                before=model(x)
                repeated=model(x)
                repeat_errors.append(float((before-repeated).abs().max()))
                with transformer_diagnostics(model.transformer) as diag: out=model(x)
                errors.append(float((before-out).abs().max()))
                changed.append(int((before.argmax(1)!=out.argmax(1)).sum()))
                assert errors[-1]<=1e-5 and changed[-1]==0, 'Diagnostic output mismatch'
                assert model.hierarchy.num_tokens==len(x)==183
                result[phase]=diag
            with np.load(f'splits/cornell_split_0.6_0.2_{split}.npz') as f:
                val=torch.as_tensor(f['val_mask'],dtype=torch.bool,device=device)
                test=torch.as_tensor(f['test_mask'],dtype=torch.bool,device=device)
            vl=F.nll_loss(out[val],y[val]).item()
            accuracy=(out[test].argmax(1)==y[test]).double().mean().item()
            assert np.isclose(vl,r['validation_loss'],rtol=1e-5,atol=1e-5)
            assert accuracy==r['test_accuracy']
        result.update(tokens=183,validation_loss=vl,test_accuracy=accuracy,diagnostic_output_error=max(errors), repeated_forward_error=max(repeat_errors), diagnostic_changed_predictions=sum(changed))
        write_json(output/(key+'.json'),result)
        reps=result['selected']['representations']
        summary=dict(split=split,accuracy=accuracy,cos_before=reps['before']['mean_pairwise_cosine'],
            cos_after=reps['after']['mean_pairwise_cosine'],
            centered_before=reps['before']['centered_energy_fraction'],centered_after=reps['after']['centered_energy_fraction'])
        summaries.append(summary);print(json.dumps(summary),flush=True)
        del model
    assert len(summaries)==10
    write_json(output/'audit.json',dict(status='passed',checkpoints=10,tokens=183,data_hashes_verified=True,
        source_hashes_verified=True,checkpoint_metrics_reproduced=True,diagnostic_changed_predictions=0,summaries=summaries))
if __name__=='__main__':main()
