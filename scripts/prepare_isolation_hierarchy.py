"""Freeze the previous validation-selected Sheaf trees for a common-tree matrix."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
import torch.nn.functional as F
from augmentation_ops import make_augmented
from compare_pegfan import digest,write_json
from pegfan_baseline import prepare_pegfan_channels
from process import full_load_data
from run_augmentation import records


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    source=Path('results/architecture_augmentation');out=Path('results/gradient_isolation/hierarchies')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((source/'protocol.json').read_text())
    for name,sha in protocol['sources'].items():assert digest(name)==sha,name
    previous={(r['dataset'],r['split']):r for r in records(source/'metrics.jsonl') if r['model']=='transformer_haar128'}
    info=[];device=torch.device('cuda:0')
    for d in ['chameleon','squirrel']:
        for name,sha in protocol['data'][d].items():assert digest(Path('new_data')/d/name)==sha
        loaded=full_load_data(d,Path('splits')/f'{d}_split_0.6_0.2_0.npz',return_sparse=True)
        base=[h.to(device) for h in prepare_pegfan_channels(loaded[3],loaded[1],loaded[2],d)]
        x,y=loaded[3].to(device),loaded[4].to(device)
        for split in [0,1,2]:
            row=previous[d,split];checkpoint=source/'checkpoints'/f'{d}_transformer_haar128_split{split}_seed42.pt'
            assert digest(checkpoint)==row['checkpoint_sha256']
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            model,_=make_augmented(loaded[-2],len(base),loaded[-1],'transformer_haar128',loaded[0],device)
            model.load_state_dict(torch.load(checkpoint,map_location=device));model.eval()
            with np.load(Path('splits')/f'{d}_split_0.6_0.2_{split}.npz') as f:
                val=torch.as_tensor(f['val_mask'],dtype=torch.bool,device=device)
            with torch.no_grad():vl=F.nll_loss(model(base,x)[val],y[val]).item()
            assert np.isclose(vl,row['validation_loss'],atol=1e-5,rtol=1e-5)
            hierarchy=model.branch.hierarchy
            parts=[np.arange(len(x),dtype=np.int64)]
            for assignment in hierarchy.assignments:parts.append(assignment[parts[-1]])
            path=out/f'{d}_split{split}.npz'
            if path.exists():
                with np.load(path) as saved:
                    assert int(saved['cut'])==hierarchy.token_level
                    for i,p in enumerate(parts):assert np.array_equal(saved[f'level{i}'],p)
            else:
                np.savez_compressed(path,cut=np.array(hierarchy.token_level),**{f'level{i}':p for i,p in enumerate(parts)})
            entry=dict(dataset=d,split=split,path=str(path),sha256=digest(path),source_checkpoint=str(checkpoint),
                source_checkpoint_sha256=digest(checkpoint),source_validation_loss=row['validation_loss'],replayed_validation_loss=vl,
                token_level=hierarchy.token_level,node_counts=hierarchy.node_counts,
                largest_region_fraction=float(np.bincount(parts[hierarchy.token_level]).max()/len(x)))
            info.append(entry);print(json.dumps(entry),flush=True)
            del model
    write_json(out/'manifest.json',dict(status='verified',selection='Previous same-split validation-selected Haar128 checkpoint; no new tree tuning',
        supervised_reference=True,entries=info,reference_protocol_sha256=digest(source/'protocol.json')))
if __name__=='__main__':main()
