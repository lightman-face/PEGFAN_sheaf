"""Recover frozen split-specific scorers at previously selected validation epochs.

No test labels are used here; weights were not persisted by the old benchmark.
Validation is recomputed once to audit replay, never to select a new epoch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, random, time
import numpy as np
import torch
import torch.nn.functional as F
from model import SheafHaarTransformer
from process import full_load_data
from compare_pegfan import digest, write_json


def main():
    output=Path('results/targeted_ablations/scorers');output.mkdir(parents=True,exist_ok=True)
    old=Path('results/filtration_vs_pegfan')
    protocol=json.loads((old/'protocol.json').read_text())
    for name,sha in protocol['source_sha256'].items():
        if digest(name)!=sha: raise RuntimeError('Previous benchmark source changed: '+name)
    results=[json.loads(s) for s in (old/'metrics.jsonl').read_text().splitlines()]
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0')
    for dataset in ('film','chameleon','squirrel','texas','cornell','wisconsin'):
        loaded=full_load_data(dataset,Path('splits')/f'{dataset}_split_0.6_0.2_0.npz',return_sparse=True)
        adjacency,_,_,features,labels,_,_,_,nf,nc=loaded
        features,labels=features.to(device),labels.to(device)
        for reference in sorted([r for r in results if r['dataset']==dataset and r['model']=='sheaf'],key=lambda r:r['split']):
            split,seed=reference['split'],reference['seed'];key=f'{dataset}_split{split}_seed{seed}'
            if (output/(key+'.json')).exists():continue
            write_json(output/'progress.json',dict(status='running',dataset=dataset,split=split))
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=SheafHaarTransformer(nf,64,nc,token_budget=512,dropout=.5).to(device).set_graph(adjacency)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005)
            with np.load(Path('splits')/f'{dataset}_split_0.6_0.2_{split}.npz') as masks:
                train=torch.tensor(masks['train_mask'],device=device);val=torch.tensor(masks['val_mask'],device=device)
            start=time.perf_counter()
            for epoch in range(1,reference['best_epoch']+1):
                model.train();optimizer.zero_grad();out=model(features)
                loss=F.nll_loss(out[train],labels[train]);loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():
                out=model(features);vl=F.nll_loss(out[val],labels[val]).item()
                hidden=F.gelu(model.input_projection(features))
                _,energy=model.sheaf(hidden,model.edge_sources,model.edge_targets,model.edge_weights)
            torch.cuda.synchronize()
            np.savez_compressed(output/(key+'.npz'),sources=model.edge_sources.cpu().numpy(),targets=model.edge_targets.cpu().numpy(),
                                energy=energy.cpu().numpy(),hidden=hidden.cpu().numpy(),weights=model.edge_weights.cpu().numpy())
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output/(key+'.pt'))
            audit=dict(dataset=dataset,split=split,seed=seed,epoch=reference['best_epoch'],reference_validation_loss=reference['validation_loss'],
                       replay_validation_loss=vl,validation_loss_difference=vl-reference['validation_loss'],
                       numerically_matching=bool(np.isclose(vl,reference['validation_loss'],atol=1e-5,rtol=1e-5)),
                       seconds=time.perf_counter()-start,split_sha256=reference['split_sha256'],
                       checkpoint_sha256=digest(output/(key+'.pt')),scores_sha256=digest(output/(key+'.npz')),
                       node_counts=model.hierarchy.node_counts,max_region_fraction=float(model.hierarchy.masses[model.hierarchy.token_level].max()/len(features)))
            write_json(output/(key+'.json'),audit);print(json.dumps(audit),flush=True)
            del optimizer,model;torch.cuda.empty_cache()
    write_json(output/'progress.json',dict(status='complete',completed=60))

if __name__=='__main__':main()
