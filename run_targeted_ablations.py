"""Three one-factor PEGFAN ablations; frozen learned hierarchy evaluated first."""
import argparse,csv,gc,json,random,time,platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ablation_ops import (original_partitions,validate_cached_tree,new_partitions,
    pegfan_replacement_channels,OriginalHierarchyTransformer,partition_diagnostics,label_diagnostics)
from framelets_utils import get_spatial_framelets_list
from pegfan_baseline import prepare_pegfan_channels,make_pegfan
from process import full_load_data
from compare_pegfan import digest,write_json,synchronize

VARIANTS=['hierarchy_only','projector_only','transformer_only']
DATASETS=['film','chameleon','squirrel','texas','cornell','wisconsin']


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def train(args,dataset,split,variant,loaded,channels,partitions):
    adjacency,_,_,features,labels,_,_,_,nf,nc=loaded
    seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    device=torch.device('cuda:0');torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    with np.load(Path('splits')/f'{dataset}_split_0.6_0.2_{split}.npz') as masks:
        train_mask,val_mask,test_mask=[torch.tensor(masks[k],dtype=torch.bool,device=device) for k in ('train_mask','val_mask','test_mask')]
    if ((train_mask&val_mask)|(train_mask&test_mask)|(val_mask&test_mask)).any():raise ValueError('overlapping masks')
    labels=labels.to(device);channels=[x.to(device) for x in channels]
    classifier,optimizer=make_pegfan(nf,len(channels),64,nc,.5,device)
    if variant=='transformer_only':
        model=OriginalHierarchyTransformer(classifier,channels,partitions)
        for module in (model.input_adapter,model.transformer,model.output_adapter):
            optimizer.add_param_group(dict(params=module.parameters(),lr=.001,weight_decay=.0005))
        forward=lambda:model()
    else:
        model=classifier;forward=lambda:model(channels,True)
    start=time.perf_counter();history=[];best=float('inf');best_state=None;stale=0
    for epoch in range(1,args.epochs+1):
        model.train();optimizer.zero_grad();out=forward();loss=F.nll_loss(out[train_mask],labels[train_mask])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();optimizer.step();model.eval()
        with torch.no_grad():
            out=forward();vl=F.nll_loss(out[val_mask],labels[val_mask]).item()
            va=(out[val_mask].argmax(1)==labels[val_mask]).float().mean().item()
        if not np.isfinite(vl):raise RuntimeError('nonfinite validation loss')
        history.append(dict(epoch=epoch,train_loss=loss.item(),validation_loss=vl,validation_accuracy=va))
        if vl<best:
            best=vl;best_epoch=epoch;best_va=va;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if epoch==1 or epoch%100==0:print(f'{dataset} split={split} {variant} epoch={epoch} val={vl:.5f}',flush=True)
        if stale>=args.patience:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        out=forward();restored=F.nll_loss(out[val_mask],labels[val_mask]).item()
        if not np.isclose(best,restored,atol=1e-5,rtol=1e-5):raise RuntimeError('checkpoint restore mismatch')
        accuracy=(out[test_mask].argmax(1)==labels[test_mask]).double().mean().item()
    synchronize(device);seconds=time.perf_counter()-start
    key=f'{dataset}_{variant}_split{split}_seed42'
    write_json(args.output/'histories'/(key+'.json'),history)
    torch.save(best_state,args.output/'checkpoints'/(key+'.pt'))
    record=dict(dataset=dataset,model=variant,split=split,seed=seed,test_accuracy=accuracy,validation_loss=best,
                validation_accuracy=best_va,best_epoch=best_epoch,epochs=epoch,training_seconds=seconds,
                parameters=sum(p.numel() for p in model.parameters()),classifier_parameters=sum(p.numel() for p in classifier.parameters()),
                channels=len(channels),node_counts=[int(p.max()+1) for p in partitions],
                split_sha256=digest(Path('splits')/f'{dataset}_split_0.6_0.2_{split}.npz'),
                peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20,
                checkpoint_sha256=digest(args.output/'checkpoints'/(key+'.pt')))
    if variant=='transformer_only':record.update(tokens=len(model.coarse_source),token_level=model.cut)
    del model,optimizer,classifier,channels,best_state,out;gc.collect();torch.cuda.empty_cache()
    return record


def summarize(output,results,old):
    summary=[];paired=[]
    for dataset in DATASETS:
        baseline={r['split']:r for r in old if r['dataset']==dataset and r['model']=='pegfan'}
        for name in ['pegfan','sheaf']+VARIANTS:
            group=[r for r in (old if name in ('pegfan','sheaf') else results) if r['dataset']==dataset and r['model']==name]
            if not group:continue
            x=np.array([r['test_accuracy']*100 for r in group]);d=np.array([(r['test_accuracy']-baseline[r['split']]['test_accuracy'])*100 for r in group])
            summary.append(dict(dataset=dataset,model=name,runs=len(x),mean_pct=float(x.mean()),std_pct=float(x.std(ddof=1)) if len(x)>1 else 0.,
                                delta_pp=float(d.mean()),wins=int(sum(d>1e-8)),ties=int(sum(abs(d)<=1e-8)),losses=int(sum(d< -1e-8))))
            for r in group:
                paired.append(dict(dataset=dataset,model=name,split=r['split'],pegfan_pct=baseline[r['split']]['test_accuracy']*100,
                                   accuracy_pct=r['test_accuracy']*100,delta_pp=(r['test_accuracy']-baseline[r['split']]['test_accuracy'])*100))
    for filename,data in [('summary.csv',summary),('paired_splits.csv',paired)]:
        with (output/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# Targeted PEGFAN ablations','','Same 10 repository splits and seed 42. Validation-loss selection, max 500 epochs, patience 50; no hyperparameter search.',
           'PEGFAN and full sheaf results are reused from the previous benchmark. Frozen scorers are replayed at its validation-selected epochs.',
           'Hierarchy-only uses original per-level region counts as filtration targets to keep FSGNN channel count and parameter count unchanged. Whole-score ties may undershoot targets.',
           'Original equal-child pairwise Haar is evaluated through an algebraically identical implicit operator; projector-only changes these weights to sqrt(child mass / parent mass).',
           'Transformer-only inserts a residual 64-wide, 2-layer attention bottleneck into AX at the original tree\'s finest existing cut <=512. Original local details and all A^kX channels remain.',
           'The residual output adapter starts at zero. New attention parameters use lr 0.001; original FSGNN keeps lr 0.01 and channel-attention lr 0.02.',
           'The hard hierarchy is frozen during FSGNN training; its supervised scorer adds pretraining cost. Scorer replay audits record numerical differences from the previous run; original weights were not saved.', '',
           '| Dataset | Variant | Runs | Accuracy (%) | Delta (pp) | Win/tie/loss |','|---|---|---:|---:|---:|---|']
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['runs']} | {r['mean_pct']:.2f} ± {r['std_pct']:.2f} | {r['delta_pp']:+.2f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=['',f'Completed new trials: {len(results)}. Partial groups are provisional.','']
    (output/'report.md').write_text('\n'.join(lines))


def final_label_diagnostics(args):
    # Run only after all classifiers/checkpoints are fixed. No output from this
    # function flows into training, early stopping, or hierarchy construction.
    result=[]
    for dataset in (d for d in args.datasets if d in ('film','chameleon','squirrel')):
        loaded=full_load_data(dataset,Path('splits')/f'{dataset}_split_0.6_0.2_0.npz',return_sparse=True)
        labels=loaded[4].numpy()
        for split in args.splits:
            with np.load(args.output/'partitions'/f'{dataset}_split{split}.npz') as saved:
                masks=np.load(Path('splits')/f'{dataset}_split_0.6_0.2_{split}.npz')
                for key in saved.files:
                    for scope,mask in [('all_posthoc',np.ones(len(labels),dtype=bool)),('test_only_posthoc',masks['test_mask'])]:
                        row=label_diagnostics(saved[key],labels,mask)
                        row.update(dataset=dataset,split=split,partition=key,label_scope=scope);result.append(row)
                masks.close()
    write_json(args.output/'label_diagnostics_posthoc.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/targeted_ablations'))
    p.add_argument('--datasets',nargs='+',default=DATASETS);p.add_argument('--splits',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    p.add_argument('--epochs',type=int,default=500);p.add_argument('--patience',type=int,default=50);p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    for sub in ('histories','checkpoints','partitions','diagnostics'): (args.output/sub).mkdir(parents=True,exist_ok=True)
    previous=Path('results/filtration_vs_pegfan');old=rows(previous/'metrics.jsonl')
    old_protocol=json.loads((previous/'protocol.json').read_text())
    for name,sha in old_protocol['source_sha256'].items():
        if digest(name)!=sha:raise RuntimeError('Previous experiment sources changed: '+name)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    protocol=dict(config=config,seed=42,source_sha256={name:digest(name) for name in ['ablation_ops.py','run_targeted_ablations.py','scripts/replay_sheaf_scorers.py']},
                  previous_protocol_sha256=digest(previous/'protocol.json'),previous_metrics_sha256=digest(previous/'metrics.jsonl'),
                  scorer_directory='results/targeted_ablations/scorers',torch=torch.__version__,python=platform.python_version())
    path=args.output/'protocol.json'
    if path.exists():
        if not args.resume or json.loads(path.read_text())!=protocol:raise ValueError('Protocol mismatch or missing --resume')
    else:write_json(path,protocol)
    results=rows(args.output/'metrics.jsonl');done={(r['dataset'],r['split'],r['model']) for r in results}
    expected=len(args.datasets)*len(args.splits)*len(args.variants)
    try:
        # Finish hierarchy-only over all data before either architectural ablation.
        for variant in args.variants:
            for dataset in args.datasets:
                if all((dataset,s,variant) in done for s in args.splits):continue
                loaded=full_load_data(dataset,Path('splits')/f'{dataset}_split_0.6_0.2_0.npz',return_sparse=True)
                adjacency,a,ai,x,labels,_,_,_,nf,nc=loaded
                base=prepare_pegfan_channels(x,a,ai,dataset)
                frames,_=get_spatial_framelets_list(None,dataset,8);original=original_partitions(frames)
                errors=validate_cached_tree(frames,original)
                equal=pegfan_replacement_channels(base,original)
                channel_errors=[float((u-v).abs().max()) for u,v in zip(base,equal)]
                if max(channel_errors)>3e-6:raise RuntimeError('Original signal channels not reproduced')
                write_json(args.output/f'{dataset}_operator_audit.json',dict(band_probe_errors=errors,feature_channel_errors=channel_errors,
                                                                           framelet_cache_sha256=digest(Path('framelets/8')/(dataset+'.pickle'))))
                del equal,frames
                mass=pegfan_replacement_channels(base,original,'mass') if variant=='projector_only' else None
                for split in args.splits:
                    if (dataset,split,variant) in done:continue
                    key=f'{dataset}_split{split}_seed42';scorer=Path('results/targeted_ablations/scorers')
                    scores_path=scorer/(key+'.npz')
                    if not (scorer/(key+'.json')).exists():raise RuntimeError('Scorer is not ready: '+key)
                    with np.load(scores_path) as f:scores={k:f[k] for k in f.files}
                    matched,dyadic,meta=new_partitions(len(x),scores,[int(z.max()+1) for z in original])
                    if variant=='hierarchy_only':
                        partitions=matched;channels=pegfan_replacement_channels(base,matched)
                        np.savez_compressed(args.output/'partitions'/f'{dataset}_split{split}.npz',
                            **{f'{name}_level{i}':v for name,levels in [('original',original),('matched',matched),('dyadic',dyadic)] for i,v in enumerate(levels)})
                        diagnostics=[]
                        if dataset in ('film','chameleon','squirrel'):
                            for name,levels in [('original',original),('matched',matched),('dyadic',dyadic)]:
                                for level,z in enumerate(levels):
                                    row=partition_diagnostics(z,x.numpy(),scores['sources'],scores['targets'],scores['energy'],scores['hidden'])
                                    row.update(dataset=dataset,split=split,hierarchy=name,level=level,
                                               is_budget_cut=level==(meta['token_level'] if name=='dyadic' else next(j for j,v in enumerate(levels) if v.max()+1<=512)))
                                    diagnostics.append(row)
                            write_json(args.output/'diagnostics'/f'{dataset}_split{split}.json',diagnostics)
                        write_json(args.output/'partitions'/f'{dataset}_split{split}.json',meta)
                    else:partitions=original;channels=mass if variant=='projector_only' else base
                    write_json(args.output/'progress.json',dict(status='running',completed=len(results),expected=expected,dataset=dataset,split=split,variant=variant))
                    record=train(args,dataset,split,variant,loaded,channels,partitions)
                    if variant=='hierarchy_only':record.update(scorer_sha256=digest(scores_path),scorer_audit=json.loads((scorer/(key+'.json')).read_text()))
                    reference=next(r for r in old if r['dataset']==dataset and r['split']==split and r['model']=='pegfan')
                    if record['classifier_parameters']!=reference['parameters']:raise RuntimeError('Changed original classifier capacity')
                    if record['split_sha256']!=reference['split_sha256']:raise RuntimeError('Changed split')
                    with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
                    results.append(record);done.add((dataset,split,variant));summarize(args.output,results,old)
                    print('RESULT '+json.dumps(record),flush=True)
                    del scores,channels;gc.collect()
                del base,mass,loaded;gc.collect()
        if 'hierarchy_only' in args.variants:final_label_diagnostics(args)
        write_json(args.output/'progress.json',dict(status='complete',completed=len(results),expected=expected))
    except Exception as e:
        write_json(args.output/'progress.json',dict(status='failed',completed=len(results),expected=expected,error=repr(e)));raise

if __name__=='__main__':main()
