"""Fixed-protocol causal path experiments, prioritized on Film/Chameleon/Squirrel."""
import argparse,csv,gc,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ablation_ops import original_partitions,torch_transfers
from causal_ops import pool_lift_inputs,FeatureRoundTrip,FixedHierarchyChain,CHAIN_MODES,make_full_with_akx
from framelets_utils import get_spatial_framelets_list
from pegfan_baseline import prepare_pegfan_channels,make_pegfan
from process import full_load_data
from compare_pegfan import digest,write_json,synchronize

VARIANTS=['pool_global','pool_sheaf_global','svd64','learned64']+CHAIN_MODES+['full_no_akx','full_with_akx']


def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []

def reset(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()


def fixed_parts(dataset,split):
    p=Path('results/targeted_ablations/partitions')/f'{dataset}_split{split}.npz'
    with np.load(p) as f:
        keys=sorted([k for k in f.files if k.startswith('dyadic_level')],key=lambda k:int(k.split('level')[1]))
        parts=[f[k] for k in keys]
    cut=next(i for i,p in enumerate(parts) if p.max()+1<=512)
    return parts,cut,digest(p)


def summary(out,rs):
    rows=[]
    for d in dict.fromkeys(r['dataset'] for r in rs):
        for name in VARIANTS:
            group=[r for r in rs if r['dataset']==d and r['model']==name]
            if not group:continue
            x=np.array([r['test_accuracy']*100 for r in group])
            rows.append(dict(dataset=d,model=name,runs=len(group),mean_pct=float(x.mean()),std_pct=float(x.std(ddof=1)) if len(x)>1 else 0.,mean_train_seconds=float(np.mean([r['training_seconds'] for r in group]))))
    if rows:
        with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def reconstruction_audit(args,d,split,loaded,base,original):
    device=torch.device('cuda:0');nf,nc=loaded[-2:];classifier,_=make_pegfan(nf,len(base),64,nc,.5,device)
    primary=Path('results/causal_ablations/no_akx')
    path=primary/'checkpoints'/f'{d}_pegfan_control_split{split}_seed42.pt'
    classifier.load_state_dict(torch.load(path,map_location=device));classifier.eval();base_gpu=[h.to(device) for h in base]
    with np.load(Path('splits')/f'{d}_split_0.6_0.2_{split}.npz') as f:test=torch.tensor(f['test_mask'],dtype=torch.bool,device=device)
    labels=loaded[4].to(device);new,cut,_=fixed_parts(d,split)
    cases=[('original_equal',original,next(i for i,p in enumerate(original) if p.max()+1<=512),'equal'),('sheaf_mass',new,cut,'mass')]
    output=[]
    with torch.no_grad():
        expected=classifier(base_gpu,True)
        for name,parts,cut,norm in cases:
            ps=torch_transfers(parts,norm,device)[:cut]
            coarse,reconstructed,errors=pool_lift_inputs(base_gpu,ps)
            observed=classifier(reconstructed,True);coarse_out=classifier(coarse,True)
            row=dict(dataset=d,split=split,hierarchy=name,tokens=int(parts[cut].max()+1),checkpoint_sha256=digest(path),
                     channel_errors=errors,logit_max_abs_error=float((expected-observed).abs().max()),
                     changed_predictions=int((expected.argmax(1)!=observed.argmax(1)).sum()),
                     baseline_accuracy=float((expected[test].argmax(1)==labels[test]).double().mean()),
                     reconstructed_accuracy=float((observed[test].argmax(1)==labels[test]).double().mean()),
                     coarse_only_frozen_accuracy=float((coarse_out[test].argmax(1)==labels[test]).double().mean()))
            if max(x['max_abs_error'] for x in errors)>3e-6 or row['logit_max_abs_error']>1e-5:raise RuntimeError('Reconstruction failed')
            output.append(row)
    write_json(args.output/'reconstruction'/f'{d}_split{split}.json',output)
    del classifier,base_gpu;gc.collect();torch.cuda.empty_cache()


def run_trial(args,d,split,variant,loaded,base,original,basis):
    reset();device=torch.device('cuda:0');adj,_,_,x,y,_,_,_,nf,nc=loaded
    with np.load(Path('splits')/f'{d}_split_0.6_0.2_{split}.npz') as f:
        train,val,test=[torch.tensor(f[k],dtype=torch.bool,device=device) for k in ('train_mask','val_mask','test_mask')]
    if ((train&val)|(train&test)|(val&test)).any():raise ValueError('Overlapping masks')
    y=y.to(device);x=x.to(device);metadata={};channels=[h.to(device) for h in base]
    if variant in ['pool_global','pool_sheaf_global','svd64','learned64']:
        model,optimizer=make_pegfan(nf,len(channels),64,nc,.5,device)
        if variant.startswith('pool_'):
            if variant=='pool_global':parts=original;cut=next(i for i,p in enumerate(parts) if p.max()+1<=512);norm='equal'
            else:parts,cut,checksum=fixed_parts(d,split);norm='mass';metadata['fixed_partition_sha256']=checksum
            with torch.no_grad():channels,_,_=pool_lift_inputs(channels,torch_transfers(parts,norm,device)[:cut])
            metadata.update(tokens=int(parts[cut].max()+1),pooled_scope='all original PEGFAN inputs')
        else:
            v=basis.to(device)
            if variant=='svd64':
                with torch.no_grad():channels=[h@v@v.T for h in channels]
            else:
                model=FeatureRoundTrip(model,channels,v)
                optimizer.add_param_group(dict(params=list(model.encode.parameters())+list(model.decode.parameters()),lr=.001,weight_decay=.0005))
        forward=(lambda:model()) if variant=='learned64' else (lambda:model(channels,True))
    elif variant in CHAIN_MODES:
        parts,cut,checksum=fixed_parts(d,split)
        model=FixedHierarchyChain(nf,64,nc,parts,cut,variant).to(device).set_graph(adj)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005)
        forward=lambda:model(x);metadata.update(tokens=int(parts[cut].max()+1),fixed_partition_sha256=checksum)
    elif variant in ('full_no_akx','full_with_akx'):
        model,optimizer=make_full_with_akx(nf,64,nc,adj,channels[1:4],variant=='full_with_akx',device)
        forward=lambda:model(x);metadata['dynamic_hierarchy']=True
    else:raise ValueError(variant)
    start=time.perf_counter();history=[];best=float('inf');stale=0;best_state=None
    for epoch in range(1,args.epochs+1):
        model.train();optimizer.zero_grad();out=forward();loss=F.nll_loss(out[train],y[train])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
        loss.backward();optimizer.step();model.eval()
        with torch.no_grad():
            out=forward();vl=F.nll_loss(out[val],y[val]).item();va=(out[val].argmax(1)==y[val]).float().mean().item()
        if not np.isfinite(vl):raise RuntimeError('Nonfinite val loss')
        history.append(dict(epoch=epoch,train_loss=loss.item(),validation_loss=vl,validation_accuracy=va))
        if vl<best:
            best=vl;best_epoch=epoch;best_va=va;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if epoch==1 or epoch%100==0:print(f'{d} {variant} split={split} epoch={epoch} val={vl:.5f}',flush=True)
        if stale>=args.patience:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        out=forward();restored=F.nll_loss(out[val],y[val]).item()
        if not np.isclose(restored,best,atol=1e-5,rtol=1e-5):raise RuntimeError('Restored val loss mismatch')
        acc=(out[test].argmax(1)==y[test]).double().mean().item()
        if variant=='chain_transformer_global':
            check=model(x,haar_identity_check=True)
            metadata.update(haar_sum_max_abs_error=model.last_haar_error,haar_sum_logit_error=float((out-check).abs().max()),
                            haar_sum_changed_predictions=int((out.argmax(1)!=check.argmax(1)).sum()))
        if variant.startswith('full_'):
            metadata.update(tokens=model.hierarchy.num_tokens,node_counts=model.hierarchy.node_counts,
                            max_region_fraction=float(model.hierarchy.masses[model.hierarchy.token_level].max()/len(x)))
    synchronize(device);elapsed=time.perf_counter()-start
    key=f'{d}_{variant}_split{split}_seed42';torch.save(best_state,args.output/'checkpoints'/(key+'.pt'))
    write_json(args.output/'histories'/(key+'.json'),history)
    row=dict(dataset=d,model=variant,split=split,seed=42,test_accuracy=acc,validation_loss=best,validation_accuracy=best_va,
             best_epoch=best_epoch,epochs=epoch,training_seconds=elapsed,parameters=sum(p.numel() for p in model.parameters()),
             split_sha256=digest(Path('splits')/f'{d}_split_0.6_0.2_{split}.npz'),checkpoint_sha256=digest(args.output/'checkpoints'/(key+'.pt')),
             peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20,**metadata)
    del model,optimizer,channels,best_state,out,x;gc.collect();torch.cuda.empty_cache();return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',default=['film','chameleon','squirrel'])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS);p.add_argument('--splits',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--epochs',type=int,default=500);p.add_argument('--patience',type=int,default=50)
    p.add_argument('--output',type=Path,default=Path('results/causal_ablations/paths'));p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    for sub in ('histories','checkpoints','reconstruction','bases'): (args.output/sub).mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    protocol=dict(config=config,seed=42,sources={s:digest(s) for s in ['run_causal_paths.py','causal_ops.py','model.py','ablation_ops.py','framelets_utils.py','pegfan_baseline.py','process.py']},
                  data={d:{f:digest(Path('new_data')/d/f) for f in ['out1_node_feature_label.txt','out1_graph_edges.txt']} for d in args.datasets},
                  description='Stages use same frozen sheaf hierarchy except full_no_akx/full_with_akx, which use the full dynamic hierarchy. Pure global/detail stages zero the h skip; chain_full_skip restores it. Original FSGNN already has width 64 per channel; F-to-64-to-F bottleneck is inserted before FSGNN.')
    pp=args.output/'protocol.json'
    if pp.exists():
        if not args.resume or json.loads(pp.read_text())!=protocol:raise RuntimeError('Protocol changed or --resume required')
    else:write_json(pp,protocol)
    rp=args.output/'metrics.jsonl';rs=records(rp);done={(r['dataset'],r['split'],r['model']) for r in rs};expected=len(args.datasets)*len(args.splits)*len(args.variants)
    try:
        for variant in args.variants:
            for d in args.datasets:
                if all((d,s,variant) in done for s in args.splits):continue
                loaded=full_load_data(d,Path('splits')/f'{d}_split_0.6_0.2_0.npz',return_sparse=True)
                base=prepare_pegfan_channels(loaded[3],loaded[1],loaded[2],d)
                original=original_partitions(get_spatial_framelets_list(None,d,8)[0]);basis=None
                if variant in ('svd64','learned64'):
                    path=args.output/'bases'/(d+'.pt')
                    if path.exists():basis=torch.load(path)
                    else:
                        # Exact leading right-singular subspace, no labels or test selection.
                        _,singular,vh=torch.linalg.svd(loaded[3],full_matrices=False);basis=vh[:64].T.contiguous()
                        torch.save(basis,path);write_json(args.output/'bases'/(d+'.json'),dict(rank=64,energy_retained=float(singular[:64].square().sum()/singular.square().sum()),basis_sha256=digest(path)))
                for split in args.splits:
                    if (d,split,variant) in done:continue
                    if variant=='pool_global' and not (args.output/'reconstruction'/f'{d}_split{split}.json').exists():
                        reconstruction_audit(args,d,split,loaded,base,original)
                    write_json(args.output/'progress.json',dict(status='running',completed=len(rs),expected=expected,dataset=d,split=split,variant=variant))
                    row=run_trial(args,d,split,variant,loaded,base,original,basis)
                    with rp.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
                    rs.append(row);done.add((d,split,variant));summary(args.output,rs);print('RESULT '+json.dumps(row),flush=True)
                del loaded,base,basis;gc.collect()
        write_json(args.output/'progress.json',dict(status='complete',completed=len(rs),expected=expected))
    except Exception as e:
        write_json(args.output/'progress.json',dict(status='failed',completed=len(rs),expected=expected,error=repr(e)));raise
if __name__=='__main__':main()
