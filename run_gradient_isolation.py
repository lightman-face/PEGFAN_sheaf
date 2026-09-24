"""Exactly 2 datasets x 3 splits x 6 variants: nonlocal prediction vs co-training."""
import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from gradient_isolation_ops import (VARIANTS, make_matrix_model, raw_logits,
    isolated_losses, freeze_backbone, contribution_metrics)
from run_augmentation import records,tensor_digest,summary
from compare_pegfan import digest,write_json,synchronize
from pegfan_baseline import prepare_pegfan_channels
from process import full_load_data


def state_copy(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fixed_parts(dataset,split):
    path=Path('results/gradient_isolation/hierarchies')/f'{dataset}_split{split}.npz'
    with np.load(path) as f:
        keys=sorted([k for k in f.files if k.startswith('level')],key=lambda k:int(k[5:]))
        return [f[k] for k in keys],int(f['cut']),digest(path)


def trial(args,d,split,variant,loaded,base,completed):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    device=torch.device('cuda:0');mode,_=VARIANTS[variant]
    adj,_,_,x,y,_,_,_,nf,nc=loaded;x,y=x.to(device),y.to(device)
    channels=[h.to(device) for h in base]
    with np.load(Path('splits')/f'{d}_split_0.6_0.2_{split}.npz') as f:
        train,val,test=[torch.as_tensor(f[k],dtype=torch.bool,device=device) for k in ('train_mask','val_mask','test_mask')]
    assert torch.all(train.int()+val.int()+test.int()==1)
    parts,cut,partition_sha=fixed_parts(d,split)
    model,optimizer=make_matrix_model(nf,len(channels),nc,variant,adj,parts,cut,device)
    backbone=model if mode=='baseline' else model.backbone
    initial_hash=tensor_digest(backbone.state_dict())
    model.eval()
    with torch.no_grad():
        local=raw_logits(backbone,channels)
        expected=F.log_softmax(local,dim=-1)
        actual=backbone(channels,True) if mode=='baseline' else model(channels,x)
        assert torch.equal(expected,actual)
    reference=None;ref_state=None;ref_logits=None;ref_history=None
    local_stop_epoch=None;reference_sha=None
    if mode in ('stopgrad','frozen'):
        reference=next(r for r in completed if r['dataset']==d and r['split']==split and r['model']=='pegfan')
        key=f'{d}_pegfan_split{split}_seed42'
        ref_path=args.output/'checkpoints'/(key+'.pt');reference_sha=digest(ref_path)
        assert reference_sha==reference['checkpoint_sha256']
        ref_state=torch.load(ref_path,map_location='cpu')
        ref_history=json.loads((args.output/'histories'/(key+'.json')).read_text())
        local_stop_epoch=reference['best_epoch']
        probe=copy.deepcopy(backbone).eval();probe.load_state_dict(ref_state)
        with torch.no_grad():ref_logits=raw_logits(probe,channels).detach()
        assert np.isclose(F.nll_loss(F.log_softmax(ref_logits,dim=-1)[val],y[val]).item(),reference['validation_loss'],atol=1e-6)
        del probe
        if mode=='frozen':
            backbone.load_state_dict(ref_state);freeze_backbone(backbone)
    # Include the intact validation-selected baseline as an explicitly labelled
    # gate-zero candidate for isolated/frozen models. No test labels select it.
    best=float('inf');best_epoch=None;best_state=None;best_va=None;stale=0
    selection_source='trained_epoch';candidate=None
    if reference is not None:
        best=reference['validation_loss'];best_epoch=0;best_va=reference['validation_accuracy']
        best_state=state_copy(model)
        for k,v in ref_state.items():best_state['backbone.'+k]=v.clone()
        best_state['alpha'].zero_();selection_source='baseline_reference'
        candidate=dict(kind='baseline_reference',epoch=0,validation_loss=best,source_checkpoint_sha256=reference_sha)
    start=time.perf_counter();history=[];warmup_checks=0;freeze_checked=False
    for epoch in range(1,args.epochs+1):
        local_active=mode in ('baseline','joint') or (mode=='stopgrad' and epoch<=local_stop_epoch)
        model.train()
        if not local_active:freeze_backbone(backbone)
        optimizer.zero_grad(set_to_none=True)
        local=raw_logits(backbone,channels) if local_active else ref_logits
        if mode=='baseline':
            out=F.log_softmax(local,dim=-1);loss=F.nll_loss(out[train],y[train])
        else:
            global_logits=model.branch(x)
            loss,out=isolated_losses(local,global_logits,model.alpha,y,train,mode,local_active)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        train_loss=F.nll_loss(out[train],y[train]).item()
        loss.backward();gate_grad=None if mode=='baseline' else float(model.alpha.grad)
        optimizer.step();local_hash=tensor_digest(backbone.state_dict())
        if mode=='stopgrad' and local_active:
            expected_hash=ref_history[epoch-1]['backbone_sha256']
            if local_hash!=expected_hash:raise RuntimeError(f'Local trajectory changed at {d}/{split}/{epoch}')
            warmup_checks+=1
            if epoch==local_stop_epoch:
                assert local_hash==reference['backbone_selected_sha256']
                freeze_backbone(backbone);freeze_checked=True
        if mode=='frozen' or (mode=='stopgrad' and epoch>=local_stop_epoch):
            assert local_hash==reference['backbone_selected_sha256']
        model.eval()
        with torch.no_grad():
            eval_local=ref_logits if reference is not None and (mode=='frozen' or epoch>=local_stop_epoch) else raw_logits(backbone,channels)
            if mode=='baseline':eval_out=F.log_softmax(eval_local,dim=-1)
            else:
                eval_global=model.branch(x)
                eval_out=F.log_softmax(eval_local+model.alpha*eval_global,dim=-1)
            vl=F.nll_loss(eval_out[val],y[val]).item();va=(eval_out[val].argmax(1)==y[val]).float().mean().item()
            if not np.isfinite(vl):raise RuntimeError('Nonfinite validation loss')
            entry=dict(epoch=epoch,train_loss=train_loss,optimization_loss=float(loss),validation_loss=vl,
                validation_accuracy=va,backbone_sha256=local_hash,local_active=local_active,
                selection_eligible=mode!='stopgrad' or epoch>=local_stop_epoch)
            if mode!='baseline':
                entry.update(alpha=float(model.alpha),alpha_gradient=gate_grad,
                    contribution_all=contribution_metrics(eval_local,model.alpha*eval_global),
                    contribution_validation=contribution_metrics(eval_local[val],model.alpha*eval_global[val]))
        history.append(entry)
        if entry['selection_eligible']:
            if vl<best:
                best=vl;best_va=va;best_epoch=epoch;best_state=state_copy(model);stale=0;selection_source='trained_epoch'
            else:stale+=1
        if epoch==1 or epoch%100==0:
            print(f'{d} {variant} split={split} epoch={epoch} val={vl:.5f}'+
                  (f' alpha={float(model.alpha):+.5f}' if mode!='baseline' else ''),flush=True)
        if stale>=args.patience:break
    synchronize(device);seconds=time.perf_counter()-start
    if mode=='stopgrad' and not freeze_checked:raise RuntimeError('Stopgrad warm-up did not reach reference checkpoint')
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        local=raw_logits(backbone,channels);off=F.log_softmax(local,dim=-1)
        if mode=='baseline':out=off;gate=None
        else:
            global_logits=model.branch(x);increment=model.alpha*global_logits
            out=F.log_softmax(local+increment,dim=-1)
            gate={name:contribution_metrics(local[mask],increment[mask]) for name,mask in (
                ('all',torch.ones(len(x),dtype=torch.bool,device=device)),('train',train),('validation',val),('test',test))}
        restored=F.nll_loss(out[val],y[val]).item()
        assert np.isclose(restored,best,atol=1e-5,rtol=1e-5),(restored,best)
        acc=(out[test].argmax(1)==y[test]).double().mean().item()
        off_acc=(off[test].argmax(1)==y[test]).double().mean().item()
        selected_hash=tensor_digest(backbone.state_dict())
        if reference is not None:
            assert selected_hash==reference['backbone_selected_sha256']
            assert torch.equal(local,ref_logits)
            assert off_acc==reference['test_accuracy']
    key=f'{d}_{variant}_split{split}_seed42';ckpt=args.output/'checkpoints'/(key+'.pt')
    torch.save(best_state,ckpt);write_json(args.output/'histories'/(key+'.json'),history)
    if gate is not None:
        write_json(args.output/'gates'/(key+'.json'),dict(dataset=d,split=split,model=variant,alpha=float(model.alpha),
            fusion_space='raw class logits',scopes=gate,gate_on_accuracy=acc,gate_off_accuracy=off_acc,
            gate_on_validation_loss=best,gate_off_validation_loss=F.nll_loss(off[val],y[val]).item(),
            checkpoint_sha256=digest(ckpt)))
    row=dict(dataset=d,model=variant,mode=mode,split=split,seed=42,test_accuracy=acc,validation_loss=best,
        validation_accuracy=best_va,best_epoch=best_epoch,epochs=epoch,selection_source=selection_source,
        baseline_candidate=candidate,local_stop_epoch=local_stop_epoch,warmup_parameter_hash_checks=warmup_checks,
        backbone_initial_sha256=initial_hash,backbone_selected_sha256=selected_hash,
        backbone_reference_checkpoint_sha256=reference_sha,initial_output_max_error=0.,
        backbone_parameters=sum(p.numel() for p in backbone.parameters()),parameters=sum(p.numel() for p in model.parameters()),
        local_input_width=nf,local_channels=len(channels),gate_off_accuracy=off_acc,
        gate_changed_test_predictions=int((out[test].argmax(1)!=off[test].argmax(1)).sum()),
        alpha=None if mode=='baseline' else float(model.alpha),
        gate_metrics_all=None if gate is None else gate['all'],gate_metrics_test=None if gate is None else gate['test'],
        tokens=int(parts[cut].max()+1),node_counts=[int(p.max()+1) for p in parts],
        partition_sha256=partition_sha,checkpoint_sha256=digest(ckpt),
        split_sha256=digest(Path('splits')/f'{d}_split_0.6_0.2_{split}.npz'),training_seconds=seconds,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20)
    del model,optimizer,backbone,channels,best_state,x,y,out,local,ref_logits
    gc.collect();torch.cuda.empty_cache();return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',default=['chameleon','squirrel'])
    p.add_argument('--splits',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument('--epochs',type=int,default=500);p.add_argument('--patience',type=int,default=50)
    p.add_argument('--output',type=Path,default=Path('results/gradient_isolation'))
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    for sub in ('checkpoints','histories','gates'):(args.output/sub).mkdir(parents=True,exist_ok=True)
    sources=['run_gradient_isolation.py','gradient_isolation_ops.py','augmentation_ops.py','run_augmentation.py',
        'ablation_ops.py','pegfan_baseline.py','model.py','framelets_utils.py','sheaf_coarsening.py','process.py','utils.py','compare_pegfan.py']
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    manifest=Path('results/gradient_isolation/hierarchies/manifest.json')
    protocol=dict(config=config,seed=42,torch=torch.__version__,numpy=np.__version__,
        sources={s:digest(s) for s in sources},hierarchy_manifest_sha256=digest(manifest),
        data={d:{f:digest(Path('new_data')/d/f) for f in ['out1_node_feature_label.txt','out1_graph_edges.txt']} for d in args.datasets},
        framelet_caches={d:digest(Path('framelets/8')/(d+'.pickle')) for d in args.datasets},
        description='36-run fixed common-tree matrix. Haar retains direct Z_T plus band residual. Stopgrad: standalone local CE and detached-local fused CE; local follows audited baseline trajectory to its validation-selected epoch and freezes. Branch selection only after that milestone, with gate-zero baseline candidate. Frozen begins from same baseline checkpoint. Metrics in raw and class-centered logit space.')
    pp=args.output/'protocol.json'
    if pp.exists():
        if not args.resume or json.loads(pp.read_text())!=protocol:raise RuntimeError('Protocol changed or --resume missing')
    else:write_json(pp,protocol)
    path=args.output/'metrics.jsonl';rs=records(path);done={(r['dataset'],r['split'],r['model']) for r in rs}
    expected=len(args.datasets)*len(args.splits)*len(args.variants)
    try:
        for variant in args.variants:
            for d in args.datasets:
                if all((d,s,variant) in done for s in args.splits):continue
                loaded=full_load_data(d,Path('splits')/f'{d}_split_0.6_0.2_0.npz',return_sparse=True)
                base=prepare_pegfan_channels(loaded[3],loaded[1],loaded[2],d)
                for split in args.splits:
                    if (d,split,variant) in done:continue
                    write_json(args.output/'progress.json',dict(status='running',completed=len(rs),expected=expected,dataset=d,split=split,variant=variant))
                    r=trial(args,d,split,variant,loaded,base,rs)
                    with path.open('a') as f:f.write(json.dumps(r)+'\n');f.flush()
                    rs.append(r);done.add((d,split,variant));summary(args.output,rs);print('RESULT '+json.dumps(r),flush=True)
                del loaded,base;gc.collect()
        write_json(args.output/'progress.json',dict(status='complete',completed=len(rs),expected=expected))
    except Exception as e:
        write_json(args.output/'progress.json',dict(status='failed',completed=len(rs),expected=expected,error=repr(e)));raise
if __name__=='__main__':main()
