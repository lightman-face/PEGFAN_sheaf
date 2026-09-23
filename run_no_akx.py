"""Priority causal intervention: null only the three explicit propagation inputs."""
import argparse,json,csv,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from process import full_load_data
from pegfan_baseline import prepare_pegfan_channels
from framelets_utils import get_spatial_framelets_list
from ablation_ops import original_partitions
from compare_pegfan import write_json,digest
from run_targeted_ablations import train


def summarize(out,records):
    rows=[];paired=[]
    for d in dict.fromkeys(r['dataset'] for r in records):
        control={r['split']:r['test_accuracy'] for r in records if r['dataset']==d and r['model']=='pegfan_control'}
        for name in ('pegfan_control','no_akx'):
            group=[r for r in records if r['dataset']==d and r['model']==name]
            if not group:continue
            x=np.array([r['test_accuracy']*100 for r in group]);delta=[100*(r['test_accuracy']-control[r['split']]) for r in group if r['split'] in control]
            rows.append(dict(dataset=d,model=name,runs=len(group),mean_pct=float(x.mean()),std_pct=float(x.std(ddof=1)) if len(x)>1 else 0.,
                             delta_pp=float(np.mean(delta)) if delta else None,wins=sum(v>1e-8 for v in delta),ties=sum(abs(v)<=1e-8 for v in delta),losses=sum(v< -1e-8 for v in delta)))
            if name=='no_akx':paired.extend(dict(dataset=d,split=r['split'],baseline_pct=control[r['split']]*100,no_akx_pct=r['test_accuracy']*100,delta_pp=100*(r['test_accuracy']-control[r['split']])) for r in group if r['split'] in control)
    for name,data in [('summary.csv',rows),('paired_splits.csv',paired)]:
        if data:
            with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['film','chameleon','squirrel','texas','wisconsin'])
    p.add_argument('--output',type=Path,default=Path('results/causal_ablations/no_akx'));p.add_argument('--resume',action='store_true');args=p.parse_args()
    args.epochs=500;args.patience=50
    for sub in ('histories','checkpoints'): (args.output/sub).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    protocol=dict(datasets=args.datasets,splits=list(range(10)),seed=42,epochs=500,patience=50,
                  intervention='Only input channels 1,2,3 set to zero; X and original cached framelet projections of AX unchanged. Biases, channel weights and all classifier parameters retained.',
                  sources={s:digest(s) for s in ['run_no_akx.py','run_targeted_ablations.py','ablation_ops.py','pegfan_baseline.py','model.py','process.py']},
                  data={d:{f:digest(Path('new_data')/d/f) for f in ['out1_node_feature_label.txt','out1_graph_edges.txt']} for d in args.datasets})
    path=args.output/'protocol.json'
    if path.exists():
        if not args.resume or json.loads(path.read_text())!=protocol:raise RuntimeError('Protocol differs or --resume required')
    else:write_json(path,protocol)
    result=args.output/'metrics.jsonl';records=[json.loads(s) for s in result.read_text().splitlines()] if result.exists() else []
    done={(r['dataset'],r['split'],r['model']) for r in records}
    try:
        for d in args.datasets:
            loaded=full_load_data(d,Path('splits')/f'{d}_split_0.6_0.2_0.npz',return_sparse=True)
            base=prepare_pegfan_channels(loaded[3],loaded[1],loaded[2],d)
            ablated=[base[0]]+[torch.zeros_like(x) for x in base[1:4]]+base[4:]
            parts=original_partitions(get_spatial_framelets_list(None,d,8)[0])
            for split in range(10):
                for name,channels in [('pegfan_control',base),('no_akx',ablated)]:
                    if (d,split,name) in done:continue
                    write_json(args.output/'progress.json',dict(status='running',dataset=d,split=split,variant=name,completed=len(records),expected=20*len(args.datasets)))
                    r=train(args,d,split,name,loaded,channels,parts)
                    with result.open('a') as f:f.write(json.dumps(r)+'\n');f.flush()
                    records.append(r);done.add((d,split,name));summarize(args.output,records);print('RESULT '+json.dumps(r),flush=True)
            del base,ablated,loaded;gc.collect()
        write_json(args.output/'progress.json',dict(status='complete',completed=len(records),expected=20*len(args.datasets)))
    except Exception as e:
        write_json(args.output/'progress.json',dict(status='failed',error=repr(e),completed=len(records)));raise
if __name__=='__main__':main()
