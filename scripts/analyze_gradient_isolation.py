"""Audit the fixed 36-run gradient matrix and report its two decisive contrasts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradient_isolation_ops import VARIANTS
from run_augmentation import records,tensor_digest,summary
from compare_pegfan import digest,write_json

ROOT=Path('results/gradient_isolation')
DATASETS=['chameleon','squirrel']
MODELS=list(VARIANTS)
LABELS=['PEGFAN','+ T128','+ T128 + Haar','stop-grad T128','stop-grad T128 + Haar','frozen PEGFAN + T128 + Haar']


def csv_write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    torch.set_num_threads(1)
    progress=json.loads((ROOT/'progress.json').read_text());assert progress['status']=='complete' and progress['completed']==36
    protocol=json.loads((ROOT/'protocol.json').read_text());cfg=protocol['config']
    assert cfg['datasets']==DATASETS and cfg['splits']==[0,1,2] and cfg['variants']==MODELS
    for source,sha in protocol['sources'].items():assert digest(source)==sha,source
    for d,files in protocol['data'].items():
        for name,sha in files.items():assert digest(Path('new_data')/d/name)==sha
    for d,sha in protocol['framelet_caches'].items():assert digest(Path('framelets/8')/(d+'.pickle'))==sha
    assert digest(ROOT/'hierarchies/manifest.json')==protocol['hierarchy_manifest_sha256']
    manifest=json.loads((ROOT/'hierarchies/manifest.json').read_text())
    hierarchy={(r['dataset'],r['split']):r for r in manifest['entries']};assert len(hierarchy)==6
    for r in hierarchy.values():
        assert digest(r['path'])==r['sha256']
        assert digest(r['source_checkpoint'])==r['source_checkpoint_sha256']
    rs=records(ROOT/'metrics.jsonl');lookup={(r['dataset'],r['model'],r['split']):r for r in rs}
    assert len(rs)==len(lookup)==36
    old={(r['dataset'],r['split']):r for r in records(Path('results/architecture_augmentation/metrics.jsonl')) if r['model']=='pegfan'}
    histories={};states={};paired=[];gate_rows=[];warmup_checks=0
    for r in rs:
        d,m,s=r['dataset'],r['model'],r['split'];base=lookup[d,'pegfan',s]
        key=f'{d}_{m}_split{s}_seed42'
        path=ROOT/'checkpoints'/(key+'.pt');assert digest(path)==r['checkpoint_sha256']
        state=torch.load(path,map_location='cpu');local=state if m=='pegfan' else {k[9:]:v for k,v in state.items() if k.startswith('backbone.')}
        assert tensor_digest(local)==r['backbone_selected_sha256'];states[d,m,s]=local
        assert digest(Path('splits')/f'{d}_split_0.6_0.2_{s}.npz')==r['split_sha256']
        assert r['partition_sha256']==hierarchy[d,s]['sha256'] and r['node_counts']==hierarchy[d,s]['node_counts']
        assert r['tokens']<=512 and r['initial_output_max_error']==0
        for k in ('backbone_initial_sha256','backbone_parameters','local_input_width','local_channels'):assert r[k]==base[k]
        hist=json.loads((ROOT/'histories'/(key+'.json')).read_text());histories[d,m,s]=hist;assert len(hist)==r['epochs']
        candidate=r['baseline_candidate'];best=candidate['validation_loss'] if candidate else float('inf');epoch=0 if candidate else None;stale=0
        for e in hist:
            eligible=r['mode']!='stopgrad' or e['epoch']>=r['local_stop_epoch']
            assert e['selection_eligible']==eligible
            if eligible:
                if e['validation_loss']<best:best=e['validation_loss'];epoch=e['epoch'];stale=0
                else:stale+=1
        assert best==r['validation_loss'] and epoch==r['best_epoch']
        assert r['epochs']==cfg['epochs'] or stale==cfg['patience']
        assert r['selection_source']==('baseline_reference' if epoch==0 else 'trained_epoch')
        if m=='pegfan':
            assert r['test_accuracy']==old[d,s]['test_accuracy'] and r['best_epoch']==old[d,s]['best_epoch']
            previous=Path('results/architecture_augmentation/checkpoints')/(key+'.pt')
            old_state=torch.load(previous,map_location='cpu')
            assert all(torch.equal(local[k],v) for k,v in old_state.items())
        else:
            gate=json.loads((ROOT/'gates'/(key+'.json')).read_text());assert gate['checkpoint_sha256']==r['checkpoint_sha256']
            assert gate['scopes']['all']==r['gate_metrics_all'] and gate['scopes']['test']==r['gate_metrics_test']
            assert float(state['alpha'])==r['alpha']
            for scope,metrics in gate['scopes'].items():
                for prefix in ('raw','centered'):
                    ratio,cosine=metrics[prefix+'_rg'],metrics[prefix+'_cosine']
                    assert ratio is not None and np.isfinite(ratio) and ratio>=0
                    if cosine is not None:assert np.isfinite(cosine) and abs(cosine)<=1+1e-10
                    if r['alpha']==0:assert ratio==0 and cosine is None
                gate_rows.append(dict(dataset=d,model=m,split=s,scope=scope,alpha=r['alpha'],
                    gate_on_accuracy_pct=100*r['test_accuracy'],gate_off_accuracy_pct=100*r['gate_off_accuracy'],
                    gate_delta_pp=100*(r['test_accuracy']-r['gate_off_accuracy']),selection_source=r['selection_source'],**metrics))
        paired.append(dict(dataset=d,model=m,split=s,accuracy_pct=100*r['test_accuracy'],
            pegfan_pct=100*base['test_accuracy'],delta_pp=100*(r['test_accuracy']-base['test_accuracy']),
            gate_off_pct=100*r['gate_off_accuracy'],gate_on_minus_off_pp=100*(r['test_accuracy']-r['gate_off_accuracy']),
            best_epoch=r['best_epoch'],selection_source=r['selection_source']))
    for r in rs:
        if r['mode'] not in ('stopgrad','frozen'):continue
        d,m,s=r['dataset'],r['model'],r['split'];base=lookup[d,'pegfan',s]
        assert r['backbone_reference_checkpoint_sha256']==base['checkpoint_sha256']
        assert r['backbone_selected_sha256']==base['backbone_selected_sha256'] and r['gate_off_accuracy']==base['test_accuracy']
        assert all(torch.equal(v,states[d,'pegfan',s][k]) for k,v in states[d,m,s].items())
        hist=histories[d,m,s]
        if r['mode']=='stopgrad':
            assert r['local_stop_epoch']==base['best_epoch'] and r['warmup_parameter_hash_checks']==base['best_epoch']
            warmup_checks+=r['warmup_parameter_hash_checks']
            for e in hist[:base['best_epoch']]:assert e['backbone_sha256']==histories[d,'pegfan',s][e['epoch']-1]['backbone_sha256']
        for e in hist:
            if r['mode']=='frozen' or e['epoch']>=base['best_epoch']:assert e['backbone_sha256']==base['backbone_selected_sha256']
    candidate_rows=[]
    for r in rs:
        if r['baseline_candidate'] is None:continue
        h=histories[r['dataset'],r['model'],r['split']]
        trained=min((e for e in h if e['selection_eligible']),key=lambda e:e['validation_loss'])
        candidate_rows.append(dict(dataset=r['dataset'],model=r['model'],split=r['split'],
            reference_validation_loss=r['baseline_candidate']['validation_loss'],
            best_trained_epoch=trained['epoch'],best_trained_validation_loss=trained['validation_loss'],
            trained_minus_reference_nll=trained['validation_loss']-r['baseline_candidate']['validation_loss'],
            trained_alpha=trained['alpha'],selected_source=r['selection_source']))
    csv_write(ROOT/'candidate_comparison.csv',candidate_rows)
    summary(ROOT,rs);csv_write(ROOT/'paired_splits.csv',paired);csv_write(ROOT/'gate_metrics.csv',gate_rows)
    with (ROOT/'summary.csv').open() as f:stats={(r['dataset'],r['model']):r for r in csv.DictReader(f)}
    def cell(d,m):
        r=stats[d,m];return f"{float(r['mean_pct']):.2f} ± {float(r['std_pct']):.2f}"
    contrasts=[]
    for d in DATASETS:
        for name,a,b in [('T versus PEGFAN','t128','pegfan'),('Haar increment joint','t128_haar','t128'),
                         ('stopgrad T versus PEGFAN','stopgrad_t128','pegfan'),
                         ('Haar increment stopgrad','stopgrad_t128_haar','stopgrad_t128'),
                         ('frozen Haar versus PEGFAN','frozen_t128_haar','pegfan')]:
            ds=np.array([100*(lookup[d,a,s]['test_accuracy']-lookup[d,b,s]['test_accuracy']) for s in [0,1,2]])
            contrasts.append(dict(dataset=d,contrast=name,mean_delta_pp=float(ds.mean()),
                split0_pp=float(ds[0]),split1_pp=float(ds[1]),split2_pp=float(ds[2]),
                wins=int(sum(ds>1e-8)),ties=int(sum(abs(ds)<=1e-8)),losses=int(sum(ds< -1e-8))))
    csv_write(ROOT/'contrasts.csv',contrasts)
    ordering={}
    for d in DATASETS:
        avg=lambda m:float(stats[d,m]['mean_pct'])
        ordering[d]=dict(mean_strict_order=avg('stopgrad_t128_haar')>avg('stopgrad_t128')>avg('pegfan'),
            per_split_strict_order=[lookup[d,'stopgrad_t128_haar',s]['test_accuracy']>lookup[d,'stopgrad_t128',s]['test_accuracy']>lookup[d,'pegfan',s]['test_accuracy'] for s in [0,1,2]])
    audit=dict(status='passed',training_trials=36,source_data_cache_partition_split_checkpoint_hashes=True,
        baseline_weights_match_previous_six=True,validation_selection_and_early_stopping=True,
        common_partition_per_split=True,stopgrad_warmup_parameter_hash_checks=warmup_checks,
        isolated_final_backbone_matches_baseline=18,isolated_gate_off_matches_baseline=18,
        frozen_all_epoch_parameter_hashes=True,zero_gate_initial_output_error=0.,
        baseline_reference_selections=sum(r['selection_source']=='baseline_reference' for r in rs),
        requested_ordering=ordering)
    write_json(ROOT/'audit.json',audit)
    lines=['# 36次梯度隔离矩阵：独立预测价值与post-T Haar增量','',
        '完成 Chameleon/Squirrel × split 0–2 × 六种配置，共36次正式训练。seed=42，500 epoch上限，patience=50，validation NLL选模；没有追加模型或调参。准确率是三个相同split的均值 ± 样本标准差（%）。',
        '[完整协议](../../docs/gradient_isolation_protocol.md)。同split所有branch共用前轮已验证的固定Sheaf partition；原PEGFAN完整保留。128维只作用于新增分支。Haar组使用 Z_T + learned Haar band fusion，明确保留直接Transformer信号。',
        '', '本轮没有出现所要求的严格顺序。12个stop-grad run均由validation选回零gate baseline；不是训练gate始终为零。frozen Haar有约0.13–0.15 pp的小幅正信号，但尚不足以支撑显著的全局预测增量，也没有证明Haar优于Transformer-alone。', '', '## 结果矩阵','', '| Variant | Chameleon | Squirrel |','|---|---:|---:|']
    for m,label in zip(MODELS,LABELS):lines.append(f'| {label} | {cell("chameleon",m)} | {cell("squirrel",m)} |')
    lines+=['','## 两个决定性对照','',
        '| 数据集 | 对照 | 平均差值 pp | split 0 / 1 / 2 差值 pp | 胜/平/负 |','|---|---|---:|---|---|']
    for r in contrasts:
        lines.append(f"| {r['dataset']} | {r['contrast']} | {r['mean_delta_pp']:+.3f} | {r['split0_pp']:+.3f} / {r['split1_pp']:+.3f} / {r['split2_pp']:+.3f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    for d,o in ordering.items():
        lines+=['',f"{d}：要求的均值顺序 `stop-grad(T128+Haar) > stop-grad(T128) > PEGFAN` {'成立' if o['mean_strict_order'] else '不成立'}；三个split分别为 {o['per_split_strict_order']}。"]
    lines+=['','## stop-grad 的实际含义与核验','',
        '当前两分支只共享原始X，输入detach本身无效。stop-grad用独立CE(L,y)更新PEGFAN，用CE(detach(L)+alpha G,y)更新branch。PEGFAN逐epoch沿独立baseline轨迹训练，到其自己选择的最佳epoch后固定；branch checkpoint只在此后选择。frozen组则从一开始冻结同一baseline。',
        f"累计核验 {warmup_checks} 个stop-grad warm-up主干参数hash；全部与baseline逐epoch一致。12个stop-grad与6个frozen的最终主干参数及gate-off logits均严格等于相应baseline。因此这18个run的gate-on增量没有混入不同主干或不同local选模epoch。",
        f"隔离组保留alpha=0的完整baseline作为validation候选，本轮选回该候选 {audit['baseline_reference_selections']} 次。零gate只表示当前训练配置未被validation选出增量，不证明所有全局分支都无效。",
        '', '## Gate的范数与方向','',
        '实际融合空间是class logits：H_PEG=L。R_g为||alpha G||_F/||L||_F；cosine为两个矩阵展平后的Frobenius夹角。下面为全图指标的逐run均值；cosine仅平均有定义的run。centered版本对每个节点沿class维去均值，消除softmax不敏感的公共偏置。逐节点平均cosine另保存在CSV，未与Frobenius cosine混用。',
        '', '| 数据集 | Variant | mean abs(alpha) | raw R_g | raw cosine | centered R_g | centered cosine | 有定义cosine数 | gate-on − off pp |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def mean_optional(values):
        values=[v for v in values if v is not None];return f'{np.mean(values):+.4f}' if values else 'undefined'
    for d in DATASETS:
        for m,label in zip(MODELS[1:],LABELS[1:]):
            group=[g for g in gate_rows if g['dataset']==d and g['model']==m and g['scope']=='all']
            lines.append(f"| {d} | {label} | {np.mean([abs(g['alpha']) for g in group]):.5f} | {mean_optional([g['raw_rg'] for g in group])} | {mean_optional([g['raw_cosine'] for g in group])} | {mean_optional([g['centered_rg'] for g in group])} | {mean_optional([g['centered_cosine'] for g in group])} | {sum(g['raw_cosine'] is not None for g in group)}/3 | {np.mean([g['gate_delta_pp'] for g in group]):+.3f} |")
    opposing=[g for g in gate_rows if g['scope']=='test' and g['centered_cosine'] is not None and g['centered_cosine']<0 and g['gate_delta_pp']< -1e-8]
    lines+=['',f"测试scope中，centered cosine为负且gate-on accuracy下降的run共有 {len(opposing)} 个（共30个branch run）。这只是方向与结果的联合诊断；负cosine本身不含labels，不独立证明有害。joint组gate-off仍是联合训练的主干，只有stop-grad/frozen组能直接等同于独立baseline。",
        '所有18个非零gate run的测试centered cosine均为正；本轮未观察到整体负cosine的抵消模式。较大的贡献范数仍可能降低accuracy，不能仅用正cosine判断是否有益。frozen组raw R_g均值约1.74% / 2.27%，对应增益很小。', '', '当前证据更适合把Transformer/post-T Haar保留为可选研究分支；不宜作为已经成立的主贡献。这个判断限于本轮固定结构和训练协议，不是对所有global attention结构的否定。', '', '## 实验范围与交付','',
        '固定树来源于前轮同split的validation-selected Haar128模型，带有预训练成本及该参考模型的偏向；本轮不宣称Sheaf score带来了增益。树的生成规则没有改动，但本轮固定实际partition。Haar融合也显式保留了Z_T，因此不拿前轮动态树/仅band fusion的数字代替本轮对照。',
        '本轮检验当前固定tree、128维分支与训练配置下的独立预测价值；三个重叠split、一个seed不能证明普遍性。没有frozen T-only行，不能单凭frozen Haar的表现断言严格冻结时Haar优于Transformer-alone；Haar增量主要由两个stop-grad配置直接比较。',
        '52项回归测试及12次两epoch GPU smoke test通过；smoke与正式36次训练分目录保存。六个baseline的完整参数、准确率与最佳epoch均复现前轮同split参考。所有checkpoint、源码、数据、原Haar cache、共同partition、split、验证选模和early stopping均通过审计。',
        '`paired_splits.csv`：逐split结果；`contrasts.csv`：五组配对对照；`gate_metrics.csv`：全图/train/validation/test的raw和centered R_g/cosine；`candidate_comparison.csv`：训练候选与完整baseline的验证损失差，供核查零gate选择；`histories/`：逐epoch损失、gate指标与主干hash；`hierarchies/manifest.json`：共同分区来源。','']
    (ROOT/'analysis.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    names=['Joint T','Joint T+Haar','SG T','SG T+Haar','Frozen T+Haar']
    for ax,d in zip(axes,DATASETS):
        for split in [0,1,2]:
            delta=[100*(lookup[d,m,split]['test_accuracy']-lookup[d,'pegfan',split]['test_accuracy']) for m in MODELS[1:]]
            ax.plot(range(5),delta,'o-',alpha=.7,label=f'Split {split}')
        mean=[float(stats[d,m]['delta_pp']) for m in MODELS[1:]]
        ax.plot(range(5),mean,'kD--',linewidth=2,label='Mean');ax.axhline(0,color='gray',linewidth=1)
        ax.set_xticks(range(5));ax.set_xticklabels(names,rotation=15);ax.set_ylabel('Accuracy delta vs PEGFAN (pp)');ax.set_title(d.capitalize());ax.legend(fontsize=8)
    fig.suptitle('Fixed common Sheaf tree, width 128: gradient-isolation matrix (seed 42)')
    fig.savefig(ROOT/'paired_comparison.png',dpi=180);fig.savefig(ROOT/'paired_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,d in zip(axes,DATASETS):
        gs=[g for g in gate_rows if g['dataset']==d and g['scope']=='test']
        for mode,marker in [('joint','^'),('stopgrad','o'),('frozen','s')]:
            points=[g for g in gs if VARIANTS[g['model']][0]==mode and g['centered_cosine'] is not None]
            if points:
                scatter=ax.scatter([g['centered_rg'] for g in points],[g['gate_delta_pp'] for g in points],
                    c=[g['centered_cosine'] for g in points],vmin=-1,vmax=1,cmap='coolwarm',marker=marker,s=65,label=mode,edgecolors='black',linewidths=.4)
        ax.axhline(0,color='gray',linewidth=1);ax.set_xlabel('Class-centered R_g (test nodes)');ax.set_ylabel('Gate-on minus gate-off accuracy (pp)');ax.set_title(d.capitalize());ax.legend(fontsize=8)
    from matplotlib.cm import ScalarMappable
    fig.colorbar(ScalarMappable(norm=plt.Normalize(-1,1),cmap='coolwarm'),ax=list(axes),label='Class-centered Frobenius cosine')
    fig.suptitle('Contribution magnitude, direction and prediction change; zero-gate cosines omitted')
    fig.savefig(ROOT/'gate_diagnostics.png',dpi=180);fig.savefig(ROOT/'gate_diagnostics.pdf');plt.close(fig)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
