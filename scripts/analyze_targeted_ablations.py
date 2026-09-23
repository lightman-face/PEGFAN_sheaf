"""Final, post-selection audit, tables and scientific plots for targeted ablations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_pegfan import digest,write_json

OUT=Path('results/targeted_ablations')
DATASETS=['film','chameleon','squirrel','texas','cornell','wisconsin']
VARIANTS=['hierarchy_only','projector_only','transformer_only']


def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    progress=json.loads((OUT/'progress.json').read_text())
    if progress['status']!='complete' or progress['completed']!=180:raise RuntimeError('Wait for all 180 trials')
    new=records(OUT/'metrics.jsonl');old=records(Path('results/filtration_vs_pegfan/metrics.jsonl'))
    protocol=json.loads((OUT/'protocol.json').read_text());previous=json.loads(Path('results/filtration_vs_pegfan/protocol.json').read_text())
    assert len(new)==len({(r['dataset'],r['model'],r['split']) for r in new})==180
    assert {(r['dataset'],r['model'],r['split']) for r in new}=={(d,m,s) for d in DATASETS for m in VARIANTS for s in range(10)}
    for name,sha in {**protocol['source_sha256'],**previous['source_sha256']}.items():assert digest(name)==sha,name
    assert digest('results/filtration_vs_pegfan/metrics.jsonl')==protocol['previous_metrics_sha256']
    baseline={(r['dataset'],r['split']):r for r in old if r['model']=='pegfan'}
    for r in new:
        reference=baseline[r['dataset'],r['split']]
        assert r['split_sha256']==reference['split_sha256']
        assert r['classifier_parameters']==reference['parameters']
        assert r['channels']==reference['channels']
        key=f"{r['dataset']}_{r['model']}_split{r['split']}_seed42"
        hist=json.loads((OUT/'histories'/(key+'.json')).read_text())
        best=min(hist,key=lambda x:x['validation_loss'])
        assert best['epoch']==r['best_epoch'] and best['validation_loss']==r['validation_loss']
        assert len(hist)==r['epochs'] and (r['epochs']==500 or r['epochs']-r['best_epoch']==50)
        assert digest(OUT/'checkpoints'/(key+'.pt'))==r['checkpoint_sha256']
        if r['model']=='hierarchy_only':
            score=OUT/'scorers'/f"{r['dataset']}_split{r['split']}_seed42.npz"
            assert digest(score)==r['scorer_sha256']==r['scorer_audit']['scores_sha256']
    replays=[json.loads(p.read_text()) for p in (OUT/'scorers').glob('*_seed42.json')]
    audit=dict(completed_new_trials=len(new),old_baseline_trials=60,source_hashes_match=True,all_splits_match=True,
               classifier_parameters_and_channels_match=True,validation_selection_and_checkpoint_hashes_match=True,
               replay_count=len(replays),replay_validation_matches=sum(r['numerically_matching'] for r in replays),
               replay_max_abs_validation_difference=max(abs(r['validation_loss_difference']) for r in replays))
    excluded={'cornell'} if (OUT/'exclusions.json').exists() else set()
    DISPLAY=[d for d in DATASETS if d not in excluded]
    audit['excluded_legacy_datasets']=sorted(excluded)
    # Recheck current dataset tensors against the previous experiment's hashes.
    from process import full_load_data
    metadata=json.loads(Path('results/filtration_vs_pegfan/dataset_metadata.json').read_text())
    for d in DISPLAY:
        loaded=full_load_data(d,Path('splits')/f'{d}_split_0.6_0.2_0.npz',return_sparse=True);adj=loaded[0]
        assert hashlib.sha256(loaded[3].numpy().tobytes()).hexdigest()==metadata[d]['features_sha256']
        assert hashlib.sha256(adj.indptr.tobytes()+adj.indices.tobytes()+adj.data.tobytes()).hexdigest()==metadata[d]['adjacency_sha256']
    audit['feature_and_adjacency_hashes_match_previous']=True
    write_json(OUT/'audit.json',audit)
    diagnostics=[]
    for path in sorted((OUT/'diagnostics').glob('*_split*.json')):diagnostics.extend(json.loads(path.read_text()))
    with (OUT/'partition_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostics[0]));w.writeheader();w.writerows(diagnostics)
    labels=json.loads((OUT/'label_diagnostics_posthoc.json').read_text())
    labels={(r['dataset'],r['split'],r['partition'],r['label_scope']):r for r in labels}
    cuts=[]
    for r in diagnostics:
        if not r['is_budget_cut']:continue
        r=dict(r)
        for scope in ('all_posthoc','test_only_posthoc'):
            lab=labels[r['dataset'],r['split'],f"{r['hierarchy']}_level{r['level']}",scope]
            r.update({scope+'_'+k:lab[k] for k in ['cluster_label_entropy','cluster_label_purity','labeled_nodes','observed_regions']})
        cuts.append(r)
    with (OUT/'budget_cut_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(cuts[0]));w.writeheader();w.writerows(cuts)
    with (OUT/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    lookup={(r['dataset'],r['model']):r for r in summary}
    def cell(d,m):
        r=lookup[d,m];return f"{float(r['mean_pct']):.2f} ± {float(r['std_pct']):.2f}"
    lines=['# 定向消融结果与诊断','','已完成 180 次新增训练，复用上一轮 60 次原 PEGFAN 与 60 次完整新模型结果。固定 seed=42、同十个 split、同验证损失选模型；未调参。',
           '详细控制方式见 [协议](../../docs/targeted_ablation_protocol.md)。accuracy 的 ± 为十个 split 的样本标准差，差值单位为百分点。','',
           '| 数据集 | PEGFAN | 只换 hierarchy | 只换 Haar 归一化 | 只加残差 Transformer | 上轮完整新模型 |','|---|---:|---:|---:|---:|---:|']
    for d in DISPLAY:lines.append('| '+d+('†' if d=='cornell' else '')+' | '+' | '.join(cell(d,m) for m in ['pegfan']+VARIANTS+['sheaf'])+' |')
    lines+=['','† 本地 Cornell 特征/标签文件与 Texas 逐字节相同，原 Git HEAD 中亦如此。上游哈希比对确认本地 Texas 匹配 Geom-GCN Texas，而本地 Cornell 不匹配上游 Cornell；因此 Cornell 行仅作仓库内部控制，不能作为标准 Cornell benchmark 分数。详见 webkb_data_audit.json。',
            '', '## 只换 hierarchy 的配对证据','','| 数据集 | 平均差值 pp | 胜 / 平 / 负 |','|---|---:|---|']
    for d in DISPLAY:
        r=lookup[d,'hierarchy_only'];lines.append(f"| {d} | {float(r['delta_pp']):+.2f} | {r['wins']} / {r['ties']} / {r['losses']} |")
    lines+=['','只换 hierarchy 使用原层级 region count 作为 filtration 查询目标，保持通道数与 FSGNN 参数量完全相同；它与完整新模型的 dyadic schedule 有意分开。Haar 是原 equal-child 算子，隐式求值已对照原缓存逐 band 和实际 AX 输入校验。',
            '这些结果只能判断新 partition 在保留原 PEGFAN 信息通路时的作用，不能排除 giant region 与瓶颈、fusion、特征降维的交互影响。',
            '', '## Film 的十个 paired split','','| split | PEGFAN % | 上轮完整模型 Δ | 只换 hierarchy Δ | 只换归一化 Δ | 只加 Transformer Δ |','|---|---:|---:|---:|---:|---:|']
    allrows=old+new
    for s in range(10):
        b=baseline['film',s]['test_accuracy']*100
        ds=[next(r['test_accuracy']*100-b for r in allrows if r['dataset']=='film' and r['split']==s and r['model']==m) for m in ['sheaf']+VARIANTS]
        lines.append(f'| {s} | {b:.2f} | '+' | '.join(f'{v:+.2f}' for v in ds)+' |')
    lines+=['','上一轮完整模型 Film 10/10 提升，平均 +1.80 pp，中位数 +1.78 pp；不是少数 split 拉高均值。但仍只有一个 seed，十个重叠数据划分不等于十个独立数据集。',
            '', '## 分区诊断：预算 cut','','以下为十个 scorer split 的均值。`matched` 在原 hierarchy 的相同层级目标 count 上取新 cut；`dyadic` 使用完整新模型的预算 cut。原 tree 的已有层可能远小于 512，所以 original 与 dyadic 的区域数并不相同。',
            '', '| 数据集 | hierarchy | 区域数 | 最大区 % | singleton/区 % | Gini | 归一化 size entropy | feature 方差占比 | hidden 方差占比 | 内部边 sheaf energy | all-label purity % | test-label purity % |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cutmeans={}
    keys=['regions','largest_cluster_ratio','singleton_cluster_ratio','cluster_size_gini','cluster_size_entropy_normalized',
          'feature_variance_fraction','hidden_variance_fraction','within_cluster_sheaf_energy','all_posthoc_cluster_label_purity','test_only_posthoc_cluster_label_purity']
    for d in DISPLAY[:3]:
        for h in ('original','matched','dyadic'):
            group=[r for r in cuts if r['dataset']==d and r['hierarchy']==h]
            mu={k:float(np.mean([r[k] for r in group])) for k in keys};cutmeans[d,h]=mu
            lines.append(f"| {d} | {h} | {mu['regions']:.1f} | {100*mu['largest_cluster_ratio']:.2f} | {100*mu['singleton_cluster_ratio']:.2f} | {mu['cluster_size_gini']:.3f} | {mu['cluster_size_entropy_normalized']:.3f} | {mu['feature_variance_fraction']:.3f} | {mu['hidden_variance_fraction']:.3f} | {mu['within_cluster_sheaf_energy']:.5f} | {100*mu['all_posthoc_cluster_label_purity']:.2f} | {100*mu['test_only_posthoc_cluster_label_purity']:.2f} |")
    lines+=['','### 如何解释这些结果','',
            '只换 hierarchy 后，Film/Chameleon/Squirrel 的均值变化分别为 +0.02/+0.53/+0.29 pp，配对胜负并不一致。因此没有证据把完整新模型的大幅下降直接归因于 partition 单独替换，也没有充分证据证明新 partition 有稳定分类优势。',
            '原 hierarchy 上 mass-normalized Haar 的影响较小；它与原 equal-child Haar 确实不同，但不是本轮观察到的几十个百分点下降的充分解释。残差 Transformer 的结果只能评价当前插入方式，不能据此免责完整新模型的 replacement/fusion 路径。',
            '新分区的 giant region 占比很高，但其 mean-reconstruction feature distortion 并不比原 hierarchy 更高。例如同样以 474 个区域为目标的 Chameleon，新分区的残差方差比例约 0.365，原分区约 0.755。由此不能保证一个全局平均失真阈值会拒绝 giant region；需要分析每个候选 region 的失真、保留的局部细节以及任务相关差异。',
            '标签混合与低特征失真可以同时发生：Chameleon 在相同 474 个区域目标下，全节点 post-hoc purity 从原分区的 84.76% 降为 43.13%，label entropy 从 0.310 升为 1.249。仅 test-label purity 受每区 test 样本数影响，须与全节点口径及有观测的区域数一并看；全部标签诊断仅作最终分析。',
            '下一项优先定位应是完整模型的信息通路：在保留原 PEGFAN 框架时分别测试 A^kX 通道移除、局部更新/特征降维、粗尺度输出与细节融合。cluster-level gate 仍是改善分辨率分配的独立方向，不应在原因未隔离时作为唯一修复。',
            '', '完整每层、每 split 数据：`partition_diagnostics.csv`；预算层两种 labels 口径及 entropy/purity：`budget_cut_diagnostics.csv`。全部 labels 诊断都在本轮所有 checkpoint 固定后生成，未用于训练、分区或选择。',
            'feature/hidden 方差是均值压缩失真占全局方差的比例，不等于带完整 detail 的 PEGFAN 信息损失。巨型 region、singleton 比例与低 size entropy 揭示分辨率分配失衡；不能单凭这一项推出分类下降的原因。',
            '', '## 数值与范围限制','',
            f"- 原 scorer checkpoint 未保存；本轮按原配置在原选定 epoch 重放，{audit['replay_validation_matches']}/60 次验证损失匹配，最大绝对差 {audit['replay_max_abs_validation_difference']:.6f}。重放权重与逐次偏差现已保存。",
            '- hierarchy-only 的 scorer 经过同 split 训练监督预训练，分类器训练期间冻结。额外训练成本不能忽略，也不能称为完全无监督 hierarchy。',
            '- Transformer 消融是带零初始化输出适配器的残差插入，保留原细节与传播通道；它不能单独证明完整模型中非残差替换式 Transformer 的优劣。',
            '- 本轮没有加入 aggregate distortion gate。下一轮若加入，必须同时声明预算不可达时的规则；内部边平均 sheaf energy 本身仍可能允许 chaining。',
            '- 经典 Chameleon/Squirrel 的重复节点问题见 [Platonov et al.](https://arxiv.org/abs/2302.11640)，后续正式 benchmark 需清理版或更新的数据集。',
            '',f"审计通过：{audit['completed_new_trials']} 个新 checkpoint 的哈希、最佳验证 epoch、early stopping、划分、FSGNN 参数量、源码，以及与上一轮的数据哈希。",'']
    (OUT/'analysis.md').write_text('\n'.join(lines))
    # Exportable plots; no smoothing or hidden runs.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    models=['pegfan']+VARIANTS+['sheaf'];names=['PEGFAN','Hierarchy only','Mass Haar only','Residual attention only','Full new model']
    colors=['#475569','#2563eb','#0d9488','#d97706','#c026d3'];xx=np.arange(len(DISPLAY));width=.15
    for i,(m,name,color) in enumerate(zip(models,names,colors)):
        mean=[float(lookup[d,m]['mean_pct']) for d in DISPLAY];sd=[float(lookup[d,m]['std_pct']) for d in DISPLAY]
        axes[0,0].bar(xx+(i-2)*width,mean,width,label=name,color=color,yerr=sd,error_kw={'lw':.7,'capsize':1.5})
    axes[0,0].set_xticks(xx);axes[0,0].set_xticklabels([d.capitalize() for d in DISPLAY],rotation=15)
    axes[0,0].set_ylabel('Test accuracy (%)');axes[0,0].set_ylim(0,100);axes[0,0].set_title('Single-factor ablations: mean and sample SD')
    axes[0,0].legend(fontsize=8,ncol=2)
    for m,name,color in zip(models[1:],names[1:],colors[1:]):
        delta=[100*(next(r['test_accuracy'] for r in allrows if r['dataset']=='film' and r['split']==s and r['model']==m)-baseline['film',s]['test_accuracy']) for s in range(10)]
        axes[0,1].plot(range(10),delta,'o-',lw=1,color=color,label=name)
    axes[0,1].axhline(0,color='#64748b',lw=.8);axes[0,1].set_xticks(range(10));axes[0,1].set_xlabel('Film split');axes[0,1].set_ylabel('Accuracy difference vs PEGFAN (pp)')
    axes[0,1].set_title('All paired Film splits');axes[0,1].legend(fontsize=8)
    for ax,key,title in [(axes[1,0],'largest_cluster_ratio','Largest region: fraction of original nodes'),(axes[1,1],'cluster_size_entropy_normalized','Normalized region-size entropy')]:
        for j,h in enumerate(['original','matched','dyadic']):
            ax.bar(np.arange(3)+(j-1)*.24,[cutmeans[d,h][key] for d in DISPLAY[:3]],.24,label=h)
        ax.set_xticks(range(3));ax.set_xticklabels(['Film','Chameleon','Squirrel']);ax.set_ylim(0,1.05);ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Fixed configuration; 10 splits, seed 42. *Cornell source-file anomaly.',fontsize=12)
    fig.savefig(OUT/'ablation_comparison.png',dpi=180);fig.savefig(OUT/'ablation_comparison.pdf');plt.close(fig)
    print(json.dumps(audit,indent=2));print('Wrote analysis.md and figures')

if __name__=='__main__':main()
