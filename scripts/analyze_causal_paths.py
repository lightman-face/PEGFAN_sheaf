"""Audit the completed information-path interventions and publish valid results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_pegfan import digest,write_json
from causal_ops import CHAIN_MODES

ROOT=Path('results/causal_ablations')
PRIORITY=['film','chameleon','squirrel']
DIRS={'no_akx':100,'paths':390,'cornell_repaired':20,'cornell_full_repaired':20}


def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def csv_write(path,rows):
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    allrows=[];audits={}
    for folder,expected in DIRS.items():
        root=ROOT/folder;progress=json.loads((root/'progress.json').read_text())
        if progress['status']!='complete' or progress['completed']!=expected:raise RuntimeError('Incomplete '+folder)
        protocol=json.loads((root/'protocol.json').read_text());rs=records(root/'metrics.jsonl')
        assert len(rs)==len({(r['dataset'],r['model'],r['split']) for r in rs})==expected
        for source,sha in protocol['sources'].items():assert digest(source)==sha,source
        for dataset,files in protocol['data'].items():
            for filename,sha in files.items():assert digest(Path('new_data')/dataset/filename)==sha
        config=protocol.get('config',protocol);cap=config['epochs'];patience=config['patience']
        for r in rs:
            d,s,m=r['dataset'],r['split'],r['model'];key=f'{d}_{m}_split{s}_seed42'
            hist=json.loads((root/'histories'/(key+'.json')).read_text());best=min(hist,key=lambda x:x['validation_loss'])
            assert len(hist)==r['epochs'] and best['epoch']==r['best_epoch'] and best['validation_loss']==r['validation_loss']
            assert r['epochs']==cap or r['epochs']-r['best_epoch']==patience
            assert digest(root/'checkpoints'/(key+'.pt'))==r['checkpoint_sha256']
            assert digest(Path('splits')/f'{d}_split_0.6_0.2_{s}.npz')==r['split_sha256']
            if 'tokens' in r:assert r['tokens']<=512
            if 'fixed_partition_sha256' in r:assert digest(Path('results/targeted_ablations/partitions')/f'{d}_split{s}.npz')==r['fixed_partition_sha256']
        allrows+=rs;audits[folder]=dict(runs=len(rs),source_data_split_checkpoint_hashes=True,validation_selection=True)
    assert len(allrows)==530
    baseline={(r['dataset'],r['split']):r for r in allrows if r['model']=='pegfan_control'}
    lookup={(r['dataset'],r['model'],r['split']):r for r in allrows}
    summary=[];pairs=[]
    for d,m in dict.fromkeys((r['dataset'],r['model']) for r in allrows):
        group=[r for r in allrows if r['dataset']==d and r['model']==m];assert len(group)==10
        accuracy=np.array([100*r['test_accuracy'] for r in group])
        delta=np.array([100*(r['test_accuracy']-baseline[d,r['split']]['test_accuracy']) for r in group])
        summary.append(dict(dataset=d,model=m,runs=10,mean_pct=float(accuracy.mean()),std_pct=float(accuracy.std(ddof=1)),
                            delta_vs_pegfan_pp=float(delta.mean()),wins=int(sum(delta>1e-8)),ties=int(sum(abs(delta)<=1e-8)),losses=int(sum(delta< -1e-8))))
        pairs.extend(dict(dataset=d,model=m,split=r['split'],accuracy_pct=100*r['test_accuracy'],pegfan_pct=100*baseline[d,r['split']]['test_accuracy'],delta_pp=100*(r['test_accuracy']-baseline[d,r['split']]['test_accuracy'])) for r in group)
    csv_write(ROOT/'summary.csv',summary);csv_write(ROOT/'paired_splits.csv',pairs)
    stats={(r['dataset'],r['model']):r for r in summary}
    def cell(d,m):
        r=stats[d,m];return f"{r['mean_pct']:.2f} ± {r['std_pct']:.2f}"
    reconstruction=[]
    for path in sorted((ROOT/'paths/reconstruction').glob('*.json')):reconstruction+=json.loads(path.read_text())
    assert len(reconstruction)==60
    assert len({(r['dataset'],r['split'],r['hierarchy']) for r in reconstruction})==60
    for r in reconstruction:
        reference=baseline[r['dataset'],r['split']]
        assert r['checkpoint_sha256']==reference['checkpoint_sha256']
        assert r['baseline_accuracy']==r['reconstructed_accuracy']==reference['test_accuracy']
        assert r['changed_predictions']==0 and r['logit_max_abs_error']<=1e-5
        assert max(e['max_abs_error'] for e in r['channel_errors'])<=3e-6
    reconstruction_flat=[]
    for r in reconstruction:
        row={k:v for k,v in r.items() if k!='channel_errors'}
        row['input_max_abs_error']=max(x['max_abs_error'] for x in r['channel_errors'])
        row['input_max_relative_l2_error']=max(x['relative_l2_error'] for x in r['channel_errors'])
        reconstruction_flat.append(row)
    csv_write(ROOT/'reconstruction_checks.csv',reconstruction_flat)
    haar=[r for r in allrows if r['model']=='chain_transformer_global'];assert len(haar)==30
    for r in haar:
        assert r['haar_sum_changed_predictions']==0
        assert r['haar_sum_max_abs_error']<=1e-5 and r['haar_sum_logit_error']<=1e-5
    for d in PRIORITY:
        info=json.loads((ROOT/'paths/bases'/(d+'.json')).read_text())
        assert digest(ROOT/'paths/bases'/(d+'.pt'))==info['basis_sha256']
    csv_write(ROOT/'haar_identity_checks.csv',[{k:r[k] for k in ['dataset','split','haar_sum_max_abs_error','haar_sum_logit_error','haar_sum_changed_predictions']} for r in haar])
    hybrid=[]
    for d in PRIORITY+['cornell']:
        delta=[]
        for s in range(10):
            off,on=lookup[d,'full_no_akx',s],lookup[d,'full_with_akx',s]
            assert off['parameters']==on['parameters']
            v=100*(on['test_accuracy']-off['test_accuracy']);delta.append(v)
            hybrid.append(dict(dataset=d,split=s,full_control_pct=100*off['test_accuracy'],full_with_akx_pct=100*on['test_accuracy'],delta_pp=v))
    csv_write(ROOT/'full_akx_paired.csv',hybrid)
    # The pre-repair Cornell rows are never used below.
    old=records(Path('results/filtration_vs_pegfan/metrics.jsonl'))
    diffs=[abs(r['test_accuracy']-baseline[r['dataset'],r['split']]['test_accuracy']) for r in old if r['model']=='pegfan' and r['dataset']!='cornell']
    audit=dict(status='passed',trials=530,groups=audits,legacy_cornell_excluded=True,
               repeated_pegfan_max_accuracy_difference=float(max(diffs)),
               reconstruction_cases=60,reconstruction_changed_predictions=sum(r['changed_predictions'] for r in reconstruction),
               reconstruction_max_input_error=max(r['input_max_abs_error'] for r in reconstruction_flat),
               reconstruction_max_logit_error=max(r['logit_max_abs_error'] for r in reconstruction),
               haar_identity_cases=30,haar_identity_changed_predictions=sum(r['haar_sum_changed_predictions'] for r in haar),
               haar_identity_max_signal_error=max(r['haar_sum_max_abs_error'] for r in haar),
               haar_identity_max_logit_error=max(r['haar_sum_logit_error'] for r in haar))
    repair=json.loads(Path('results/data_repairs/cornell/audit.json').read_text());assert repair['status']=='repaired'
    for r in repair['files']:assert digest(r['path'])==r['upstream_sha256']
    for h,r in repair['regenerated_caches'].items():assert digest(Path('framelets')/h/'cornell.pickle')==r['sha256']
    cache_hashes={}
    for d in PRIORITY+['texas','wisconsin']:
        current=digest(Path('framelets/8')/(d+'.pickle'))
        reference=next(r['framelet_cache_sha256'] for r in old if r['dataset']==d and r['model']=='pegfan')
        assert current==reference;cache_hashes[d]=current
    cache_hashes['cornell']=repair['regenerated_caches']['8']['sha256']
    import platform,torch,scipy
    audit['environment']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__)
    audit['framelet_cache_hashes']=cache_hashes
    execution=ROOT/'paths/parallel_tail/execution.json'
    if execution.exists():audit['parallel_tail_execution']=json.loads(execution.read_text())
    assert audit['repeated_pegfan_max_accuracy_difference']==0
    audit['cornell_repair_revision']=repair['commit'];write_json(ROOT/'audit.json',audit)
    lines=['# 信息通路因果消融：完整结果','',
           '完成 530 次固定配置训练：优先传播通道消融 100 次，三个重点数据集的 13 项路径实验 390 次，修复后 Cornell 独立重跑 40 次。另有 60 项同 checkpoint pooling 重构检查和 30 项 post-Transformer Haar 恒等性检查。',
           '同 seed=42、十个 split、验证损失选 checkpoint、500 epoch 上限及 patience=50。表中 accuracy 为均值 ± 样本标准差（%）。[详细干预协议](../../docs/causal_information_path_protocol.md)。',
           '', '核心判断：原 PEGFAN 的显式传播通路对 Chameleon/Squirrel 很关键；64 维固定投影和无损 pool/detail/lift 都不能解释完整模型的全部退化。新 hierarchy 在保留原 PEGFAN 局部通路时可用，但在 pure coarse 模式下损失明显。Film 的正信号在 local-only 就已存在，尚不能归功于 Transformer。',
           '', '## 1. 原 PEGFAN 只移除额外 A^kX','',
           '只将 AX/A²X/A³X 三个显式输入槽位置零，保留 X、原 Haar 对 AX 的所有频带、原 FSGNN 全部参数和 bias。这里没有移除 framelet 中已有的 AX 信息。',
           '', '| 数据集 | PEGFAN | 去除额外 A^kX | 差值 pp | 胜/平/负 |','|---|---:|---:|---:|---|']
    for d in PRIORITY+['texas','wisconsin','cornell']:
        r=stats[d,'no_akx'];label=d+('（已修复）' if d=='cornell' else '')
        lines.append(f"| {label} | {cell(d,'pegfan_control')} | {cell(d,'no_akx')} | {r['delta_vs_pegfan_pp']:+.2f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=['','Chameleon/Squirrel 的额外传播通道具有明确作用，移除后 10/10 splits 均下降约 10–12 pp；Film 基本不变。因此它是重要因素，但不能把此差值与其他干预差值简单相加，或宣称已经解释完整模型的全部下降。',
            '', '## 2. 只 pool → immediate lift','',
            '| 数据集 | 原 PEGFAN | 原树 coarse-only | Sheaf 树 coarse-only | 加回全部 detail 的预测变化 |','|---|---:|---:|---:|---:|']
    for d in PRIORITY:
        changed=sum(r['changed_predictions'] for r in reconstruction if r['dataset']==d)
        lines.append(f"| {d} | {cell(d,'pegfan_control')} | {cell(d,'pool_global')} | {cell(d,'pool_sheaf_global')} | {changed} |")
    lines += ['',f"同 checkpoint 重构的最大输入绝对误差 {audit['reconstruction_max_input_error']:.3g}、最大 logit 误差 {audit['reconstruction_max_logit_error']:.3g}；全节点变化预测共 {audit['reconstruction_changed_predictions']} 个。已逐 split 验证重构后的测试准确率。没有发现 pooling/detail/lifting 的信息遗漏。",
              'coarse-only 是对原 classifier 的每个输入 H 施加 P Pᵀ；原图传播与 Haar 通道先按原定义计算，没有新增 Transformer 或 learned fusion。原树使用原 equal-child transfers，新树使用完整模型的 mass transfers。',
              '原树实际 tokens 为 Film 274、Chameleon 474、Squirrel 218；新树约 512。两者的预算是上界，并未强行取相同 token 数。Chameleon 原树 cut 只有一级，equal-child 与 mass normalization 在此相同；其他图的两种 transfer normalization 并不等同。',
              '这组揭示了保留完整局部通路与只保留粗信号的差别：新 hierarchy 在原 PEGFAN 中不崩，并不意味着它适合作为纯 coarse bottleneck。',
              '', '## 3. F→64→F feature bottleneck','',
              '原 FSGNN 每个 channel 本就投影到 hidden=64；此处在它之前增加共享的 feature rank bottleneck，保留原 classifier。固定 SVD 使用无标签的精确 top-64 子空间；learned 版本从同一子空间初始化，新增 lr=0.001。',
              '', '| 数据集 | PEGFAN | 固定 SVD64 | 可学习 rank64 | 原 X 的 SVD64 能量保留 % |','|---|---:|---:|---:|---:|']
    for d in PRIORITY:
        info=json.loads((ROOT/'paths/bases'/(d+'.json')).read_text())
        lines.append(f"| {d} | {cell(d,'pegfan_control')} | {cell(d,'svd64')} | {cell(d,'learned64')} | {100*info['energy_retained']:.2f} |")
    lines+=['','64 维本身并不普遍导致崩塌。固定与可学习映射的差别也说明优化/参数化影响很大，不能把某个训练版本的掉点视为所有 64 维表示的必然上限。',
            '', '## 4. 逐项构造链条','',
            '此表使用同 split 固定的 Sheaf hierarchy、同完整模型初始化/优化器。除了最后一行，其余各行都关闭原节点 hidden skip；global-only 才不会暗含局部旁路。所有模块均实例化，未启用模块不进入输出。',
            '', '| 阶段 | Film | Chameleon | Squirrel |','|---|---:|---:|---:|']
    names=['仅 local sheaf signal','Pool→Lift，global-only','加 Transformer，global-only','再加 post-Haar learned fusion，global-only','仅 local details','global＋detail','再加回原节点 hidden skip']
    for mode,name in zip(CHAIN_MODES,names):lines.append('| '+name+' | '+' | '.join(cell(d,mode) for d in PRIORITY)+' |')
    lines+=['',f"另对第三行同 checkpoint 做 Haar 分解后直接求和：最大 signal 误差 {audit['haar_identity_max_signal_error']:.3g}、logit 误差 {audit['haar_identity_max_logit_error']:.3g}、预测变化 {audit['haar_identity_changed_predictions']}。恒等分析/合成与后续 learned band fusion 是两件事；第四行测的是后者的介入。",
            'Film 的 local-only 已能出现强信号，因此完整模型优于原 PEGFAN 并不能单独证明 Transformer 或多尺度全局分支有贡献。Chameleon/Squirrel 的 local-only 本身已经明显落后，也不能把所有差距归咎于后续粗化。',
            '本阶段固定 hierarchy 来源于前轮同 split 监督训练 scorer 的重放，不是随机未训练分区；它与下一组的动态 hierarchy 区分报告。',
            '', '## 5. 完整动态模型接回 local A^kX','',
            'on/off 两组均为完整动态 Sheaf 模型，使用相同新增模块和参数量。三个传播输入经独立投影和归一化接入最终 classifier，新连接零初始化，初始函数等于原完整模型。关闭组只将新增 local features 置零。',
            '', '| 数据集 | 关闭 local A^kX | 接回 local A^kX | 配对差值 pp | 胜/平/负 | 原 PEGFAN |','|---|---:|---:|---:|---|---:|']
    for d in PRIORITY+['cornell']:
        delta=np.array([r['delta_pp'] for r in hybrid if r['dataset']==d])
        lines.append(f"| {d}{'（已修复）' if d=='cornell' else ''} | {cell(d,'full_no_akx')} | {cell(d,'full_with_akx')} | {delta.mean():+.2f} | {sum(delta>1e-8)}/{sum(abs(delta)<=1e-8)}/{sum(delta< -1e-8)} | {cell(d,'pegfan_control')} |")
    recovery='；'.join(f"{d}：{cell(d,'full_no_akx')} → {cell(d,'full_with_akx')}，相对原 PEGFAN {stats[d,'full_with_akx']['delta_vs_pegfan_pp']:+.2f} pp" for d in PRIORITY)
    lines+=['',recovery+'。这是当前接法和固定训练配置下的结果；它评估传播分支是否有用，尚未证明 Transformer 分支在保留完整 PEGFAN 基础上有额外收益。', '', '该组是重新训练的匹配 on/off control；新增模块初始化消耗的随机数与前轮不同，因此使用本组 control 作配对，而不是把不同训练轨迹冒充完全相同的 checkpoint。新增 projection/通道权重采用 PEGFAN 原学习率 0.01/0.02，原完整模型参数仍为 0.001。',
            '', '## Cornell 修复及结果有效性','',
            f"固定上游提交 `{repair['commit']}`；校验 183 nodes、1703 features、5 labels、298 edge rows。修复了 {repair['changed_feature_entries']} 个 feature entries 和 {repair['changed_labels']} 个 labels；旧图也与上游不一致，因此 h=4/8 的 Ward Haar 均重新生成。十个 split 文件 checksum 与上游一致。",
            '旧 Cornell 已从此前两轮正式表格及图表剔除，错误数据/旧缓存及原结果只在归档中保留。这里的 Cornell 行全部来自新数据版本的独立重跑，不能与旧 Cornell 的数字作架构因果比较。',
            '数据来源、每个文件 checksum、缓存重构误差及构建依赖版本见 [修复审计](../data_repairs/cornell/audit.json)。训练主环境没有升级 NumPy；Ward 在隔离环境重建。',
            '', '## 审计与复现','',
            '所有 530 个新增 checkpoint 的文件哈希、源码、数据版本、split、最佳验证 epoch 和 early stopping 均核验。首组五个有效数据集重跑 PEGFAN 与上一轮 test accuracy 完全一致。',
            '完整逐 split 数据：`paired_splits.csv`；接回传播通道的 on/off 差值：`full_akx_paired.csv`；重构和 Haar 恒等检查均有逐 split CSV。',
            '最后一组的独立 split 使用两个训练进程执行，调用原 run_trial，模型/选模规则未变；部分训练也与 Cornell 重跑并行，耗时不作严格速度比较。',
            '本轮使用一个 seed、十个重叠划分；不能替代多 seed 稳定性及清理版 heterophily benchmark。所有干预和解释边界见协议。','']
    (ROOT/'analysis.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for ax,models,title in [
        (axes[0,0],['pegfan_control','no_akx'],'Remove only explicit propagation inputs'),
        (axes[0,1],['pegfan_control','pool_global','pool_sheaf_global'],'Coarse-only spatial bottleneck'),
        (axes[1,0],['pegfan_control','svd64','learned64'],'Shared feature bottleneck F -> 64 -> F'),
        (axes[1,1],['pegfan_control','full_no_akx','full_with_akx'],'Full dynamic model: local propagation off/on')]:
        width=.8/len(models)
        for i,m in enumerate(models):
            ax.bar(np.arange(3)+(i-(len(models)-1)/2)*width,[stats[d,m]['mean_pct'] for d in PRIORITY],width,
                   yerr=[stats[d,m]['std_pct'] for d in PRIORITY],label=m,capsize=2)
        ax.set_xticks(range(3));ax.set_xticklabels([d.capitalize() for d in PRIORITY]);ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Information-path ablations: 10 splits, seed 42; validation-loss selection')
    fig.savefig(ROOT/'causal_comparison.png',dpi=180);fig.savefig(ROOT/'causal_comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    for d in PRIORITY:ax.plot(range(len(CHAIN_MODES)),[stats[d,m]['mean_pct'] for m in CHAIN_MODES],'o-',label=d)
    ax.set_xticks(range(len(CHAIN_MODES)));ax.set_xticklabels(['Local','Pool global','+Transformer','+Haar fusion','Detail only','Global+detail','+h skip'],rotation=15)
    ax.set_ylabel('Test accuracy (%)');ax.set_ylim(0,100);ax.set_title('Fixed hierarchy, controlled information paths');ax.legend()
    fig.savefig(ROOT/'chain_stages.png',dpi=180);fig.savefig(ROOT/'chain_stages.pdf');plt.close(fig)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
