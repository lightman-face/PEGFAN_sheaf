"""Audit the 48-run architecture screen and summarize Cornell representation probes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from augmentation_ops import VARIANTS
from compare_pegfan import digest,write_json
from run_augmentation import records,summary

ROOT=Path('results/architecture_augmentation')
DATASETS=['film','chameleon','squirrel','cornell']
MODELS=list(VARIANTS)
NAMES={'pegfan':'PEGFAN','transformer64':'+ T64','transformer_haar64':'+ T + Haar64','transformer_haar128':'+ T + Haar128'}


def csv_write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    progress=json.loads((ROOT/'progress.json').read_text())
    assert progress['status']=='complete' and progress['completed']==48
    protocol=json.loads((ROOT/'protocol.json').read_text());cfg=protocol['config']
    assert cfg['splits']==[0,1,2] and cfg['variants']==MODELS and cfg['datasets']==DATASETS
    for name,sha in protocol['sources'].items():assert digest(name)==sha,name
    for d,files in protocol['data'].items():
        for name,sha in files.items():assert digest(Path('new_data')/d/name)==sha
    for d,sha in protocol['framelet_caches'].items():assert digest(Path('framelets/8')/(d+'.pickle'))==sha
    rs=records(ROOT/'metrics.jsonl');lookup={(r['dataset'],r['model'],r['split']):r for r in rs}
    assert len(rs)==len(lookup)==48
    old=records(Path('results/causal_ablations/no_akx/metrics.jsonl'))+records(Path('results/causal_ablations/cornell_repaired/metrics.jsonl'))
    previous={(r['dataset'],r['split']):r for r in old if r['model']=='pegfan_control'}
    paired=[];gates=[]
    for r in rs:
        d,m,s=r['dataset'],r['model'],r['split'];key=f'{d}_{m}_split{s}_seed42'
        hist=json.loads((ROOT/'histories'/(key+'.json')).read_text());best=min(hist,key=lambda e:e['validation_loss'])
        assert len(hist)==r['epochs'] and best['epoch']==r['best_epoch'] and best['validation_loss']==r['validation_loss']
        assert r['epochs']==cfg['epochs'] or r['epochs']-r['best_epoch']==cfg['patience']
        assert digest(ROOT/'checkpoints'/(key+'.pt'))==r['checkpoint_sha256']
        assert digest(Path('splits')/f'{d}_split_0.6_0.2_{s}.npz')==r['split_sha256']
        base=lookup[d,'pegfan',s]
        assert r['backbone_initial_sha256']==base['backbone_initial_sha256']
        assert r['backbone_parameters']==base['backbone_parameters']
        assert r['local_input_width']==base['local_input_width'] and r['local_channels']==base['local_channels']
        assert r['initial_output_max_error']==0
        if m=='pegfan':
            assert r['test_accuracy']==previous[d,s]['test_accuracy']
            assert r['best_epoch']==previous[d,s]['best_epoch']
        else:
            assert r['tokens']<=512
            if d=='cornell':assert r['tokens']==183
            gates.append(dict(dataset=d,model=m,split=s,alpha=r['alpha'],
                logit_norm_ratio=r['gated_to_local_logit_norm_ratio'],
                selected_full_accuracy_pct=100*r['test_accuracy'],
                joint_backbone_gate_off_pct=100*r['gate_disabled_accuracy'],
                gate_test_delta_pp=100*(r['test_accuracy']-r['gate_disabled_accuracy']),
                changed_test_predictions=r['gate_changed_test_predictions'],tokens=r['tokens'],
                largest_region_fraction=r['largest_region_fraction']))
        paired.append(dict(dataset=d,model=m,split=s,accuracy_pct=100*r['test_accuracy'],
                           pegfan_pct=100*base['test_accuracy'],delta_pp=100*(r['test_accuracy']-base['test_accuracy'])))
    csv_write(ROOT/'paired_splits.csv',paired);csv_write(ROOT/'gates.csv',gates);summary(ROOT,rs)
    with (ROOT/'summary.csv').open() as f:stats={(r['dataset'],r['model']):r for r in csv.DictReader(f)}
    def cell(d,m):
        r=stats[d,m];return f"{float(r['mean_pct']):.2f} ± {float(r['std_pct']):.2f}"
    representation=[];attention=[];probes=[]
    paths=list((ROOT/'diagnostics').glob('cornell_*.json'))+list((ROOT/'diagnostics/previous_full').glob('cornell_*.json'))
    assert len(paths)==19
    old_full=records(Path('results/causal_ablations/cornell_full_repaired/metrics.jsonl'))
    old_lookup={r['split']:r for r in old_full if r['model']=='full_no_akx'}
    for path in sorted(paths):
        probe=json.loads(path.read_text());m=probe['model'];s=probe['split']
        reference=old_lookup[s] if m=='previous_full' else lookup['cornell',m,s]
        assert probe['checkpoint_sha256']==reference['checkpoint_sha256'] and probe['tokens']==183
        for phase in ('initial','selected'):
            for stage,values in probe[phase]['representations'].items():
                representation.append(dict(model=m,split=s,phase=phase,stage=stage,**values))
            for a in probe[phase]['attention']:
                assert -1e-8<=a['normalized_entropy']<=1+1e-6
                attention.append(dict(model=m,split=s,phase=phase,**a))
            assert len(probe[phase]['attention'])==8
        probes.append(probe)
    old_audit=json.loads((ROOT/'diagnostics/previous_full/audit.json').read_text());assert old_audit['status']=='passed'
    csv_write(ROOT/'cornell_representations.csv',representation);csv_write(ROOT/'cornell_attention.csv',attention)
    groups=['previous_full']+MODELS[1:]
    diagnostic_summary=[]
    for m in groups:
        ps=[p for p in probes if p['model']==m]
        for phase in ('initial','selected'):
            before=[p[phase]['representations']['before'] for p in ps]
            after=[p[phase]['representations']['after'] for p in ps]
            row=dict(model=m,phase=phase,checkpoints=len(ps))
            for k in ('feature_variance','mean_pairwise_cosine','centered_energy_fraction'):
                row[k+'_before']=float(np.mean([p[k] for p in before]));row[k+'_after']=float(np.mean([p[k] for p in after]))
            row['cosine_increased_splits']=sum(a['mean_pairwise_cosine']>b['mean_pairwise_cosine'] for a,b in zip(after,before))
            for layer in (0,1):row[f'layer{layer}_normalized_entropy']=float(np.mean([a['normalized_entropy'] for p in ps for a in p[phase]['attention'] if a['layer']==layer]))
            diagnostic_summary.append(row)
    csv_write(ROOT/'cornell_diagnostic_summary.csv',diagnostic_summary)
    audit=dict(status='passed',new_training_trials=48,splits=[0,1,2],seed=42,source_data_cache_split_checkpoint_hashes=True,
               selection_and_early_stopping=True,backbone_initialization_identical=True,local_features_unchanged=True,
               zero_gate_initial_output_error=0.,pegfan_reference_accuracy_reproduced=True,
               cornell_new_diagnostic_checkpoints=9,cornell_reused_checkpoints=10,cornell_tokens=183,
               previous_probe_max_output_error=max(p['diagnostic_output_error'] for p in probes if p['model']=='previous_full'),
               previous_probe_max_repeated_forward_error=max(p['repeated_forward_error'] for p in probes if p['model']=='previous_full'),
               previous_probe_changed_predictions=sum(p['diagnostic_changed_predictions'] for p in probes if p['model']=='previous_full'))
    write_json(ROOT/'audit.json',audit)
    lines=['# Exact PEGFAN + 零门控非局部分支：架构诊断','',
           '48 次正式训练全部完成。只用固定 split 0/1/2、seed=42，validation loss 选 checkpoint，500 epochs / patience=50；没有追加调参。下面均为同三个 split 的 mean ± sample std（%），不与前轮十 split 均值混作配对。',
           '[实现与控制变量协议](../../docs/architecture_augmentation_protocol.md)。训练主干为原 FSGNN，所有原输入、framelets 和 optimizer 保留。新分支只在自身进行 F→64/128 压缩，融合为 log_softmax(L_PEGFAN + alpha G)，alpha 初始为0。',
           '', '## 配对准确率','', '| 数据集 | PEGFAN | + T64 | + T + Haar64 | + T + Haar128 |','|---|---:|---:|---:|---:|']
    for d in DATASETS:lines.append('| '+d+' | '+' | '.join(cell(d,m) for m in MODELS)+' |')
    lines+=['','| 数据集 | 增量配置 | 相对 PEGFAN pp | 胜/平/负 |','|---|---|---:|---|']
    for d in DATASETS:
        for m in MODELS[1:]:
            r=stats[d,m];lines.append(f"| {d} | {NAMES[m]} | {float(r['delta_pp']):+.2f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=['','## post-T Haar 与分支宽度','',
            '两个64维配置共享初始主干、Transformer、Sheaf maps、输出头和实例化模块，区别是是否执行 post-T Haar + learned fusion。独立联合训练后 hierarchy 会随各自参数变化；这是整条分支的配置对照，不是固定 checkpoint 的纯线性算子比较。',
            '', '| 数据集 | Haar64 − T64 pp | Haar128 − Haar64 pp |','|---|---:|---:|']
    for d in DATASETS:
        diff=lambda a,b:float(stats[d,a]['mean_pct'])-float(stats[d,b]['mean_pct'])
        lines.append(f"| {d} | {diff('transformer_haar64','transformer64'):+.2f} | {diff('transformer_haar128','transformer_haar64'):+.2f} |")
    lines+=['','原 local channels 的输入 feature dimension 全部保留。第三项“取消全局 bottleneck”已体现在所有新配置中：变化的只是新分支宽度，不是把 PEGFAN 主干先压到64维。三个 split 不能支持稳定增益或显著性结论。本轮没有 T128-only 对照，因此 Haar128 的结果不能单独证明 post-T Haar 的必要性。',
            '', '## 门控是否实际起作用','',
            '| 数据集 | 分支 | mean alpha / mean abs(alpha) | mean gated/local logit norm | 同 checkpoint 开门控 − 关门控 pp |','|---|---|---:|---:|---:|']
    for d in DATASETS:
        for m in MODELS[1:]:
            gs=[g for g in gates if g['dataset']==d and g['model']==m]
            lines.append(f"| {d} | {NAMES[m]} | {np.mean([g['alpha'] for g in gs]):+.5f} / {np.mean([abs(g['alpha']) for g in gs]):.5f} | {np.mean([g['logit_norm_ratio'] for g in gs]):.4f} | {np.mean([g['gate_test_delta_pp'] for g in gs]):+.2f} |")
    lines+=['','这里关闭门控后的主干已经联合训练，不能把它当作独立 PEGFAN control。范数比对每个节点的 class logits 去均值后计算，排除 softmax 不敏感的公共偏置；alpha 有符号，mean abs(alpha) 防止不同 split 的正负号相互抵消；其大小不能单独衡量贡献。',
            '', '## Cornell：Transformer 前后表示','',
            'Cornell 为修复后的183节点，所有诊断 tokens=183。旧 replacement 模型复用前轮10个 full_no_akx checkpoint；本轮每个增量配置3个 checkpoint。另保存各自初始化时的相同诊断。两个时期均在 eval 下测量，未使用诊断结果选 epoch 或超参数。',
            '', '| 已选 checkpoint | 数量 | feature variance 前→后 | mean pair cosine 前→后 | centered energy fraction 前→后 | 两层 normalized attention entropy |','|---|---:|---|---|---|---|']
    for r in diagnostic_summary:
        if r['phase']!='selected':continue
        name='旧完整 replacement 模型' if r['model']=='previous_full' else NAMES[r['model']]
        lines.append(f"| {name} | {r['checkpoints']} | {r['feature_variance_before']:.5g} → {r['feature_variance_after']:.5g} | {r['mean_pairwise_cosine_before']:.5f} → {r['mean_pairwise_cosine_after']:.5f} | {r['centered_energy_fraction_before']:.5f} → {r['centered_energy_fraction_after']:.5f} | {r['layer0_normalized_entropy']:.5f} / {r['layer1_normalized_entropy']:.5f} |")
    lines+=['','旧完整模型必须逐 split 看：','', '| split | 准确率 % | cosine 前→后 | centered energy 前→后 |','|---:|---:|---|---|']
    for p in sorted([p for p in probes if p['model']=='previous_full'],key=lambda p:p['split']):
        a=p['selected']['representations']['before'];b=p['selected']['representations']['after']
        lines.append(f"| {p['split']} | {p['test_accuracy']*100:.2f} | {a['mean_pairwise_cosine']:.5f} → {b['mean_pairwise_cosine']:.5f} | {a['centered_energy_fraction']:.5f} → {b['centered_energy_fraction']:.5f} |")
    lines+=['','旧完整模型的 split 2、4 在 Transformer 后接近一致表示，且输入已很相似；另7个 split 的 cosine 反而下降。不能把所有 Cornell 低准确率统一归因于 global oversmoothing。',
            '本轮 Cornell 的 Haar64/Haar128 增量分支，在 Transformer 输入处就已高度同向（平均 cosine 0.99961 / 0.99996），输出进一步接近一致（0.99976 / 0.99998）。因此这两组的表示退化已发生在 Transformer 之前，不能仅归因于 attention。相比之下，T64 输出 cosine 约0.06889，节点间差异仍然很大。',
            '', '## 本轮支持的结论', '',
            '1. 保留完整 PEGFAN 主干后，先前的大幅性能下降消失；零门控增量结构适合作为后续验证基础。但这只说明避免了替换主干造成的信息损失，不等于新增分支已获得普遍增益。',
            '2. post-T Haar64 相对 T64 在不同图上有升有降，当前未证明其稳定必要性。Haar128 在 Chameleon 的三个 split 都有小幅提升，在 Squirrel 两升一降；仍需区分分支宽度与 Haar 的作用。',
            '3. Chameleon 的 Haar128 虽比独立 PEGFAN 高0.88 pp，但同一 checkpoint 关门控后反而比开门控高1.32 pp。训练耦合与选模变化也可能贡献差异，不能直接把高出的0.88 pp解释为新增非局部信息的预测收益。',
            '4. 门控不是停在零；Cornell Haar分支也能明显改变预测，但其表示近乎常量，且输入已经退化。因此下一步若验证非局部支路的独立增量，应优先固定已训练 PEGFAN 主干，仅训练分支/门控，并使用相同 validation 规则；本轮没有追加这项实验。',

            'feature variance 受 LayerNorm 缩放影响，不能只凭绝对方差增大或减小判断过平滑；normalized attention entropy 高也不充分，残差通路能保留差异。因此同时报告 cosine、相对 centered energy 和各 residual block 输出。',
            '', '## 审计与文件','',
            '48个 checkpoint、源码/图/特征/原 Haar cache/split 哈希、validation checkpoint 选择与 early stopping 均通过审计。12个原 PEGFAN 对照复现前轮相同 split 的 accuracy 与最佳 epoch；36个增量模型的初始输出与主干逐值一致。46项单元测试通过，包括零门控时连续训练轨迹与原 PEGFAN 相同、门控开启后的梯度、分支置换等变性、逐头 attention 概率与原实现一致。',
            '`paired_splits.csv`：逐 split 相对独立 PEGFAN 的差值；`gates.csv`：门控及同 checkpoint 关门控诊断；`cornell_representations.csv`：每层表示；`cornell_attention.csv`：逐层逐 head entropy；`diagnostics/`：初始化与最终 checkpoint 原始诊断。',
            '8次两 epoch GPU smoke test单独存放，不计正式训练；旧 Cornell checkpoint 只读诊断不计新增训练。经典 Chameleon/Squirrel 和单 seed 小样本诊断不作为最终论文 benchmark。','']
    (ROOT/'analysis.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    for i,m in enumerate(MODELS):
        ax.bar(np.arange(4)+(i-1.5)*.2,[float(stats[d,m]['mean_pct']) for d in DATASETS],.2,
               yerr=[float(stats[d,m]['std_pct']) for d in DATASETS],capsize=2,label=NAMES[m])
    ax.set_xticks(range(4));ax.set_xticklabels(['Film','Chameleon','Squirrel','Cornell (repaired)'])
    ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');ax.legend(ncol=4,fontsize=9)
    ax.set_title('Exact PEGFAN + gated global branch: fixed splits 0–2, seed 42')
    fig.savefig(ROOT/'architecture_comparison.png',dpi=180);fig.savefig(ROOT/'architecture_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    colors=plt.get_cmap('tab10')
    for i,m in enumerate(groups):
        for p in [p for p in probes if p['model']==m]:
            a=p['selected']['representations']['before'];b=p['selected']['representations']['after']
            axes[0].plot([0,1],[a['mean_pairwise_cosine'],b['mean_pairwise_cosine']],'-o',color=colors(i),alpha=.5)
            axes[1].plot([0,1],[a['centered_energy_fraction'],b['centered_energy_fraction']],'-o',color=colors(i),alpha=.5)
        axes[0].plot([],[],color=colors(i),label='Previous full (10 splits)' if m=='previous_full' else NAMES[m]+' (3 splits)')
        values=[r for r in attention if r['model']==m and r['phase']=='selected']
        axes[2].plot([0,1],[np.mean([r['normalized_entropy'] for r in values if r['layer']==j]) for j in (0,1)],'o-',color=colors(i))
    for ax,title in zip(axes[:2],['Mean off-diagonal cosine','Centered energy fraction']):
        ax.set_xticks([0,1]);ax.set_xticklabels(['Before Transformer','After Transformer']);ax.set_ylim(0,1.03);ax.set_title(title)
    axes[0].legend(fontsize=8,loc='lower left');axes[2].set_xticks([0,1]);axes[2].set_xticklabels(['Layer 1','Layer 2']);axes[2].set_ylim(0,1.03);axes[2].set_title('Attention entropy / log(183)')
    fig.suptitle('Cornell: residual blocks can retain differences despite diffuse attention')
    fig.savefig(ROOT/'cornell_transformer_diagnostics.png',dpi=180);fig.savefig(ROOT/'cornell_transformer_diagnostics.pdf');plt.close(fig)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
