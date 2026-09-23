"""Audit completed paired runs and render an experiment table and figure."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    root = args.directory
    progress = json.loads((root/'progress.json').read_text())
    if progress['status'] != 'complete':
        raise RuntimeError('Wait for the experiment to complete before producing the final analysis')
    protocol = json.loads((root/'protocol.json').read_text())
    config = protocol['config']
    if set(config['models']) != {'pegfan', 'sheaf'}:
        raise ValueError('Paired analysis requires both pegfan and sheaf results')
    results = [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    keys = [(r['dataset'],r['split'],r['seed'],r['model']) for r in results]
    expected = {(d,s,k,m) for d in config['datasets'] for s in config['splits']
                for k in config['seeds'] for m in config['models']}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    for source, checksum in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == checksum, source
    excluded={'cornell'} if (root/'exclusions.json').exists() else set()
    display_datasets=[d for d in config['datasets'] if d not in excluded]
    rows = []
    lines = ['# 首轮异亲图对比实验', '',
             f"完成 {len(results)} 次训练；{len(config['datasets'])} 个数据集，"
             f"每个 {len(config['splits'])} 个官方划分，seed={config['seeds']}。", '',
             '两种模型按验证损失选取最佳权重，恢复后评估测试集。表中为测试准确率均值 ± 样本标准差（%）。'
             '这是固定超参数的首轮对比，未进行参数搜索，也不代表 PEGFAN 论文调优后的结果。', '',
             '| 数据集 | PEGFAN | Sheaf filtration | 差值（百分点） | 平均实际 tokens |',
             '|---|---:|---:|---:|---:|']
    for dataset in display_datasets:
        groups = {model:[r for r in results if r['dataset']==dataset and r['model']==model]
                  for model in ('pegfan','sheaf')}
        left = {(r['split'],r['seed']):r for r in groups['pegfan']}
        for result in groups['sheaf']:
            assert result['tokens'] <= config['token_budget']
            assert result['tokens'] == result['node_counts'][result['token_level']]
            assert result['split_sha256'] == left[result['split'],result['seed']]['split_sha256']
        row = {'dataset':dataset}
        for model,group in groups.items():
            scores = np.array([r['test_accuracy'] for r in group])*100
            row[model+'_mean'] = scores.mean()
            row[model+'_std'] = scores.std(ddof=1) if len(scores)>1 else 0.
            row[model+'_epoch_seconds'] = np.mean([r['mean_epoch_seconds'] for r in group])
            row[model+'_training_seconds'] = np.mean([r['training_seconds'] for r in group])
        row['tokens'] = np.mean([r['tokens'] for r in groups['sheaf']])
        row['max_region_pct'] = 100*np.mean([r['max_region_fraction'] for r in groups['sheaf']])
        row['singleton_tokens'] = np.mean([r['singleton_tokens'] for r in groups['sheaf']])
        rows.append(row)
        lines.append(f"| {dataset} | {row['pegfan_mean']:.2f} ± {row['pegfan_std']:.2f} | "
                     f"{row['sheaf_mean']:.2f} ± {row['sheaf_std']:.2f} | "
                     f"{row['sheaf_mean']-row['pegfan_mean']:+.2f} | {row['tokens']:.1f} |")
    lines += ['', '## 耗时与分区诊断', '',
              '训练耗时包含训练和验证；sheaf 每次前向重建 filtration。PEGFAN 的静态通道预处理耗时'
              '另存于 dataset_metadata.json，并在各划分间复用。两者参数量不同，逐次记录在 metrics.jsonl。', '',
              '| 数据集 | PEGFAN 秒/轮 | Sheaf 秒/轮 | 最大区域占原图比例（%） | 单节点 tokens |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['pegfan_epoch_seconds']:.4f} | "
                     f"{row['sheaf_epoch_seconds']:.4f} | {row['max_region_pct']:.2f} | {row['singleton_tokens']:.1f} |")
    lines += ['', '## 解释范围', '',
              '- 等变性、嵌套性、预算上限与重构测试通过，不等价于分类精度得到提升。',
              '- 节点数已小于预算的数据集没有在 Transformer 前压缩；其精度差距不能归因于预算压缩。',
              '- 大区域与大量单节点区域并存，符合 threshold-CC 的单链连接效应；这只是结构诊断，不能单独证明性能差距的因果来源。',
              '- 下一轮应独立比较输入/分支归一化、Transformer 与 Haar 消融、静态/动态层次以及评分学习；使用验证集选择配置，不能凭本轮测试结果直接挑选参数。', '']
    if excluded: lines.insert(2, '旧 Cornell 已从正式表格和图中排除；原始记录仅保留审计。')
    (root/'analysis.md').write_text('\n'.join(lines))
    audit = dict(status='passed', runs=len(results), unique_keys=True, source_hashes=True,
                 paired_split_hashes=True, all_token_budgets_satisfied=True)
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = np.arange(len(rows))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for model,offset,color,label in [('pegfan',-.19,'#3975ad',f"PEGFAN (type {config['pegfan_type']}, h={config['pegfan_h']})"),
                                         ('sheaf',.19,'#d58238','Sheaf filtration + Transformer')]:
            axes[0].bar(x+offset,[r[model+'_mean'] for r in rows],.38,
                        yerr=[r[model+'_std'] for r in rows],capsize=3,color=color,label=label)
            axes[1].bar(x+offset,[r[model+'_epoch_seconds'] for r in rows],.38,color=color,label=label)
        for axis in axes:
            axis.set_xticks(x)
            axis.set_xticklabels([r['dataset'].capitalize() for r in rows],rotation=25,ha='right')
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
        axes[0].set_ylim(0,100)
        axes[0].set_ylabel('Test accuracy (%)')
        run_count = len(config['splits']) * len(config['seeds'])
        axes[0].set_title(f'Mean and sample SD over {run_count} runs')
        axes[1].set_yscale('log')
        axes[1].set_ylabel('Seconds per training + validation epoch (log scale)')
        axes[1].set_title('Measured runtime; dynamic hierarchy included')
        axes[0].legend(loc='upper right',fontsize=8)
        fig.suptitle(f"Fixed-configuration first comparison (seeds: {config['seeds']})",fontsize=12)
        fig.tight_layout()
        fig.savefig(root/'comparison.png',dpi=180)
        fig.savefig(root/'comparison.pdf')
        plt.close(fig)
    print(json.dumps(audit))
    print((root/'analysis.md').read_text())


if __name__=='__main__':
    main()
