"""Quarantine legacy Cornell observations without rewriting raw run records."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for root in [Path('results/filtration_vs_pegfan'),Path('results/targeted_ablations')]:
    archive=root/'legacy_with_invalid_cornell';archive.mkdir(exist_ok=True)
    for name in ['summary.csv','paired_splits.csv','report.md','analysis.md','comparison.png','comparison.pdf','ablation_comparison.png','ablation_comparison.pdf']:
        path=root/name
        if not path.exists():continue
        if not (archive/name).exists():shutil.copy2(path,archive/name)
        if path.suffix=='.csv':
            with path.open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if r.get('dataset')!='cornell']
            with path.open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        elif path.suffix=='.md':
            lines=[line for line in path.read_text().splitlines() if not line.lower().startswith('| cornell')]
            path.write_text('> **Cornell 已从正式统计排除。原始记录仅作审计保留；不得用于架构判断。修复后的 Cornell 将在新数据版本单独复跑。**\n\n'+'\n'.join(lines)+'\n')
    records=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
    for name,excluded in [('valid_metrics.jsonl',False),('excluded_cornell_metrics.jsonl',True)]:
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in records if (r['dataset']=='cornell')==excluded))
    (root/'exclusions.json').write_text(json.dumps(dict(dataset='cornell',reason='Legacy feature/label file is Texas data, not upstream Cornell',exclude_all_legacy_runs=True,raw_metrics_retained_for_audit=True),indent=2)+'\n')
    valid=[r for r in records if r['dataset']!='cornell'];datasets=['film','chameleon','squirrel','texas','wisconsin']
    models=list(dict.fromkeys(r['model'] for r in valid));fig,ax=plt.subplots(figsize=(11,5));width=.8/len(models)
    for i,m in enumerate(models):
        arrays=[np.array([r['test_accuracy']*100 for r in valid if r['dataset']==d and r['model']==m]) for d in datasets]
        ax.bar(np.arange(5)+(i-(len(models)-1)/2)*width,[x.mean() for x in arrays],width,yerr=[x.std(ddof=1) for x in arrays],label=m,capsize=2)
    ax.set_xticks(range(5));ax.set_xticklabels(datasets);ax.set_ylabel('Test accuracy (%)');ax.set_ylim(0,100)
    ax.set_title('Validated datasets only: legacy Cornell excluded');ax.legend();fig.tight_layout()
    stem='comparison' if root.name=='filtration_vs_pegfan' else 'ablation_comparison'
    fig.savefig(root/(stem+'.png'),dpi=180);fig.savefig(root/(stem+'.pdf'));plt.close(fig)
    print(root,'valid runs',len(valid),'excluded',len(records)-len(valid))
