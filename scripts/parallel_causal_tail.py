"""Run remaining full_with_akx splits in two independent training processes.

Calls the already-frozen run_trial implementation without changing training
semantics. Only the parent appends metrics, preventing concurrent log writes.
"""
import os,sys,json,time,signal,subprocess,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
from compare_pegfan import digest,write_json
from run_causal_paths import run_trial,records,summary
from pegfan_baseline import prepare_pegfan_channels
from process import full_load_data
from ablation_ops import original_partitions
from framelets_utils import get_spatial_framelets_list
ROOT=Path('results/causal_ablations/paths');TAIL=ROOT/'parallel_tail'

def worker(index):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    jobs=json.loads((TAIL/'jobs.json').read_text())[index::2]
    config=json.loads((ROOT/'protocol.json').read_text())['config'];config['output']=ROOT;args=SimpleNamespace(**config)
    for d in dict.fromkeys(j['dataset'] for j in jobs):
        loaded=full_load_data(d,Path('splits')/f'{d}_split_0.6_0.2_0.npz',return_sparse=True)
        base=prepare_pegfan_channels(loaded[3],loaded[1],loaded[2],d)
        parts=original_partitions(get_spatial_framelets_list(None,d,8)[0])
        for job in [j for j in jobs if j['dataset']==d]:
            split=job['split'];key=f'{d}_full_with_akx_split{split}_seed42'
            result=run_trial(args,d,split,'full_with_akx',loaded,base,parts,None)
            write_json(TAIL/(key+'.json'),result);print('RESULT '+json.dumps(result),flush=True)
        del loaded,base;gc.collect()

def parent():
    TAIL.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for source,sha in protocol['sources'].items():assert digest(source)==sha
    pid=json.loads(Path('results/causal_ablations/launcher_paths.json').read_text())['pid']
    while True:
        progress=json.loads((ROOT/'progress.json').read_text())
        if progress['status']=='failed':raise RuntimeError(progress)
        if progress['status']=='complete':return
        if progress['completed']>=360:break
        time.sleep(1)
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'run_causal_paths.py' not in command:raise RuntimeError('Unexpected process identity')
    os.kill(pid,signal.SIGINT)
    for _ in range(100):
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('Sequential process did not stop')
    rs=records(ROOT/'metrics.jsonl');done={(r['dataset'],r['split'],r['model']) for r in rs}
    pending=[dict(dataset=d,split=s) for d in protocol['config']['datasets'] for s in protocol['config']['splits'] if (d,s,'full_with_akx') not in done]
    if any((d,s,m) not in done for d in protocol['config']['datasets'] for s in protocol['config']['splits'] for m in protocol['config']['variants'] if m!='full_with_akx'):
        raise RuntimeError('Earlier stages not complete')
    write_json(TAIL/'jobs.json',pending)
    write_json(TAIL/'execution.json',dict(workers=2,unchanged_training_function='run_causal_paths.run_trial',source_sha256=digest(__file__),completed_before_parallel=len(rs),pending=len(pending),timings_not_comparable=True))
    procs=[]
    try:
        for i in range(2):
            log=(TAIL/f'worker{i}.log').open('w')
            procs.append(subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(i)],stdout=log,stderr=subprocess.STDOUT))
        while True:
            for path in sorted(TAIL.glob('*_seed42.json')):
                r=json.loads(path.read_text());key=r['dataset'],r['split'],r['model']
                if key in done:continue
                with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps(r)+'\n');f.flush()
                rs.append(r);done.add(key);summary(ROOT,rs)
                print('COLLECTED',key,'completed',len(rs),flush=True)
            write_json(ROOT/'progress.json',dict(status='running',completed=len(rs),expected=390,variant='full_with_akx',workers=2))
            if all(p.poll() is not None for p in procs):
                if any(p.returncode for p in procs):raise RuntimeError('Parallel training worker failed; inspect worker logs')
                # Result files are written before each worker exits and have
                # been collected above after the final process boundary.
                if len(rs)!=390:
                    time.sleep(.1);continue
                break
            time.sleep(1)
        write_json(ROOT/'progress.json',dict(status='complete',completed=len(rs),expected=390,tail_workers=2))
    except BaseException as e:
        for p in procs:
            if p.poll() is None:p.terminate()
        for p in procs:p.wait()
        write_json(ROOT/'progress.json',dict(status='failed',completed=len(rs),expected=390,error=repr(e)));raise

if __name__=='__main__':
    if '--worker' in sys.argv:worker(int(sys.argv[-1]))
    else:parent()
