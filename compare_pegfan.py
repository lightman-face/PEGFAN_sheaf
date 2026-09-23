"""Paired fixed-configuration benchmark on the repository's official splits.

Both models select checkpoints by validation loss and evaluate test labels
once after restoring the checkpoint. This is not a reproduction of a tuned
paper result: the original PEGFAN architecture/channels are retained, while
both models share the evaluation protocol and epoch/patience limits.
"""
import argparse
import csv
import gc
import hashlib
import json
import platform
from pathlib import Path
import random
import time

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from model import SheafHaarTransformer
from pegfan_baseline import prepare_pegfan_channels, make_pegfan
from process import full_load_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['texas', 'cornell', 'wisconsin', 'chameleon', 'film', 'squirrel'])
    p.add_argument('--splits', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--models', nargs='+', choices=['pegfan', 'sheaf'], default=['pegfan', 'sheaf'])
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=.5)
    p.add_argument('--token-budget', type=int, default=512)
    p.add_argument('--sheaf-lr', type=float, default=.001)
    p.add_argument('--pegfan-lr', type=float, default=.01)
    p.add_argument('--pegfan-att-lr', type=float, default=.02)
    p.add_argument('--weight-decay', type=float, default=.0005)
    p.add_argument('--pegfan-h', type=int, choices=[4,8], default=8)
    p.add_argument('--pegfan-type', choices=['a','b','c'], default='c')
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', type=Path, default=Path('results/filtration_vs_pegfan'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.token_budget, args.threads) < 1:
        p.error('epochs, patience, budget and threads must be positive')
    if any(split not in range(10) for split in args.splits):
        p.error('official splits are 0..9')
    return args


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_trial(args, dataset, split, seed, name, data, channels_cpu, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    adjacency, features_cpu, labels_cpu, nfeat, nclass = data
    with np.load(Path('splits') / f'{dataset}_split_0.6_0.2_{split}.npz') as masks:
        train_mask, val_mask, test_mask = [torch.as_tensor(masks[key], dtype=torch.bool, device=device)
                                         for key in ('train_mask','val_mask','test_mask')]
    labels = labels_cpu.to(device)
    if any(mask.shape != labels.shape or not mask.any().item() for mask in (train_mask,val_mask,test_mask)):
        raise ValueError('invalid or empty split masks')
    if ((train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)).any().item():
        raise ValueError('overlapping split masks')
    if name == 'pegfan':
        channels = [channel.to(device) for channel in channels_cpu]
        model, optimizer = make_pegfan(nfeat, len(channels), args.hidden, nclass, args.dropout,
                                       device, args.pegfan_lr, args.pegfan_att_lr, args.weight_decay)
        forward = lambda: model(channels, True)
    else:
        features = features_cpu.to(device)
        model = SheafHaarTransformer(nfeat, args.hidden, nclass, token_budget=args.token_budget,
                                     dropout=args.dropout).to(device).set_graph(adjacency)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.sheaf_lr, weight_decay=args.weight_decay)
        forward = lambda: model(features)
    synchronize(device)
    setup_seconds = time.perf_counter() - start
    train_start = time.perf_counter()
    best_loss, best_epoch, best_state = float('inf'), 0, None
    best_val_accuracy, stale = 0., 0
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        output = forward()
        loss = F.nll_loss(output[train_mask], labels[train_mask])
        if not torch.isfinite(loss).item():
            raise RuntimeError('non-finite training loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            output = forward()
            val_loss = F.nll_loss(output[val_mask], labels[val_mask]).item()
            val_accuracy = (output[val_mask].argmax(1) == labels[val_mask]).float().mean().item()
        if not np.isfinite(val_loss):
            raise RuntimeError('non-finite validation loss')
        if val_loss < best_loss:
            best_loss, best_epoch, best_val_accuracy = val_loss, epoch, val_accuracy
            best_state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        synchronize(device)
        history.append(dict(epoch=epoch, train_loss=loss.item(), val_loss=val_loss,
                            val_accuracy=val_accuracy, seconds=time.perf_counter()-epoch_start,
                            tokens=model.hierarchy.num_tokens if name == 'sheaf' else None))
        if epoch == 1 or epoch % 50 == 0:
            print(f'{dataset} split={split} seed={seed} {name} epoch={epoch} '
                  f'train={loss.item():.4f} val={val_loss:.4f} val_acc={val_accuracy:.4f}', flush=True)
        if stale >= args.patience:
            break
    synchronize(device)
    training_seconds = time.perf_counter() - train_start
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = forward()
        restored_val_loss = F.nll_loss(output[val_mask], labels[val_mask]).item()
        # GPU scatter reductions may differ at floating-point precision.
        if not np.isclose(restored_val_loss, best_loss, atol=1e-5, rtol=1e-5):
            raise RuntimeError('restored checkpoint does not reproduce validation loss')
        test_accuracy = (output[test_mask].argmax(1) == labels[test_mask]).double().mean().item()
    synchronize(device)
    result = dict(dataset=dataset, model=name, split=split, seed=seed, nodes=len(features_cpu),
                  best_epoch=best_epoch, epochs=epoch, validation_loss=best_loss,
                  validation_accuracy=best_val_accuracy, test_accuracy=test_accuracy,
                  parameters=sum(p.numel() for p in model.parameters()),
                  setup_seconds=setup_seconds, training_seconds=training_seconds,
                  mean_epoch_seconds=training_seconds/epoch,
                  total_trial_seconds=time.perf_counter()-start,
                  peak_gpu_memory_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                  split_sha256=digest(Path('splits') / f'{dataset}_split_0.6_0.2_{split}.npz'))
    if name == 'sheaf':
        h = model.hierarchy
        result.update(tokens=h.num_tokens, token_level=h.token_level, node_counts=h.node_counts,
                      full_filtration_events=len(h.filtration.event_counts),
                      selected_events=list(h.selection.events), num_components=h.filtration.num_components,
                      virtual_merge_levels=h.virtual_merge_levels,
                      token_region_max_mass=int(h.masses[h.token_level].max()),
                      max_region_fraction=float(h.masses[h.token_level].max()/len(features_cpu)),
                      singleton_tokens=int((h.masses[h.token_level] == 1).sum()))
    else:
        result.update(channels=len(channels_cpu), framelet_cache_sha256=digest(
            Path('framelets') / str(args.pegfan_h) / f'{dataset}.pickle'))
    trial_id = f'{dataset}_{name}_split{split}_seed{seed}'
    write_json(args.output / 'histories' / (trial_id + '.json'), history)
    return result


def summarize(args, results):
    fields = ['dataset','model','runs','test_mean_pct','test_std_pct','validation_mean_pct',
              'mean_training_seconds','mean_epoch_seconds','mean_parameters','mean_tokens']
    rows=[]
    for dataset in args.datasets:
        for name in args.models:
            group=[r for r in results if r['dataset']==dataset and r['model']==name]
            if not group:
                continue
            accuracies=np.array([r['test_accuracy'] for r in group])*100
            rows.append(dict(dataset=dataset, model=name, runs=len(group),
                             test_mean_pct=float(accuracies.mean()),
                             test_std_pct=float(accuracies.std(ddof=1)) if len(group)>1 else 0.,
                             validation_mean_pct=float(np.mean([r['validation_accuracy'] for r in group])*100),
                             mean_training_seconds=float(np.mean([r['training_seconds'] for r in group])),
                             mean_epoch_seconds=float(np.mean([r['mean_epoch_seconds'] for r in group])),
                             mean_parameters=float(np.mean([r['parameters'] for r in group])),
                             mean_tokens=float(np.mean([r['tokens'] for r in group])) if name=='sheaf' else None))
    with (args.output / 'summary.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines=['# Sheaf filtration vs original PEGFAN', '',
           'Fixed-configuration first comparison; no hyperparameter search or claim of reproducing tuned paper scores.', '',
           'Both models use identical official masks, row-normalized features, paired seeds, validation-loss checkpoint selection, '
           'and one final test evaluation. The baseline retains original type-'+args.pegfan_type+
           f' channels, h={args.pegfan_h} cached framelets and FSGNN. Epoch cap={args.epochs}, patience={args.patience}.', '',
           'PEGFAN keeps its original normalized adjacency channel semantics; the sheaf branch uses an undirected graph. '
           'Training time includes dynamic sheaf hierarchy construction; static baseline channel preparation is recorded separately '
           'in dataset_metadata.json and reused across splits. Parameter counts are not matched. '
           'Standard deviation is the sample standard deviation across completed runs.', '',
           '| Dataset | Model | Completed runs | Test accuracy (%) | Train seconds/run | Tokens |',
           '|---|---|---:|---:|---:|---:|']
    for row in rows:
        token='—' if row['mean_tokens'] is None else f"{row['mean_tokens']:.1f}"
        lines.append(f"| {row['dataset']} | {row['model']} | {row['runs']} | "
                     f"{row['test_mean_pct']:.2f} ± {row['test_std_pct']:.2f} | {row['mean_training_seconds']:.2f} | {token} |")
    lines.extend(['', '## Paired differences', '', '| Dataset | Paired runs | Sheaf minus PEGFAN (percentage points) |', '|---|---:|---:|'])
    for dataset in args.datasets:
        baseline={(r['split'],r['seed']):r['test_accuracy'] for r in results if r['dataset']==dataset and r['model']=='pegfan'}
        differences=[100*(r['test_accuracy']-baseline[(r['split'],r['seed'])]) for r in results
                     if r['dataset']==dataset and r['model']=='sheaf' and (r['split'],r['seed']) in baseline]
        if differences:
            lines.append(f'| {dataset} | {len(differences)} | {np.mean(differences):+.2f} |')
    expected=len(args.datasets)*len(args.splits)*len(args.seeds)*len(args.models)
    lines.extend(['',f'Progress: {len(results)}/{expected} runs. Partial rows are provisional.', ''])
    (args.output/'report.md').write_text('\n'.join(lines))


def main():
    args=parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'histories').mkdir(exist_ok=True)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items() if key!='resume'}
    sources=['sheaf_coarsening.py','model.py','framelets_utils.py','process.py','utils.py','pegfan_baseline.py','compare_pegfan.py']
    protocol=dict(config=config, source_sha256={name:digest(name) for name in sources},
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   torch=torch.__version__, device=str(device),
                                   gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None))
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if not args.resume:
            raise ValueError('output already has a protocol; use --resume or a new output directory')
        if previous != protocol:
            raise ValueError('resume configuration, source or environment differs from recorded protocol')
    else:
        write_json(protocol_path, protocol)
    result_path=args.output/'metrics.jsonl'
    results=[json.loads(line) for line in result_path.read_text().splitlines() if line.strip()] if result_path.exists() else []
    done={(r['dataset'],r['split'],r['seed'],r['model']) for r in results}
    metadata_path=args.output/'dataset_metadata.json'
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    expected=len(args.datasets)*len(args.splits)*len(args.seeds)*len(args.models)
    summarize(args, results)
    try:
        for dataset in args.datasets:
            if all((dataset,split,seed,name) in done for split in args.splits for seed in args.seeds for name in args.models):
                continue
            start=time.perf_counter()
            loaded=full_load_data(dataset, Path('splits')/f'{dataset}_split_0.6_0.2_{args.splits[0]}.npz', return_sparse=True)
            adjacency, normalized, normalized_i, features, labels, _, _, _, nfeat, nclass=loaded
            data=(adjacency, features, labels, nfeat, nclass)
            feature_type='homophily' if dataset in ('cora','citeseer','pubmed') else 'heterophily'
            prep_start=time.perf_counter()
            channels_cpu=prepare_pegfan_channels(features, normalized, normalized_i, dataset,
                                                h=args.pegfan_h, channel_type=args.pegfan_type,
                                                feature_type=feature_type) if 'pegfan' in args.models else None
            metadata[dataset]=dict(nodes=len(features), directed_stored_edges=int(adjacency.nnz), nfeat=nfeat, nclass=nclass,
                                   pegfan_channel_preparation_seconds=time.perf_counter()-prep_start,
                                   total_data_preparation_seconds=time.perf_counter()-start,
                                   features_sha256=hashlib.sha256(features.numpy().tobytes()).hexdigest(),
                                   adjacency_sha256=hashlib.sha256(adjacency.indptr.tobytes()+adjacency.indices.tobytes()+adjacency.data.tobytes()).hexdigest())
            write_json(metadata_path, metadata)
            for split in args.splits:
                for seed in args.seeds:
                    for name in args.models:
                        key=(dataset,split,seed,name)
                        if key in done:
                            continue
                        write_json(args.output/'progress.json', dict(status='running', completed=len(results), expected=expected,
                                                                    dataset=dataset, split=split, seed=seed, model=name))
                        result=run_trial(args,dataset,split,seed,name,data,channels_cpu,device)
                        with result_path.open('a') as file:
                            file.write(json.dumps(result)+'\n'); file.flush()
                        results.append(result); done.add(key)
                        summarize(args,results)
                        print('RESULT '+json.dumps(result),flush=True)
                        gc.collect()
                        if device.type=='cuda':
                            torch.cuda.empty_cache()
            del channels_cpu, loaded, data
            gc.collect()
        write_json(args.output/'progress.json',dict(status='complete',completed=len(results),expected=expected))
    except Exception as error:
        write_json(args.output/'progress.json',dict(status='failed',completed=len(results),expected=expected,error=repr(error)))
        raise


if __name__=='__main__':
    main()
