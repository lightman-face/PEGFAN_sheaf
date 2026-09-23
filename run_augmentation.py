"""Small, fixed architecture probe: exact PEGFAN plus zero-gated global signals."""
import argparse
import csv
import gc
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from augmentation_ops import VARIANTS, make_augmented, transformer_diagnostics
from compare_pegfan import digest, write_json, synchronize
from pegfan_baseline import prepare_pegfan_channels
from process import full_load_data


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def tensor_digest(state):
    hasher = hashlib.sha256()
    for key, value in sorted(state.items()):
        hasher.update(key.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def summary(out, rows):
    controls = {(r['dataset'], r['split']): r for r in rows if r['model'] == 'pegfan'}
    results = []
    for dataset, variant in dict.fromkeys((r['dataset'], r['model']) for r in rows):
        group = [r for r in rows if r['dataset'] == dataset and r['model'] == variant]
        acc = np.array([100 * r['test_accuracy'] for r in group])
        delta = [100 * (r['test_accuracy'] - controls[dataset, r['split']]['test_accuracy'])
                 for r in group if (dataset, r['split']) in controls]
        results.append(dict(dataset=dataset, model=variant, runs=len(group), mean_pct=float(acc.mean()),
            std_pct=float(acc.std(ddof=1)) if len(acc) > 1 else 0.,
            delta_pp=float(np.mean(delta)) if delta else None,
            wins=sum(d > 1e-8 for d in delta), ties=sum(abs(d) <= 1e-8 for d in delta), losses=sum(d < -1e-8 for d in delta)))
    if results:
        with (out / 'summary.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)


def trial(args, dataset, split, variant, loaded, base):
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    device = torch.device('cuda:0')
    adj, _, _, x, y, _, _, _, nf, nc = loaded
    x, y = x.to(device), y.to(device)
    channels = [h.to(device) for h in base]
    with np.load(Path('splits') / f'{dataset}_split_0.6_0.2_{split}.npz') as f:
        train, val, test = [torch.as_tensor(f[k], dtype=torch.bool, device=device) for k in ('train_mask','val_mask','test_mask')]
    if not torch.all(train.int() + val.int() + test.int() == 1):
        raise ValueError('Masks must form a disjoint partition')
    model, optimizer = make_augmented(nf, len(channels), nc, variant, adj, device)
    augmented = variant != 'pegfan'
    backbone = model.backbone if augmented else model
    initial_hash = tensor_digest(backbone.state_dict())
    forward = (lambda: model(channels, x)) if augmented else (lambda: model(channels, True))
    key = f'{dataset}_{variant}_split{split}_seed42'
    diagnostic = {}
    model.eval()
    with torch.no_grad():
        reference = backbone(channels, True)
        if augmented and dataset == 'cornell':
            with transformer_diagnostics(model.branch.transformer) as diag:
                initial = forward()
            diagnostic['initial'] = diag
        else:
            initial = forward()
        initial_error = float((reference - initial).abs().max())
        if not torch.equal(reference, initial):
            raise RuntimeError('Zero gate changed PEGFAN initial function')
    history = []; best = float('inf'); stale = 0; best_state = None
    synchronize(device); start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(); out = forward(); loss = F.nll_loss(out[train], y[train])
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite train loss')
        loss.backward()
        gate_grad = float(model.alpha.grad) if augmented else None
        optimizer.step(); model.eval()
        with torch.no_grad():
            out = forward(); vl = F.nll_loss(out[val], y[val]).item()
            va = (out[val].argmax(1) == y[val]).float().mean().item()
        if not np.isfinite(vl): raise RuntimeError('Nonfinite val loss')
        record = dict(epoch=epoch, train_loss=loss.item(), validation_loss=vl, validation_accuracy=va)
        if augmented: record.update(alpha=float(model.alpha), alpha_gradient=gate_grad)
        history.append(record)
        if vl < best:
            best = vl; best_epoch = epoch; best_va = va; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if epoch == 1 or epoch % 100 == 0:
            print(f'{dataset} {variant} split={split} epoch={epoch} val={vl:.5f}' +
                  (f' alpha={float(model.alpha):+.5f}' if augmented else ''), flush=True)
        if stale >= args.patience: break
    synchronize(device); seconds = time.perf_counter() - start
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        if augmented and dataset == 'cornell':
            with transformer_diagnostics(model.branch.transformer) as diag:
                out = forward()
            diagnostic['selected'] = diag
        else: out = forward()
        restored = F.nll_loss(out[val], y[val]).item()
        if not np.isclose(restored, best, atol=1e-5, rtol=1e-5): raise RuntimeError('Restore mismatch')
        acc = (out[test].argmax(1) == y[test]).double().mean().item()
        metadata = dict(initial_output_max_error=initial_error, backbone_initial_sha256=initial_hash,
                        local_input_width=nf, local_channels=len(channels),
                        backbone_parameters=sum(p.numel() for p in backbone.parameters()))
        if augmented:
            base_out = F.log_softmax(model.last_base_logits, dim=-1)
            b = model.last_base_logits - model.last_base_logits.mean(-1, keepdim=True)
            g = model.alpha * model.last_branch_logits
            g = g - g.mean(-1, keepdim=True)
            metadata.update(alpha=float(model.alpha), branch_width=model.branch.width,
                gate_disabled_accuracy=float((base_out[test].argmax(1) == y[test]).double().mean()),
                gate_changed_test_predictions=int((base_out[test].argmax(1) != out[test].argmax(1)).sum()),
                gated_to_local_logit_norm_ratio=float(g.norm() / b.norm().clamp_min(1e-12)),
                tokens=model.branch.hierarchy.num_tokens, node_counts=model.branch.hierarchy.node_counts,
                largest_region_fraction=float(model.branch.hierarchy.masses[model.branch.hierarchy.token_level].max() / len(x)))
            if dataset == 'cornell' and metadata['tokens'] != len(x):
                raise RuntimeError('Cornell should not be spatially pooled')
    ckpt = args.output / 'checkpoints' / (key + '.pt')
    torch.save(best_state, ckpt)
    write_json(args.output / 'histories' / (key + '.json'), history)
    if diagnostic:
        diagnostic.update(dataset=dataset, split=split, model=variant, checkpoint_sha256=digest(ckpt), tokens=metadata['tokens'])
        write_json(args.output / 'diagnostics' / (key + '.json'), diagnostic)
    row = dict(dataset=dataset, model=variant, split=split, seed=42, test_accuracy=acc,
        validation_loss=best, validation_accuracy=best_va, epochs=epoch, best_epoch=best_epoch,
        training_seconds=seconds, parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_sha256=digest(ckpt), split_sha256=digest(Path('splits') / f'{dataset}_split_0.6_0.2_{split}.npz'),
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20, **metadata)
    del optimizer, model, backbone, channels, x, y, best_state, out
    gc.collect(); torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['film','chameleon','squirrel','cornell'])
    p.add_argument('--splits', nargs='+', type=int, default=[0,1,2])
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--epochs', type=int, default=500); p.add_argument('--patience', type=int, default=50)
    p.add_argument('--output', type=Path, default=Path('results/architecture_augmentation'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    for sub in ('checkpoints','histories','diagnostics'): (args.output/sub).mkdir(parents=True, exist_ok=True)
    sources = ['run_augmentation.py','augmentation_ops.py','model.py','pegfan_baseline.py','process.py',
               'sheaf_coarsening.py','framelets_utils.py','utils.py','compare_pegfan.py']
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != 'resume'}
    protocol = dict(config=config, seed=42, torch=torch.__version__, numpy=np.__version__,
        sources={s:digest(s) for s in sources},
        data={d:{f:digest(Path('new_data')/d/f) for f in ['out1_node_feature_label.txt','out1_graph_edges.txt']} for d in args.datasets},
        framelet_caches={d:digest(Path('framelets/8')/(d+'.pickle')) for d in args.datasets},
        description='Unmodified PEGFAN backbone, original full-width channels. Add alpha*global_logits, alpha0=0. Dynamic sheaf hierarchy only in global branch. Branch RNG isolated. Fixed split 0-2 architecture screen; no hyperparameter selection.')
    pp = args.output/'protocol.json'
    if pp.exists():
        if not args.resume or json.loads(pp.read_text()) != protocol: raise RuntimeError('Protocol changed or --resume missing')
    else: write_json(pp, protocol)
    path = args.output/'metrics.jsonl'; rows = records(path)
    done = {(r['dataset'],r['split'],r['model']) for r in rows}
    expected = len(args.datasets)*len(args.splits)*len(args.variants)
    try:
        for variant in args.variants:
            for dataset in args.datasets:
                if all((dataset,s,variant) in done for s in args.splits): continue
                loaded = full_load_data(dataset, Path('splits')/f'{dataset}_split_0.6_0.2_0.npz', return_sparse=True)
                base = prepare_pegfan_channels(loaded[3],loaded[1],loaded[2],dataset)
                for split in args.splits:
                    if (dataset,split,variant) in done: continue
                    write_json(args.output/'progress.json',dict(status='running',completed=len(rows),expected=expected,dataset=dataset,split=split,variant=variant))
                    row = trial(args,dataset,split,variant,loaded,base)
                    with path.open('a') as f: f.write(json.dumps(row)+'\n'); f.flush()
                    rows.append(row); done.add((dataset,split,variant)); summary(args.output,rows)
                    print('RESULT '+json.dumps(row),flush=True)
                del loaded,base; gc.collect()
        write_json(args.output/'progress.json',dict(status='complete',completed=len(rows),expected=expected))
    except Exception as e:
        write_json(args.output/'progress.json',dict(status='failed',completed=len(rows),expected=expected,error=repr(e))); raise
if __name__ == '__main__': main()
