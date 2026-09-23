"""Train the sheaf-hierarchy / coarse Transformer / Haar node classifier."""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from model import SheafHaarTransformer
from process import full_load_data
from utils import accuracy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='cora')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--token-budget', type=int, default=512)
    parser.add_argument('--cluster-size', type=int, default=4)
    parser.add_argument('--residual-quantile', type=float, default=0.75)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--transformer-layers', type=int, default=2)
    parser.add_argument('--sheaf-rank', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:N')
    parser.add_argument('--checkpoint', type=Path, help='Optional best-model checkpoint path')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error('epochs and patience must be positive')
    if not 0 <= args.split <= 9:
        parser.error('split must be between 0 and 9')
    return args


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    )
    split_path = Path('splits') / f'{args.data}_split_0.6_0.2_{args.split}.npz'
    adjacency, _, _, features, labels, train_mask, val_mask, test_mask, nfeat, nclass = full_load_data(
        args.data, split_path, return_sparse=True,
    )
    features, labels = features.to(device), labels.to(device)
    train_mask, val_mask, test_mask = [mask.to(device) for mask in (train_mask, val_mask, test_mask)]
    masks = (train_mask, val_mask, test_mask)
    if any(mask.shape != labels.shape or not mask.any().item() for mask in masks):
        raise ValueError('train/validation/test masks must be nonempty and match node count')
    if (train_mask & val_mask | train_mask & test_mask | val_mask & test_mask).any().item():
        raise ValueError('train/validation/test masks must be disjoint')
    model = SheafHaarTransformer(
        nfeat, args.hidden, nclass, token_budget=args.token_budget,
        max_cluster_size=args.cluster_size, residual_quantile=args.residual_quantile,
        nhead=args.heads, transformer_layers=args.transformer_layers,
        sheaf_rank=args.sheaf_rank, dropout=args.dropout,
    ).to(device).set_graph(adjacency)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_accuracy, best_epoch = float('inf'), 0.0, 0
    best_state, stale_epochs = None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss = F.nll_loss(output[train_mask], labels[train_mask])
        if not torch.isfinite(loss).item():
            raise RuntimeError('non-finite training loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(features)
            val_loss = F.nll_loss(output[val_mask], labels[val_mask]).item()
            val_accuracy = accuracy(output[val_mask], labels[val_mask]).item()
        if not np.isfinite(val_loss):
            raise RuntimeError('non-finite validation loss')
        if val_loss < best_loss:
            best_loss, best_accuracy, best_epoch = val_loss, val_accuracy, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch:04d} train_loss={loss.item():.4f} '
                  f'val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} '
                  f'tokens={model.hierarchy.num_tokens}/{args.token_budget}', flush=True)
        if stale_epochs >= args.patience:
            break

    # Select by validation loss, then evaluate the test labels once.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(features)
        test_accuracy = accuracy(output[test_mask], labels[test_mask]).item()
    hierarchy = model.hierarchy
    result = dict(best_epoch=best_epoch, validation_accuracy=best_accuracy,
                  test_accuracy=test_accuracy, node_counts=hierarchy.node_counts,
                  token_level=hierarchy.token_level, virtual_merge_levels=hierarchy.virtual_merge_levels)
    print(f'Best epoch={best_epoch} val_acc={best_accuracy:.4f} test_acc={test_accuracy:.4f}')
    print(f'Hierarchy={hierarchy.node_counts}; token_level={hierarchy.token_level}; '
          f'virtual_merge_levels={hierarchy.virtual_merge_levels}')
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        torch.save(dict(model_state=best_state, config=config, metrics=result), args.checkpoint)
    return result


if __name__ == '__main__':
    train(parse_args())
