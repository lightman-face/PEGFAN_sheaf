"""Original PEGFAN channels and FSGNN, reusable under a common evaluator."""
from pathlib import Path
import torch
from framelets_utils import get_spatial_framelets_list
from model import FSGNN


def prepare_pegfan_channels(features, adjacency, adjacency_with_loops, dataset,
                            h=8, channel_type='c', feature_type='heterophily', hops=3):
    """Match pegfan_node_class.py's a/b/c channel construction exactly.

    Use the repository's original framelets; no sheaf partition enters this
    baseline. The caller controls the device and can cache these static inputs.
    """
    if channel_type not in ('a', 'b', 'c'):
        raise ValueError('channel_type must be a, b, or c')
    if feature_type not in ('homophily', 'heterophily', 'all'):
        raise ValueError('invalid feature_type')
    if channel_type == 'c' and feature_type == 'all':
        raise ValueError('original type c requires homophily or heterophily')
    cache = Path('framelets') / str(h) / (dataset + '.pickle')
    if not cache.is_file():
        raise FileNotFoundError(f'Original PEGFAN framelet cache missing: {cache}')
    framelets, transposes = get_spatial_framelets_list(None, dataset, h)
    if any(matrix.shape[1] != features.shape[0] for matrix in framelets):
        raise ValueError('framelet cache does not match the dataset node count')
    channels = [features]
    no_loop, loop = features, features
    if channel_type != 'a':
        for _ in range(hops):
            no_loop = torch.sparse.mm(adjacency, no_loop)
            loop = torch.sparse.mm(adjacency_with_loops, loop)
            channels.extend((no_loop, loop))
        if feature_type == 'homophily':
            channels = [channels[index] for index in [0] + [2*i for i in range(1, hops+1)]]
        elif feature_type == 'heterophily':
            channels = [channels[index] for index in [0] + [2*i-1 for i in range(1, hops+1)]]
    x = features
    if channel_type == 'c':
        operator = adjacency_with_loops if feature_type == 'homophily' else adjacency
        x = torch.sparse.mm(operator, features)
    for matrix, transpose in zip(framelets, transposes):
        matrix, transpose = matrix.to(features.device), transpose.to(features.device)
        channels.append(torch.sparse.mm(transpose, torch.sparse.mm(matrix, x)))
    return channels


def make_pegfan(num_features, num_channels, hidden, num_classes, dropout, device,
                 lr_fc=.01, lr_attention=.02, weight_decay=.0005):
    model = FSGNN(num_features, num_channels, hidden, num_classes, dropout).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.fc2.parameters(), 'weight_decay': weight_decay, 'lr': lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': weight_decay, 'lr': lr_fc},
        {'params': [model.att], 'weight_decay': weight_decay, 'lr': lr_attention},
    ])
    return model, optimizer
