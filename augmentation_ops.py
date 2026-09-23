"""Exact PEGFAN plus a separately compressed, zero-gated nonlocal branch."""
import math
from contextlib import contextmanager
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from model import LightweightSheaf, MultiscaleFusion
from pegfan_baseline import make_pegfan
from sheaf_coarsening import (_as_symmetric_adjacency, build_sheaf_filtration,
                              select_hierarchy_levels, materialize_projections)
from framelets_utils import haar_pool_details, haar_lift, haar_framelet_projections
from utils import sparse_mx_to_torch_sparse_tensor

VARIANTS = {'pegfan': None, 'transformer64': (64, False),
            'transformer_haar64': (64, True), 'transformer_haar128': (128, True)}


def rng_devices(device):
    device = torch.device(device)
    return [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []


class NonlocalBranch(nn.Module):
    """Only this branch projects F to d_T; all local PEGFAN inputs stay intact."""
    def __init__(self, nfeat, width, nclass, post_haar, budget=512, dropout=.5):
        super().__init__()
        self.width, self.post_haar, self.budget = width, post_haar, budget
        self.input_projection = nn.Linear(nfeat, width)
        self.sheaf = LightweightSheaf(width, 16)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=dropout), 2)
        # Instantiate both variants identically; only use fusion when requested.
        self.fusion = MultiscaleFusion(width)
        self.output = nn.Linear(width, nclass)
        for name, dtype in [('sources', torch.long), ('targets', torch.long), ('weights', torch.float32)]:
            self.register_buffer(name, torch.empty(0, dtype=dtype), persistent=False)
        self.adjacency = None
        self.hierarchy = None
        self.training_calls = 0

    def set_graph(self, adjacency):
        self.adjacency = _as_symmetric_adjacency(adjacency)
        s, t = sp.triu(self.adjacency, k=1).nonzero()
        device = self.input_projection.weight.device
        self.sources = torch.as_tensor(s, dtype=torch.long, device=device)
        self.targets = torch.as_tensor(t, dtype=torch.long, device=device)
        self.weights = torch.as_tensor(np.asarray(self.adjacency[s, t]).reshape(-1),
                                       dtype=torch.float32, device=device)
        return self

    def _forward(self, x):
        if self.adjacency is None:
            raise RuntimeError('Call set_graph before forward')
        h = F.gelu(self.input_projection(x))
        local, energy = self.sheaf(h, self.sources, self.targets, self.weights)
        filtration = build_sheaf_filtration(len(x), self.sources.cpu().numpy(),
            self.targets.cpu().numpy(), -energy.detach().cpu().numpy(), 'virtual_root')
        self.hierarchy = materialize_projections(filtration, select_hierarchy_levels(filtration, self.budget))
        ps = [sparse_mx_to_torch_sparse_tensor(p).to(x.device) for p in self.hierarchy.projections]
        cut = self.hierarchy.token_level
        z, _ = haar_pool_details(local, ps[:cut])
        if len(z) > self.budget:
            raise RuntimeError('Attention budget exceeded')
        transformed = self.transformer(z.unsqueeze(1)).squeeze(1)
        if self.post_haar:
            bands = haar_framelet_projections(transformed, ps[cut:])
            counts = self.hierarchy.node_counts[cut:]
            scales = [math.log2(counts[0] / counts[-1])]
            scales += [math.log2(counts[0] / c) for c in counts[1:]]
            transformed = self.fusion(bands, scales)
        # Coarse information only; original PEGFAN already retains local details.
        return self.output(haar_lift(transformed, ps[:cut]))

    def forward(self, x):
        if not self.training:
            return self._forward(x)
        # Branch dropout cannot shift PEGFAN's original dropout RNG stream.
        with torch.random.fork_rng(devices=rng_devices(x.device)):
            seed = 100042 + self.training_calls
            torch.manual_seed(seed)
            if x.is_cuda:
                torch.cuda.manual_seed_all(seed)
            out = self._forward(x)
        self.training_calls += 1
        return out


class AugmentedPEGFAN(nn.Module):
    def __init__(self, backbone, branch):
        super().__init__()
        self.backbone, self.branch = backbone, branch
        self.alpha = nn.Parameter(torch.zeros(()))
        self.last_base_logits = self.last_branch_logits = None

    def forward(self, channels, x):
        # Read raw logits from the unmodified FSGNN; its input path and classifier
        # execute exactly as before. Temporary hook keeps no graph between calls.
        captured = []
        handle = self.backbone.fc2.register_forward_hook(lambda module, inputs, out: captured.append(out))
        try:
            self.backbone(channels, True)
        finally:
            handle.remove()
        base = captured[0]
        global_logits = self.branch(x)
        self.last_base_logits = base.detach()
        self.last_branch_logits = global_logits.detach()
        return F.log_softmax(base + self.alpha * global_logits, dim=-1)


def make_augmented(nfeat, channels, nclass, variant, adjacency, device):
    if variant not in VARIANTS:
        raise ValueError(variant)
    backbone, optimizer = make_pegfan(nfeat, channels, 64, nclass, .5, device)
    if variant == 'pegfan':
        return backbone, optimizer
    width, haar = VARIANTS[variant]
    # Baseline and augmented variants start with identical local parameters AND
    # identical RNG state for subsequent PEGFAN dropout.
    with torch.random.fork_rng(devices=rng_devices(device)):
        branch = NonlocalBranch(nfeat, width, nclass, haar).to(device).set_graph(adjacency)
    model = AugmentedPEGFAN(backbone, branch).to(device)
    optimizer.add_param_group(dict(params=branch.parameters(), lr=.001, weight_decay=.0005))
    optimizer.add_param_group(dict(params=[model.alpha], lr=.001, weight_decay=0.))
    return model, optimizer


def representation_stats(signal):
    h = signal.detach().reshape(-1, signal.shape[-1]).double()
    n = len(h)
    centered = h - h.mean(0, keepdim=True)
    u = F.normalize(h, dim=-1)
    cosine = ((u.sum(0).square().sum() - u.square().sum()) / (n * (n - 1))) if n > 1 else h.new_tensor(0.)
    return dict(nodes=n, width=h.shape[1], feature_variance=float(centered.square().mean()),
                mean_pairwise_cosine=float(cosine),
                centered_energy_fraction=float(centered.square().sum() / h.square().sum().clamp_min(1e-30)),
                mean_squared_norm=float(h.square().sum(1).mean()),
                zero_rows=int((h.norm(dim=-1) < 1e-12).sum()))


def attention_probabilities(module, query):
    """Exact pre-dropout, per-head attention for these unmasked self-attentions."""
    if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError('Unsupported attention augmentation')
    q, k, _ = F.linear(query, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
    n, b, d = q.shape
    heads, dh = module.num_heads, d // module.num_heads
    q = q.reshape(n, b, heads, dh).permute(1, 2, 0, 3) / math.sqrt(dh)
    k = k.reshape(n, b, heads, dh).permute(1, 2, 0, 3)
    return torch.softmax(q @ k.transpose(-2, -1), dim=-1)


@contextmanager
def transformer_diagnostics(transformer):
    """Read-only hooks: input/output, each residual block, each attention head."""
    if transformer.training:
        raise ValueError('Diagnostics must run in eval mode')
    result = dict(representations={}, attention=[])
    reps, handles = result['representations'], []
    handles.append(transformer.register_forward_pre_hook(
        lambda module, inputs: reps.update(before=representation_stats(inputs[0]))))
    handles.append(transformer.register_forward_hook(
        lambda module, inputs, out: reps.update(after=representation_stats(out))))
    for index, layer in enumerate(transformer.layers):
        def attention_hook(module, inputs, out, index=index):
            probs = attention_probabilities(module, inputs[0]).double()
            entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
            normalizer = math.log(probs.shape[-1]) if probs.shape[-1] > 1 else 1.
            for head in range(module.num_heads):
                result['attention'].append(dict(layer=index, head=head,
                    entropy_nats=float(entropy[:, head].mean()),
                    normalized_entropy=float(entropy[:, head].mean() / normalizer),
                    mean_max_probability=float(probs[:, head].max(-1).values.mean())))
            reps[f'layer{index}_attention_output'] = representation_stats(out[0])
        handles.append(layer.self_attn.register_forward_hook(attention_hook))
        handles.append(layer.register_forward_hook(
            lambda module, inputs, out, index=index: reps.update({f'layer{index}_output': representation_stats(out)})))
    try:
        with torch.no_grad():
            yield result
    finally:
        for handle in handles:
            handle.remove()
