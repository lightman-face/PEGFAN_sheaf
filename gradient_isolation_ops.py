"""A fixed-tree, 128-wide matrix with explicit isolation of backbone gradients."""
import math
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from augmentation_ops import NonlocalBranch, AugmentedPEGFAN, rng_devices
from pegfan_baseline import make_pegfan
from ablation_ops import torch_transfers
from framelets_utils import haar_pool_details, haar_lift, haar_framelet_projections

VARIANTS = {
    'pegfan': ('baseline', False),
    't128': ('joint', False),
    't128_haar': ('joint', True),
    'stopgrad_t128': ('stopgrad', False),
    'stopgrad_t128_haar': ('stopgrad', True),
    'frozen_t128_haar': ('frozen', True),
}


class FixedTreeBranch(NonlocalBranch):
    """Reuse one previously learned tree; never regenerate partitions in training.

    The local sheaf update stays differentiable. Its scores do not move the
    fixed partition. Haar fusion retains a direct Transformer signal:
    Z_TF = Z_T + learned_fusion(projector_Haar_bands(Z_T)).
    """
    def set_partition(self, partitions, cut, device):
        self.fixed_projections = torch_transfers(partitions, 'mass', device)
        self.cut = int(cut)
        self.counts = [int(p.max()+1) for p in partitions]
        self.region_fraction = float(np.bincount(partitions[self.cut]).max()/len(partitions[0]))
        if self.counts[self.cut] > self.budget:
            raise ValueError('Fixed partition exceeds attention budget')
        self.hierarchy = SimpleNamespace(num_tokens=self.counts[self.cut], node_counts=self.counts)
        return self

    def _forward(self, x):
        h = F.gelu(self.input_projection(x))
        local, _ = self.sheaf(h, self.sources, self.targets, self.weights)
        ps, cut = self.fixed_projections, self.cut
        z, _ = haar_pool_details(local, ps[:cut])
        transformed = self.transformer(z.unsqueeze(1)).squeeze(1)
        if self.post_haar:
            bands = haar_framelet_projections(transformed, ps[cut:])
            counts = self.counts[cut:]
            scales = [math.log2(counts[0]/counts[-1])] + [math.log2(counts[0]/n) for n in counts[1:]]
            transformed = transformed + self.fusion(bands, scales)
        return self.output(haar_lift(transformed, ps[:cut]))


def raw_logits(backbone, channels):
    captured = []
    handle = backbone.fc2.register_forward_hook(lambda module, inputs, out: captured.append(out))
    try:
        backbone(channels, True)
    finally:
        handle.remove()
    return captured[0]


def make_matrix_model(nfeat, nchannels, nclass, variant, adjacency, partitions, cut, device):
    mode, haar = VARIANTS[variant]
    backbone, optimizer = make_pegfan(nfeat, nchannels, 64, nclass, .5, device)
    if mode == 'baseline':
        return backbone, optimizer
    with torch.random.fork_rng(devices=rng_devices(device)):
        branch = FixedTreeBranch(nfeat, 128, nclass, haar).to(device).set_graph(adjacency)
        branch.set_partition(partitions, cut, device)
    model = AugmentedPEGFAN(backbone, branch).to(device)
    optimizer.add_param_group(dict(params=branch.parameters(), lr=.001, weight_decay=.0005))
    optimizer.add_param_group(dict(params=[model.alpha], lr=.001, weight_decay=0.))
    return model, optimizer


def isolated_losses(local, branch, alpha, labels, mask, mode, local_active=True):
    """Return optimization loss and the actual predictive log probabilities.

    Detaching G alone would NOT isolate the local gradient: the CE residual
    still depends on G. Stopgrad uses CE(L,y) for the local learner and
    CE(stopgrad(L)+alpha*G,y) for the nonlocal learner.
    """
    if mode == 'joint':
        out = F.log_softmax(local + alpha*branch, dim=-1)
        return F.nll_loss(out[mask], labels[mask]), out
    if mode not in ('stopgrad', 'frozen'):
        raise ValueError(mode)
    out = F.log_softmax(local.detach() + alpha*branch, dim=-1)
    loss = F.nll_loss(out[mask], labels[mask])
    if mode == 'stopgrad' and local_active:
        loss = loss + F.nll_loss(F.log_softmax(local, dim=-1)[mask], labels[mask])
    return loss, out


def freeze_backbone(backbone):
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
        p.grad = None


def contribution_metrics(local, increment):
    """Frobenius ratio and flattened cosine at the actual logit fusion point.

    Also report class-centered quantities (softmax invariant) and mean row
    cosine. Undefined cosines at a zero gate are None rather than fake zeros.
    """
    local, increment = local.detach().double(), increment.detach().double()
    result = {}
    for prefix, a, b in (
        ('raw', local, increment),
        ('centered', local-local.mean(-1,keepdim=True), increment-increment.mean(-1,keepdim=True))):
        an, bn = a.norm(), b.norm()
        result[prefix+'_rg'] = float(bn/an) if an > 1e-30 else None
        result[prefix+'_cosine'] = float((a*b).sum()/(an*bn)) if an > 1e-30 and bn > 1e-30 else None
        row_a, row_b = a.norm(dim=-1), b.norm(dim=-1)
        valid = (row_a>1e-30)&(row_b>1e-30)
        result[prefix+'_cosine_valid_nodes'] = int(valid.sum())
        result[prefix+'_mean_node_cosine'] = float(((a*b).sum(-1)[valid]/(row_a[valid]*row_b[valid])).mean()) if valid.any() else None
    return result
