import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sheaf_coarsening import (
    _as_symmetric_adjacency, build_sheaf_filtration,
    select_hierarchy_levels, materialize_projections,
)
from framelets_utils import haar_pool_details, haar_lift, haar_framelet_projections
from utils import sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp



class FSGNN(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dropout):
        super(FSGNN,self).__init__()
        self.fc2 = nn.Linear(nhidden*nlayers,nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)



    def forward(self,list_mat,layer_norm):

        mask = self.sm(self.att)
        list_out = list()
        for ind, mat in enumerate(list_mat):
            tmp_out = self.fc1[ind](mat)
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)

            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.fc2(out)


        return F.log_softmax(out, dim=1)


class FSGNN_Large(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dp1,dp2):
        super(FSGNN_Large,self).__init__()
        self.wt1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.fc2 = nn.Linear(nhidden*nlayers,nhidden)
        self.fc3 = nn.Linear(nhidden,nclass)
        self.dropout1 = dp1 
        self.dropout2 = dp2 
        self.act_fn = nn.ReLU()
        
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)


    def forward(self,list_adj,layer_norm,st=0,end=0):

        mask = self.sm(self.att)
        mask = torch.mul(len(list_adj),mask)

        list_out = list()
        for ind, mat in enumerate(list_adj):
            mat = mat[st:end,:].cuda()
            tmp_out = self.wt1[ind](mat)
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)

            list_out.append(tmp_out)


        final_mat = torch.cat(list_out, dim=1)

        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout1,training=self.training)
        out = self.fc2(out)

        out = self.act_fn(out)
        out = F.dropout(out,self.dropout2,training=self.training)
        out = self.fc3(out)

        return F.log_softmax(out, dim=1)




class LightweightSheaf(nn.Module):
    """Feature-conditioned diagonal cellular-sheaf restriction maps.

    A shared endpoint MLP makes swapping edge orientation swap the maps.
    Restrictions stay in [0.5, 1.5], avoiding the all-zero-map solution.
    The differentiable local update gives maps a classification gradient;
    the detached hard partition alone would not train them.
    """

    def __init__(self, hidden, rank=16):
        super().__init__()
        self.restriction = nn.Sequential(
            nn.Linear(2 * hidden, rank), nn.Tanh(), nn.Linear(rank, hidden),
        )
        nn.init.zeros_(self.restriction[-1].weight)
        nn.init.zeros_(self.restriction[-1].bias)
        self.step_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features, sources, targets, edge_weights):
        source, target = features[sources], features[targets]
        rho_source = 1 + 0.5 * torch.tanh(self.restriction(torch.cat((source, target), dim=-1)))
        rho_target = 1 + 0.5 * torch.tanh(self.restriction(torch.cat((target, source), dim=-1)))
        difference = rho_source * source - rho_target * target
        residual = difference.square().sum(dim=-1)
        weighted = edge_weights.unsqueeze(-1) * difference
        laplacian = torch.zeros_like(features)
        laplacian.index_add_(0, sources, rho_source * weighted)
        laplacian.index_add_(0, targets, -rho_target * weighted)
        degree = features.new_zeros(features.shape[0])
        degree.index_add_(0, sources, edge_weights)
        degree.index_add_(0, targets, edge_weights)
        local = features - torch.sigmoid(self.step_logit) * laplacian / degree.clamp_min(1).unsqueeze(-1)
        return local, residual


class MultiscaleFusion(nn.Module):
    """Fuse a variable number of Haar bands with learned per-node scale gates."""

    def __init__(self, hidden):
        super().__init__()
        self.lowpass = nn.Linear(hidden, hidden, bias=False)
        self.highpass = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(hidden + 1, 1)

    def forward(self, bands, scales):
        outputs, scores = [], []
        for index, (band, scale) in enumerate(zip(bands, scales)):
            transformed = self.lowpass(band) if index == 0 else self.highpass(band)
            descriptor = torch.cat((transformed, transformed.new_full((len(band), 1), scale)), dim=-1)
            outputs.append(transformed)
            scores.append(self.score(descriptor))
        weights = torch.softmax(torch.stack(scores, dim=0), dim=0)
        return (weights * torch.stack(outputs, dim=0)).sum(dim=0)


class SheafHaarTransformer(nn.Module):
    """Sheaf hierarchy -> budgeted Transformer -> Haar bands -> node logits.

    Call ``set_graph`` once (also after loading weights). Each forward rebuilds
    the hard hierarchy from current residuals by default. Graph and hierarchy
    are derived, label-free state, not learnable checkpoint parameters.
    """

    def __init__(self, nfeat, nhidden, nclass, token_budget=512,
                 disconnected_policy="virtual_root",
                 nhead=4, transformer_layers=2, sheaf_rank=16, dropout=0.5):
        super().__init__()
        if nhead < 1 or nhidden < 1 or nhidden % nhead:
            raise ValueError("nhidden must be positive and divisible by nhead")
        if token_budget < 1 or transformer_layers < 1 or sheaf_rank < 1:
            raise ValueError("budget/layers/rank must be positive")
        if isinstance(token_budget, bool) or not isinstance(token_budget, (int, np.integer)):
            raise ValueError("token_budget must be a positive integer")
        if disconnected_policy not in ("virtual_root", "forest"):
            raise ValueError("invalid disconnected policy")
        self.token_budget = token_budget
        self.disconnected_policy = disconnected_policy
        self.input_projection = nn.Linear(nfeat, nhidden)
        self.sheaf = LightweightSheaf(nhidden, sheaf_rank)
        # Sequence-first layout also works with the original PEGFAN torch 1.7.
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(nhidden, nhead, 4 * nhidden, dropout=dropout),
            transformer_layers,
        )
        self.fusion = MultiscaleFusion(nhidden)
        self.detail_projection = nn.Linear(nhidden, nhidden, bias=False)
        nn.init.eye_(self.detail_projection.weight)
        self.classifier = nn.Sequential(
            nn.Linear(2 * nhidden, nhidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(nhidden, nclass),
        )
        self.register_buffer("edge_sources", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_targets", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_weights", torch.empty(0), persistent=False)
        self._adjacency = None
        self.hierarchy = None
        self._torch_projections = None
        self._projection_key = None

    def set_graph(self, adjacency):
        """Set an undirected, loop-free graph without densifying adjacency."""
        self._adjacency = _as_symmetric_adjacency(adjacency)
        sources, targets = sp.triu(self._adjacency, k=1).nonzero()
        device = self.input_projection.weight.device
        self.edge_sources = torch.as_tensor(sources, dtype=torch.long, device=device)
        self.edge_targets = torch.as_tensor(targets, dtype=torch.long, device=device)
        self.edge_weights = torch.as_tensor(
            np.asarray(self._adjacency[sources, targets]).reshape(-1) if len(sources) else np.empty(0),
            dtype=self.input_projection.weight.dtype, device=device,
        )
        self.hierarchy = None
        self._torch_projections = None
        self._projection_key = None
        return self

    def forward(self, features, rebuild_hierarchy=True):
        if self._adjacency is None:
            raise RuntimeError("call set_graph(adjacency) before forward")
        if features.shape[0] != self._adjacency.shape[0]:
            raise ValueError("features and graph must have the same node count")
        hidden = F.gelu(self.input_projection(features))
        local, residual = self.sheaf(hidden, self.edge_sources, self.edge_targets, self.edge_weights)
        if rebuild_hierarchy or self.hierarchy is None:
            filtration = build_sheaf_filtration(
                len(features), self.edge_sources.cpu().numpy(), self.edge_targets.cpu().numpy(),
                -residual.detach().cpu().numpy(), self.disconnected_policy,
            )
            self.hierarchy = materialize_projections(
                filtration, select_hierarchy_levels(filtration, self.token_budget),
            )
            self._torch_projections = None
        key = (features.device, hidden.dtype)
        if self._torch_projections is None or key != self._projection_key:
            self._torch_projections = [
                sparse_mx_to_torch_sparse_tensor(p).to(device=features.device, dtype=hidden.dtype)
                for p in self.hierarchy.projections
            ]
            self._projection_key = key
        projections = self._torch_projections
        cut = self.hierarchy.token_level
        tokens, local_details = haar_pool_details(local, projections[:cut])
        if tokens.shape[0] > self.token_budget:
            raise RuntimeError("hierarchy exceeded the Transformer token budget")
        # All attention happens at the budget cut, before framelet analysis.
        transformed = self.transformer(tokens.unsqueeze(1)).squeeze(1)
        bands = haar_framelet_projections(transformed, projections[cut:])
        counts = self.hierarchy.node_counts[cut:]
        scales = [math.log2(counts[0] / counts[-1])]
        scales.extend(math.log2(counts[0] / count) for count in counts[1:])
        global_signal = self.fusion(bands, scales)
        details = [self.detail_projection(detail) for detail in local_details]
        lifted = haar_lift(global_signal, projections[:cut], details)
        return F.log_softmax(self.classifier(torch.cat((lifted, hidden), dim=-1)), dim=-1)


if __name__ == '__main__':
    pass




