"""Information-path interventions with explicit reconstruction controls."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model import SheafHaarTransformer
from framelets_utils import haar_pool_details,haar_lift,haar_framelet_projections
from ablation_ops import torch_transfers


def null_propagation_channels(channels):
    if len(channels)<5:raise ValueError('Expected X, AX, A2X, A3X and original Haar bands')
    return [channels[0]]+[torch.zeros_like(h) for h in channels[1:4]]+channels[4:]


def pool_lift_inputs(channels,projections):
    global_only=[];reconstructed=[];errors=[]
    for h in channels:
        z,details=haar_pool_details(h,projections)
        g=haar_lift(z,projections);r=haar_lift(z,projections,details)
        global_only.append(g);reconstructed.append(r)
        errors.append(dict(max_abs_error=float((h-r).abs().max()),relative_l2_error=float((h-r).norm()/h.norm().clamp_min(1e-12)),
                           removed_detail_energy_fraction=float((h-g).square().sum()/h.square().sum().clamp_min(1e-12))))
    return global_only,reconstructed,errors


class FeatureRoundTrip(nn.Module):
    """Shared learned linear F -> 64 -> F map, initialized to rank-64 SVD.

    Applied to every original PEGFAN input channel before the unchanged
    FSGNN. No residual bypass: this genuinely imposes a feature-rank limit.
    """
    def __init__(self,classifier,channels,basis):
        super().__init__();self.classifier=classifier;self.channels=channels
        f,r=basis.shape;self.encode=nn.Linear(f,r,bias=False).to(basis.device)
        self.decode=nn.Linear(r,f,bias=False).to(basis.device)
        with torch.no_grad():self.encode.weight.copy_(basis.T);self.decode.weight.copy_(basis)
    def forward(self):return self.classifier([self.decode(self.encode(h)) for h in self.channels],True)


CHAIN_MODES=['chain_local','chain_pool_global','chain_transformer_global','chain_haar_global',
             'chain_detail_only','chain_global_detail','chain_full_skip']

class FixedHierarchyChain(SheafHaarTransformer):
    """Same full-model modules/initialization; frozen tree separates path effects.

    Pure global/detail/global+detail comparisons ALL disable the separate h
    skip by replacing it with zeros. The final full-skip stage restores it.
    Analysis followed by summing bands is an identity, audited separately.
    """
    def __init__(self,nfeat,hidden,nclass,partitions,cut,mode,dropout=.5):
        super().__init__(nfeat,hidden,nclass,dropout=dropout)
        if mode not in CHAIN_MODES:raise ValueError(mode)
        self.mode=mode;self.partitions=partitions;self.cut=cut;self.fixed_projections=None
        self.counts=[int(np.max(p)+1) for p in partitions];self.last_haar_error=None
    def forward(self,features,haar_identity_check=False):
        if self._adjacency is None:raise RuntimeError('Call set_graph')
        if self.fixed_projections is None or self.fixed_projections[0].device!=features.device:
            self.fixed_projections=torch_transfers(self.partitions,'mass',features.device)
        ps=self.fixed_projections;cut=self.cut
        h=F.gelu(self.input_projection(features))
        local,_=self.sheaf(h,self.edge_sources,self.edge_targets,self.edge_weights)
        tokens,details=haar_pool_details(local,ps[:cut])
        if len(tokens)>512:raise RuntimeError('Exceeded attention budget')
        if self.mode=='chain_local':path=local
        elif self.mode=='chain_detail_only':
            path=haar_lift(torch.zeros_like(tokens),ps[:cut],[self.detail_projection(d) for d in details])
        else:
            z=tokens
            if self.mode!='chain_pool_global':z=self.transformer(z.unsqueeze(1)).squeeze(1)
            if self.mode in ('chain_haar_global','chain_global_detail','chain_full_skip'):
                bands=haar_framelet_projections(z,ps[cut:]);counts=self.counts[cut:]
                scales=[math.log2(counts[0]/counts[-1])]+[math.log2(counts[0]/n) for n in counts[1:]]
                z=self.fusion(bands,scales)
            elif haar_identity_check:
                bands=haar_framelet_projections(z,ps[cut:]);reconstructed=sum(bands)
                self.last_haar_error=float((reconstructed-z).abs().max());z=reconstructed
            use_detail=self.mode in ('chain_global_detail','chain_full_skip')
            path=haar_lift(z,ps[:cut],[self.detail_projection(d) for d in details] if use_detail else None)
        skip=h if self.mode=='chain_full_skip' else torch.zeros_like(h)
        return F.log_softmax(self.classifier(torch.cat((path,skip),-1)),-1)


class AkXClassifier(nn.Module):
    """Append learned local AkX features to the existing full-model classifier.

    Existing classifier weights/biases are copied. All appended columns start
    at zero, so the full-model function is unchanged at initialization. Off/on
    controls have identical modules and parameter counts, differing only in
    whether the appended per-node features are supplied or zeroed.
    """
    def __init__(self,classifier,propagation,nfeat,hidden,enabled):
        super().__init__();self.enabled=enabled;self.propagation=propagation
        self.proj=nn.ModuleList([nn.Linear(nfeat,hidden) for _ in range(3)])
        self.att=nn.Parameter(torch.ones(3));self.trunk=classifier
        old=self.trunk[0];new=nn.Linear(5*hidden,hidden)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:2*hidden].copy_(old.weight);new.bias.copy_(old.bias)
        self.trunk[0]=new
    def forward(self,signal):
        weights=torch.softmax(self.att,dim=0)
        local=torch.cat([F.normalize(layer(h),p=2,dim=1)*weights[i] for i,(layer,h) in enumerate(zip(self.proj,self.propagation))],-1)
        local=F.relu(local)
        if not self.enabled:local=torch.zeros_like(local)
        return self.trunk(torch.cat((signal,local),-1))


def make_full_with_akx(nfeat,hidden,nclass,adjacency,propagation,enabled,device):
    model=SheafHaarTransformer(nfeat,hidden,nclass,dropout=.5)
    model.classifier=AkXClassifier(model.classifier,propagation,nfeat,hidden,enabled)
    model=model.to(device).set_graph(adjacency)
    extra_ids={id(p) for p in model.classifier.proj.parameters()}|{id(model.classifier.att)}
    optimizer=torch.optim.Adam([
        dict(params=[p for p in model.parameters() if id(p) not in extra_ids],lr=.001,weight_decay=.0005),
        dict(params=model.classifier.proj.parameters(),lr=.01,weight_decay=.0005),
        dict(params=[model.classifier.att],lr=.02,weight_decay=.0005)])
    return model,optimizer
