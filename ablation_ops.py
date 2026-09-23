"""Controlled PEGFAN ablations: topology, Haar normalization, and attention.

An implicit evaluation of original pairwise Haar Psi.T@Psi is exact because
sum_{i<j}(e_i-e_j)(e_i-e_j).T/k = I - 11.T/k. This uses EQUAL CHILD
weights, not the mass weighting introduced by the full new model.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from framelets_utils import get_spatial_framelets_list, haar_framelet_projections, haar_lift
from sheaf_coarsening import build_sheaf_filtration, select_hierarchy_levels
from utils import sparse_mx_to_torch_sparse_tensor


def canonical(labels):
    _,first,inverse=np.unique(np.asarray(labels),return_index=True,return_inverse=True)
    remap=np.empty(len(first),dtype=np.int64)
    remap[np.argsort(first)]=np.arange(len(first))
    return remap[inverse]


def original_partitions(framelets):
    """Recover the actual cached tree, without rerunning a Ward implementation.

    Each highpass row has support on two sibling scaling vectors. The union of
    all row supports at that level identifies parents; singleton parents carry
    over. The cache must be a hierarchical pairwise Haar frame (audited below).
    """
    n=framelets[0].shape[1];parent=list(range(n));size=[1]*n
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:
            if size[a]<size[b]:a,b=b,a
            parent[b]=a;size[a]+=size[b]
    partitions=[np.arange(n)]
    for band in framelets[1:]:
        band=band.coalesce().cpu();indices=band.indices().numpy();values=band.values().numpy()
        active=values!=0;rows,cols=indices[:,active]
        if len(rows):
            order=np.argsort(rows,kind='stable');rows,cols=rows[order],cols[order]
            begin=0
            for end in np.r_[np.flatnonzero(np.diff(rows))+1,len(rows)]:
                anchor=int(cols[begin])
                for v in cols[begin+1:end]:union(anchor,int(v))
                begin=end
        partitions.append(canonical([root(v) for v in range(n)]))
    if len(np.unique(partitions[-1]))!=1:
        raise ValueError('Cached bands do not form a rooted hierarchy')
    return partitions


def transfers(partitions,normalization='equal'):
    """Fine-to-coarse isometries, retaining repeated cuts as identity levels."""
    if normalization not in ('equal','mass'):raise ValueError('unknown normalization')
    partitions=[canonical(p) for p in partitions]
    n=len(partitions[0])
    if not np.array_equal(partitions[0],np.arange(n)):raise ValueError('first partition must be singleton')
    answer=[]
    for fine,coarse in zip(partitions,partitions[1:]):
        if fine.shape!=coarse.shape:raise ValueError('partition size mismatch')
        _,first=np.unique(fine,return_index=True);assignment=coarse[first]
        if not np.array_equal(assignment[fine],coarse):raise ValueError('partitions are not nested')
        nf,nc=int(fine.max()+1),int(coarse.max()+1)
        if normalization=='equal':values=1/np.sqrt(np.bincount(assignment)[assignment])
        else:values=np.sqrt(np.bincount(fine)/np.bincount(coarse)[assignment])
        answer.append(sp.csr_matrix((values,(np.arange(nf),assignment)),shape=(nf,nc)))
    return answer


def torch_transfers(partitions,normalization='equal',device='cpu',dtype=torch.float32):
    return [sparse_mx_to_torch_sparse_tensor(p).to(device=device,dtype=dtype) for p in transfers(partitions,normalization)]


def validate_cached_tree(framelets,partitions):
    """Check every cached band against equal-child implicit Haar on probes."""
    g=torch.Generator().manual_seed(927)
    probe=torch.randn(framelets[0].shape[1],7,generator=g)
    computed=haar_framelet_projections(probe,torch_transfers(partitions))
    errors=[]
    for matrix,band in zip(framelets,computed):
        expected=torch.sparse.mm(matrix.transpose(0,1),torch.sparse.mm(matrix,probe))
        errors.append((expected-band).abs().max().item())
        if not torch.allclose(expected,band,atol=3e-6,rtol=3e-5):
            raise ValueError('Recovered hierarchy does not reproduce original cached Haar operators')
    return errors


def new_partitions(n,scores,original_counts,budget=512):
    filtration=build_sheaf_filtration(n,scores['sources'],scores['targets'],-scores['energy'])
    counts=filtration.event_counts
    # Preserve the number of channels/parameters. Whole ties may undershoot a
    # target. Repeated cuts give zero detail bands, not arbitrary tie breaking.
    events=[int(np.searchsorted(-counts,-int(k))) for k in original_counts]
    matched=[canonical(filtration.partition(e)) for e in events]
    selection=select_hierarchy_levels(filtration,budget)
    dyadic=[canonical(filtration.partition(e)) for e in selection.events]
    return matched,dyadic,dict(matched_events=events,dyadic_events=list(selection.events),
                              token_level=selection.token_level,virtual_event=filtration.virtual_event,
                              connected_components=filtration.num_components)


def pegfan_replacement_channels(base_channels,partitions,normalization='equal'):
    # Type-c heterophily: base channels 0..3 are X, AX, A^2X, A^3X.
    bands=haar_framelet_projections(base_channels[1],torch_transfers(partitions,normalization,base_channels[1].device))
    if len(bands)!=len(base_channels)-4:raise ValueError('Ablation changed FSGNN channel count')
    return base_channels[:4]+bands


class OriginalHierarchyTransformer(nn.Module):
    """Residual bottleneck attention on the original hierarchy's budget cut.

    The original FSGNN, A^kX channels and equal-child Haar stay intact. A
    64-wide, 2-layer Transformer changes only the coarse framelet source AX.
    Its zero-initialized output adapter starts at the exact PEGFAN function.
    Saved local details are retained. This is a residual insertion ablation,
    not a claim to reproduce the full new model's replacement-style branch.
    """
    def __init__(self,classifier,base_channels,partitions,budget=512,hidden=64,dropout=.5):
        super().__init__();self.classifier=classifier;self.base_channels=base_channels
        device=base_channels[0].device;nfeat=base_channels[0].shape[1]
        self.projections=torch_transfers(partitions,device=device)
        counts=[int(p.max()+1) for p in partitions]
        self.cut=next(i for i,k in enumerate(counts) if k<=budget)
        z=base_channels[1]
        for p in self.projections[:self.cut]:z=torch.sparse.mm(p.transpose(0,1),z)
        self.register_buffer('coarse_source',z,persistent=False)
        self.input_adapter=nn.Linear(nfeat,hidden).to(device)
        self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,4,hidden*4,dropout),2).to(device)
        self.output_adapter=nn.Linear(hidden,nfeat).to(device)
        nn.init.zeros_(self.output_adapter.weight);nn.init.zeros_(self.output_adapter.bias)
    def forward(self):
        delta=self.output_adapter(self.transformer(self.input_adapter(self.coarse_source).unsqueeze(1)).squeeze(1))
        upper=haar_framelet_projections(delta,self.projections[self.cut:])
        channels=list(self.base_channels)
        indices=[4]+list(range(5+self.cut,len(channels)))
        for index,band in zip(indices,upper):
            channels[index]=channels[index]+haar_lift(band,self.projections[:self.cut])
        return self.classifier(channels,True)


def partition_diagnostics(partition,features,sources,targets,energy,hidden=None):
    """Label-free statistics. Entropies use natural logarithms.

    Feature variance is mean per-node squared Euclidean reconstruction error;
    sheaf energy is mean over undirected internal edges (and macro cluster mean).
    Singleton ratio is given both per region and per original node.
    """
    a=canonical(partition);n=len(a);sizes=np.bincount(a);k=len(sizes);prob=sizes/n
    incidence=sp.csr_matrix((np.ones(n),(a,np.arange(n))),shape=(k,n))
    def variance(x):
        x=np.asarray(x,dtype=np.float64);sums=incidence@x
        sse=float((x*x).sum()-(sums*sums/sizes[:,None]).sum())
        total=float(((x-x.mean(axis=0))**2).sum())
        return max(sse/n,0.),max(sse,0.)/total if total>0 else 0.
    fv,fr=variance(features)
    ordered=np.sort(sizes);gini=2*np.dot(np.arange(1,k+1),ordered)/(k*n)-(k+1)/k
    inside=a[sources]==a[targets];edge_count=np.bincount(a[sources[inside]],minlength=k)
    edge_sum=np.bincount(a[sources[inside]],weights=energy[inside],minlength=k)
    active=edge_count>0
    row=dict(nodes=n,regions=k,largest_cluster_ratio=float(sizes.max()/n),singleton_cluster_ratio=float(np.mean(sizes==1)),
             singleton_node_ratio=float(np.sum(sizes==1)/n),cluster_size_gini=float(gini),
             cluster_size_entropy=float(-(prob*np.log(prob)).sum()),
             cluster_size_entropy_normalized=float(-(prob*np.log(prob)).sum()/np.log(k)) if k>1 else 0.,
             within_cluster_feature_variance=fv,feature_variance_fraction=fr,
             internal_edges=int(inside.sum()),total_edges=len(sources),internal_edge_fraction=float(inside.mean()) if len(inside) else 0.,
             within_cluster_sheaf_energy=float(energy[inside].mean()) if inside.any() else None,
             macro_cluster_sheaf_energy=float((edge_sum[active]/edge_count[active]).mean()) if active.any() else None,
             clusters_without_internal_edges=int((~active).sum()))
    if hidden is not None:
        row['within_cluster_hidden_variance'],row['hidden_variance_fraction']=variance(hidden)
    return row


def label_diagnostics(partition,labels,mask):
    """Post-hoc only; this function must never be called during model selection."""
    a=canonical(partition);k=int(a.max()+1);labels=np.asarray(labels);mask=np.asarray(mask,dtype=bool)
    contingency=np.zeros((k,int(labels.max()+1)),dtype=np.int64)
    np.add.at(contingency,(a[mask],labels[mask]),1)
    totals=contingency.sum(1);present=totals>0;contingency=contingency[present];totals=totals[present]
    if not totals.size:return dict(labeled_nodes=0,observed_regions=0,cluster_label_entropy=None,cluster_label_purity=None)
    p=contingency/totals[:,None];logp=np.zeros_like(p);np.log(p,out=logp,where=p>0)
    entropy=-(p*logp).sum(1)
    return dict(labeled_nodes=int(totals.sum()),observed_regions=int(present.sum()),
                cluster_label_entropy=float(np.dot(totals,entropy)/totals.sum()),
                cluster_label_purity=float(contingency.max(1).sum()/totals.sum()))
