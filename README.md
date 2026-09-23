Implementation of PEGFAN (Permutation Equivariant Graph Framelet Augmented Network), adapted from [FSGNN](https://github.com/sunilkmaurya/FSGNN) . Please refer to [our paper](https://ieeexplore.ieee.org/document/10466590)  ([arXiv link](https://arxiv.org/abs/2306.04265)) for details.

Corrigendum: Let $\textbf{S}:= \textbf{D}^{-1/2}\textbf{A}\textbf{D}^{-1/2}$ and $\tilde{\textbf{S}}:= \textbf{D}^{-1/2}(\textbf{A}+\textbf{I})\textbf{D}^{-1/2}$. The matrices $\tilde{\textbf{A}}$ and $\textbf{A}$ in the channel types (see Paragraph 3 and Remark 5 of Section IV of our paper) should be replaced by $\tilde{\textbf{S}}$ and $\textbf{S}$ for homophilous and heterophilous graphs, respectively. Results of (Chamelon, h = 8, Type a) and (Squrriel, h = 8, Type a) in Table 3 were obtained by splitting the matrix $\textbf{SX}$ instead of $\textbf{X}$ in Type-a channels. If replaced by $\textbf{X}$, the results are similar to those of (h = 4, Type a).

It would be very appreciated if you mention our work or apply our codes and cite
```
@article{li2024permutation,
  title={Permutation equivariant graph framelets for heterophilous graph learning},
  author={Li, Jianfei and Zheng, Ruigang and Feng, Han and Li, Ming and Zhuang, Xiaosheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```

A list of Python packages in our virtual environment is contained in packages.txt.
Other specs: Ubuntu 18.04, CUDA Version 11.0, Graphics Card: NVIDIA RTX 3090

To obtain result on Chameleon (Ours,Type c, h=4) in Table III, run the following command in shell:

bash grid_search_scripts/heterophily_search.sh chameleon

and select test-set result (right column in csv files) according to the best validation set result (left column in csv files).


To obtain other results, modify the hyperparameter in bash scripts.
(use homophily_search.sh for homophilous datasets(cora,citeseer,pubmed) and syn_search.sh for synthetic data, see the scripts for details)

Most of the results in our paper are collected in folder "result_collections", new results after running the codes will be stored in folder "results" and "syn_results".

Framelets are stored in folder "framelets". Delete the files then the framelets will regenerate by running the code.

## Sheaf-aware hierarchy + coarse Transformer + Haar framelets

The experimental backbone is implemented by `SheafHaarTransformer` in
`model.py`, with its training entry point in `sheaf_pegfan_node_class.py`:

```text
G, X -> learned diagonal sheaf restrictions -> residual-aware hierarchy
     -> pool to <= M regions, saving local details
     -> Transformer over region tokens
     -> Haar framelet projections of Transformer output on the same hierarchy
     -> learned multiscale fusion -> lift + saved details -> node classification
```

**Sheaf learning and partitioning.** An endpoint-shared MLP learns diagonal
restriction maps, initialized to identity and bounded between 0.5 and 1.5.
For each undirected edge it computes
`s_e = ||rho_(u,e) H_u - rho_(v,e) H_v||^2`.
Low-residual edges merge first, with a per-level child-count bound; higher
residuals delay merging. Coarse boundary scores average the original edge
residuals using edge weights. Residual is a learned relation-disagreement
score, not a class-boundary label or a guarantee of heterophily detection.

Hard partitions use detached scores and are rebuilt on every forward pass by
default. The same restrictions also drive a differentiable local sheaf
Laplacian update before pooling, so training-label classification loss trains
the restriction maps. There is no gradient through the discrete partition and
no objective that forces every edge residual to zero. Validation/test labels
are never used by the hierarchy. `sheaf_rank` controls the restriction MLP's
bottleneck width; it is not the rank of a full restriction matrix.

**One hierarchy, one budget cut.** Features stop at the first hierarchy level
with at most `M` regions (default 512). The final merge in that level stops at
exactly `M` when the original graph has more than `M` nodes. The hierarchy
continues above this cut solely to define multiscale Haar bands on the token
graph. The Transformer receives only the tokens at the cut, with no index-based
positional encoding. The implementation uses PyTorch's sequence-first
[`TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).

**Haar projections and local details.** Each assignment matrix `P_l` has
orthonormal columns, with entries `1 / sqrt(number_of_children)`. Pooling
computes `H_(l+1) = P_l.T H_l` and saves
`D_l = H_l - P_l H_(l+1)`. Lifting uses the same `P_l` and adds back a learned
linear transform of `D_l`. Above the token cut, the pairwise tight Haar
highpass projection is evaluated implicitly as `I - P_l P_l.T`, equivalent to
PEGFAN's pairwise `(e_i-e_j)/sqrt(k)` framelets within a parent's `k` children.
The lifted lowpass and detail bands sum to the Transformer output before
learned fusion. Singleton groups contribute zero detail. This avoids a dense
framelet basis or an independently generated partition.

**Boundary cases.** If the graph already fits the budget, feature pooling is
skipped. Once no inter-region edges remain, disconnected regions are grouped
deterministically if further hierarchy levels are needed. No graph edges are
invented; these potentially disconnected regions are reported through
`virtual_merge_levels`. A hard budget can require merging across high-residual
boundaries. Equal-residual ties and disconnected grouping use node order, so
this implementation does not claim exact permutation equivariance in those
cases. A budget of one yields a single token and only a lowpass global band.

Run from the repository root in an environment with these dependencies:

```bash
python -m pip install -r requirements-sheaf.txt
python sheaf_pegfan_node_class.py --data cora --split 0 --token-budget 512
python sheaf_pegfan_node_class.py --data texas --split 0 --token-budget 64 --device cpu
python -m unittest discover -s tests -v
```

The new entry point works with CPU or CUDA, keeps raw adjacency sparse, uses
validation loss for checkpoint selection, and evaluates test labels only after
restoring the best weights. `--checkpoint results/sheaf_cora.pt` optionally
saves weights, configuration, and metrics. Restore the model with the same
configuration and call `set_graph(adjacency)` to reconstruct graph-dependent
state. The original `pegfan_node_class.py` and `syn_pegfan_node_class.py` remain
the baseline entry points; the new branch does not use their framelet cache.

The tests cover boundary preservation, hard token limits (including isolated
nodes), Haar reconstruction and energy conservation, equivalence to pairwise
framelet projections, edge-orientation invariance, classifier gradients into
restriction maps, and weight reloads. No experimental accuracy improvement is
claimed without running the training comparisons.
