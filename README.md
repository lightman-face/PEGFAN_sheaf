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

## Sheaf-guided equivariant filtration + Transformer + Haar

The new backbone is `SheafHaarTransformer` in `model.py`. Its hierarchy follows
[the theory and implementation specification](docs/sheaf_guided_equivariant_coarsening.md).

```text
G, X -> learned diagonal restrictions -> q_e = -sheaf_residual_e
     -> threshold connected-component filtration (whole equal-score batches)
     -> finest partition with <= token_budget regions
     -> Transformer -> Haar bands on selected coarser filtration levels
     -> learned fusion -> lift with saved local details -> node classification
```

`build_sheaf_filtration` stores a compact multiway merge forest independent of
any token budget. It never splits equal-score batches to obtain an exact token
count. `select_hierarchy_levels` selects the budget cut and dyadic local/global
scales, removing duplicate selections when a merge skips several targets.
`materialize_projections` creates sparse transfers only at those selected levels.
The original Ward-based framelet cache is used only by the PEGFAN baseline.

For disconnected graphs, the default `virtual_root` policy groups all final
components into one abstract root simultaneously, without adding graph edges.
This guarantees a feasible cut but can yield just one token when the number of
components exceeds the budget. The optional `forest` policy preserves the
components and rejects an unreachable budget. Graphs already below budget do
not pool features before attention.

Each transfer has entries `sqrt(child_node_count / parent_node_count)` and
orthonormal columns. Pooling saves `D = H - P(P.T H)`; lifting uses the same P.
This mass normalization makes skipped-level transfers equal their composition.
Haar analysis applies complementary projectors implicitly, so a large equal-score
merge requires no dense pairwise framelet basis or cluster-size cap.
The selected global scale count is at most `1 + floor(log2(actual_tokens))`;
the full filtration can have linear depth. Its construction sorts fixed edge
scores once and records only effective merge events, without saving a full
node assignment at every event or rebuilding every coarse adjacency.

Restrictions are diagonal, initialized at identity and bounded in [0.5, 1.5].
`sheaf_rank` is the restriction MLP bottleneck width. The hard filtration uses
detached scores and is rebuilt on every forward pass by default. Classification
gradients train the maps through the differentiable local sheaf update; gradients
do not pass through sorting or connected-component membership. No labels enter
the hierarchy. The Transformer has no node-ID positional encoding.

The complete deterministic forward map is permutation equivariant in exact
arithmetic; floating-point reductions are checked with tolerances. Independent
training-time dropout masks are equivariant in distribution. A high-residual
edge's endpoints may still connect via a low-residual path (single-link chaining),
so neither guaranteed class-boundary preservation nor perturbation stability is
claimed. Learned band fusion need not preserve the analysis energy identity.

Run a single training split or the tests:

```bash
python -m pip install -r requirements-sheaf.txt
python sheaf_pegfan_node_class.py --data texas --split 0 --token-budget 64 --device cpu
python sheaf_pegfan_node_class.py --data chameleon --split 0 --token-budget 512
python -m unittest discover -s tests -v
```

The entry point supports CPU/CUDA and an optional `--checkpoint` path. Checkpoint
selection uses validation loss, with test labels evaluated after best-weight
restoration. Restoring weights requires the same model configuration and
`set_graph(adjacency)`; the default hierarchy is then reconstructed from scores.
The old `--cluster-size` and `--residual-quantile` options have been removed from
this branch because they conflict with the threshold-filtration definition.

### Paired PEGFAN comparison

`compare_pegfan.py` runs both models on identical official splits and paired
seeds, with validation-loss selection and one final test evaluation per run.
`pegfan_baseline.py` retains the original FSGNN and original a/b/c channel
construction, loading the supplied h=4 or h=8 framelets. No sheaf hierarchy is
used in the baseline. `pegfan_node_class.py` remains the original entry point.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python compare_pegfan.py \
  --datasets texas cornell wisconsin chameleon film squirrel \
  --splits 0 1 2 3 4 5 6 7 8 9 --seeds 42 \
  --epochs 500 --patience 50 --token-budget 512 --device cuda:0 \
  --output results/filtration_vs_pegfan
```

Defaults: hidden width 64 and dropout 0.5 for both; PEGFAN type c, h=8,
three propagation hops, layer normalization, FC learning rate 0.01 and attention
learning rate 0.02; sheaf learning rate 0.001; weight decay 0.0005 for both.
These are fixed first-round configurations, not a hyperparameter search or a
reproduction of tuned paper scores. Parameter counts are recorded, not matched.
The baseline retains its original normalized adjacency channel semantics;
the sheaf branch uses an undirected graph.

Outputs include `protocol.json` (configuration, environment, source hashes),
`dataset_metadata.json`, per-epoch histories, per-run `metrics.jsonl`,
`summary.csv`, `report.md`, and `progress.json`. Static baseline channel
preparation is timed separately and reused across splits; training times include
all dynamic sheaf hierarchy builds. Actual token counts, largest region size,
virtual merges and peak GPU allocation are recorded. Add `--resume` to skip
completed runs; changed code/configuration/environment is rejected to prevent
mixing experiments.

Tests cover whole tie batches, arbitrary node/edge order, disconnected policies,
budget-independent filtration, depth bounds, unequal cluster masses and skipped
levels, Haar reconstruction/energy, full-model permutation equivariance,
classification gradients, reloads, and original PEGFAN channel formulas.

After all runs finish, audit the paired results and generate `analysis.md`,
`audit.json`, and `comparison.png` / `comparison.pdf`:

```bash
python scripts/summarize_comparison.py results/filtration_vs_pegfan
```

The optional plots require matplotlib (or use `--no-plot`).

### Targeted single-factor ablations

`run_targeted_ablations.py` tests three controls, in priority order:

1. Frozen, split-specific sheaf filtration in original PEGFAN, preserving
   equal-child Haar, all `A^kX` channels, FSGNN channel count and parameter count.
2. Original cached hierarchy with mass-normalized Haar.
3. Original cached hierarchy with residual intermediate-scale Transformer,
   retaining local details and propagation channels.

The first control queries the new filtration at the original tree's region
count targets; the full model's dyadic schedule is diagnosed separately. The
original tree is recovered from cached wavelet supports and verified against
every cached band, avoiding a new Ward run. Implicit equal-child evaluation
is algebraically identical to the original pairwise Haar operator.

```bash
# Recover frozen scorers at the earlier validation-selected epochs.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/replay_sheaf_scorers.py
# Three variants x six datasets x ten splits; saves model checkpoints.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_targeted_ablations.py
# Resume an interrupted run with the same source/configuration.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_targeted_ablations.py --resume
# Final checkpoint/hash audit, all diagnostics, paired tables and figures.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/analyze_targeted_ablations.py
```

These commands use the existing `results/filtration_vs_pegfan` reference run
and CUDA. Full protocol and interpretation limits are in
[docs/targeted_ablation_protocol.md](docs/targeted_ablation_protocol.md).
Results are in `results/targeted_ablations/analysis.md`; `paired_splits.csv`
contains every accuracy difference. Film/Chameleon/Squirrel diagnostics cover
all levels of the original, resolution-matched and dyadic trees. Label entropy
and purity are computed only after all trial checkpoints are fixed.

The earlier experiment did not save weights. Scorers are replays, with numerical
validation differences recorded per split, not exact recovered checkpoints.
Scorer pretraining uses the same split's training supervision and adds cost.
The Transformer control uses a zero-initialized residual output adapter; its
initial function equals PEGFAN, but its insertion differs from the full model's
replacement-style Transformer path.

Historical data audit: the originally supplied Texas and Cornell feature/label files were byte-identical,
including in the original Git HEAD. Upstream hash comparison confirmed Texas matched Geom-GCN but Cornell did not.
Those Cornell scores are now excluded; the working data and caches have been repaired as described below. The classic Chameleon/Squirrel data
also need cleaned-data/new-benchmark follow-up before publication.

### Causal information-path experiments and Cornell repair

The current causal experiments and interpretation boundaries are documented in
[docs/causal_information_path_protocol.md](docs/causal_information_path_protocol.md).
The priority intervention nulls only the three explicit propagation inputs in
PEGFAN, retaining the original framelets of AX and every FSGNN parameter.
Further controls test exact pooling/detail/lifting reconstruction, shared rank-64
feature bottlenecks, fixed-hierarchy global/detail paths, and adding local AkX
features back to the full dynamic model.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_no_akx.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_causal_paths.py
python scripts/analyze_causal_paths.py
```

Existing result directories require `--resume` for the two runners. Results,
per-split differences, checkpoints and reconstruction diagnostics are under
[results/causal_ablations/analysis.md](results/causal_ablations/analysis.md), with raw artifacts
in `results/causal_ablations/`. All ten splits and seed 42 are retained; the extended
path controls prioritize Film, Chameleon and Squirrel.

Cornell is now repaired against pinned Geom-GCN revision
`1124af17444d7fd09686504ec46caa3f39f4f632`. Its old graph also differed from
upstream, so both h=4 and h=8 Ward Haar caches were regenerated. The old Cornell
runs are excluded from formal tables and plots; raw observations remain only
for audit. Corrected data and old observations must not be mixed.
The full source/cache checksum audit is `results/data_repairs/cornell/audit.json`.
Corrected Cornell reruns are separate:

```bash
python run_no_akx.py --datasets cornell --output results/causal_ablations/cornell_repaired --resume
python run_causal_paths.py --datasets cornell --variants full_no_akx full_with_akx \
  --output results/causal_ablations/cornell_full_repaired --resume
```

For a fresh checkout that needs the pinned repair/cache regeneration, use the
isolated Ward dependency directory; the main experiment NumPy version is unchanged:

```bash
python -m pip install --target /tmp/pegfan_ward_deps --no-deps numpy==1.24.4 scikit-network==0.28.3
PYTHONPATH=/tmp/pegfan_ward_deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/repair_cornell.py
```

The legacy targeted runner uses pre-repair frozen scorer artifacts. Do not use
its old Cornell artifacts with the corrected graph/features; use the current
causal runners for this repaired dataset version.


### Exact PEGFAN with a gated nonlocal branch

`augmentation_ops.py` adds a separately compressed Sheaf/Transformer branch to
the unmodified PEGFAN logits. A shared scalar gate starts at zero; initialization
and branch dropout preserve the original backbone RNG stream. Original features,
propagation channels, cached Haar operators and FSGNN optimization remain intact.
Only the new branch uses width 64 or 128; its optional Haar analysis happens after
the Transformer and uses the same hierarchy as pooling and lifting.

This architecture screen deliberately uses only splits 0, 1 and 2 on Film,
Chameleon, Squirrel and repaired Cornell: four configurations including the
original control, 48 formal training runs. It is a diagnostic screen, not a
replacement for the ten-split benchmark. Cornell additionally records per-layer
variance, pairwise cosine similarity and per-head attention entropy, including
read-only probes of the previous repaired-data checkpoints.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_augmentation.py
# Use --resume to continue an existing run with identical source/configuration.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/diagnose_cornell_transformer.py
python scripts/analyze_augmentation.py
```

See the [architecture protocol](docs/architecture_augmentation_protocol.md) and
[results report](results/architecture_augmentation/analysis.md), including paired
split deltas, gate diagnostics and initialization/selected-checkpoint attention
measurements. Existing experiment modules and earlier result artifacts are kept
unchanged by this screen.
