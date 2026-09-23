"""Sheaf residuals and a shared hierarchy for pooling, Haar bands and lifting."""

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


def _as_symmetric_adjacency(adjacency):
    adjacency = sp.csr_matrix(adjacency, dtype=np.float32)
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.isfinite(adjacency.data).all() or (adjacency.data < 0).any():
        raise ValueError("adjacency must have finite, nonnegative weights")
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def edge_residual_energy(features, adjacency, restriction=None, weighted=False):
    """Compute relation-aware residual energy for every undirected graph edge.

    With the default identity restriction, this is the baseline sheaf model
    R_u = R_v = I. A callable restriction returns the two mapped endpoint
    vectors. By default scores are the raw squared residuals s_e; weighted=True
    retains the edge-weighted energy used by the initial single-level helper.
    Pass raw scores to build_sheaf_hierarchy, which weights boundary averages.
    """
    features = np.asarray(features, dtype=np.float32)
    adjacency = _as_symmetric_adjacency(adjacency)
    rows, cols = sp.triu(adjacency, k=1).nonzero()
    weights = np.asarray(adjacency[rows, cols]).reshape(-1) if len(rows) else np.empty(0)

    residual = np.empty(rows.shape[0], dtype=np.float32)
    for edge_index, (source, target) in enumerate(zip(rows, cols)):
        if restriction is None:
            source_relation = features[source]
            target_relation = features[target]
        else:
            source_relation, target_relation = restriction(
                features[source], features[target], source, target
            )
        difference = source_relation - target_relation
        residual[edge_index] = np.dot(difference, difference)
        if weighted:
            residual[edge_index] *= weights[edge_index]

    return rows, cols, residual


def node_residual_energy(num_nodes, edge_sources, edge_targets, edge_residual):
    """Aggregate incident edge residuals into a normalized node score."""
    scores = np.zeros(num_nodes, dtype=np.float32)
    degree = np.zeros(num_nodes, dtype=np.float32)
    for source, target, residual in zip(edge_sources, edge_targets, edge_residual):
        scores[source] += residual
        scores[target] += residual
        degree[source] += 1
        degree[target] += 1
    return scores / np.maximum(degree, 1)


def adaptive_partition(
    adjacency,
    edge_sources,
    edge_targets,
    edge_residual,
    max_cluster_size=4,
    residual_quantile=0.75,
    min_clusters=1,
):
    """Build a residual-aware partition using low-residual edges first.

    High-residual edges are protected from early merging. The greedy rule is
    deterministic and intentionally non-learned for controlled ablations.
    """
    adjacency = _as_symmetric_adjacency(adjacency)
    num_nodes = adjacency.shape[0]
    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be positive")
    if not 0 <= residual_quantile <= 1:
        raise ValueError("residual_quantile must be between 0 and 1")
    if min_clusters < 1:
        raise ValueError("min_clusters must be positive")

    threshold = np.quantile(edge_residual, residual_quantile) if len(edge_residual) else 0
    parent = np.arange(num_nodes, dtype=np.int64)
    cluster_size = np.ones(num_nodes, dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first, second):
        first = find(first)
        second = find(second)
        if first == second or cluster_size[first] + cluster_size[second] > max_cluster_size:
            return False
        if cluster_size[first] < cluster_size[second]:
            first, second = second, first
        parent[second] = first
        cluster_size[first] += cluster_size[second]
        return True

    order = np.argsort(edge_residual, kind="stable")
    num_clusters = num_nodes
    for edge_index in order:
        if num_clusters <= min_clusters:
            break
        if edge_residual[edge_index] <= threshold:
            num_clusters -= int(union(int(edge_sources[edge_index]), int(edge_targets[edge_index])))

    roots = np.array([find(node) for node in range(num_nodes)])
    _, assignment = np.unique(roots, return_inverse=True)
    return assignment.astype(np.int64)


def build_coarse_graph(adjacency, assignment):
    """Return normalized assignment matrix and weighted coarse adjacency."""
    adjacency = _as_symmetric_adjacency(adjacency)
    assignment = np.asarray(assignment, dtype=np.int64)
    num_nodes = adjacency.shape[0]
    if assignment.shape != (num_nodes,) or (assignment < 0).any():
        raise ValueError("assignment must contain one nonnegative cluster ID per node")
    if num_nodes and not np.array_equal(np.unique(assignment), np.arange(assignment.max() + 1)):
        raise ValueError("cluster IDs must be contiguous starting at zero")
    num_clusters = int(assignment.max()) + 1 if num_nodes else 0
    cluster_sizes = np.bincount(assignment, minlength=num_clusters).astype(np.float32)
    rows = np.arange(num_nodes)
    values = 1.0 / np.sqrt(cluster_sizes[assignment])
    projection = sp.csr_matrix(
        (values, (rows, assignment)), shape=(num_nodes, num_clusters)
    )
    coarse_adjacency = projection.T @ adjacency @ projection
    coarse_adjacency = coarse_adjacency.tocsr()
    coarse_adjacency.setdiag(0)
    coarse_adjacency.eliminate_zeros()
    return projection, coarse_adjacency


@dataclass
class SheafHierarchy:
    """Fine-to-coarse partitions; features stop at ``token_level``.

    The remaining partitions form Haar bands *on* the token graph. Projection
    columns are orthonormal, so their transposes pool and the same matrices lift.
    """

    assignments: list
    projections: list
    adjacencies: list
    token_level: int
    virtual_merge_levels: list

    @property
    def node_counts(self):
        return [adj.shape[0] for adj in self.adjacencies]

    @property
    def num_tokens(self):
        return self.node_counts[self.token_level]


def build_sheaf_hierarchy(
    adjacency, edge_sources, edge_targets, edge_residual,
    token_budget=512, max_cluster_size=4, residual_quantile=0.75,
):
    """Merge low-residual neighbors first, stopping feature pooling at budget.

    Residuals are unweighted squared sheaf differences, one per undirected
    edge. Coarse boundary scores are edge-weighted means of those original
    residuals, rather than differences of pooled features (which can cancel).
    High scores delay merging, but cannot forbid it under a hard token cap.

    If only disconnected regions remain, deterministic virtual groups ensure
    the cap and a complete Haar hierarchy. These add no graph edges and are
    reported in ``virtual_merge_levels``. Equal scores use input-index order;
    exact permutation equivariance is not guaranteed for such ties.
    """
    if not isinstance(token_budget, (int, np.integer)) or token_budget < 1:
        raise ValueError("token_budget must be a positive integer")
    if not isinstance(max_cluster_size, (int, np.integer)) or max_cluster_size < 2:
        raise ValueError("max_cluster_size must be an integer >= 2")
    if not 0 <= residual_quantile <= 1:
        raise ValueError("residual_quantile must be between 0 and 1")
    adjacency = _as_symmetric_adjacency(adjacency)
    n = adjacency.shape[0]
    if not n:
        raise ValueError("cannot build a hierarchy for an empty graph")
    sources = np.asarray(edge_sources, dtype=np.int64)
    targets = np.asarray(edge_targets, dtype=np.int64)
    residual = np.asarray(edge_residual, dtype=np.float32)
    expected_sources, expected_targets = sp.triu(adjacency, k=1).nonzero()
    if (sources.shape != targets.shape or sources.shape != residual.shape
            or not np.array_equal(sources, expected_sources)
            or not np.array_equal(targets, expected_targets)):
        raise ValueError("residuals must follow the upper-triangular adjacency edge order")
    if not np.isfinite(residual).all() or (residual < 0).any():
        raise ValueError("edge residuals must be finite and nonnegative")
    weights = np.asarray(adjacency[sources, targets]).reshape(-1) if len(sources) else np.empty(0)
    energy = sp.csr_matrix((weights * residual, (sources, targets)), shape=(n, n))
    energy = energy + energy.T
    boundary_weights = adjacency.copy()
    hierarchy = SheafHierarchy([], [], [adjacency], 0 if n <= token_budget else -1, [])

    while adjacency.shape[0] > 1:
        n = adjacency.shape[0]
        rows, cols = sp.triu(boundary_weights, k=1).nonzero()
        if len(rows):
            weight = np.asarray(boundary_weights[rows, cols]).reshape(-1)
            scores = np.asarray(energy[rows, cols]).reshape(-1) / np.maximum(weight, 1e-30)
        else:
            scores = np.empty(0, dtype=np.float32)
        min_clusters = token_budget if hierarchy.token_level < 0 else 1
        assignment = adaptive_partition(
            adjacency, rows, cols, scores, max_cluster_size,
            residual_quantile, min_clusters=min_clusters,
        )
        if assignment.max() + 1 == n:
            if len(rows):
                assignment = adaptive_partition(
                    adjacency, rows, cols, scores, max_cluster_size,
                    1.0, min_clusters=min_clusters,
                )
            else:
                # No graph edges remain: group components only as a last resort.
                target_count = max(min_clusters, (n + max_cluster_size - 1) // max_cluster_size)
                assignment = np.arange(n, dtype=np.int64) * target_count // n
                hierarchy.virtual_merge_levels.append(len(hierarchy.projections))

        projection, coarse_adjacency = build_coarse_graph(adjacency, assignment)
        membership = sp.csr_matrix(
            (np.ones(n, dtype=np.float32), (np.arange(n), assignment)),
            shape=projection.shape,
        )
        boundary_weights = (membership.T @ boundary_weights @ membership).tocsr()
        energy = (membership.T @ energy @ membership).tocsr()
        for matrix in (boundary_weights, energy):
            matrix.setdiag(0)
            matrix.eliminate_zeros()
        hierarchy.assignments.append(assignment)
        hierarchy.projections.append(projection)
        hierarchy.adjacencies.append(coarse_adjacency)
        adjacency = coarse_adjacency
        if hierarchy.token_level < 0 and adjacency.shape[0] <= token_budget:
            hierarchy.token_level = len(hierarchy.projections)

    return hierarchy
