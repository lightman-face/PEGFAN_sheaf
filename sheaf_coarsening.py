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
    Pass raw scores to build_sheaf_hierarchy; it uses q = -residual.
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
    """Legacy greedy helper (not used by the filtration model).

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


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class MergeFiltration:
    """Budget-independent, multiway merge forest; IDs are not semantic labels."""

    num_nodes: int
    children: list
    parent: np.ndarray
    masses: np.ndarray
    birth_events: np.ndarray
    event_counts: np.ndarray
    thresholds: np.ndarray
    roots: tuple
    num_components: int
    virtual_event: object = None

    def partition(self, event):
        """Materialize one leaf partition in O(n), primarily for diagnostics."""
        if not isinstance(event, (int, np.integer)) or not 0 <= event < len(self.event_counts):
            raise ValueError("invalid filtration event")
        blocks, stack = [], list(self.roots)
        while stack:
            node = stack.pop()
            if self.birth_events[node] <= event:
                blocks.append(node)
            else:
                stack.extend(self.children[node])
        assignment = np.empty(self.num_nodes, dtype=np.int64)
        for label, root in enumerate(blocks):
            stack = [root]
            while stack:
                node = stack.pop()
                if node < self.num_nodes:
                    assignment[node] = label
                else:
                    stack.extend(self.children[node])
        return assignment


@dataclass
class HierarchySelection:
    events: tuple
    token_level: int
    token_budget: int


@dataclass
class SheafHierarchy:
    assignments: list
    projections: list
    node_counts: list
    masses: list
    cluster_nodes: list
    token_level: int
    virtual_merge_levels: list
    filtration: MergeFiltration
    selection: HierarchySelection

    @property
    def num_tokens(self):
        return self.node_counts[self.token_level]


def build_sheaf_filtration(num_nodes, edge_sources, edge_targets, merge_scores,
                           disconnected_policy="virtual_root"):
    """Threshold CCs with complete tie batches and no child-count constraint.

    Larger scores merge earlier. Scores remain fixed for the entire build.
    Parallel edges are allowed; self-loops have no effect. Only changed CC
    partitions are recorded, as a compact multiway tree, never n-by-events
    assignments. A virtual root groups ALL final components simultaneously.
    """
    _positive_integer(num_nodes, "num_nodes")
    if disconnected_policy not in ("virtual_root", "forest"):
        raise ValueError("disconnected_policy must be virtual_root or forest")
    sources, targets = np.asarray(edge_sources), np.asarray(edge_targets)
    scores = np.asarray(merge_scores, dtype=np.float64)
    if sources.ndim != 1 or sources.shape != targets.shape or sources.shape != scores.shape:
        raise ValueError("edges and scores must be equally sized vectors")
    if len(sources) and (not np.issubdtype(sources.dtype, np.integer)
                         or not np.issubdtype(targets.dtype, np.integer)):
        raise ValueError("edge endpoints must be integers")
    sources, targets = sources.astype(np.int64), targets.astype(np.int64)
    if ((sources < 0).any() or (sources >= num_nodes).any()
            or (targets < 0).any() or (targets >= num_nodes).any()):
        raise ValueError("edge endpoint outside node range")
    if not np.isfinite(scores).all():
        raise ValueError("merge scores must be finite")

    # Python integers avoid NumPy scalar overhead in the sequential UF loop.
    source_list, target_list = sources.tolist(), targets.tolist()
    uf_parent = list(range(num_nodes))
    uf_size = [1] * num_nodes
    component_node = list(range(num_nodes))
    children = [()] * num_nodes
    parent, mass, birth = [-1] * num_nodes, [1] * num_nodes, [0] * num_nodes
    counts, thresholds = [num_nodes], [np.inf]
    count = num_nodes

    def find(node):
        while uf_parent[node] != node:
            uf_parent[node] = uf_parent[uf_parent[node]]
            node = uf_parent[node]
        return int(node)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(order)].tolist()
    order = order.tolist()
    for start, end in zip(starts[:-1], starts[1:]):
        if start == end:
            continue
        # Remember the OLD tree nodes; sequential UF operations are internal
        # bookkeeping, not binary nodes exposed to the Haar hierarchy.
        touched = {}
        for index in order[start:end]:
            first, second = find(source_list[index]), find(target_list[index])
            if first == second:
                continue
            touched.setdefault(first, int(component_node[first]))
            touched.setdefault(second, int(component_node[second]))
            if uf_size[first] < uf_size[second]:
                first, second = second, first
            uf_parent[second] = first
            uf_size[first] += uf_size[second]
            count -= 1
        if not touched:
            continue
        groups = {}
        for old_root, tree_node in touched.items():
            groups.setdefault(find(old_root), []).append(tree_node)
        event = len(counts)
        for root, group in groups.items():
            node = len(children)
            children.append(tuple(group))
            mass.append(sum(mass[child] for child in group))
            birth.append(event)
            parent.append(-1)
            for child in group:
                parent[child] = node
            component_node[root] = node
        counts.append(count)
        thresholds.append(float(scores[order[start]]))
        if count == 1:
            break

    roots = tuple(int(component_node[i]) for i in range(num_nodes) if uf_parent[i] == i)
    num_components = len(roots)
    virtual_event = None
    if num_components > 1 and disconnected_policy == "virtual_root":
        virtual_event = len(counts)
        node = len(children)
        children.append(roots)
        parent.append(-1)
        mass.append(num_nodes)
        birth.append(virtual_event)
        for child in roots:
            parent[child] = node
        roots = (node,)
        counts.append(1)
        thresholds.append(-np.inf)
    return MergeFiltration(num_nodes, children, np.asarray(parent), np.asarray(mass),
                           np.asarray(birth), np.asarray(counts), np.asarray(thresholds),
                           roots, num_components, virtual_event)


def select_hierarchy_levels(filtration, token_budget=512):
    """Finest budget-feasible cut, with dyadic local and global level queries."""
    _positive_integer(token_budget, "token_budget")
    counts = filtration.event_counts
    if counts[-1] > token_budget:
        raise ValueError(f"budget {token_budget} is unreachable: graph has "
                         f"{filtration.num_components} connected components; "
                         "use virtual_root or increase the budget")

    def first_at_most(target):
        return int(np.searchsorted(-counts, -target, side="left"))

    cut = first_at_most(token_budget)
    events = {0, cut}
    target = filtration.num_nodes // 2
    while target >= 1:
        event = first_at_most(target)
        if event >= cut:
            break
        events.add(event)
        target //= 2
    target = int(counts[cut]) // 2
    terminal = int(counts[-1])
    while target >= terminal:
        events.add(first_at_most(target))
        target //= 2
    events.add(len(counts) - 1)
    events = tuple(sorted(events))
    return HierarchySelection(events, events.index(cut), int(token_budget))


def materialize_projections(filtration, selection):
    """Mass-normalized projections only at selected events.

    Internal merge nodes are visited once as the active cut advances. Between
    consecutive selected cuts, tree traversal stops at the previous cut, so
    skipped levels do not require full original-node assignment snapshots.
    """
    if not selection.events or selection.events[0] != 0:
        raise ValueError("selection must start at the singleton partition")
    active = set(range(filtration.num_nodes))
    fine = np.arange(filtration.num_nodes, dtype=np.int64)
    cursor = filtration.num_nodes
    assignments, projections = [], []
    clusters, masses = [fine], [filtration.masses[fine]]
    virtual_levels = []
    for level, event in enumerate(selection.events[1:]):
        while cursor < len(filtration.children) and filtration.birth_events[cursor] <= event:
            active.difference_update(filtration.children[cursor])
            active.add(cursor)
            cursor += 1
        coarse = np.fromiter(active, dtype=np.int64, count=len(active))
        fine_index = {int(node): index for index, node in enumerate(fine)}
        assignment = np.empty(len(fine), dtype=np.int64)
        for label, root in enumerate(coarse):
            stack = [int(root)]
            while stack:
                node = stack.pop()
                index = fine_index.get(node)
                if index is not None:
                    assignment[index] = label
                else:
                    stack.extend(filtration.children[node])
        fine_mass, coarse_mass = filtration.masses[fine], filtration.masses[coarse]
        values = np.sqrt(fine_mass / coarse_mass[assignment]).astype(np.float32)
        projection = sp.csr_matrix((values, (np.arange(len(fine)), assignment)),
                                   shape=(len(fine), len(coarse)))
        if (filtration.virtual_event is not None
                and selection.events[level] < filtration.virtual_event <= event):
            virtual_levels.append(level)
        assignments.append(assignment)
        projections.append(projection)
        clusters.append(coarse)
        masses.append(coarse_mass)
        fine = coarse
    return SheafHierarchy(assignments, projections, [len(c) for c in clusters], masses,
                          clusters, selection.token_level, virtual_levels, filtration, selection)


def build_sheaf_hierarchy(adjacency, edge_sources, edge_targets, edge_residual,
                           token_budget=512, disconnected_policy="virtual_root"):
    """Convenience wrapper: nonnegative sheaf energies become q=-energy."""
    adjacency = _as_symmetric_adjacency(adjacency)
    sources = np.array(edge_sources, dtype=np.int64, copy=True)
    targets = np.array(edge_targets, dtype=np.int64, copy=True)
    residual = np.asarray(edge_residual)
    expected_sources, expected_targets = sp.triu(adjacency, k=1).nonzero()
    if (sources.shape != targets.shape or sources.shape != residual.shape
            or not np.array_equal(sources, expected_sources)
            or not np.array_equal(targets, expected_targets)):
        raise ValueError("residuals must follow the upper-triangular adjacency edge order")
    if not np.isfinite(residual).all() or (residual < 0).any():
        raise ValueError("edge residuals must be finite and nonnegative")
    filtration = build_sheaf_filtration(adjacency.shape[0], sources, targets, -residual,
                                        disconnected_policy)
    return materialize_projections(filtration, select_hierarchy_levels(filtration, token_budget))
