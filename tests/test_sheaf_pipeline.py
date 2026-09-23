"""Mathematical invariants and task-gradient checks for the new backbone."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from framelets_utils import haar_pool_details, haar_lift, haar_framelet_projections
from model import LightweightSheaf, SheafHaarTransformer
from sheaf_coarsening import adaptive_partition, build_coarse_graph, build_sheaf_hierarchy, edge_residual_energy
from utils import sparse_mx_to_torch_sparse_tensor


def chain(n):
    return sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format='csr')


def hierarchy_for(adjacency, budget=4, residual=None):
    rows, cols = sp.triu(adjacency, k=1).nonzero()
    if residual is None:
        residual = np.arange(len(rows), dtype=np.float32) + 1
    return build_sheaf_hierarchy(adjacency, rows, cols, residual, token_budget=budget)


class HierarchyTests(unittest.TestCase):
    def test_residual_uses_both_endpoint_restrictions(self):
        adjacency = sp.csr_matrix([[0, 3], [3, 0]])
        features = np.array([[1, 2], [2, 4]], dtype=np.float32)
        restriction = lambda source, target, u, v: (2 * source, target)
        _, _, residual = edge_residual_energy(features, adjacency, restriction)
        np.testing.assert_allclose(residual, [0])
        _, _, raw = edge_residual_energy(features, adjacency)
        _, _, weighted = edge_residual_energy(features, adjacency, weighted=True)
        np.testing.assert_allclose(raw, [5])
        np.testing.assert_allclose(weighted, [15])

    def test_high_residual_boundary_survives_early_partition(self):
        adjacency = chain(4)
        assignment = adaptive_partition(adjacency, [0, 1, 2], [1, 2, 3],
                                        np.array([0.1, 10.0, 0.2]), residual_quantile=0.5)
        self.assertEqual(assignment[0], assignment[1])
        self.assertEqual(assignment[2], assignment[3])
        self.assertNotEqual(assignment[1], assignment[2])

    def test_budget_orthonormality_and_strict_progress(self):
        for n, budget in ((17, 5), (5, 512), (12, 1), (1, 1)):
            hierarchy = hierarchy_for(chain(n), budget)
            self.assertEqual(hierarchy.num_tokens, min(n, budget))
            self.assertEqual(hierarchy.node_counts[-1], 1)
            self.assertTrue(all(a > b for a, b in zip(hierarchy.node_counts, hierarchy.node_counts[1:])))
            for projection in hierarchy.projections:
                np.testing.assert_allclose((projection.T @ projection).toarray(),
                                           np.eye(projection.shape[1]), atol=1e-6)

    def test_isolates_still_respect_budget(self):
        hierarchy = hierarchy_for(sp.csr_matrix((13, 13)), budget=3)
        self.assertEqual(hierarchy.num_tokens, 3)
        self.assertTrue(hierarchy.virtual_merge_levels)
        self.assertTrue(all(adjacency.nnz == 0 for adjacency in hierarchy.adjacencies))

    def test_invalid_inputs(self):
        adjacency = chain(3)
        for budget in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                hierarchy_for(adjacency, budget)
        with self.assertRaises(ValueError):
            hierarchy_for(adjacency, residual=[1, np.nan])
        with self.assertRaises(ValueError):
            build_sheaf_hierarchy(adjacency, [0, 1], [1, 2], [1, 2], max_cluster_size=1)


class HaarTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.hierarchy = hierarchy_for(chain(19), budget=6)
        self.projections = [sparse_mx_to_torch_sparse_tensor(p) for p in self.hierarchy.projections]

    def test_pool_detail_reconstruction_and_energy(self):
        signal = torch.randn(19, 5)
        lowpass, details = haar_pool_details(signal, self.projections)
        reconstructed = haar_lift(lowpass, self.projections, details)
        self.assertTrue(torch.allclose(signal, reconstructed, atol=1e-6))
        energy = lowpass.square().sum() + sum(detail.square().sum() for detail in details)
        self.assertTrue(torch.allclose(signal.square().sum(), energy, atol=1e-5))

    def test_bands_reconstruct_transformer_scale_and_backpropagate(self):
        cut = self.hierarchy.token_level
        signal = torch.randn(self.hierarchy.num_tokens, 5, requires_grad=True)
        bands = haar_framelet_projections(signal, self.projections[cut:])
        self.assertTrue(torch.allclose(sum(bands), signal, atol=1e-6))
        sum(band.square().sum() for band in bands).backward()
        self.assertTrue(torch.allclose(signal.grad, 2 * signal, atol=2e-6))

    def test_implicit_band_matches_pairwise_haar_framelets(self):
        # An unequal-size partition with a singleton exercises PEGFAN's local
        # pairwise construction; the singleton must contribute zero highpass.
        projection, _ = build_coarse_graph(chain(6), np.array([0, 0, 0, 1, 1, 2]))
        projection = sparse_mx_to_torch_sparse_tensor(projection)
        signal = torch.randn(6, 3)
        rows = []
        for children in ((0, 1, 2), (3, 4), (5,)):
            for i, source in enumerate(children):
                for target in children[i + 1:]:
                    row = torch.zeros(6)
                    row[source], row[target] = len(children) ** -0.5, -len(children) ** -0.5
                    rows.append(row)
        psi = torch.stack(rows)
        _, details = haar_pool_details(signal, [projection])
        self.assertTrue(torch.allclose(details[0], psi.T @ psi @ signal, atol=1e-6))


class ModelTests(unittest.TestCase):
    def test_framelet_analysis_receives_transformer_output(self):
        model = SheafHaarTransformer(3, 8, 2, token_budget=4, nhead=2,
                                     transformer_layers=1, dropout=0).set_graph(chain(10))
        model.eval()
        transformed = []
        handle = model.transformer.register_forward_hook(
            lambda module, inputs, output: transformed.append(output.detach().clone())
        )
        with torch.no_grad(), patch('model.haar_framelet_projections', wraps=haar_framelet_projections) as analyze:
            model(torch.randn(10, 3))
        handle.remove()
        self.assertEqual(analyze.call_count, 1)
        signal, projections = analyze.call_args[0]
        self.assertTrue(torch.equal(signal, transformed[0].squeeze(1)))
        self.assertEqual(len(projections), len(model.hierarchy.projections) - model.hierarchy.token_level)
        self.assertEqual(projections[0].shape[0], model.hierarchy.num_tokens)

    def test_sheaf_edge_orientation_does_not_change_signal_or_energy(self):
        torch.manual_seed(11)
        sheaf = LightweightSheaf(8)
        torch.nn.init.normal_(sheaf.restriction[-1].weight, std=0.1)
        features = torch.randn(5, 8)
        source, target = torch.arange(4), torch.arange(1, 5)
        weights = torch.rand(4)
        local, energy = sheaf(features, source, target, weights)
        flipped_local, flipped_energy = sheaf(features, target, source, weights)
        self.assertTrue(torch.allclose(local, flipped_local, atol=1e-6))
        self.assertTrue(torch.allclose(energy, flipped_energy, atol=1e-6))

    def test_budget_actual_attention_input_and_classification_gradients(self):
        torch.manual_seed(3)
        model = SheafHaarTransformer(5, 8, 3, token_budget=4, nhead=2,
                                     transformer_layers=1, dropout=0).set_graph(chain(13))
        seen_tokens = []
        handle = model.transformer.register_forward_pre_hook(
            lambda module, inputs: seen_tokens.append(inputs[0].shape[0])
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        features, labels = torch.randn(13, 5), torch.arange(13) % 3
        previous = model.sheaf.restriction[-1].weight.detach().clone()
        output = model(features)
        self.assertEqual(tuple(output.shape), (13, 3))
        self.assertEqual(seen_tokens, [4])
        self.assertTrue(torch.isfinite(output).all().item())
        F.nll_loss(output[:7], labels[:7]).backward()
        for parameters in (model.sheaf.restriction.parameters(), model.transformer.parameters(),
                           model.fusion.parameters(), model.detail_projection.parameters()):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all().item()
                                and p.grad.abs().sum().item() > 0 for p in parameters))
        optimizer.step()
        self.assertFalse(torch.equal(previous, model.sheaf.restriction[-1].weight))
        handle.remove()
        model.eval()
        with torch.no_grad():
            reference = model(features)
            restored = SheafHaarTransformer(5, 8, 3, token_budget=4, nhead=2,
                                            transformer_layers=1, dropout=0).set_graph(chain(13))
            restored.load_state_dict(model.state_dict())
            restored.eval()
            self.assertTrue(torch.allclose(reference, restored(features), atol=1e-6))

    def test_small_and_edgeless_graphs(self):
        for n, budget in ((1, 512), (7, 2), (7, 512)):
            model = SheafHaarTransformer(3, 8, 2, token_budget=budget, nhead=2,
                                         transformer_layers=1, dropout=0).set_graph(sp.csr_matrix((n, n)))
            output = model(torch.randn(n, 3))
            self.assertEqual(tuple(output.shape), (n, 2))
            self.assertLessEqual(model.hierarchy.num_tokens, budget)
            self.assertTrue(torch.isfinite(output).all().item())


if __name__ == '__main__':
    unittest.main()
