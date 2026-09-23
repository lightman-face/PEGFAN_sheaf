"""Check the reusable baseline against PEGFAN's stated channel formulas."""
import unittest
from unittest.mock import patch
import torch
from pegfan_baseline import prepare_pegfan_channels


class BaselineChannelsTests(unittest.TestCase):
    def test_channel_types_and_graph_normalization_choices(self):
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        a = torch.tensor([[0., .5, 0.], [.25, 0., .75], [0., 1., 0.]])
        a_i = (a + torch.eye(3)) / 2
        low = torch.ones(1, 3) / 3**.5
        high = torch.eye(3) - low.T @ low
        framelets = [low.to_sparse(), high.to_sparse()]
        transposes = [m.transpose(0, 1) for m in framelets]
        for kind in ('a', 'b', 'c'):
            for feature_type, operator in (('heterophily', a), ('homophily', a_i)):
                with patch('pegfan_baseline.get_spatial_framelets_list', return_value=(framelets, transposes)):
                    channels = prepare_pegfan_channels(x, a.to_sparse(), a_i.to_sparse(),
                                                        'texas', channel_type=kind, feature_type=feature_type)
                expected = [x]
                if kind != 'a':
                    expected.extend([torch.matrix_power(operator, power) @ x for power in (1, 2, 3)])
                signal = operator @ x if kind == 'c' else x
                expected.extend([low.T @ low @ signal, high.T @ high @ signal])
                self.assertEqual(len(channels), len(expected))
                for actual, reference in zip(channels, expected):
                    self.assertTrue(torch.allclose(actual, reference, atol=2e-6))

    def test_original_type_c_rejects_unspecified_propagation(self):
        x = torch.ones(3, 2)
        a = torch.eye(3).to_sparse()
        with self.assertRaises(ValueError):
            prepare_pegfan_channels(x, a, a, 'texas', feature_type='all', channel_type='c')
