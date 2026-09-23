import copy
import unittest
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from augmentation_ops import (make_augmented, attention_probabilities,
                              representation_stats, transformer_diagnostics)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.x = torch.randn(9, 5)
        self.channels = [self.x, self.x*2, self.x*.3, self.x-1, self.x+1]
        self.graph = sp.csr_matrix(np.ones((9,9)) - np.eye(9))

    def make(self, variant):
        torch.manual_seed(42)
        return make_augmented(5,5,3,variant,self.graph,torch.device('cpu'))

    def test_zero_gate_exact_and_original_initialization_rng(self):
        base, _ = self.make('pegfan'); state = torch.get_rng_state()
        base.eval(); expected = base(self.channels,True)
        for variant in ('transformer64','transformer_haar64','transformer_haar128'):
            model, _ = self.make(variant)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            for k,v in base.state_dict().items(): self.assertTrue(torch.equal(v,model.backbone.state_dict()[k]))
            model.eval(); self.assertTrue(torch.equal(expected,model(self.channels,self.x)))
            self.assertEqual(model.backbone.fc1[0].in_features,5)

    def test_closed_gate_preserves_training_trajectory(self):
        base, opt_base = self.make('pegfan')
        model, opt_model = self.make('transformer_haar64'); model.alpha.requires_grad_(False)
        base.train(); model.train(); y = torch.arange(9) % 3
        for _ in range(3):
            rng = torch.get_rng_state()
            opt_base.zero_grad(); a = base(self.channels,True); F.nll_loss(a,y).backward(); opt_base.step()
            after = torch.get_rng_state(); torch.set_rng_state(rng)
            opt_model.zero_grad(); b = model(self.channels,self.x); F.nll_loss(b,y).backward(); opt_model.step()
            self.assertTrue(torch.equal(a,b)); self.assertTrue(torch.equal(after,torch.get_rng_state()))
            for k,v in base.state_dict().items(): self.assertTrue(torch.equal(v,model.backbone.state_dict()[k]))

    def test_gate_then_branch_receive_gradients(self):
        model, opt = self.make('transformer_haar64'); model.eval(); y = torch.arange(9) % 3
        loss = F.nll_loss(model(self.channels,self.x),y); loss.backward()
        self.assertGreater(abs(float(model.alpha.grad)),1e-8)
        self.assertEqual(float(model.branch.output.weight.grad.abs().sum()),0.)
        opt.step(); opt.zero_grad()
        F.nll_loss(model(self.channels,self.x),y).backward()
        self.assertGreater(float(model.branch.output.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.branch.sheaf.restriction[-1].weight.grad.abs().sum()),0.)

    def test_active_branch_permutation_and_budget(self):
        for variant in ('transformer64','transformer_haar64'):
            left,_ = self.make(variant); left.alpha.data.fill_(.3); left.branch.budget=3; left.eval()
            perm=torch.tensor([8,1,4,0,7,2,6,3,5]); right=copy.deepcopy(left)
            right.branch.set_graph(self.graph[perm.numpy()][:,perm.numpy()]); right.eval()
            a=left(self.channels,self.x)[perm]; b=right([h[perm] for h in self.channels],self.x[perm])
            self.assertTrue(torch.allclose(a,b,atol=2e-6)); self.assertLessEqual(left.branch.hierarchy.num_tokens,3)

    def test_attention_entropy_matches_native_weights(self):
        model,_=self.make('transformer64'); module=model.branch.transformer.layers[0].self_attn; module.eval()
        x=torch.randn(7,1,64); _, native=module(x,x,x,need_weights=True)
        p=attention_probabilities(module,x)
        self.assertTrue(torch.allclose(p.mean(1),native,atol=1e-6))
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=1e-6))

    def test_diagnostic_no_change_and_all_heads(self):
        model,_=self.make('transformer64'); model.alpha.data.fill_(.2); model.eval()
        expected=model(self.channels,self.x)
        with transformer_diagnostics(model.branch.transformer) as diag:
            actual=model(self.channels,self.x)
        self.assertTrue(torch.equal(expected,actual)); self.assertEqual(len(diag['attention']),8)
        self.assertEqual(diag['representations']['before']['nodes'],9)
        for r in diag['attention']: self.assertGreaterEqual(r['normalized_entropy'],0); self.assertLessEqual(r['normalized_entropy'],1+1e-6)

    def test_representation_stats_known_cases(self):
        same=representation_stats(torch.ones(5,3))
        self.assertEqual(same['feature_variance'],0.)
        self.assertAlmostEqual(same['mean_pairwise_cosine'],1.)
        eye=representation_stats(torch.eye(4))
        self.assertAlmostEqual(eye['mean_pairwise_cosine'],0.)
        self.assertAlmostEqual(eye['feature_variance'],.1875)

    def test_paired_64_branches_and_post_transformer_haar_order(self):
        from unittest.mock import patch
        from framelets_utils import haar_framelet_projections
        plain,_=self.make('transformer64'); framed,_=self.make('transformer_haar64')
        for k,v in plain.branch.state_dict().items():
            self.assertTrue(torch.equal(v,framed.branch.state_dict()[k]))
        seen=[]
        for model in (plain,framed):
            model.eval(); model.branch.budget=3
            handle=model.branch.transformer.register_forward_hook(
                lambda module,inputs,out: seen.append(out.squeeze(1).detach().clone()))
            with patch('augmentation_ops.haar_framelet_projections', wraps=haar_framelet_projections) as haar:
                model(self.channels,self.x)
                if model.branch.post_haar:
                    self.assertEqual(haar.call_count,1)
                    self.assertTrue(torch.equal(haar.call_args[0][0],seen[-1]))
                    self.assertLessEqual(len(seen[-1]),3)
                else:
                    self.assertEqual(haar.call_count,0)
            handle.remove()
