import unittest
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from gradient_isolation_ops import (make_matrix_model,raw_logits,isolated_losses,
                                    freeze_backbone,contribution_metrics)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        self.x=torch.randn(8,5);self.channels=[self.x,self.x*2,self.x*.3,self.x-1,self.x+1]
        self.y=torch.arange(8)%3;self.mask=torch.ones(8,dtype=torch.bool)
        self.graph=sp.csr_matrix(np.ones((8,8))-np.eye(8))
        self.parts=[np.arange(8),np.array([0,0,0,1,1,2,2,2]),np.zeros(8,dtype=int)]
    def make(self,variant):
        torch.manual_seed(42)
        return make_matrix_model(5,5,3,variant,self.graph,self.parts,1,torch.device('cpu'))

    def test_stopgrad_removes_ce_coupling_not_just_direct_path(self):
        l=torch.randn(8,3,requires_grad=True);g=torch.randn(8,3,requires_grad=True);a=torch.tensor(.7,requires_grad=True)
        reference=torch.autograd.grad(F.cross_entropy(l,self.y),l,retain_graph=True)[0]
        loss,_=isolated_losses(l,g,a,self.y,self.mask,'stopgrad')
        grad_l,grad_g=torch.autograd.grad(loss,(l,g),retain_graph=True)
        self.assertTrue(torch.equal(reference,grad_l));self.assertGreater(float(grad_g.norm()),0.)
        naive=torch.autograd.grad(F.cross_entropy(l+a*g.detach(),self.y),l)[0]
        self.assertFalse(torch.allclose(reference,naive))

    def test_stopgrad_active_gate_preserves_full_local_updates_and_rng(self):
        baseline,base_opt=self.make('pegfan')
        model,opt=self.make('stopgrad_t128_haar');model.alpha.data.fill_(.5)
        for epoch in range(4):
            baseline.train();model.train();rng=torch.get_rng_state()
            base_opt.zero_grad(set_to_none=True)
            F.nll_loss(baseline(self.channels,True),self.y).backward();base_opt.step()
            after=torch.get_rng_state();torch.set_rng_state(rng)
            opt.zero_grad(set_to_none=True)
            local=raw_logits(model.backbone,self.channels);g=model.branch(self.x)
            loss,_=isolated_losses(local,g,model.alpha,self.y,self.mask,'stopgrad')
            loss.backward();opt.step()
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            for k,v in baseline.state_dict().items():self.assertTrue(torch.equal(v,model.backbone.state_dict()[k]),k)

    def test_frozen_parameters_stay_fixed_while_branch_learns(self):
        model,opt=self.make('frozen_t128_haar');freeze_backbone(model.backbone)
        before={k:v.clone() for k,v in model.backbone.state_dict().items()}
        model.branch.train()
        with torch.no_grad():local=raw_logits(model.backbone,self.channels)
        for _ in range(3):
            opt.zero_grad(set_to_none=True);g=model.branch(self.x)
            loss,_=isolated_losses(local,g,model.alpha,self.y,self.mask,'frozen',False)
            loss.backward();opt.step()
        for k,v in before.items():self.assertTrue(torch.equal(v,model.backbone.state_dict()[k]))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(abs(float(model.alpha)),0.)

    def test_common_initialization_and_zero_gate(self):
        base,_=self.make('pegfan');base.eval();expected=base(self.channels,True)
        first=None
        for variant in ('t128','t128_haar','stopgrad_t128','stopgrad_t128_haar','frozen_t128_haar'):
            model,_=self.make(variant);model.eval()
            self.assertTrue(torch.equal(expected,model(self.channels,self.x)))
            if first is None:first={k:v.clone() for k,v in model.branch.state_dict().items()}
            for k,v in first.items():self.assertTrue(torch.equal(v,model.branch.state_dict()[k]))

    def test_fixed_tree_and_identity_transformer_path_in_haar_fusion(self):
        from unittest.mock import patch
        model,_=self.make('t128_haar');model.eval();model.alpha.data.fill_(1.)
        before=[p.to_dense().clone() for p in model.branch.fixed_projections]
        with torch.no_grad():
            model.branch.fusion.lowpass.weight.zero_();model.branch.fusion.highpass.weight.zero_()
        a=model(self.channels,self.x)
        model.branch.post_haar=False;b=model(self.channels,self.x)
        self.assertTrue(torch.equal(a,b))
        model.branch.post_haar=True
        with patch('augmentation_ops.build_sheaf_filtration',side_effect=RuntimeError('Must not regenerate tree')):
            model(self.channels,self.x*2)
        for p,q in zip(before,model.branch.fixed_projections):self.assertTrue(torch.equal(p,q.to_dense()))

    def test_rg_and_cosine_and_softmax_gauge(self):
        l=torch.tensor([[1.,-1.],[2.,-2.]])
        d=contribution_metrics(l,-.25*l)
        self.assertAlmostEqual(d['raw_rg'],.25);self.assertAlmostEqual(d['raw_cosine'],-1.)
        shifted=contribution_metrics(l+10.,-.25*l+3.)
        self.assertAlmostEqual(shifted['centered_rg'],d['centered_rg'])
        self.assertAlmostEqual(shifted['centered_cosine'],d['centered_cosine'])
        zero=contribution_metrics(l,torch.zeros_like(l))
        self.assertEqual(zero['raw_rg'],0.);self.assertIsNone(zero['raw_cosine'])
        self.assertIsNone(zero['centered_mean_node_cosine'])
