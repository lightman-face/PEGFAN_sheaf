import unittest
import numpy as np
import scipy.sparse as sp
import torch
from model import FSGNN,SheafHaarTransformer
from ablation_ops import torch_transfers
from causal_ops import null_propagation_channels,pool_lift_inputs,FeatureRoundTrip,FixedHierarchyChain,AkXClassifier

class CausalTests(unittest.TestCase):
    def setUp(self):
        self.parts=[np.arange(6),np.array([0,0,0,1,1,2]),np.zeros(6,int)]
        self.x=torch.randn(6,5);self.graph=sp.csr_matrix(np.ones((6,6))-np.eye(6))
    def test_null_only_explicit_propagation(self):
        c=[torch.randn(6,5) for _ in range(8)];a=null_propagation_channels(c)
        self.assertEqual(len(a),len(c));self.assertIs(a[0],c[0])
        for i in range(1,4):self.assertEqual(float(a[i].abs().sum()),0.)
        for i in range(4,8):self.assertIs(a[i],c[i])
    def test_pool_detail_identity_and_gradient(self):
        for norm in ('equal','mass'):
            x=self.x.clone().requires_grad_();g,r,e=pool_lift_inputs([x],torch_transfers(self.parts,norm))
            self.assertTrue(torch.allclose(x,r[0],atol=1e-6));self.assertGreater((x-g[0]).abs().max().item(),.01)
            r[0].sum().backward();self.assertTrue(torch.allclose(x.grad,torch.ones_like(x),atol=1e-6))
    def test_rank_roundtrip_and_classifier_shape(self):
        q,_=torch.linalg.qr(torch.randn(5,2));clf=FSGNN(5,2,8,3,0.)
        model=FeatureRoundTrip(clf,[self.x,self.x*2],q)
        expected=clf([self.x@q@q.T,2*self.x@q@q.T],True)
        self.assertTrue(torch.allclose(model(),expected,atol=1e-6))
        self.assertEqual(tuple(model.decode.weight.shape),(5,2))
    def test_haar_sum_is_identity_after_attention(self):
        model=FixedHierarchyChain(5,8,3,self.parts,1,'chain_transformer_global',dropout=0.).set_graph(self.graph)
        model.eval();a=model(self.x);b=model(self.x,haar_identity_check=True)
        self.assertTrue(torch.allclose(a,b,atol=1e-6));self.assertLess(model.last_haar_error,1e-6)
    def test_full_chain_matches_original_with_same_fixed_hierarchy(self):
        model=FixedHierarchyChain(5,8,3,self.parts,1,'chain_full_skip',dropout=0.).set_graph(self.graph)
        original=SheafHaarTransformer(5,8,3,dropout=0.).set_graph(self.graph);original.load_state_dict(model.state_dict())
        from types import SimpleNamespace
        original.hierarchy=SimpleNamespace(token_level=1,node_counts=[6,3,1])
        original._torch_projections=torch_transfers(self.parts,'mass');original._projection_key=(self.x.device,self.x.dtype)
        original.eval();model.eval()
        self.assertTrue(torch.allclose(original(self.x,rebuild_hierarchy=False),model(self.x),atol=1e-6))
    def test_adding_local_channels_starts_at_original_function(self):
        original=SheafHaarTransformer(5,8,3,dropout=0.).set_graph(self.graph);original.eval()
        expected=original(self.x)
        original.classifier=AkXClassifier(original.classifier,[self.x]*3,5,8,True)
        original.eval();actual=original(self.x)
        self.assertTrue(torch.equal(expected,actual))
        original.train();original(self.x).square().sum().backward()
        self.assertGreater(original.classifier.trunk[0].weight.grad[:,16:].abs().sum().item(),0.)

    def test_active_added_propagation_branch_is_permutation_equivariant(self):
        from causal_ops import make_full_with_akx
        a=torch.tensor(self.graph.toarray(),dtype=torch.float32)/5
        channels=[a@self.x,a@a@self.x,a@a@a@self.x]
        left,_=make_full_with_akx(5,8,3,self.graph,channels,True,torch.device('cpu'))
        with torch.no_grad():left.classifier.trunk[0].weight[:,16:].normal_(std=.1)
        order=torch.tensor([4,1,5,0,3,2])
        graph=self.graph[order.numpy()][:,order.numpy()]
        right,_=make_full_with_akx(5,8,3,graph,[h[order] for h in channels],True,torch.device('cpu'))
        right.load_state_dict(left.state_dict());left.eval();right.eval()
        self.assertTrue(torch.allclose(left(self.x)[order],right(self.x[order]),atol=2e-6))
