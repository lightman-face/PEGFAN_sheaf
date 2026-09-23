import unittest
import numpy as np
import torch
from ablation_ops import (transfers,torch_transfers,original_partitions,validate_cached_tree,
                         OriginalHierarchyTransformer,partition_diagnostics,label_diagnostics,new_partitions)
from framelets_utils import haar_framelet_projections,get_spatial_framelets_list
from model import FSGNN

class AblationTests(unittest.TestCase):
    def test_original_cache_recovery(self):
        for name in ('texas','cornell','wisconsin','chameleon','film','squirrel'):
            f,_=get_spatial_framelets_list(None,name,8);parts=original_partitions(f)
            self.assertLess(max(validate_cached_tree(f,parts)),3e-6)
    def test_equal_children_are_original_pairwise_haar_not_mass(self):
        parts=[np.arange(4),np.array([0,0,0,1]),np.zeros(4,dtype=int)]
        p=transfers(parts,'equal');m=transfers(parts,'mass')
        a=np.array([[1.,-1.]])/np.sqrt(2)
        self.assertTrue(np.allclose(a.T@a,np.eye(2)-(p[1]@p[1].T).toarray()))
        self.assertFalse(np.allclose(p[1].toarray(),m[1].toarray()))
        x=torch.randn(4,3)
        for norm in ('equal','mass'):
            self.assertTrue(torch.allclose(sum(haar_framelet_projections(x,torch_transfers(parts,norm))),x,atol=1e-6))
    def test_transformer_zero_adapter_is_original_model(self):
        parts=[np.arange(4),np.array([0,0,1,1]),np.zeros(4,dtype=int)]
        base=[torch.randn(4,3) for _ in range(4)]
        base+=haar_framelet_projections(base[1],torch_transfers(parts))
        clf=FSGNN(3,len(base),8,2,.5);clf.eval();expected=clf(base,True)
        model=OriginalHierarchyTransformer(clf,base,parts,budget=2,hidden=8);model.eval()
        self.assertTrue(torch.equal(expected,model()))
        model.train();model().square().sum().backward()
        self.assertGreater(model.output_adapter.weight.grad.abs().sum().item(),0.)
    def test_transformer_delta_matches_explicit_pool_modify_lift(self):
        parts=[np.arange(5),np.array([0,0,1,1,2]),np.zeros(5,dtype=int)]
        base=[torch.randn(5,3) for _ in range(4)]
        ps=torch_transfers(parts);base+=haar_framelet_projections(base[1],ps)
        clf=FSGNN(3,len(base),8,2,0.)
        model=OriginalHierarchyTransformer(clf,base,parts,budget=3,hidden=8,dropout=0.)
        torch.nn.init.normal_(model.output_adapter.weight);model.eval()
        z=model.coarse_source;delta=model.output_adapter(model.transformer(model.input_adapter(z).unsqueeze(1)).squeeze(1))
        from framelets_utils import haar_lift
        source=base[1]+haar_lift(delta,ps[:model.cut])
        explicit=clf(base[:4]+haar_framelet_projections(source,ps),True)
        self.assertTrue(torch.allclose(model(),explicit,atol=1e-6))
    def test_tied_new_cuts_preserve_channel_count_without_splitting_ties(self):
        scores=dict(sources=np.array([0,1,2]),targets=np.array([1,2,3]),energy=np.ones(3))
        matched,_,meta=new_partitions(4,scores,[4,3,2,1],2)
        self.assertEqual([int(p.max()+1) for p in matched],[4,1,1,1])
        self.assertEqual(len(transfers(matched)),3)
    def test_diagnostic_definitions(self):
        a=np.array([0,0,0,1]);x=np.array([[0.],[1.],[2.],[10.]])
        s=np.array([0,1,2]);t=np.array([1,2,3]);e=np.array([1.,3.,10.])
        row=partition_diagnostics(a,x,s,t,e)
        self.assertEqual(row['largest_cluster_ratio'],.75)
        self.assertEqual(row['singleton_cluster_ratio'],.5)
        self.assertEqual(row['singleton_node_ratio'],.25)
        self.assertAlmostEqual(row['cluster_size_gini'],.25)
        self.assertAlmostEqual(row['within_cluster_feature_variance'],.5)
        self.assertEqual(row['within_cluster_sheaf_energy'],2.)
        lab=label_diagnostics(a,np.array([0,0,1,1]),np.ones(4,bool))
        self.assertEqual(lab['cluster_label_purity'],.75)
        empty=label_diagnostics(a,np.array([0,0,1,1]),np.zeros(4,bool))
        self.assertIsNone(empty['cluster_label_purity'])
    def test_non_nested_partitions_rejected(self):
        with self.assertRaises(ValueError):transfers([np.arange(4),[0,0,1,1],[0,1,0,1]])
