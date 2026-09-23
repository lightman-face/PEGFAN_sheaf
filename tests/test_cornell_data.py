"""Regression protection for the confirmed Cornell source mix-up."""
import csv,hashlib,json,unittest
from pathlib import Path
import numpy as np

class CornellDataTests(unittest.TestCase):
    def test_pinned_upstream_files_and_masks(self):
        audit=json.loads(Path('results/data_repairs/cornell/audit.json').read_text())
        self.assertEqual(audit['status'],'repaired')
        self.assertEqual(audit['commit'],'1124af17444d7fd09686504ec46caa3f39f4f632')
        for row in audit['files']:
            self.assertEqual(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),row['upstream_sha256'])
        self.assertNotEqual(Path('new_data/cornell/out1_node_feature_label.txt').read_bytes(),Path('new_data/texas/out1_node_feature_label.txt').read_bytes())
        for i in range(10):
            with np.load(f'splits/cornell_split_0.6_0.2_{i}.npz') as f:
                masks=[np.asarray(f[k],dtype=bool) for k in ('train_mask','val_mask','test_mask')]
                self.assertTrue(np.all(sum(m.astype(int) for m in masks)==1))
    def test_legacy_cornell_excluded_from_formal_tables(self):
        for directory in ('filtration_vs_pegfan','targeted_ablations'):
            with Path('results',directory,'summary.csv').open() as f:
                self.assertNotIn('cornell',[r['dataset'] for r in csv.DictReader(f)])
    def test_regenerated_haar_caches_match_repair_audit(self):
        audit=json.loads(Path('results/data_repairs/cornell/audit.json').read_text())
        for h,row in audit['regenerated_caches'].items():
            self.assertEqual(hashlib.sha256(Path('framelets',h,'cornell.pickle').read_bytes()).hexdigest(),row['sha256'])
            self.assertLess(max(row['band_errors']),3e-6)
