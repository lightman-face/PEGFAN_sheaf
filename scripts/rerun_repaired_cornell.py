"""Rerun corrected Cornell only after the pinned-source repair audit passes."""
from pathlib import Path
import json,subprocess,hashlib
p=Path('results/data_repairs/cornell/audit.json');audit=json.loads(p.read_text());assert audit['status']=='repaired'
for r in audit['files']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['upstream_sha256']
for r in audit['regenerated_caches'].values():assert max(r['band_errors'])<3e-6
for command in [
    ['python','-u','run_no_akx.py','--datasets','cornell','--output','results/causal_ablations/cornell_repaired'],
    ['python','-u','run_causal_paths.py','--datasets','cornell','--variants','full_no_akx','full_with_akx','--output','results/causal_ablations/cornell_full_repaired']
]:subprocess.run(command,check=True)
