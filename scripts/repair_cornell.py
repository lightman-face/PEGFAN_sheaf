"""Verify Cornell against a pinned Geom-GCN revision, then install verified data."""
import concurrent.futures,hashlib,json,shutil,urllib.request
from pathlib import Path
import numpy as np

ROOT=Path('results/data_repairs/cornell');ROOT.mkdir(parents=True,exist_ok=True)
META=ROOT/'audit.json'
UPSTREAM_REVISION='1124af17444d7fd09686504ec46caa3f39f4f632'

def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'PEGFAN-data-audit'})
    with urllib.request.urlopen(request,timeout=30) as f:return f.read()
def parse_nodes(content):
    lines=content.decode().splitlines()[1:];parts=[s.split('\t') for s in lines if s]
    ids=np.array([int(s[0]) for s in parts]);x=np.array([[int(v) for v in s[1].split(',')] for s in parts]);y=np.array([int(s[2]) for s in parts])
    assert np.array_equal(ids,np.arange(183));assert x.shape==(183,1703);assert np.isfinite(x).all();assert set(np.unique(y))==set(range(5))
    return x,y

def main():
    if META.exists():
        previous=json.loads(META.read_text())
        files_ok=all(Path(r['path']).is_file() and sha(Path(r['path']).read_bytes())==r['upstream_sha256'] for r in previous.get('files',[]))
        caches_ok=all((Path('framelets')/h/'cornell.pickle').is_file() and sha((Path('framelets')/h/'cornell.pickle').read_bytes())==r['sha256'] for h,r in previous.get('regenerated_caches',{}).items())
        if previous.get('status')=='repaired' and files_ok and caches_ok:
            print(META.read_text());return
    commit=UPSTREAM_REVISION
    paths=['new_data/cornell/out1_node_feature_label.txt','new_data/cornell/out1_graph_edges.txt']+[f'splits/cornell_split_0.6_0.2_{i}.npz' for i in range(10)]
    def download(path):
        url=f'https://raw.githubusercontent.com/bingzhewei/geom-gcn/{commit}/{path}';return path,url,fetch(url)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:items=list(pool.map(download,paths))
    archive=ROOT/'legacy';staging=ROOT/'verified_upstream';records=[]
    for path,url,content in items:
        dest=staging/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        previous=archive/path if (archive/path).exists() else Path(path)
        records.append(dict(path=path,url=url,upstream_sha256=sha(content),previous_sha256=sha(previous.read_bytes())))
    x,y=parse_nodes((staging/paths[0]).read_bytes());oldx,oldy=parse_nodes((archive/paths[0] if (archive/paths[0]).exists() else Path(paths[0])).read_bytes())
    edges=np.loadtxt(staging/paths[1],skiprows=1,dtype=int)
    assert edges.ndim==2 and edges.shape[1]==2 and edges.min()>=0 and edges.max()<183
    local_edges=np.loadtxt(archive/paths[1] if (archive/paths[1]).exists() else paths[1],skiprows=1,dtype=int)
    same_graph=set(map(tuple,edges))==set(map(tuple,local_edges))
    # Changed graph is installed together with regenerated Haar caches below.
    for path in paths[2:]:
        with np.load(staging/path) as z:
            masks=[np.asarray(z[k],dtype=bool) for k in ('train_mask','val_mask','test_mask')]
            assert all(m.shape==(183,) and m.any() for m in masks)
            assert np.all(sum(m.astype(int) for m in masks)==1)
    audit=dict(status='verified',repository='https://github.com/bingzhewei/geom-gcn',commit=commit,nodes=183,features=1703,classes=5,
               edges=len(edges),graph_matches_previous=same_graph,cached_haar_reusable=same_graph,
               changed_feature_entries=int(np.sum(x!=oldx)),changed_labels=int(np.sum(y!=oldy)),
               feature_sha256=sha(x.tobytes()),label_sha256=sha(y.tobytes()),label_counts=np.bincount(y).tolist(),files=records)
    for path,_,content in items:
        backup=archive/path;backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copy2(path,backup)
        destination=Path(path);temp=destination.with_suffix(destination.suffix+'.verified_tmp');temp.write_bytes(content);temp.replace(destination)
    if not same_graph:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        import torch
        from process import full_load_data
        from framelets_utils import get_spatial_framelets_list
        from ablation_ops import original_partitions,validate_cached_tree
        torch.set_num_threads(1)
        loaded=full_load_data('cornell','splits/cornell_split_0.6_0.2_0.npz',return_sparse=True)
        cache_audit={}
        for h in (4,8):
            path=Path('framelets')/str(h)/'cornell.pickle';backup=archive/path;backup.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                if not backup.exists():shutil.copy2(path,backup)
                path.unlink()
            frames,_=get_spatial_framelets_list(loaded[0].toarray(),'cornell',h)
            parts=original_partitions(frames)
            cache_audit[str(h)]=dict(sha256=sha(path.read_bytes()),node_counts=[int(p.max()+1) for p in parts],band_errors=validate_cached_tree(frames,parts))
        audit['regenerated_caches']=cache_audit
    import sknetwork
    audit['cache_build_environment']=dict(numpy=np.__version__,scikit_network=sknetwork.__version__)
    audit['status']='repaired';META.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
