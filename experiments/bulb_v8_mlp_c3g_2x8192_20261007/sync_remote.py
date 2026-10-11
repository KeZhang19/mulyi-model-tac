"""Copy verified source files into a new deployment; never modify the base run."""
from pathlib import Path
import hashlib,json,os,shutil,sys
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
base=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v7-lstm-20261007')
out=root/'experiments/bulb_v8_mlp_c3g_2x8192_20261007'
manifest=json.loads((out/'sync_manifest.json').read_text())

def matches(p,spec):
    if 'symlink' in spec:return p.is_symlink() and os.readlink(p)==spec['symlink']
    return p.is_file() and not p.is_symlink() and p.stat().st_size==spec['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']

if sys.argv[1]=='copy':
    mismatch=[]; copied=0
    for rel,spec in manifest.items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        src,dest=base/rel,root/rel
        if not matches(src,spec):mismatch.append(rel);continue
        assert not dest.exists() and not dest.is_symlink(),rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        if 'symlink' in spec:dest.symlink_to(spec['symlink'])
        else:shutil.copy2(src,dest)
        copied+=1
    (out/'overlay_files.json').write_text(json.dumps(mismatch,indent=2)+'\n')
    print(json.dumps({'copied_verified':copied,'overlay_needed':mismatch}))
elif sys.argv[1]=='verify':
    bad=[rel for rel,spec in manifest.items() if not matches(root/rel,spec)]
    result={'root':str(root),'count':len(manifest),'verified':len(manifest)-len(bad),'mismatches':bad,'passed':not bad}
    (out/'sync_verified.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    assert not bad,bad
else:raise ValueError(sys.argv[1])
