from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007');out=root/'experiments/bulb_v10_entry_region_20261011'
manifest=json.loads((out/'docs_manifest.json').read_text())
for row in manifest:
 if row['expected_remote_before']:
  assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['expected_remote_before'],row['path']
backup=out/'before_docs.tar.gz';assert not backup.exists()
with tarfile.open(backup,'w:gz') as a:
 for row in manifest:
  p=root/row['path']
  if p.exists():a.add(p,arcname=row['path'])
with tarfile.open(out/'docs_payload.tar.gz') as a:
 assert all((root/m.name).resolve().is_relative_to(root) for m in a.getmembers())
 a.extractall(root)
checks={row['path']:hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest}
report={'passed':all(checks.values()),'checks':checks,'backup_sha256':hashlib.sha256(backup.read_bytes()).hexdigest()}
(out/'docs_verified.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'files':len(checks)}));assert report['passed']
