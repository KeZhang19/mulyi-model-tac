from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out=root/'experiments/bulb_v10_entry_region_20261011'
manifest=json.loads((out/'sync_manifest.json').read_text())
mismatches=[]
for row in manifest:
 if row['check_before']:
  p=root/row['path'];current=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
  if current!=row['before_sha256']:mismatches.append({'path':row['path'],'current':current,'expected':row['before_sha256']})
assert not mismatches, json.dumps(mismatches)
active=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if (p/'cwd').resolve()==root:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if any(s in cmd for s in ('train_bulb','torch.distributed','split_gpu','record_teacher','play_bulb')):active.append({'pid':p.name,'cmd':cmd})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not active,active
backup=out/'v9_replaced_files_backup.tar.gz'
assert not backup.exists(), 'Backup already exists; do not repeat apply blindly'
before={}
with tarfile.open(backup,'w:gz') as archive:
 for row in manifest:
  p=root/row['path']
  if p.exists():
   before[row['path']]=hashlib.sha256(p.read_bytes()).hexdigest()
   archive.add(p,arcname=row['path'])
with tarfile.open(out/'sync_payload.tar.gz') as archive:
 members=archive.getmembers()
 assert all((root/m.name).resolve().is_relative_to(root) for m in members)
 archive.extractall(root)
checks={row['path']:hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest}
report={'passed':all(checks.values()),'files':len(checks),'checks':checks,'v9_backup':str(backup),'backup_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'before_sha256':before,'no_active_job_before_sync':True}
(out/'sync_verified.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'files':report['files'],'v9_backup':str(backup)},indent=2))
assert report['passed']
