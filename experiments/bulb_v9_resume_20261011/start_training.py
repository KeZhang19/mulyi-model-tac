from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,subprocess,time
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out=root/'experiments/bulb_v9_resume_20261011'
selection=json.loads((out/'checkpoint_selection.json').read_text())
checkpoint=Path(selection['selected_checkpoint'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==selection['selected_sha256']
assert json.loads((out/'sync_verified.json').read_text())['passed']
cpu=(out/'cpu_checks_remote.log').read_text();assert 'Ran 58 tests' in cpu and '\nOK\n' in cpu,cpu[-2500:]
active=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if (p/'cwd').resolve()==root:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if any(s in cmd for s in ('train_bulb','torch.distributed','split_gpu','record_teacher','play_bulb')):active.append({'pid':p.name,'cmd':cmd})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not active,active
run=root/'logs/rsl_rl/bulb_grasp_teacher_v9_mlp/2026-10-11_resume_model900_2x8192'
assert not run.exists()
log=out/'train.log';assert not log.exists()
with log.open('wb') as output:
 process=subprocess.Popen(['bash',str(out/'launch.sh')],cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
report={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'launcher_pid':process.pid,'root':str(root),'run':str(run),'log':str(log),'checkpoint':str(checkpoint),'checkpoint_sha256':selection['selected_sha256'],'iterations_additional':29100,'target_iteration':30000,'num_envs_per_gpu':8192,'gpus':[0,1],'observation_history':5,'lstm':False,'v8_to_v9_migration':True,'process_poll':process.poll()}
(out/'launch.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));assert process.poll() is None
