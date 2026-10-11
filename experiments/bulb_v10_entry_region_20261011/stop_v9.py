"""Stop only the verified v9 job; preserve saved models and record resume point."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,re,signal,time
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out=root/'experiments/bulb_v10_entry_region_20261011'
run=root/'logs/rsl_rl/bulb_grasp_teacher_v9_mlp/2026-10-11_resume_model900_2x8192'
expected={4515:'389651',4531:'390061',4532:'390061'}

def process(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 try:
  stat=(p/'stat').read_text().split(') ',1)[1].split()
  if stat[0]=='Z':return None
  return {'pid':pid,'start_ticks':stat[19],'cwd':str((p/'cwd').resolve()),'cmd':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
 except (FileNotFoundError,ProcessLookupError):return None

before=[process(pid) for pid in expected]
assert all(p and p['start_ticks']==expected[p['pid']] and p['cwd']==str(root) and str(run) in p['cmd'] for p in before),before
log=root/'experiments/bulb_v9_resume_20261011/train.log'
iterations=list(map(int,re.findall(r'Learning iteration (\d+)/',log.read_text(errors='replace'))))
models=sorted((run/'rank0').glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
assert models
# SIGINT to torchrun forwards to its own workers even though they have separate sessions.
os.kill(4515,signal.SIGINT)
for _ in range(25):
 if all(process(pid) is None for pid in expected):break
 time.sleep(1)
remaining=[process(pid) for pid in expected if process(pid)]
for p in remaining:
 assert p['start_ticks']==expected[p['pid']] and p['cwd']==str(root) and str(run) in p['cmd']
 os.kill(p['pid'],signal.SIGTERM)
for _ in range(20):
 if all(process(pid) is None for pid in expected):break
 time.sleep(1)
assert all(process(pid) is None for pid in expected),[process(pid) for pid in expected]
models=sorted((run/'rank0').glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
selected=models[-1];iteration=int(selected.stem.split('_')[1])
inventory=[{'name':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in models]
import torch
state=torch.load(selected,map_location='cpu',weights_only=False)
assert state['iter']==iteration
assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
steps=sorted(set(int(v['step']) for v in state['optimizer_state_dict']['state'].values()))
report={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'stopped':True,'processes_before':before,'last_logged_iteration_before_stop':iterations[-1],'selected_checkpoint':str(selected),'selected_iteration':iteration,'selected_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),'optimizer_steps':steps,'checkpoint_inventory':inventory,'iterations_additional':30000-iteration,'target_iteration':30000,'note':'Latest saved v9 full-state resume; unsaved tail is not restored. Task migration v9->v10, not identical-MDP continuation.'}
(out/'checkpoint_selection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('time','stopped','last_logged_iteration_before_stop','selected_checkpoint','selected_iteration','optimizer_steps','iterations_additional')},indent=2))
