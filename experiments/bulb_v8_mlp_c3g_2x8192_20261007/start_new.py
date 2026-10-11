"""Launch the prepared v8 run only on the inspected c3g host with both GPUs free."""
from pathlib import Path
import subprocess,json,socket
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out=root/'experiments/bulb_v8_mlp_c3g_2x8192_20261007'
preflight=json.loads((out/'preflight_old_job.json').read_text())
assert socket.gethostname()==preflight['hostname']
assert json.loads((out/'sync_verified.json').read_text())['passed']
assert json.loads((out/'cpu_verified.json').read_text())['passed']
assert json.loads((out/'old_stopped.json').read_text())['stopped']
assert not (out/'train.log').exists() and not (out/'launch.json').exists()
used=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not used,used
mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
assert len(mem)==2 and all(int(v)<1000 for v in mem),mem
with (out/'train.log').open('xb') as log:
 p=subprocess.Popen(['bash',str(out/'launch.sh')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'hostname':socket.gethostname(),'instance':'dsw-c3g3e0ieowszisvolv','pid':p.pid,'root':str(root),'run':str(root/'logs/rsl_rl/bulb_grasp_teacher_v8_mlp/2026-10-07_scratch_2x8192'),'num_envs_per_rank':8192,'total_envs':16384,'lstm':False,'observation_history_length':5,'mdp_version':8,'iterations':30000,'from_scratch':True}
(out/'launch.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
