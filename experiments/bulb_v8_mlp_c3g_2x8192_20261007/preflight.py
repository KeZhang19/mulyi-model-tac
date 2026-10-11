"""Record the specific old job and available dependencies without stopping it."""
from pathlib import Path
import hashlib,json,subprocess,socket
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out=root/'experiments/bulb_v8_mlp_c3g_2x8192_20261007'
old=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v5-20261006')
run=old/'logs/rsl_rl/bulb_grasp_teacher_v5/2026-10-06_scratch_2x8192/rank0'
processes=[]
for pid in (948,1009,1010):
 p=Path('/proc')/str(pid)
 if p.exists():
  processes.append({'pid':pid,'cwd':str((p/'cwd').resolve()),'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),'start_ticks':(p/'stat').read_text().split()[21]})
checkpoints=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
latest=checkpoints[-1]
report={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'hostname':socket.gethostname(),
 'requested_instance':'dsw-c3g3e0ieowszisvolv','old_root':str(old),'old_processes':processes,
 'latest_checkpoint':str(latest),'latest_sha256':hashlib.sha256(latest.read_bytes()).hexdigest(),
 'checkpoints':[{'path':str(p),'bytes':p.stat().st_size} for p in checkpoints],
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
(out/'preflight_old_job.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='checkpoints'}))
