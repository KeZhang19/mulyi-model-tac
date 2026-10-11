"""Run only after user approval: stop the identified old v5 launcher, preserving files."""
from pathlib import Path
import hashlib,json,os,signal,time,socket
from datetime import datetime
from zoneinfo import ZoneInfo
out=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007/experiments/bulb_v8_mlp_c3g_2x8192_20261007')
preflight=json.loads((out/'preflight_old_job.json').read_text())
assert socket.gethostname()==preflight['hostname']
processes=preflight['old_processes']
assert {p['pid'] for p in processes}=={948,1009,1010}
for data in processes:
 p=Path('/proc')/str(data['pid'])
 assert str((p/'cwd').resolve())==preflight['old_root']
 assert (p/'stat').read_text().split()[21]==data['start_ticks']
 assert (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()==data['cmdline']
launcher=next(p for p in processes if p['pid']==948)
assert '-m torch.distributed.run' in launcher['cmdline']
run=Path(preflight['latest_checkpoint']).parent
checkpoints=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
latest=checkpoints[-1]; digest=hashlib.sha256(latest.read_bytes()).hexdigest()
request={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'processes':processes,'latest_checkpoint':str(latest),'sha256_before':digest,'checkpoint_count':len(checkpoints)}
(out/'old_stop_requested.json').write_text(json.dumps(request,indent=2)+'\n')
os.kill(948,signal.SIGTERM)
for _ in range(60):
 live=[p['pid'] for p in processes if (Path('/proc')/str(p['pid'])).exists()]
 if not live:break
 time.sleep(.5)
assert not live,live
assert latest.is_file() and hashlib.sha256(latest.read_bytes()).hexdigest()==digest
assert all(p.is_file() for p in checkpoints)
result={**request,'stopped':True,'sha256_after':digest,'stopped_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
(out/'old_stopped.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
