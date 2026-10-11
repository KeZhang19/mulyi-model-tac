from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,subprocess,hashlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
run=root/'logs/rsl_rl/bulb_grasp_teacher_v8_mlp/2026-10-07_scratch_2x8192/rank0'
out=root/'experiments/bulb_v9_resume_20261011'
out.mkdir(parents=True,exist_ok=True)
acc=EventAccumulator(str(run),size_guidance={'scalars':0});acc.Reload()
tags=['Metrics/object_pose/stage1_success','Metrics/object_pose/stage2_success','Metrics/object_pose/stage3_success']
series={t:{v.step:v.value for v in acc.Scalars(t)} for t in tags}
rows=[]
for step in range(0,1801,100):
 row={'step':step}
 for tag in tags:
  values=[v for s,v in series[tag].items() if max(0,step-99)<=s<=step]
  row[tag.rsplit('/',1)[-1]+'_mean100']=sum(values)/len(values)
 rows.append(row)
selected=run/'model_900.pt';saved=torch.load(selected,map_location='cpu',weights_only=False)
assert saved['iter']==900
assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
assert tuple(saved['model_state_dict']['actor.0.weight'].shape)==(512,1900)
report={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'root':str(root),'selected_checkpoint':str(selected),'selected_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),'saved_iteration':saved['iter'],'optimizer_steps':sorted(set(int(s['step']) for s in saved['optimizer_state_dict']['state'].values())),'latest_checkpoint_sha256':hashlib.sha256((run/'model_6900.pt').read_bytes()).hexdigest(),'checkpoint_inventory':[{'name':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(run.glob('model_*.pt'))],'candidate_windows':rows,'stage2_positive_updates_through900':sum(v>0 for s,v in series[tags[1]].items() if s<=900),'selection_reason':'Earliest saved checkpoint with last-100 stage1 mean >=10%, avoiding later extensive stage2 learning. Zero promotion does not prove zero stage2 exposure/bias.','metric_scope':'rank0 reset-batch means, not independent evaluation; rank1 has no event history.'}
(out/'checkpoint_selection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoint_inventory','candidate_windows')},indent=2))
