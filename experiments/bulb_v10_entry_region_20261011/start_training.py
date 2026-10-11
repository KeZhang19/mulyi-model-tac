from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,subprocess,time
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out=root/'experiments/bulb_v10_entry_region_20261011'
selection=json.loads((out/'checkpoint_selection.json').read_text())
checkpoint=Path(selection['selected_checkpoint'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==selection['selected_sha256']
assert json.loads((out/'sync_verified.json').read_text())['passed']
cpu=(out/'cpu_checks_remote.log').read_text();assert 'Ran 60 tests' in cpu and '\nOK\n' in cpu,cpu[-2500:]
active=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if (p/'cwd').resolve()==root:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if any(s in cmd for s in ('train_bulb','torch.distributed','split_gpu','record_teacher','play_bulb')):active.append({'pid':p.name,'cmd':cmd})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not active,active
iteration=selection['selected_iteration']
run=root/f'logs/rsl_rl/bulb_grasp_teacher_v10_mlp/2026-10-11_resume_model{iteration}_2x8192'
launch=f'''#!/usr/bin/env bash
set -Eeuo pipefail
cd {root}
export CUDA_VISIBLE_DEVICES=0,1 OMNI_KIT_ALLOW_ROOT=1 OMP_NUM_THREADS=8
export ITERATIONS={selection['iterations_additional']}
export RUN_DIR={run}
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to reuse existing run: $RUN_DIR" >&2
    exit 1
fi
exec bash scripts/rsl_rl/launch_bulb_grasp_teacher_2gpu.sh --num_envs 8192 --initial_adr_difficulty 0 --physx_buffer_scale 2 --physx_collision_stack_size 4026531840 --checkpoint {checkpoint} --migrate-v9-entry-window
'''
(out/'launch.sh').write_text(launch)
assert not run.exists()
log=out/'train.log';assert not log.exists()
with log.open('wb') as output:
 process=subprocess.Popen(['bash',str(out/'launch.sh')],cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
report={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'launcher_pid':process.pid,'root':str(root),'run':str(run),'log':str(log),'checkpoint':str(checkpoint),'checkpoint_sha256':selection['selected_sha256'],'iterations_additional':selection['iterations_additional'],'target_iteration':30000,'num_envs_per_gpu':8192,'gpus':[0,1],'observation_history':5,'lstm':False,'v9_to_v10_migration':True,'process_poll':process.poll()}
(out/'launch.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));assert process.poll() is None
