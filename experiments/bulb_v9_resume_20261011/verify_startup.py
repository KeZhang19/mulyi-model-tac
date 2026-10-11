"""Read-only audit after at least ten v9 PPO updates; never stops the job."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,re,subprocess
import torch,yaml
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out=root/'experiments/bulb_v9_resume_20261011'
run=root/'logs/rsl_rl/bulb_grasp_teacher_v9_mlp/2026-10-11_resume_model900_2x8192'
log=(out/'train.log').read_text(errors='replace')
iterations=sorted(set(map(int,re.findall(r'Learning iteration (\d+)/',log))))
required=[run/f'rank{r}'/name for r in (0,1) for name in ('launch_contract.json','distributed_verified.json','params/env.yaml','params/agent.yaml','target_contract_migration.json')]+[run/'rank0/model_900.pt']
missing=[str(p) for p in required if not p.exists()]
if missing or len(iterations)<10:
 print(json.dumps({'ready':False,'iterations':iterations[-12:],'missing':missing,'tail':log.splitlines()[-10:]}));raise SystemExit(0)
checks={};contracts=[];ddps=[];migrations=[]
for rank in (0,1):
 p=run/f'rank{rank}'
 cfg=yaml.load((p/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
 agent=yaml.load((p/'params/agent.yaml').read_text(),Loader=yaml.BaseLoader)
 contract=json.loads((p/'launch_contract.json').read_text());contracts.append(contract)
 ddp=json.loads((p/'distributed_verified.json').read_text());ddps.append(ddp)
 migration=json.loads((p/'target_contract_migration.json').read_text());migrations.append(migration)
 checks[f'rank{rank}_v9']=int(cfg['bulb_grasp_curriculum_version'])==9 and cfg['rewards']['position_tracking']['func'].endswith(':curriculum_position_tracking_reward')
 checks[f'rank{rank}_settings']=(int(cfg['scene']['num_envs'])==8192 and float(cfg['sim']['dt'])==1/120 and int(cfg['decimation'])==4 and float(cfg['curriculum']['adr']['params']['init_difficulty'])==0 and int(agent['max_iterations'])==29100 and all(int(cfg['observations'][g]['history_length'])==5 for g in ('policy','proprio','perception')) and agent['policy']['class_name']=='ActorCritic')
 cmd=cfg['commands']['object_pose']
 checks[f'rank{rank}_thresholds']=(float(cmd['insertion_depth_m'])==.0055 and list(map(float,cmd['insertion_depth_range_m']))==[.005,.006] and float(cmd['shallow_insertion_tilt_tolerance_deg'])==30 and float(cmd['insertion_tilt_tolerance_deg'])==5 and float(cmd['insertion_hold_s'])==.2)
 checks[f'rank{rank}_ddp']=(ddp['verified'] and ddp['gradient_reductions']>0 and len(set(ddp['parameter_sha256_by_rank']))==1 and contract['total_envs']==16384 and contract['num_envs_per_rank']==8192 and not contract['from_scratch'] and contract['seed']==42+rank and contract['cuda_visible_devices']==str(rank))
 checks[f'rank{rank}_migration']=(migration['source_version']==8 and migration['target_version']==9 and migration['v8_to_v9_migration'] and migration['restore_mode']=='full_state_resume')
 checks[f'rank{rank}_capacity']=int(cfg['sim']['physx']['gpu_collision_stack_size'])==4026531840 and int(cfg['sim']['physx']['gpu_max_rigid_contact_count'])==16777216
selected=json.loads((out/'checkpoint_selection.json').read_text())
parent=Path(selected['selected_checkpoint']);original=torch.load(parent,map_location='cpu',weights_only=False)
fresh_path=run/'rank0/model_900.pt';fresh=torch.load(fresh_path,map_location='cpu',weights_only=False)
steps=sorted(set(int(v['step']) for v in fresh['optimizer_state_dict']['state'].values()))
checks['restored_optimizer_then_updated']=steps==[selected['optimizer_steps'][0]+20]
checks['finite_model_optimizer']=all(torch.isfinite(v).all() for v in fresh['model_state_dict'].values()) and all(torch.isfinite(v).all() for d in fresh['optimizer_state_dict']['state'].values() for v in d.values() if isinstance(v,torch.Tensor))
checks['model_is_same_width_mlp']=all(tuple(fresh['model_state_dict'][f'{side}.0.weight'].shape)==(512,1900) for side in ('actor','critic')) and not any(k.startswith('memory_') for k in fresh['model_state_dict'])
counts={side:{'parent':original['model_state_dict'][f'{side}_obs_normalizer.count'].item(),'new':fresh['model_state_dict'][f'{side}_obs_normalizer.count'].item()} for side in ('actor','critic')}
checks['normalizers_restored_then_updated']=all(c['new']>c['parent']>0 for c in counts.values())
checks['policy_parameters_updated']=any(not torch.equal(fresh['model_state_dict'][k],v) for k,v in original['model_state_dict'].items() if k.startswith('actor.') or k.startswith('critic.'))
checks['selected_checkpoint_preserved']=hashlib.sha256(parent.read_bytes()).hexdigest()==selected['selected_sha256']
old=parent.parent
checks['old_inventory_preserved']=all((old/r['name']).stat().st_size==r['bytes'] and (old/r['name']).stat().st_mtime_ns==r['mtime_ns'] for r in selected['checkpoint_inventory'])
checks['latest_checkpoint_preserved']=hashlib.sha256((old/'model_6900.pt').read_bytes()).hexdigest()==selected['latest_checkpoint_sha256']
checks['gravity_state_restored']=fresh.get('v2_gravity_curriculum_state')==original.get('v2_gravity_curriculum_state')
native_paths=re.findall(r'^\[Info\] \[carb\] Logging to file: (.+)$',log,re.M)
fatal=re.compile(r'Traceback \(most recent call last\)|CUDA.*out of memory|CUDA error|PhysX error|\[Error\].*(?:physx|omni\.physx)|(?:contact|patch|pair|collision|foundLost).*(?:overflow|capacity.*exceed|buffer.*too small|lost contacts|dropped contacts)',re.I)
scan={}
for p in [out/'train.log',*map(Path,native_paths)]:
 lines=p.read_text(errors='replace').splitlines();scan[str(p)]={'fatal':[s for s in lines if fatal.search(s)][:20],'errors':[s for s in lines if '[Error]' in s][:20]}
checks['no_fatal_capacity_error']=not any(v['fatal'] for v in scan.values())
checks['no_error_lines']=not any(v['errors'] for v in scan.values())
checks['two_native_logs']=len(native_paths)==2
report={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'passed':all(checks.values()),'checks':checks,'iterations':iterations,'optimizer_steps':steps,'normalizer_counts':counts,'contracts':contracts,'distributed':ddps,'migrations':migrations,'log_scan':scan,'new_checkpoint_sha256':hashlib.sha256(fresh_path.read_bytes()).hexdigest(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'recent_metrics':[s.strip() for s in log.splitlines()[-180:] if any(t in s for t in ('stage1_success:','stage2_success:','stage3_success:','GravityCurriculum/gravity_z:'))],'limitation':'Initial resume/PPO/DDP/capacity evidence; no independent task-success evaluation.'}
(out/'startup_verified.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'ready':True,'passed':report['passed'],'checks':checks,'iterations':iterations[-12:],'optimizer_steps':steps,'gpu':report['gpu']},indent=2))
