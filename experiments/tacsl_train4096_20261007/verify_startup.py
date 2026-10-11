import json,pathlib,re,subprocess,hashlib,datetime
import torch
root=pathlib.Path('/home/brainco1/workspace/xinyu/tacsl_insertion_lab')
job=root/'reports/train4096_20261007_v3'
log=(job/'train.log').read_text()
service=subprocess.check_output(['systemctl','--user','show','tacsl-train4096-20261007-v3','--property=ActiveState,SubState,MainPID,MemoryCurrent,RuntimeMaxUSec'],text=True)
initial=torch.load(job/'train/initial.pt',map_location='cpu',weights_only=False)
check=torch.load(job/'train/model_0.pt',map_location='cpu',weights_only=False)
changes={}
for role in ('actor','critic'):
 a=initial[role+'_state_dict'];b=check[role+'_state_dict']
 assert all(torch.isfinite(v).all() for v in b.values())
 changes[role]=sum(float((b[k]-v).abs().sum()) for k,v in a.items() if k.startswith('mlp.'))
 assert changes[role]>0
states=check['optimizer_state_dict']['state']
assert all(torch.isfinite(v).all() for s in states.values() for v in s.values() if torch.is_tensor(v))
episodes=[json.loads(x) for x in (job/'train/episodes.jsonl').read_text().splitlines()]
critical=[line for line in log.splitlines() if re.search(r'\[Error\]|\[Fatal\]|Traceback|GUARD_ABORT|Contacts have been dropped',line)]
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
acc=EventAccumulator(str(job/'train')).Reload()
scalars={tag:[{'step':e.step,'value':e.value} for e in acc.Scalars(tag)] for tag in acc.Tags()['scalars']}
assert all(torch.isfinite(torch.tensor([e['value'] for e in values])).all() for values in scalars.values())
assert not critical,critical[:3]
args=json.loads((job/'train/arguments.json').read_text())
assert args['num_envs']==4096 and args['collision_stack_size']==4293918720
report={'verified_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'service':service,'job':str(job),'arguments':args,'iteration_numbers':[int(i) for i in re.findall(r'Learning iteration (\d+)/1500',log)],
 'critical_log_count':len(critical),'checkpoint':'model_0.pt','checkpoint_sha256':hashlib.sha256((job/'train/model_0.pt').read_bytes()).hexdigest(),
 'network_weight_l1_change':changes,'optimizer_steps':sorted(set(int(s['step']) for s in states.values())),
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip(),
 'episode_batches':episodes,'scalars':scalars,'finite_checkpoint_optimizer_and_scalars':True}
(job/'startup_verified.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('scalars','arguments')},indent=2))
