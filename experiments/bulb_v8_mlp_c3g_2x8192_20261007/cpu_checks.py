"""Use the existing DSW runtime for CPU-only task and recurrent-policy checks."""
from pathlib import Path
import os,subprocess,json
root=Path('/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007')
out=root/'experiments/bulb_v8_mlp_c3g_2x8192_20261007'
prefix=Path('/home/admin/workspace/xinyu/brainco_isaaclab_2_3_2_env/.pixi/envs/default')
python=prefix/'bin/python'
version=subprocess.check_output([str(python), '-c', 'import sys; print("python%d.%d" % sys.version_info[:2])'],text=True).strip()
libs=sorted((prefix/'lib'/version/'site-packages/isaacsim/extscache').glob('omni.usd.libs-*'))
assert len(libs)==1,libs
config=os.environ.copy()
config.update(ISAACLAB_ROOT='/home/admin/workspace/xinyu/IsaacLab-d94504bc',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
config['PYTHONPATH']=str(libs[0])+':'+str(root/'runtime/bulb_grasp_teacher')
config['LD_LIBRARY_PATH']=str(libs[0]/'bin')+':'+str(prefix/'lib')+':'+config.get('LD_LIBRARY_PATH','')
results={}
for name in ['check_bulb_grasp_curriculum']:
 with (out/(name+'.log')).open('w') as log:
  r=subprocess.run([str(python),str(root/'scripts/diagnostics'/ (name+'.py'))],cwd=root,env=config,stdout=log,stderr=subprocess.STDOUT)
 results[name]={'exit_code':r.returncode,'tail':(out/(name+'.log')).read_text(errors='replace').splitlines()[-5:]}
 print(json.dumps({name:results[name]}),flush=True)
 assert r.returncode==0,results[name]
(out/'cpu_verified.json').write_text(json.dumps({'passed':True,'results':results},indent=2)+'\n')
