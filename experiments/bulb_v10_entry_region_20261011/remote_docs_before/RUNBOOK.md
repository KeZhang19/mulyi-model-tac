## 2026-10-07：本地v8端点浅插判据

课程CPU入口仍为`check_bulb_grasp_curriculum.py`，当前52/52通过；[复现命令与日志](../experiments/bulb_stage2_tip_entry_20261007/README.md)。
二阶段<30°/无速度或保持，位置奖励跟踪实际螺纹端点；三阶段原稳定条件保持。
本地train/play使用v8；远端已有v7作业未更新。旧模型在v8下继续学习/评测是变更MDP，需保留原源码回放旧协议。

## 2026-10-07：v7 LSTM双卡各8192启动

指定wpmt DSW使用独立项目`/home/admin/workspace/xinyu/mulyi-model-tac-v7-lstm-20261007`。
已执行命令见[双卡launch.sh](../experiments/bulb_v7_lstm_dsw_2x8192_20261007/launch.sh)，已有作业勿重复执行。
`CUDA_VISIBLE_DEVICES=0,1 ITERATIONS=30000`，原双卡启动器追加`--lstm --num_envs 8192`、buffer scale2、collision stack4026531840。
从零运行，无旧MLP checkpoint。每rank1帧380维/LSTM256，完整v7三阶段判据保持。
只读[verify_startup.py](../experiments/bulb_v7_lstm_dsw_2x8192_20261007/verify_startup.py)用远端现有brainco Python运行；
需至少10次更新、两rank的launch/DDP契约和model0后核验，报告见[运行记录](../experiments/bulb_v7_lstm_dsw_2x8192_20261007/README.md)。
原v6已停止，检查点保留；不自动恢复旧作业。

## 2026-10-07：TacSL 4096环境后台训练

当前有效服务 `tacsl-train4096-20261007-v3`，位于5090主机 `brainco1@192.168.10.252`。
使用 `systemctl --user status` 和工程 `reports/train4096_20261007_v3/train.log` 查看；已有作业勿重复启动。
冻结[launch.sh](../experiments/tacsl_train4096_20261007/launch.sh)含4096环境、1500更新、horizon128/minibatches32、
save_interval25、容量scale4/collision_stack4293918720、without_tactile/compact_logs；保持原状态观测与接触物理。
`scripts/guard_run.py` 在仿真Error/接触丢弃/异常时停止自己的作业；不能从失败v1/v2直接宣称有效结果。
完成后自动seed1042评测1024回合；服务24h上限，异常/超时后续评测不执行。
TensorBoard仅远端loopback6007，本机SSH转发后 [6007](http://127.0.0.1:6007/#scalars) 可读。
训练/查看/停止命令和验证脚本见[本次说明](../experiments/tacsl_train4096_20261007/README.md)。

## 2026-10-07：灯泡教师单帧 LSTM 开关

专用 `train_bulb_grasp_teacher.py` / `play_bulb_grasp_teacher.py` 及对应shell入口支持 `--lstm`，
`launch_bulb_grasp_teacher_2gpu.sh`透传。启用时使用任务内置runtime、历史1帧/380维、单层LSTM256；
省略保持旧5帧MLP。首次启用不传旧MLP checkpoint，之后LSTM续训/回放均须传开关及兼容checkpoint。
训练/回放会启动Isaac/GPU；完整示例和部署文件见[任务说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md#2026-10-07可选单帧-lstm)。
已执行的CPU检查（无Kit/GPU，用真实RSL循环PPO和合成观测）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_teacher_lstm.py
```

7/7通过，不代表原生仿真、双卡或成功率验证。

## 2026-10-07：5090 独立 TacSL 插入任务

连接目标为 `brainco1@192.168.10.252`（主机名 `brainco1-ThinkStation-P3-Tower-Gen-2`），不是旧 SSH 别名 `5090`。
在 `/home/brainco1/workspace/xinyu/tacsl_insertion_lab` 使用既有 `../assembly_play_sim6/bin/python`；
入口 `scripts/run.py --mode {smoke,tactile,train,eval} --headless --output reports/新目录`。
先核实 GPU/活动进程，不覆盖已有输出。首次转换用 `scripts/convert_assets.py`，CPU 核验用 `scripts/validate_contract.py`；
详细环境变量、参数与已执行命令见[独立工程 README](/home/liuxinyu/workspace/tacsl-insertion-lab/README.md)。
本次单环境/触觉和4次PPO更新通过，seed43独立0/16；不是长期训练授权，也不自动恢复作业。
[实验及差异](../experiments/tacsl_insertion_20261007/README.md)。

## 2026-10-07：v6双卡8192从v5续训

已启动：`dsw-wpmt7kzoh4b53jpmov:/home/admin/workspace/xinyu/mulyi-model-tac-v6-20261007`。
具体命令在`experiments/bulb_v6_dsw_resume_20261007/launch.sh`，已有作业时不重复执行。
使用父v5 model9500，`ITERATIONS=20500`，`--num_envs 8192 --initial_adr_difficulty 0`
及`--physx_buffer_scale 2 --physx_collision_stack_size 4026531840`。
checkpoint不存逐环境ADR；新可选参数仅显式指定时覆盖，需按父run匹配，不能对所有旧模型都假设0。
省略参数保持原行为：从零为0，有checkpoint则沿配置默认。非法/非有限/越界值在创建环境前拒绝。
原始优化器、归一化和重力状态通过runner恢复；奖励v6不同，属于更换奖励后的继续学习。
只读验收脚本为同目录`verify_startup.py`，验证生效配置、两rank all-reduce、检查点更新和原生日志；
使用该服务器既有brainco解释器，不启动/干预训练，满10个日志迭代后生成`startup_verified.json`。
[本次44项CPU及12次PPO更新证据](../experiments/bulb_v6_dsw_resume_20261007/README.md)。

## 2026-10-06：v5 TensorBoard访问

本机 http://127.0.0.1:6006/#scalars ，run `Bulb_v5_2x8192/.`，指向新DSW独立v5运行的rank0日志。
远端TensorBoard仅监听127.0.0.1:6006；本机SSH转发命令记录在
[隧道配置](../experiments/bulb_v5_dsw_2x8192_20261006/tensorboard_tunnel.json)，服务启动参数见
[启动记录](../experiments/bulb_v5_dsw_2x8192_20261006/tensorboard_launch.json)。
断线后先核实6006端口及现有进程，再恢复转发，勿重复占端口或重启训练。
重点曲线：stage1_success为进入二阶段，stage2_success为进入三阶段，stage3_success为完整成功；
thread_down_error_deg越低越好，screw_turns为旋入进度。均为训练指标，不是独立评测。

## 2026-10-06：灯泡v5双卡每卡8192启动覆盖

DSW `dsw-c3g3e0ieowszisvolv` 的独立副本为
`/home/admin/workspace/xinyu/mulyi-model-tac-v5-20261006`；已有作业时不要重复启动。
本次实际命令见[launch.sh](../experiments/bulb_v5_dsw_2x8192_20261006/launch.sh)。
新运行需先核实GPU/作业归属并另选RUN_DIR；双卡脚本的默认4096可由末尾`--num_envs 8192`覆盖。

`train_bulb_grasp_teacher.py`新增`--physx_buffer_scale {1,2,4}`（默认1），仅倍增
rigid contact/patch、三项pair、heap、temp七项容量；`--physx_collision_stack_size BYTES`
默认不覆盖，必须满足0<值<2^32。均为每rank容量，写入launch_contract和env.yaml。
本次值为scale2及4026531840字节（3.75GiB），不修改原快照、求解器或碰撞几何。
需同时检查两rank的`distributed_verified.json`、生效配置和PhysX/CUDA日志；启动并不等于容量或学习成功。
[运行记录](../experiments/bulb_v5_dsw_2x8192_20261006/README.md)。

## 长方体贴纸v3

沿用 `python3 assets/head/tag_press_head/build.py`；当前输出为正方形端面接口、四长侧面每面2枚tag，共8枚。PDF A4/100%，裁剪尺寸均20×40 mm。旧4枚检测结果不适用，当前paper_validation.json为8/8。

## 当前长方体压头与贴纸重建（2026-10-05 更新）

`python3 assets/head/tag_press_head/build.py` 使用 CadQuery/NumPy/Trimesh/ReportLab，本地CPU重建单色STL、STEP、3MF、四面PDF、角点布局与几何报告。只重写此目录当前生成文件，不运行硬件、Isaac或训练。打印PDF用A4/100%，先核对50 mm标尺。旧双色视觉脚本仅用于归档；本次PDF用600 DPI渲染并经OpenCV核验4/4。

## 2026-10-05：灯泡抓取教师headless视频

在已安装Isaac、任务独立rsl_rl runtime及imageio/Pillow/FFmpeg的环境执行：

```bash
PYTHONPATH="$PWD/runtime/bulb_grasp_teacher${PYTHONPATH:+:$PYTHONPATH}" \
bash scripts/rsl_rl/play_bulb_grasp_teacher.sh --headless --device cuda:0 \
  --num_envs 1 --steps 960 --seed 42 --checkpoint CHECKPOINT.pt \
  --video NEW_DIR/replay.mp4 --report NEW_DIR/report.json
```

`CHECKPOINT.pt`和`NEW_DIR`需替换；远端需指定`BULB_TEACHER_PYTHON`及`ISAACLAB_ROOT`，先核实GPU资源和当前授权。
该命令启动GPU物理/RTX并写视频、逐帧trace及初始/首个二阶段PNG，不训练；拒绝覆盖既有MP4。
默认1280×960/30FPS，960步为32秒；可调`--video-resolution WIDTH HEIGHT`（偶数）、`--camera-eye X Y Z`和`--camera-lookat X Y Z`。
HUD为当前帧瞬时成功；报告terminal_episodes为回合末态，二者不能混用。视频结束可截断最后一个回合。
2026-10-05已在DSW实际录制model13100，完整解码/哈希通过；实际远端命令与独立Kit缓存路径见
[启动清单](../recordings/bulb_grasp_curriculum_20261005_1942/launch_complete.json)，结果见EXPERIMENTS对应记录。
首次Kit初始化曾等待约8分钟后自行恢复；不能仅凭数分钟无新日志判定死锁。

## 2026-10-05：压头表面和原生相机检查

`/home/liuxinyu/miniconda3/envs/brainco/bin/python calibration/assets/head/tag_cube_head/inspect_surface.py`做CPU射线检查；
同解释器执行`calibration/assets/head/tag_cube_head/inspect_isaac.py --headless --output NEW_DIR`运行有限本地GPU相机检查，要求新输出目录，不连接硬件／训练。
[结果边界与场景](../calibration/assets/head/tag_cube_head/surface_inspection/README.md)。

## 2026-10-05：压头销生成

`python3 calibration/assets/head/press_head_pin/build.py` 使用现有系统Python及CadQuery／NumPy／Trimesh／Matplotlib，重写该目录三种STL、STEP、validation.json及preview.png；仅CPU，无硬件。尺寸、试配限制见[说明](../calibration/assets/head/press_head_pin/README.md)。

## 2026-10-05：五面四标签正方体压头（当前）

`python3 calibration/assets/head/tag_cube_head/build.py`重建STL/3MF/STEP/角点；
`/home/liuxinyu/miniconda3/envs/brainco/bin/python calibration/assets/head/tag_cube_head/verify_vision.py`验证五面20枚tag、上下视角合成PnP与3MF回读。
仅本地CPU、重写本目录输出；依赖／打印／接口边界见[说明](../calibration/assets/head/tag_cube_head/README.md)。下方长方体为旧方案。

## 2026-10-05：标签压头重建与离线检查

`python3 calibration/assets/head/tag_press_head/build.py` 重写该资产目录的 STL/3MF/STEP/布局；
`/home/liuxinyu/miniconda3/envs/brainco/bin/python calibration/assets/head/tag_press_head/verify_vision.py` 从生成网格核验 tag 和合成相机 PnP，并保存图片与报告。
均为本地CPU，无硬件／训练。依赖和打印说明见[资产说明](../calibration/assets/head/tag_press_head/README.md)。

## 2026-10-05：当前Tag唯一支架frame验证

当前默认完整URDF为v5，运行CPU检查：
`/home/liuxinyu/miniconda3/envs/brainco/bin/python calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_only/verify.py`。
只读、无Isaac/硬件。下方tag_origin/verify.py属于旧v4验证，不能对当前包使用。

## 2026-10-05：真机三模态参考会话

`calibration/hardware/collect_real_trimodal.py`使用SDK虚拟环境，先确认空载再用
`--phase baseline --confirm-unloaded --output NEW_DIR`；加载用`--phase loaded --reference-session BASELINE_DIR --serial CAMERA_SN --output NEW_DIR`。
正常运行打开相机并写新目录，无手部运动/串口读写；help离线。参考哈希和确认状态在打开相机前检查。
完整命令、原生数据字段和训练转换缺口见[hardware说明](../calibration/hardware/README.md)。

## 2026-10-05：抓取教师两阶段课程CPU检查

当前本地训练/Play默认v7：新增二阶段晋级一次+10，48项CPU检查通过；未切换正在运行的远端v6。
v7继承v6朝向shaping；继承v5口上15cm→浅插5.5mm→旋入3圈；前两阶段共8s、第三阶段独立6s，首次最终成功+20。
v6只在第二阶段对success位置奖励乘朝向软系数，`rewards.success.params.orientation_floor=0.2`；1恢复旧奖励数值。
2026-10-07的CPU检查为44/44，证据`experiments/bulb_stage2_reward_20261007/cpu_checks.log`，未运行v6原生仿真或训练。
v6已于10月7日部署至新副本并从v5续训，见本手册顶部；下方初次实现的未部署说明属于历史阶段。
下方10月4日原生运行结果属于历史单阶段版本。
以下检查不启动Kit/GPU，不加载策略或执行物理仿真；使用现有Torch和Isaac附带OpenUSD：

```bash
USD_RUNTIME=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
PYTHONPATH="$USD_RUNTIME" LD_LIBRARY_PATH="$USD_RUNTIME/bin:/home/liuxinyu/miniconda3/envs/brainco/lib" \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_grasp_curriculum.py
```

v5 CPU40项通过：[日志](../experiments/bulb_grasp_screw_stage3_20261005/cpu_success_bonus2.log)。
测试默认读取`/home/liuxinyu/docker_workplace/IsaacLab`，可用`ISAACLAB_ROOT`指定；使用真实数学、CommandTerm和事件参数验证器，场景为替身。
部署需携带`assets/bulb_grasp_teacher/{socket.usd,socket.json,bulb_insert.usd,insertion_geometry.json}`及原点云/metadata。
生成新灯泡需先有旧`bulb_grasp.usd`，在上面的USD环境执行
`source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/build_insertion_asset.py`；
它仅重写bulb_insert.usd和insertion_geometry.json，原资产不改。

有限原生完整任务检查命令与原回放相同，可用`--num_envs 2 --steps 245 --report NEW_REPORT.json`，
并提供任务独立runtime的PYTHONPATH。必须读取报告passed字段：Kit关闭可能使失败进程也返回0。
v5实测[报告](../experiments/bulb_grasp_screw_stage3_20261005/native_success_bonus.json)，非成功率评测。

隔离物理探针（新报告路径、无机器人/策略/训练，启动GPU物理，先零重力目标释放，再正常重力自由落下）：

```bash
PYTHONPATH=/home/liuxinyu/docker_workplace/IsaacLab/source/isaaclab \
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/probe_bulb_shallow_insertion.py \
  --headless --device cuda:0 --report NEW_REPORT.json
```

[本次证据与局限](../experiments/bulb_grasp_insertion_20261005/README.md)。需核实本机GPU资源，报告拒绝覆盖；此探针的制造初态不可算作完整任务成功。

三圈几何检查可将上方探针命令的脚本换为`scripts/diagnostics/probe_bulb_screw_path.py`，仍须新报告路径。
它在三种倍率、13个螺旋位姿分别写入制造初态并零重力释放0.2s，无机器人/策略/引导力；
[本次39/39检查](../experiments/bulb_grasp_screw_stage3_20261005/physics_path.json)不等于策略旋入成功。

## 2026-10-04：抓取教师双卡从零入口

在服务器已配置Isaac5.1/Lab d94504bc环境和任务独立runtime时：

```bash
ITERATIONS=30000 bash scripts/rsl_rl/launch_bulb_grasp_teacher_2gpu.sh
```

默认2×4096环境、seed42/43，模型/优化器/ADR/重力触发从零开始；rank0保存模型，rank1单独保存配置和同步验证证据。
首个优化步骤后的`distributed_verified.json`记录真实gradient all-reduce计数和两rank参数SHA。
部署及停止旧训练记录见`experiments/bulb_grasp_teacher_scratch_2x4096_20261004/`。

## 2026-10-04：移植灯泡抓取教师回放（已验证）

```bash
bash scripts/rsl_rl/play_bulb_grasp_teacher.sh --num_envs 16 --real-time
```

默认使用本仓库model_22200；关闭窗口结束。无界面有限检查加`--headless --num_envs 4 --steps 180 --report NEW_REPORT.json`。
本次实际执行4环境180步，以及专用训练入口4环境1次PPO续训；两者均通过，非成功率评测。
通用train/play不支持此任务的专用runner。资产、环境变量、可选续训命令和验证限制见
[任务说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md)。

## 2026-10-04：Tag原点URDF的CPU复核

当前默认包以Tag槽底名义中心为全局原点，世界轴向保持；码纸厚度待实测。
从仓库根执行以下只读检查，不启动Isaac或控制硬件：

```bash
/home/liuxinyu/miniconda3/envs/brainco/bin/python calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_origin/verify.py
```

验证旧/新URDF的全部非根link通过同一平移对应，覆盖零位、20组非零状态、121帧原轨迹，
同时检查网格/轨迹/初态哈希、世界原点、轴向、坐标登记一致性。
本次142组/9088个变换通过，urdfdom解析通过；未做Isaac/真机验证。
`modifications/tag_origin/before/`为修订前快照，URDF相对网格仍按原urdf目录解释，不能原地加载。
旧`export_urdf.py`及`verify_touch_asset.py`是v3源包流程，尚未纳入M6×16派生网格与Tag原点；
重建当前包应按[包说明](../calibration/assets/calibration_bench_visuotactile_urdf/README.md)同步应用两项修订。
原NPZ世界系仍为lower_mount；使用manifest的`source_recording_to_current_world`左乘转换，CSV关节值不变。

## 2026-10-04：资产目录整理后的使用与复制

先按[资产入口](../assets/README.md)查模型，按[任务数据入口](../task_data/README.md)查抓姿/采样库。
旧命令继续有效：assets中的任务数据入口是相对符号链接。
新候选应显式写到task_data的新文件；不覆盖原模板，不把旧验证状态当作新结果。
复制/同步必须同时携带assets、task_data、calibration/assets并保留相对符号链接；仅复制assets会漏数据。
实体数据延续本地忽略策略；单独git clone不包含原资产或task_data数据。
本次没有执行任何远端同步，以下命令只做本地CPU检查：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_floor_contract.py
```

2026-10-04本次12项通过；会导入torch和任务的纯计算定义、读取旧兼容入口，不启动Isaac或训练。
迁移完整性检查保存在[organization_verification.json](../task_data/organization_verification.json)：
逐文件SHA256、新旧路径同一实体、来源哈希、标准/Custom D-Peg与装配真实JSON加载函数（AST隔离Isaac依赖）。
这些是文件与CPU验证，未证明动态物理、训练恢复或独立成功率；无需为目录移动启动完整训练。

## 2026-10-04：M6×16局部沉孔生成（CPU）

`python3 calibration/simulation/build_m6x16_mount.py --output NEW_DIR`

系统Python已有CadQuery、trimesh、NumPy和matplotlib；从保留V4 BREP生成四孔沉孔加深版本，
输出STL/3MF/STEP/BREP与validation.json，目录存在时拒绝覆盖；无设备／仿真／训练启动。
生成器不会自动更新URDF。当前指定完整包已集成，见
[说明](../calibration/assets/calibration_bench_visuotactile_urdf/README.md)。
旧export_urdf仍从原V4-touch导出，应用新网格时需同步visual/collision及资产／坐标登记哈希。

## 2026-10-04：标定与采集新入口

完整分类与副作用见[calibration/README.md](../calibration/README.md)；旧路径保留相对符号链接。
从仓库根目录执行：

```bash
# 离线规模预览，不启动GPU/硬件、不写数据
/home/liuxinyu/miniconda3/envs/brainco/bin/python calibration/simulation/run.py --plan-only
# 真机只读短采：打开串口与相机，无运动；目录必须全新
bash calibration/hardware/probe_local.sh --output calibration/captures/NEW_DIRECTORY
```

三模态离线推理入口迁至calibration/hardware/probe_revo3_trimodal.py，参数不变。
仿真采集／验证／导出／Isaac查看位于calibration/simulation；原始仿真数据位于calibration/outputs/simulation。
训练NPZ采集与mmap位于calibration/collection，默认新输出位于calibration/datasets；
压力工具的默认生成输出也统一在calibration/outputs。旧datasets入口链接到calibration/datasets。
不要批量执行目录内脚本；RL球探针／图像生成／回放仍依赖Isaac，不把help默认当轻量检查。
本轮仅做CPU迁移检查，不连接硬件、不启动Isaac/GPU或训练。历史命令原文保留如下。

## 2026-10-04：P3 5090 Stage0 显式低精度性能配置

目标主机 `brainco1@192.168.10.252`；项目根`/home/brainco1/workspace/xinyu/mulyi-model-tac`。
使用`/home/brainco1/workspace/xinyu/assembly_play_conda/bin/python`，Sim5.1/Lab2.3.2。
先确认GPU空闲，设置`OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`，
`PYTHONPATH`包含相邻`assembly_play_support/rsl_rl`与当前冻结工作区`source/BrainCo_DexHand`。
下列命令启动仿真，输出目录必须为新路径：

```bash
python scripts/diagnostics/probe_bulb_floor.py --headless --device cuda:0 \
  --num_envs 256 --steps 300 --warmup 60 --random-actions --physics-hz 120 \
  --solver-position 4 --solver-velocity 1 --out NEW_PROBE_DIRECTORY \
  --kit_args '--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 --/plugins/carb.tasking.plugin/threadCount=8'
```

省略两solver参数恢复原始基线；参数同时钳制场景min/max，报告读取原生USD属性。
固定动作使用独立GPU随机数生成器seed20261004，不受任务reset抽样影响。
`--gpu-partitions 4/8/16`与Kit线程数用于单因素对照；`--audit-scene`输出原生拓扑及碰撞统计。
`--lamp-asset PATH`仅用于独立碰撞诊断；本次正式候选保留原Lamp资产。
应核对`passed`、原生日志、碰撞容量、完整步数和接触质量；报告的32步折算不等于PPO迭代。
生产入口无需修改源码，在Stage0从零新run中显式传入以下四项：

```text
env.sim.physx.min_position_iteration_count=4
env.sim.physx.max_position_iteration_count=4
env.sim.physx.min_velocity_iteration_count=1
env.sim.physx.max_velocity_iteration_count=1
```

保留`env.bulb_curriculum.physics_hz=120`，不要只改sim.dt。原机器人请求32位置迭代，因此只改min无效。
4/1是快速验证候选，不宣称完整旋入精度或长程成功率已验证；物理契约改变，不绕过旧模型恢复校验。
本次用户确认“第一关”是stage0抓持；独立train.py不会自动启动stage1抬起。
后续4096同机测试：仅120Hz/4/1为1.637倍，90Hz/4/1为2.002倍，因此最终选择显式90Hz：

```text
env.bulb_curriculum.stage=0
env.bulb_curriculum.physics_hz=90
env.bulb_curriculum.fast_stage0_physics=true
```

仍使用上面四项solver覆盖，控制为30Hz、decimation=3。默认physics_hz仍120，未开标志则90会报错；
标志在stage1–4被拒绝，60Hz也未开放。90Hz检查点不能直接跨物理契约晋级，后续阶段需单独验证迁移，
不得跳过状态库/源码/物理契约检查。标准任务与其他阶段的默认配置未改。
本轮正式入口在远端`experiments/assembly_stage0_fast90_4096_20261004/launch.sh`，冻结源码在同名`.workspaces/`目录；
Stage0从零、4096环境、seed42、2000轮上限，使用原生物理错误停机守卫。新run复现应另建输出目录，
不能重复覆盖已有launch.sh里的输出。训练及守卫状态读取`train.log`、`guard_status.json`，最终验收见`startup_validation.json`。
实验与启动状态见STATUS及`ASM-PERF-P3-20261004-06`。

# 运行与验证手册

## 2026-10-03：P3 RTX5090 Isaac Sim 6.0.1 隔离环境

远端前缀为 `~/workspace/xinyu/assembly_play_sim6`，对应 Isaac Lab checkout 为
`~/workspace/xinyu/assembly_play_support/IsaacLab6`（`v3.0.0-beta2.patch1`）。不要混用下方 5.1
前缀的 Python 或 RSL-RL。官方 Cartpole 训练与 RTX 回放已通过，日志分别为
`sim6_cartpole_train.log`、`sim6_cartpole_play_video.log`，回放视频位于该 checkout 的
`logs/rsl_rl/cartpole/2026-10-03_23-14-23_sim6_rtx5090_smoke/videos/play/rl-video-step-0.mp4`。
首次启用 RTX 相机可能花约 80 秒构建缓存；应等待日志出现模型加载和 RTX streaming，而不是看到
短时高 CPU 就直接判定卡死。

这套环境可训练和回放 Lab 3 任务，但当前不能直接回放原 assembly stage0 checkpoint。原模型继续用
`assembly_play_conda` 做无界面评测；迁移到 6.0.1 前需完成任务模块的延迟运行时导入、Lab 3 启动入口、
RSL-RL 3.1.1→5.0.1 checkpoint 兼容核查，以及四元数／ProxyArray 语义验证。仅让状态字典成功加载
不能证明策略行为兼容。

## 2026-10-03：P3 RTX5090 独立 Conda 回放环境

远端 `brainco1@brainco1-ThinkStation-P3-Tower-Gen-2.local` 使用既有Miniforge和conda-forge
创建 `~/workspace/xinyu/assembly_play_conda`；Isaac包通过NVIDIA pip源安装。
解释器为该前缀的 `bin/python`，不需要更改默认环境。版本/安装日志/冻结清单位于
`~/workspace/xinyu/assembly_play_support/`，`pip check`和CUDA运算已通过。
启动命令 `bash ~/workspace/xinyu/assembly_play_support/play_stage0.sh` 会使用桌面 `DISPLAY=:1`，
执行1环境、stage0、evaluation=true、120Hz物理、30Hz控制的model1000回放，并产生导出及回合日志。
**当前GUI未通过**：595.91.07驱动下在RTX渲染初始化崩溃，尚未加载模型；命令不是已验证可用的回放流程。
原checkpoint不改，回放副本及从内嵌契约重建的sidecar在support/stage0_model1000，来源记录随附。
不得为此直接修改系统驱动或更换仿真物理版本；需先确认用户范围并验证兼容性。

## 2026-10-03：横躺装配课程v2

2026-10-04新奖励双卡作业已启动：服务器项目下冻结目录
`.workspaces/assembly_reward_ddp4096_20261004`，输出`experiments/assembly_reward_ddp4096_20261004/train`。
精确启动脚本/守卫/日志位于同一experiment父目录的`launch.sh`、`guard_training.py`、`train.log`、
`guard_status.json`；启动验收为`startup_validation.json`。原主目录缺课程前置，运行时须进入冻结目录。
命令语义仍为下方双卡各4096/每阶段2000轮/三seed晋级配置。该作业已在运行，不要重复执行launch.sh；
诊断只读日志和配置，任何恢复须先核对作业状态与契约。实际核查结果见STATUS/ASM-REWARD-DDP-20261004-04。

2026-10-04本地Stage0新增`reach_best_progress`（每新靠近1cm给+1分，保留机械臂代价）。
同日Stage0另启用any_finger_contact（>1N任意单指，+0.7分/秒）。
沿用下方CPU检查入口，现为11项；本次未运行下方仿真/训练命令或部署服务器。
修改课程源码哈希后，旧checkpoint会被严格恢复检查拒绝；旧回放使用原冻结代码，新奖励训练用新输出目录。
分值与验证边界见[课程奖励说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum/README.md)。

当前实施／证据见[课程说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum/README.md)
及[STATUS](STATUS.md)。以下新命令会启动仿真或训练；使用新输出目录，先确认GPU与已有作业。
不要在正在训练的冻结快照中改源码。

CPU行为检查（无Isaac启动）：

```bash
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_floor_contract.py
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_curriculum_distributed.py
```

服务器已验证解释器为
`/home/admin/workspace/xinyu/brainco_isaaclab_2_3_2_env/.pixi/envs/default/bin/python`，
Isaac Sim 5.1.0、IsaacLab源码标记2.3.2、PyTorch2.7.0+cu126、rsl-rl-lib3.1.1。
从冻结工作区根目录运行，设置 `PYTHONPATH="$PWD/source/BrainCo_DexHand"`、
`OMNI_KIT_ACCEPT_EULA=YES`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=1`。
下列 `python` 代指该解释器。

```bash
python scripts/diagnostics/probe_bulb_floor.py --headless --device cuda:0 \
  --num_envs 256 --steps 180 --warmup 30 --random-actions --physics-hz 120 \
  --out NEW_PROBE_DIRECTORY \
  --kit_args '--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 --/plugins/carb.tasking.plugin/threadCount=8'
```

必须读 `report.json` 的 `passed` 和每项检查；Isaac关闭可能使外层进程返回0，即便报告中有异常。
新版还读取Kit原生日志，发现碰撞栈溢出／丢接触即passed=false。旧报告必须补查外层日志。
`--snapshot --num_envs 1` 会启用相机；本服务器本次启用时渲染器段错误，无截图交付，
不能将相机模式显存／吞吐混入无渲染基准。`--legacy`只用于旧冻结源码对照，跳过水平轴／369维要求。

只训练Stage0并限定预算（本轮计划／试训用1024环境／300轮，完成情况看STATUS）：

```bash
python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0 \
  --headless --device cuda:0 --num_envs 1024 --seed 42 --max_iterations 300 \
  --logger tensorboard --log_dir NEW_TRAIN_DIRECTORY \
  env.bulb_curriculum.physics_hz=120 env.scene.sky_light.spawn.texture_file=null
```

完整课程启动器为 `train_bulb_curriculum.py --out NEW_DIR --iterations-per-stage N`；
现在直接从自由横躺stage0开始，保留真实前序状态库、原独立晋级评测和预算停机。
用户后续明确要求双卡共同训练、每卡4096环境，启动方式为：

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python scripts/rsl_rl/train_bulb_curriculum.py \
  --out NEW_DISTRIBUTED_DIRECTORY --num-processes 2 --num_envs 4096 \
  --iterations-per-stage 2000 --physics-hz 120 --seed 42 \
  --kit_args '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=8' \
  env.scene.sky_light.spawn.texture_file=null
```

此命令通过torchrun/NCCL训练同一个策略，总8192环境；环境seed为42/43，参数与归一化同步。
各阶段达预算且未通过则停止，不跳关。独立晋级评测保持单卡三seed。
4096/卡预留3.75GiB碰撞栈（uint32不能到4GiB），配对容量各2²⁴、接触点2²⁴、接触片2²⁰。
双卡检查点仅允许同world-size/seed映射恢复训练，单卡评测另设`evaluation=True`。
本轮后续已按追加要求启动双卡正式课程，每阶段2000轮上限；实际进度见STATUS与run日志。
评测使用 `evaluate_bulb_curriculum.py --checkpoint CHECKPOINT --out NEW_EVAL --episodes 100
--num_envs 16 --seed 401 --headless`，应传入相同环境配置；stage4才是完整桌面成功率。
v1检查点和旧669维Custom导出与v2不兼容。240Hz保守对照用
`env.bulb_curriculum.physics_hz=240`，不要仅改`env.sim.dt`，后者会被课程统一时钟配置覆盖。

下方保留其他工作线和历史流程。

核查日期：2026-09-20。版本范围见 [STATUS](STATUS.md)，接口来源见 [PROJECT_MAP](PROJECT_MAP.md)。
最初文档审计仅执行低开销检查；后续已按新请求运行下述灯泡初态验证。
下列其他训练、安装、下载、远程与真机命令仍是流程说明，不代表本次执行。

## 2026-10-03：Revo3真机只读探测

2026-10-04后续复测：相机重新连接后5/5通过。三模态入口为
`calibration/probe_revo3_trimodal.py --capture CAPTURE_DIR --output NEW_DIR --reference-state unverified-current`，
使用下述SDK解释器；只对指定原始采集推理，不打开硬件，45秒/路超时且拒绝覆盖。
本次35组诊断数据通过；用户未确认空载，不能用作物理标定。完整两步命令见
[机械标定说明7.6节](../calibration/机械标定台仿真搭建_README.md#76-五路三模态通路验证2026-10-04相机重新接通后)。

2026-10-04新增本机快捷入口：`bash calibration/probe_local.sh --output calibration/captures/NEW_DIRECTORY`。
默认使用下述源SDK解释器，可通过`REVO3_PYTHON`覆盖；参数、有限采集与副作用相同，
`--help`仅离线帮助。当前复测主链路成功、ViTai为0/5，返回2；相机USB恢复后须重新运行。
该入口不执行Depth／Marker推理。最新结果与三模态模板见[机械标定说明7.5节](../calibration/机械标定台仿真搭建_README.md#75-本机连接配置与三模态复测2026-10-04)。

本次已按用户授权实际运行，不发送运动／设备配置／清零命令。
前提：已连接BrainCo转换器和ViTai相机，端口未被其它程序占用，源SDK已有Python环境可用。
脚本优先使用`calibration/brainco-revo3-sdk/runtime/python/`副本；其余依赖来自源环境。

```bash
/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python \
  calibration/probe_revo3_readonly.py --output calibration/captures/new_probe
```

输出目录须不存在；会打开串口、依次采集ViTai并请求相机格式，保存JSON／NPZ／PNG／日志。
默认21关节与阵列3次快照、5相机各8帧，有分进程超时。退出码0仅表示读取完成，不代表模块全启用或健康验收。
本次301声明点中244点有效，拇指指腹禁用；健康未知及电压量纲待核实。
字段、SDK接口、时钟与复现边界见[机械标定说明第7节](../calibration/机械标定台仿真搭建_README.md#7-revo3-sdk与真机只读采集2026-10-03)。

## 2026-10-03：独立机械标定仿真

当前精简场景的URDF导出：`/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/calibration_bench/export_urdf.py --output NEW_DIRECTORY`。
默认读 `outputs/calibration_bench/visuotactile_v1/` 的配置与首回合，CPU复制网格并生成XYZ运动学链、
轨迹CSV和manifest，拒绝覆盖已有目录；完整命令与坐标编辑位置见[视触觉URDF说明](../assets/calibration_bench_visuotactile_urdf/README.md)。
新检出需先执行 `scripts/calibration_bench/build_touch_fixture.py` 生成DV2手＋V4支架；输出已存在则拒绝覆盖。
`verify_touch_asset.py` 核查标准交付路径，写资产验证与任务参考映射，不启动Isaac或硬件。
历史场景可显式 `--scene-dir outputs/calibration_bench/head_fixture_only`，无asset字段继续使用旧手。
当前导出器将压头运动系转为X=原−Y、Y=原+X、Z=原+Z；限位和CSV同步换算，
XYZ关节origin全0；法兰／压头／tip共用工作尖端原点及轴向，stage/carriage零位也位于尖端。
网格局部origin重表达保持实际几何与路径，CSV及相对行程保持。
manifest v3的`T_source_flange_shared_tip`记录原法兰到统一尖端frame的完整刚体变换，比较历史NPZ时需右乘。
该约定也适用于显式导出的历史场景；原交付文件不自动更新。

新增独立标定系统的命令、依赖、副作用和验证等级见
[scripts/calibration_bench/README.md](../scripts/calibration_bench/README.md)。
`run.py --plan-only` 不启动 Isaac/GPU、不写数据；实际采集为 CPU 受控运动模型，输出目录拒绝覆盖。
`preview_isaac.py --headless --capture --frames 30` 会启动本地 Isaac/GPU、读取生成USD并写截图和验证JSON。
默认270回合是计划规模，本轮只做有限短测；没有训练、远程执行或SDK联调。
复现8种短协议：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python \
  scripts/calibration_bench/run.py --config scripts/calibration_bench/smoke_config.json \
  --output outputs/calibration_bench/new_smoke
/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/calibration_bench/verify.py \
  outputs/calibration_bench/new_smoke
```

## 2026-10-03：Revo3标定支架保留包（不可在仓库内重建）

当前只保留 `assets/revo3_breadboard_mount/generated_calibration_v4/`。V1/V2/V3、V4生成器、
独立验证脚本和包外说明已按用户要求清理，因此 2026-10-01 的重建命令不再适用。
直接使用入口为 `urdf/lower_calibration_v4.urdf` 和
`urdf/revo3_right_calibration_lower_mount_v4.urdf`；CAD、打印 STL/3MF、手部网格、预览及
原 `validation.json` 均随包保留。清理后已复核两份 XML 可解析，143 个相对 mesh 引用无缺失；
没有重新运行 ROS urdfdom、Isaac、碰撞扫描、打印或真机标定。

2026-10-03 同时清理了根目录 `tests/`、运行日志、录像、输出和本地 benchmark 产物。
下文出现的 pytest 命令与 `logs/`、`recordings/`、`outputs/`、`benchmarks/` 本地路径均为
历史流程记录，当前工作区不再提供相应测试或证据文件。`docs/` 下带日期后缀及
experiment(s) 命名的 52 份历史日志型 Markdown 也已清理；下文对它们的链接不再可用。

## 2026-10-01：双 Y 标定台 Blender 装配导出 URDF（CPU）

2026-10-03复核：下面输入目录当前缺失，命令仅作历史记录，不能在当前工作区直接复现。

输入为 `calibration_bench_sim_package/assets/blender/dual_y_press_checked.blend`，输出目录必须
不存在；生成器不修改 Blender/STEP 原件。默认输出已位于
`calibration_bench_sim_package/generated_urdf/`，包含 base→Y→X→Z 四 link、四份米制视觉 STL、
简化碰撞盒和两份 URDF。名义 CAD 版为 X/Y/Z ±250/±150/±75 mm；软件限位版来自
`config/mechanical.yaml`，但其 Z 范围超过候选 CAD，不能当作全行程碰撞通过。

```bash
blender --background calibration_bench_sim_package/assets/blender/dual_y_press_checked.blend \
  --python calibration_bench_sim_package/tools/export_urdf.py -- \
  --output /tmp/calibration_bench_urdf

xmllint --noout /tmp/calibration_bench_urdf/urdf/*.urdf
LD_LIBRARY_PATH=/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  /opt/ros/humble/bin/check_urdf \
  /tmp/calibration_bench_urdf/urdf/dual_y_press_nominal_cad.urdf
```

本机 `~/.local/bin/check_urdf` 当前缺少 `cmeel` Python 模块，不用于验收；ROS Humble 的
`/opt/ros/humble/bin/check_urdf` 已实际通过两份 URDF。导出目录与独立临时重建逐字节一致。
这些检查只证明 XML/拓扑/路径/确定性，不证明 Isaac/Gazebo 动力学、质量参数或连续碰撞有效。
证据范围见 [实验记录](../EXPERIMENTS.md#calibration-bench-urdf-20261001-01)。

## 2026-10-01：当前工作区恢复GPU4 profile3旋出任务

本地及服务器`/home/xinyu/mulyi-model-tac`默认Randomized任务已恢复高成功率GPU4冻结配置：profile3、物理60Hz/控制30Hz/12秒、无initial_arm_pose、原joint1零位与±160°、workspace_bank_high_z_v2。
基准run `/home/xinyu/training_runs/20260928_140146_rotate_bulb_phase_phys60_30hz_12s_scratch_gpu4`，检查点`train/model_8400.pt`；本次只恢复版本，未启动训练。运行时的6144环境/seed42等完整CLI保存在原run的launch_spec.json，不能把配置类默认环境数当作当时生效环境数。
三份恢复文件已逐字匹配冻结manifest；两端各114 CPU测试及相关源码/资产hash通过，未新做原生运行。GPU6 profile6检查点属于另一资产/契约，不能在恢复后的默认任务上直接加载。
新零位文件仅保留归档用途，当前配置不引用；后续如果运行旧GPU6检查点，必须沿其冻结workspace，不要再次覆盖恢复的默认任务。
备份与恢复报告见[状态](STATUS.md)及[BULB-RESTORE-GPU4-20261001-20](../EXPERIMENTS.md#bulb-restore-gpu4-20261001-20)。

## 历史2026-09-29：GPU6 profile6，joint1零位±180/姿态weight -0.01（现已完成）

当前run `/home/xinyu/training_runs/20260929_125130_rotate_bulb_joint1zero_pose001_phys60_scratch_gpu6`；精确命令在launch_spec.json，supervisor/worker见status.json。GPU6/6144/32steps/seed42/15000轮，60Hz物理/30Hz控制/12秒，从零训练。
使用新USD层及workspace，旧Pose90 checkpoint不兼容；worker禁止load，scratch_audit确认空优化器/iteration0/weight=-0.01/joint1零位及硬软限位±π。
TensorBoard标签`Rotate-Bulb_Joint1Zero_Pose001_Phys60_Scratch_GPU6_20260929`。旧profile5已停止，GPU4保持。验证及部署文件见[记录](../logs/diagnostics/bulb_joint1_zero_weakpose_gpu6_20260929/README.md)。

## 2026-09-29：本地关节1零位±180候选，尚未替换活动训练

Randomized使用独立新USD和workspace，原物理初态保持；部署必须包含`env_cfg.py`、`joint1_zero.py`、`rizon4_joint1_zero.usda`和`workspace_bank_joint1_zero_v1.json`，新增文件不一定在旧source_manifest里。
原CPU检查：`python -m pytest -q tests/test_bulb_joint1_zero.py tests/test_bulb_bounded_actions.py tests/test_rotate_bulb_initial_arm_pose.py tests/test_rotate_bulb_randomized.py`，32项通过。
独立GPU7 baseline/centered原生流程、精确命令、9项检查及资产重建脚本见[诊断记录](../logs/diagnostics/bulb_joint1_zero_20260929/README.md)。检查含显式状态注入，不是任务评测。
旧checkpoint必须沿旧冻结坐标系回放；不能关闭契约校验在新坐标系上直接续训。GPU6 Pose90旧run与GPU4仍保留原冻结资产；新GPU6 profile6采用零位资产。

## 历史：GPU6 profile5，已被profile6替换

当前run `/home/xinyu/training_runs/20260929_104034_rotate_bulb_pose90_phys60_scratch_gpu6`，精确命令保存在launch_spec.json。GPU6/6144/32steps/seed42/15000轮/初始LR0.001，60Hz物理/30Hz控制/12秒，与前run仅deadband/profile不同。
worker安装scratch_audit，禁止checkpoint load；正式独立于pilot从零初始化。旧profile4在1754轮已停止，不能把旧目录作为当前作业恢复。
TensorBoard筛选Pose90；status.json看当前轮次，formal_verification.json与train/initialization_verification.json确认初始化与配置。
检查命令及53项CPU/12项原生/13轮pilot证据见[执行记录](../logs/diagnostics/bulb_pose90_gpu6_20260929/README.md)。

## 历史：2026-09-29 GPU6 profile4，已由profile5替换

当前run：`/home/xinyu/training_runs/20260929_071719_rotate_bulb_initial_pose_phys60_scratch_gpu6`。GPU6/6144、物理60Hz/控制30Hz/12秒、32steps、seed42、15000轮、初始LR0.001，`agent.resume=false`。
`worker.py`安装`scratch_audit.py`，禁止checkpoint加载并记录初始化证据；正式和pilot分别随机初始化。不要改为恢复旧GPU6 240Hz模型或临时model6300续训目录。
原GPU6已停，GPU4保持；只管理当前run内supervisor及worker，不使用历史PID盲目停止。
`launch_spec.json`保存精确参数；`train/initialization_verification.json`和`formal_verification.json`是本次启动证据，`status.json`为进度。
TensorBoard筛选`InitialPose_Phys60_30Hz_12s_Scratch`，新增标签`Episode_Reward/initial_arm_pose`。正式env与GPU4仅奖励及路径差异，agent仅run_name差异。
验证与停止历史见[执行记录](../logs/diagnostics/bulb_initial_pose_gpu6_launch_20260929/README.md)。

## 2026-09-29：本地profile4初始姿态弱引导验证

新增`initial_arm_pose`：权重-0.025、15度免罚/90度尺度、关节权重[2,1,1,1,.5,.5,.5]、释放后0.2；不改变现有物理或契约。GPU4/GPU6冻结作业仍profile3，编辑本地文件不会影响活动进程。
CPU检查与离线尺度复算命令见[验证记录](../logs/diagnostics/bulb_initial_pose_guidance_20260929/README.md)；68项通过。尚未进行profile4原生仿真/训练，部署时须使用新冻结快照并保存env.yaml。旧录像仍用原快照，不能把新增成本追溯为旧模型训练奖励。

## 2026-09-29：TensorBoard重启与本机自动重连转发

入口仍为 <http://127.0.0.1:6006/#scalars>。本次故障为本机SSH链路超时、6006转发消失；
服务器旧TensorBoard仍响应。按用户要求重启TensorBoard，PID66933→1185617，保留原33组run及日志目录，
GPU4/6训练worker及启动时钟保持。首次启动需重新读取历史事件，run列表出现早于全部曲线加载完成。

本机转发由用户级临时systemd服务`codex-xinyu21-tensorboard-tunnel.service`管理，断线失败10秒后重连，
仅监听127.0.0.1:6006；不依赖操作终端保持打开。认证失败/未知主机键等不会无限重试。
帮助脚本为`/home/liuxinyu/.local/state/codex/xinyu21-tensorboard/ssh_tunnel.py`，运行时使用既有SSH配置，
不复制密码到脚本或unit。控制socket保持`/tmp/codex_xinyu21_tb_live.sock`。

```bash
systemctl --user status codex-xinyu21-tensorboard-tunnel.service
systemctl --user restart codex-xinyu21-tensorboard-tunnel.service
```

此临时unit不承诺跨本机重启保留；若unit已不存在，需要重新建立服务。不要同时启动其他6006转发。
本机HTTP/标量核验记录位于`/home/liuxinyu/.local/state/codex/xinyu21-tensorboard/verification.json`；
远端重启记录位于`/home/xinyu/training_runs/20260919_115128_nomodal8192/tensorboard/restart_*.json`。
GPU4新训练筛选`Phys60_30Hz_12s_Scratch`，GPU6筛选`PhaseContact_Resume3500`。

## 2026-09-28 22:03：GPU4物理60Hz/控制30Hz/12秒，从头训练

当前run：`/home/xinyu/training_runs/20260928_140146_rotate_bulb_phase_phys60_30hz_12s_scratch_gpu4`。
用户已选择物理dt1/60、decimation2、控制30Hz、12秒/360控制步；奖励profile3与GPU6一致。
GPU4单卡6144环境、32steps/env、seed42、15000轮；显式`agent.resume=false`，初始学习率0.001。
不加载旧checkpoint，也不加载短测权重；不手改旧动作契约绕过decimation差异。
原GPU4＋7双卡run已停止，GPU6旧周期续训保留；不要重复启动这两个当前作业。

精确命令在`launch_spec.json`，运行环境为run内冻结workspace与runtime；主要源码仅覆盖Randomized配置及说明文件。
`status.json`与`train/worker.json`记录监督/训练PID、物理卡；`train/params/env.yaml`是实际生效参数。
`pilot_verification.json`记录32环境13次PPO、原生奖励与超时重置检查；`formal_verification.json`记录22:10正式到3轮的更新/保存/配置/契约及哈希核验。
监督器只管理自身进程组，遇指定致命仿真错误或Traceback停止，不自动重启；旧日志和模型均保留。
TensorBoard <http://127.0.0.1:6006/#scalars>，Runs筛选`Phys60_30Hz_12s_Scratch`。
原GPU6从model3500续训，物理240Hz/decimation8/20秒；新GPU4从头、物理60Hz/decimation2/12秒，不能直接作等条件学习收益比较。
见[启动与验证记录](../logs/diagnostics/bulb_gpu4_timing_launch_20260928/README.md)。

## 2026-09-28历史：本地Randomized临时控制周期与12秒回合

本地Randomized训练及PLAY默认改为物理60Hz、decimation2、控制30Hz、回合12秒/360步。
用户明确控制30Hz，前版保留240Hz导致120Hz控制的改动已修正。
现有GPU4＋7及GPU6作业使用冻结workspace，仍decimation8、20秒；未重启或修改运行副本。
旧checkpoint动作契约包含control_dt和decimation，虽然控制30Hz相同，decimation变化仍会被校验拒绝；
不要手改历史契约以绕过。回放旧模型使用其原冻结代码和配置。
本次仅配置/CPU验证，12秒是仿真时间，原生物理稳定性、实际墙钟耗时与显存未测量。

## 2026-09-28 20:45：GPU6分阶段接触奖励，从双卡model3500续训

独立run为`/home/xinyu/training_runs/20260928_123831_rotate_bulb_phase_contact_resume_gpu6`。
GPU6、6144环境、32steps/env，继承原Randomized旋出任务/资产/物理/分布；奖励profile版本3。
用户确认从当前双卡run最新checkpoint续训；正式来源为原run的`train/model_3500.pt`，
SHA256以`2f084913c739842d`开头。原GPU4＋7继续运行，**不要重复启动**GPU6。

profile3只调整奖励：辅助接触仅释放后支付，单侧最高0.25/s、对向最高1/s，每回合最多4分；
`randomized_grasp.weight=0`；`arm_motion`改为释放前均约束臂速，权重−0.3，
位置漂移仍只在真实抓持后约束，释放后该项为0。旋转进展仍需真实接触；搬运40、失败−6、
动作/观测/归一化结构、TGS24/4、240Hz/30Hz、三圈和最终成功判据保持。

`launch_spec.json`保存完整命令及非敏感环境。`--resume --load_run phase_resume --checkpoint model_3500.pt`
解析到新run的私有checkpoint；模型、归一化及优化器完整恢复。显式恢复有效学习率
`0.00013647718878928572`，避免adaptive LR标量回到默认值。`--max_iterations 11500`
为从3500追加11500次PPO，总编号上限15000。正式重新加载原模型，未加载pilot2更新的权重。

`status.json`为监督状态，`train/worker.json`含真实PID及物理卡；
`train/resume_verification.json`是首次更新前的逐张量恢复核验。
`train.console.log`和`train/params/env.yaml`分别是原生日志与实际环境配置。
run-local监督器只管理自己创建的进程组，发现Traceback或指定致命仿真错误即停止，不自动重试。
首次pilot的零权重检查误报发生在加载前，证据保留；pilot2原生10项及三次PPO已通过。

TensorBoard http://127.0.0.1:6006/#scalars ，Runs搜索`PhaseContact_Resume3500`。
新指标`Metrics/phase_contact/quality_when_released`和`budget_spent`分别表示当前释放且未失败
样本的平均对向接触质量与所有环境的预算均值；两者都不是任务成功率。
单卡每轮196608转移，原双卡393216；不能按相同迭代号当作等样本公平比较。
新奖励效果待后续完整回合评测，证据见
[本次实施与启动记录](../logs/diagnostics/bulb_phase_rewards_20260928/README.md)。

## 2026-09-27 22:16：接触降至0.1，从指定双卡run的model100续训

Randomized当前默认any_finger_contact=0.1、position_tracking=40、failure=−6，奖励profile版本2。
本次唯一奖励变化是接触1→0.1；Custom/Assembly、随机化、物理参数及成功判据保持。
原双卡run `20260927_041844_rotate_bulb_contact_transport_12288_ddp` 已按用户要求停在1425轮，
model1400及历史日志保留。**续训来源是该run自己的train/model_100.pt**，选择依据是已保存点之前
完整50个原始released日志点均值最高（51–100为39.2601%），不是旧GPU6的model7300。

新run为 `/home/xinyu/training_runs/20260927_141100_rotate_bulb_contact01_tracking40_resume100_ddp`，
GPU4＋7各6144环境、共12288，32steps/env，保留原双卡启动及NCCL设置。**不要重复启动**。
精确命令见新run/launch_spec.json；原模型复制至checkpoint_source，通过冻结workspace内resume100软链解析。
`--resume --load_run resume100 --checkpoint model_100.pt`恢复模型、归一化及优化器。
RSL-RL加载优化器后不会同步算法内部adaptive learning_rate标量，故命令显式使用checkpoint有效学习率
`0.00037290056934580207`，并由resume_audit.py逐rank验证两者一致，避免首次更新回到默认1e-3。
`--max_iterations 14900`表示新增14900次更新，加恢复编号100，总上限保持15000。
首轮新日志仍编号100，但保存在新目录；首个新model100应将优化器step2020推进到2040。
环境物理状态重新reset，checkpoint没有保存当时仿真现场，不能称逐轨迹连续。

TensorBoard标签为`Rotate-Bulb_Contact0.1_Tracking40_Resume100_12288_DDP_GPU4_7_20260927`，
本机入口 http://127.0.0.1:6006/#scalars ，转发依赖SSH连接存活。
`status.json`与`train/workers/rank*.json`记录状态/物理卡/PID；两rank的resume_verification.json核对来源及完整状态。
先用32×2环境完成三次PPO/同步检查，再从同一原始model100启动正式规模，未使用pilot更新后的模型。
验收证据及后续统计见[续训记录](../logs/diagnostics/bulb_contact01_resume_ddp_20260927/README.md)。

## 2026-09-27 13:01历史：Randomized接触/搬运奖励与原双卡训练

当时Randomized默认any_finger_contact=1、position_tracking=40、failure=−6，奖励profile版本1；
Custom/Assembly保留各自默认。回放旧模型时使用原run冻结配置，不凭相同观测维度推断奖励相同。
双卡run为 `/home/xinyu/training_runs/20260927_041844_rotate_bulb_contact_transport_12288_ddp`，
精确命令及环境在该run的launch_spec.json，**不要重复启动**。
用户已确认停止旧GPU4，临时GPU7单卡也已停止。4＋7每卡6144、共12288，从头共同训练；
GPU6曾同时继续，后于9月27日14:30按用户要求暂停用于性能测试；测试已结束，不自动恢复旧GPU6。
先通过32×2三轮原生PPO，`--expected_world_size 2`检查梯度均值/参数一致性，
不能用两个独立进程冒充共同训练。使用本次独立ddp_worker.py和supervise_ddp.py，
后者停止torchrun时给子进程45秒清理窗口，再按精确PID/命令/cwd核查清理所属workers。
本机NCCL2.26.2默认协议在512KiB通信处非法内存访问；仅新run设置`NCCL_PROTO=LL`，
与`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`一起保存在spec中。未修改共享软件包。
修改通信设置后须重新验证大数据通信及原生梯度同步，不能只凭启动日志判断通过。
正式规模12:51:53启动，13:00:31核验通过；结果在run/formal_verification.json。
TensorBoard6006标签为`Rotate-Bulb_Contact1_Tracking40_12288_DDP_GPU4_7_20260927`；
监督状态在status.json，真实两rank PID及物理卡映射在train/workers/rank*.json。
9月27日16:14本机已建立SSH转发，可打开 http://127.0.0.1:6006/#scalars ，
Runs搜索`Contact1_Tracking40_12288`；连接依赖SSH存活，断开后的完整重连命令及指标说明见
[本机TensorBoard查看步骤](../logs/diagnostics/bulb_training_tensorboard_20260927_1614/README.md)。
本次正式前1–5轮19.39秒/轮，不作为其他负载或后期训练的保证。
见[参数、资源安排与证据](rotate_bulb_contact_transport_2026-09-27.md)。

## 2026-09-24：装配课程扩大 PhysX 容量并恢复（GPU6）

当前run：`/home/xinyu/training_runs/20260924_100337_bulb_curriculum_buffers_4096_gpu6`。
按用户要求停止原GPU6插销，保留其 `model_13700.pt` 后迁入装配课程；GPU4旋出继续。
精确命令见run内 `launch_spec.json`，容量前后值见 `buffer_profile.json`，**不要重复启动**。
本次通过课程启动器的 `env.*` 覆盖向每个训练/独立评测进程传递容量；未改共享任务默认值或Python源码。

| 配置 | 原值 | 本次值 | 倍率 |
| --- | ---: | ---: | ---: |
| `gpu_collision_stack_size` | 2147483648 | 4026531840 | 1.875 |
| `gpu_max_rigid_contact_count` | 8388608 | 16777216 | 2 |
| `gpu_max_rigid_patch_count` | 655360 | 1310720 | 2 |
| `gpu_found_lost_pairs_capacity` | 33554432 | 67108864 | 2 |
| `gpu_found_lost_aggregate_pairs_capacity` | 33554432 | 67108864 | 2 |
| `gpu_total_aggregate_pairs_capacity` | 33554432 | 67108864 | 2 |
| `gpu_heap_capacity` | 67108864 | 134217728 | 2 |
| `gpu_temp_buffer_capacity` | 16777216 | 33554432 | 2 |

碰撞栈单位为字节，接触/配对字段为数量。碰撞栈原为2GiB，选3.75GiB保留余量；
4GiB=4294967296超过 `collision_stack_bytes` 的uint32上限4294967295。
[PhysX官方类型说明](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxGpuDynamicsMemoryConfig.html)
也将该字段定义为 `PxU32`，不能直接配置4GiB或更大。此前报错中的负建议值不能照抄。

4096环境、seed42、reward v2；使用已冻结的 `predecessor_stage0/model_3419.pt`，
`--start stage0 --checkpoint ...` 恢复策略、优化器、归一化及课程状态。
已有启动器不支持将固定Stage0检查点直接当作同阶段自由模式resume；因此先复核固定阶段，
再走原提名、三seed各100回合独立评测及相邻阶段transfer，不绕过前序学习或独立评测。
`--iterations-per-stage 9000` 是每次阶段启动的新增更新上限；恢复阶段日志序号接续原checkpoint。
任务仍会在预算用尽/评测不通过/致命物理错误时停止，监督器不自动重启。

监督状态 `status.json` 中 `training_pid` 为课程启动器；真实训练子PID需核对 `/proc`。
8192环境首试 `20260924_095053_bulb_curriculum_buffers_gpu6` 在首轮采样时CUDA OOM，已停止并留存；
没有把这一尝试算作启动通过。保留上述全部容量，减为4096环境；相对原8192运行，
每环境碰撞栈为3.75倍，其余7项容量为4倍。没有跨GPU或自动降低物理/成功标准。
TensorBoard6006标签 `Bulb_Assembly_Buffers375G_4096_GPU6_20260924/<phase>`。
核验进度见[实验记录](../EXPERIMENTS.md#asm-physx-resume-20260924-04)。

## 2026-09-22：装配 reward v2 有限短测

默认 `env.OLD_REWARD=false`；原奖励消融使用 `env.OLD_REWARD=true`，覆盖在任务工厂生效。
仅装配任务读取此开关，旋出任务不变。独立reward记录保存版本/开关/哈希；原物理契约逻辑保留，
True/False可复用同一兼容初态库，不绕过旧库/模型兼容检查。
在仓库根目录与现有brainco/Isaac环境中，以下命令运行1000环境、180控制步，然后关闭仿真：

```bash
PYTHONPATH="$PWD/source/BrainCo_DexHand" PYTHONDONTWRITEBYTECODE=1 \
  /home/liuxinyu/miniconda3/envs/brainco/bin/python \
  scripts/diagnostics/validate_bulb_reward_v2.py \
  --out REPLACE_NEW_PROBE_DIR --num_envs 1000 --steps 180 \
  --task stage0 --headless --device cuda:0 --compact-buffers
```

输出目录必须不存在；写 `report.json`、原生日志、TensorBoard、逐回合JSONL和reward契约。
使用小幅固定手部动作与零动作子集，检查总/分项reward、观测有限、原success标志一致以及必需标签回读；
不加载策略、不更新PPO、不导出成功状态库。没有实际前序成功库时不伪造Stage2/3起点。
Full检查使用 `--task full --steps 620`，保持原20秒时限并覆盖回合reset；
脚本旧奖励模式使用 `--old-reward`，自由Stage0使用 `--free`（两者需分别解释结果）。

本机8GB显存下，默认分配池的首轮1000环境曾发生PhysX OOM，已拒绝其数值有限的表面结果。
`--compact-buffers` 仅把预分配aggregate-pair容量从2^25限制为2^22并记入report，
不改生产配置、求解器、碰撞、传感器或reset。检测到PhysX/CUDA错误或容量溢出立即判失败。
本次实际结果与范围见[实施记录](rotate_bulb_reward_implementation.md)；短测通过不代表学会完整装配。

2026-09-23 GPU5的1024环境/每阶段2000轮版本已用尽Stage0预算停止，历史部署见
[GPU5记录](rotate_bulb_reward_gpu5_2026-09-23.md)。用户随后授权以8192环境重启，
并接受`--iterations-per-stage 9000`；仍按原门槛和三seed独立评测提前晋级，未达标到上限会停止。
新run为`/home/xinyu/training_runs/20260923_104131_bulb_reward_v2_8192_9000_gpu1`，
GPU1/seed42/OLD_REWARD=False，从零Stage0。GPU5已被其他用户占用，本次未干预该作业。
监督737978；停止前核对该PID的`start_ticks=39963355`与run路径，再发送SIGINT。
日志为run根目录`train.console.log`；TensorBoard 6006标签为
`Bulb_Assembly_RewardV2_8192_9000_GPU1_20260923/stage0`。
确切命令、源码哈希和最新核验见[实验记录](../EXPERIMENTS.md#asm-reward-v2-8192-20260923-03)，不要重复启动。

## 2026-09-22：模具光学面包板固定版重建（CPU）

在仓库根目录，使用已有CadQuery/OCP、NumPy、trimesh、Shapely的系统Python运行：

```bash
python 1_stl/breadboard_mount_256/build.py /tmp/new_mold_mount
```

输出目录必须不存在。输入依赖 `1_stl/root_open_256` 原生CAD及 `trimmed_256` 的局部参考曲面，
只写新输出目录；生成两份CAD/STL、合模STEP、螺丝参考包络和验证记录，不启动仿真。
本次已在独立临时目录实际重建，两STL和两BREP与交付件SHA256一致，几何检查通过。
预览图与说明不由该命令生成；几何参数、已保留的严格网格检查失败及实打边界见[交付说明](../1_stl/breadboard_mount_256/README.md)。

## 2026-09-22：独立模具几何转换（CPU）

系统 `python3` 已有 CadQuery 2.7、NumPy、trimesh；不使用 brainco/Isaac 环境。
在仓库根目录运行 `python3 assets/mold_split/build_mold_urdf.py --output /tmp/mold_split_new`。
输出目录必须不存在；命令写入新资产，不修改源 STEP，不启动仿真。默认150mm行程为建模参数。
本次默认输出实际生成于 `assets/mold_split/generated/`；README说明已验证范围与网格未闭合限制。
可直接在浏览器打开 `assets/mold_split/generated/preview.html` 查看和调整开模距离。
URDF解析实际通过：

```bash
LD_LIBRARY_PATH=/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  /opt/ros/humble/bin/check_urdf assets/mold_split/generated/mold.urdf
```

该解析检查不证明碰撞/动力学有效；质量未知，下模仍有7条网格开边。
详情和复现参数见 [资产说明](../assets/mold_split/README.md)。

## 2026-09-20：检查 Dexonomy 灯泡初态

在仓库根目录、已配置 Isaac Lab 的 `brainco` 环境运行：

```bash
python assets/rotate_bulb_custom/validate_grasp_scene.py --headless --num_envs 2 --device cuda:0
```

这是实际仿真状态检查，会使用 GPU 并写
`assets/rotate_bulb_custom/grasp_reference/dexonomy_20260920/runtime_validation.json`。
默认检查启动、三次完整 reset、扰动后的部分 reset；精简观测且不运行物理 rollout。
参考文件默认使用同目录 `simulation_state.json`；完整观测可加 `--full-observations`。
设置和来源见 [模板说明](../assets/rotate_bulb_custom/grasp_reference/dexonomy_20260920/README.md)。

## 1. 命令证据等级与共同前置条件

### 2026-09-25：Randomized回退/停滞惩罚验证（不重启训练）

参数在env.unscrew_progress_penalty（version=1）；两项日志unscrew_regression/unscrew_stall。
动作/观测契约保持；恢复旧策略时须另记奖励变更，不能比较不同奖励版本的总回报。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider tests/test_rotate_bulb_progress_penalty.py tests/test_rotate_bulb_randomized.py tests/test_rotate_bulb_segmented_guidance.py tests/test_rotate_bulb_custom_failure.py tests/test_rotate_bulb_soft_grasp_rewards.py
python scripts/diagnostics/validate_bulb_progress_penalty.py --headless --num_envs 2 --device cuda:0 --out REPLACE_NEW_PROBE_DIR
```

第一条仅CPU；第二条实际启动GPU/Isaac，需可用资源和新输出目录，包含明确标记的奖励状态注入及200步真实动力学。
本次本地/服务器CPU各134通过，本地原生212检查通过。服务器GPU当时均已有作业，未在其上额外启动仿真。
主项目同步不会热加载原训练快照，本次没有重启。见[参数与结果](rotate_bulb_progress_penalty_2026-09-25.md)。


### 2026-09-24：Randomized 手心上方奖励与高目标验证

当前Randomized任务使用v2契约及 `workspace_bank_high_z_v2.json`；相对支架目标Z为+28–42cm，
包含有限预算的手心上方奖励。旧v1库和旧run快照保留；观测同为687维不代表可以直接恢复。
本次实际执行原生检查：

```bash
python scripts/diagnostics/validate_bulb_palm_high_z.py --headless --num_envs 8 --device cuda:0 --out REPLACE_NEW_PROBE_DIR
```

需在已准备的brainco/Isaac环境和项目根目录运行，先授权对应GPU；会启动真实仿真并写新目录。
检查16次reset、目标Z抽样、固定臂/零手、实际palm刚体、奖励预算/释放门控、687维契约和40步物理。
本次96项检查通过，报告 `dynamic_success_validated=false`；启动退出码不能代替检查 `report.json.passed`。

离线目标库生成器新增 `--goal-z-min` / `--goal-z-max`，默认仍是历史0.18/0.30m。
使用已有USD Python库与NumPy/SciPy，不启动Isaac应用；拒绝覆盖已有输出。本次生成参数：

```bash
python scripts/diagnostics/build_bulb_randomized_workspace.py --out REPLACE_NEW_BANK_JSON --grid 21 --goals-per-support 8 --seed 240924 --goal-z-min 0.28 --goal-z-max 0.42
```

本次xinyu21新run为 `/home/xinyu/training_runs/20260924_081649_rotate_bulb_palm_high_z_gpu4`。
GPU4/6144env/15000轮/seed42，从头训练；实际参数以run内 `launch_spec.json` 为准，**不要重复启动**。
冻结源码及RSL runtime分别在 `workspace/`、`runtime/rsl_rl/`；状态/停旧审计/短测在
`status.json`、`old_stop_audit.json`、`native_probe/`、`pilot/`。
TensorBoard6006标签 `Rotate-Bulb_PalmAbove_HighZ_GPU4_20260924`。
旧run已于本次切换时停止且保存原检查点。验证边界见[实验记录](../EXPERIMENTS.md#bulb-palm-high-z-20260924-03)。

### 2026-09-21：Randomized 旋出验证与训练（历史，9月24日已替换）

新任务 ID 为 `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0`，含固定臂/零手与有限可行采样库。
在准备好的 brainco 环境、仓库根目录运行原生检查（会占用GPU并写新目录）：

```bash
python scripts/diagnostics/validate_bulb_randomized.py --headless --num_envs 2 --device cuda:0 --out REPLACE_NEW_PROBE_DIR
```

该命令本机/服务器均实际完成121项检查与40步物理运行，不是完整任务成功率评测。
CPU测试为 `tests/test_rotate_bulb_randomized.py`；库生成器为 `scripts/diagnostics/build_bulb_randomized_workspace.py`，
依赖已有USD CPU库、NumPy/SciPy；拒绝覆盖已有输出，重建后采样库哈希改变会阻止旧契约直接恢复。

正式训练已由用户授权在xinyu21启动；不要重复启动。run为
`/home/xinyu/training_runs/20260921_050053_rotate_bulb_randomized_6144_gpu4`，
GPU4/6144环境/15000轮/seed42/from scratch，确切命令与环境白名单在run内 `launch_spec.json`，
冻结源码在 `workspace/`，监督状态 `status.json`，原生与3轮PPO证据在 `native_probe/`、`pilot/`。
6006标签为 `Rotate-Bulb_Randomized_ZeroHand_GPU4_20260921`。
运行与验证边界见[执行记录](rotate_bulb_randomized_training_2026-09-21.md)。

| 等级 | 含义 |
| --- | --- |
| A | 本次实际执行；仅对列出的范围和结果负责 |
| B | 有可对应历史记录；适用当时配置/源码，不能推定当前也通过 |
| C | 仅核对当前源码/脚本参数；本次未执行 |
| D | 待确认或尚未实现；明确缺少的能力 |

除特别说明，工作目录均为 **Git 仓库根目录**，不是 `scripts/rsl_rl/`。
下文 `python` 必须是相应任务的已准备解释器；本次实际使用 `/usr/bin/python` 3.12.3，
其 metadata 仅找到 pytest 7.4.4，未找到 torch、isaaclab、isaacsim、isaaclab-rl、rsl-rl-lib。
这不表示机器上没有其他环境，本次未遍历个人环境目录。

所有 `REPLACE_*` 都是**必须替换的占位符**；不得原样运行。
涉及仿真的命令需要已授权 GPU/显存、可用 Isaac 环境、可解析资产和独立输出目录；
先记录具体实验预算与停止条件，不能从示例中的环境数/迭代数推导资源授权。
默认值、GPU 编号和历史远端路径不能直接移植；不启动重复作业，不覆盖旧 run。

## 2. 环境准备与轻量检查

### A：仓库与当前解释器

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
git check-ignore -v AGENTS.md
python - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError
print(sys.executable, sys.version.split()[0])
for name in ("pytest", "torch", "isaaclab", "isaacsim", "isaaclab-rl", "rsl-rl-lib"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "NOT_FOUND_IN_CURRENT_INTERPRETER")
PY
```

输入为 Git 元数据和包 metadata；只输出终端文字，不导入 torch/Isaac、不初始化 GPU。
本次确认 `main` / `4707074…`，已有 39 个已跟踪修改及 16 条未跟踪文件/目录状态项。
具体清单与保留检查见 [DOC-AUDIT-20260920-01](../EXPERIMENTS.md#doc-audit-20260920-01)。

### A：不启动仿真的最小检查

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider tests/test_play_camera.py
bash -n scripts/rsl_rl/launch_d_peg_dual_gpu.sh
bash -n scripts/rsl_rl/launch_rotate_bulb_custom_dual_gpu.sh
```

相机测试只用标准库与 pytest，验证坐标和参数拒绝逻辑；本次 **15 passed in 0.02s**。
禁用自动插件、字节码和 pytest 缓存；无模型输入、无 GPU、无仿真输出。
两条 `bash -n` 均返回 0，仅证明 shell 语法有效，不能证明解释器路径或分布式训练可用。
另对 8 个运行脚本及 4 个注册文件执行了 `ast.parse`（12/12 通过），未导入项目包。
完整范围见文档审计。未运行全套测试。

### C：开发环境安装入口（本次未执行）

已有 Isaac Lab 环境中可用包安装入口：

```bash
python -m pip install -e source/BrainCo_DexHand
```

仅安装本项目扩展，可能解析/下载 Python 依赖并修改所选环境；不是 Isaac Sim 全套安装命令。
依据 `source/BrainCo_DexHand/setup.py`：Python ≥3.10，声明 omegaconf、psutil、tensorboardX、termcolor；
`extension.toml` 依赖 isaaclab / assets / mimic / rl / tasks。
训练脚本运行时要求 `rsl-rl-lib>=3.0.1`，还使用本地 RSL-RL fork 扩展字段。
setup 分类中的 Isaac Sim 4.5/5.0 不是经验证版本锁定；**完整兼容依赖锁定为 D**。
核实方法：在获准使用的训练解释器中记录包版本和实际 import 路径，保存到具体 run，
先静态/CPU 测试，再进行有预算的仿真检查；不要为了文档核实升级环境。

## 3. C：仿真快速检查（本次未执行）

无检查点的 Assembly 环境探针：

```bash
python scripts/rsl_rl/probe_bulb_assembly.py \
  --out REPLACE_NEW_PROBE_DIR --num_envs 2 --steps 40 --seed 42 --headless
```

默认任务 `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0`；输入当前资产/配置，
输出 `report.json`、`steps.jsonl`，占用 GPU 并启动 Isaac。新输出目录必须单独选择。
应看到有限状态、reset/部分 reset 与物理检查结果；探针通过不证明策略学会装配。
**B 历史证据**：[ASM-PROBE-20260919-01](../EXPERIMENTS.md#asm-probe-20260919-01) 为 2 环境/40 步通过，
但其中 `assembly_env_cfg.py` 哈希与当前不一致，不能给上面的当前命令标 A 或直接沿用旧结论。

其他任务的快速接线可用下面的有限训练；它会更新模型，不能称为纯 reset/step 只读探针。

## 4. C：小规模与正式训练（本次未执行）

### 小规模状态观测 Repose

```bash
python scripts/rsl_rl/train.py \
  --task BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0 \
  --num_envs 4 --max_iterations 3 --seed 7 --headless --logger tensorboard \
  --log_dir REPLACE_NEW_SMOKE_DIR \
  env.tactile_policy_enabled=false
```

输入当前 revision 1 配置与资产；状态观测 152 维，无编码器要求；仍启动物理仿真并使用 GPU。
输出新 run 下 `params/`、检查点、日志及 `tactile_policy_contract.json`。
预期 3 次 PPO 更新后正常退出，张量有限、保存文件可对应；不说明策略成功或支持触觉部署。
开启触觉时需明确选择正确编码包，沿用 [Repose 接入说明](tactile_representation/repose_policy_integration.md)
中的 `env.tactile_policy_checkpoint=...`；该默认导出目录在本次本地调查中未见。

### 正式训练示例：Flexiv D-Peg Custom

```bash
python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0 \
  --num_envs REPLACE_ENV_COUNT --max_iterations REPLACE_ITERATIONS \
  --seed REPLACE_SEED --headless --logger tensorboard \
  --log_dir REPLACE_NEW_TRAIN_DIR \
  env.tactile_policy_enabled=false
```

环境数、迭代数和 seed 必须按已记录的实验方案替换；当前源码默认 32 环境/15000 迭代不是已获授权预算。
同一 `train.py` 支持项目地图列出的注册任务，换任务必须同时核对 PPO、动作/观测与资产，
不能只替换名字后恢复旧模型。训练输出与短测相同；D-Peg 增加完成回合/PPO 诊断。
应核对实际生效 YAML、契约和独立评测，不能把 reward 或 `DPeg/success` 的瞬时比例当作最终验收。

**双卡脚本为 C（仅 shell 语法为 A）**：
`launch_d_peg_dual_gpu.sh` 启动标准 Revo3 D-Peg、默认 2 卡×512 环境；不是 Flexiv Custom。
`launch_rotate_bulb_custom_dual_gpu.sh` 为 2 卡×4096，带旧奖励覆盖和 2 GiB 碰撞栈，
并硬编码历史解释器/fork 默认路径。两者会直接 `exec` 分布式训练，不能作为环境检查运行。
实际使用前核对 `D_PEG_PYTHON` 或 `BULB_PYTHON/BULB_RSL_RL_PATH`、授权 GPU、覆盖参数和总预算；
本次不把这些机器专用脚本列为通用推荐启动方式。

### Custom 旋出显存与并行环境调整

当前三圈分段引导版本见 [设计与验证](rotate_bulb_segmented_guidance_2026-09-20.md)。
旋出必须使用 `env.bulb_pure_collision=false`，初始圈数保持 `(3.0, 3.0)`。
观测已由669增至673维，新契约含 `rotation_guidance.version=1`；从头启动，不能直接恢复旧669维模型。
独立装配仍为669维，沿用其自己的启动配置和检查点。
新run应核对保存配置中 `unscrew_turns.weight=1.5`、`unscrew_progress.weight=0.5`、
两项函数均为 `RotateBulbGuidedRotation`（分别 `milestones=false/true`），
`segment_turns=0.25`、`force_scale=0.5`、`regrasp_grace_s=0.2`、`max_turn_rate=1.0`，
以及 `arm_motion.weight=-0.1`、`position_deadband=0.05`、`velocity_deadband=0.15`。
这些数值是奖励参数，不是放宽三圈物理释放或最终成功判据。

2026-09-20 的 GPU 4 扩容记录见
[BULB-MEM-RESTART-20260920-01](../EXPERIMENTS.md#bulb-mem-restart-20260920-01)。
完整命令保存在该记录的 `launch_spec.json`，其中包括原任务必需的传感器与物理覆盖；
不能只复制下面四个值就替代完整启动配置。

```text
--num_envs 6144
env.sim.physx.gpu_collision_stack_size=3221225472
env.sim.physx.gpu_found_lost_pairs_capacity=33554432
env.sim.physx.gpu_found_lost_aggregate_pairs_capacity=33554432
env.sim.physx.gpu_total_aggregate_pairs_capacity=33554432
```

配对容量的单位是条目数，碰撞栈单位是字节。此方案缩减配对缓冲的过量预留，同时随环境数提高碰撞栈。
仅修改本次 run 的启动覆盖，未改变通用默认、接触模型、自碰撞或成功判据。
新规模须检查实际 `params/env.yaml`、PPO 更新后的显存、接触容量警告及完整回合重置。
容量不足可能丢弃接触/约束，因此出现 overflow、miss interactions、CUDA/PhysX 内存错误应立即停止并重新配置，
不能继续把损坏的仿真用于训练；详见 [PhysX GPU Memory](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html#gpu-memory)。
早期启动无警告不能保证学习后所有接触状态都有足够容量；保留监督器检测和容量余量。

## 5. C：恢复训练（本次未执行）

以下来源目录须位于 `logs/rsl_rl/dexsuite_flexiv_insert_d_peg_custom_v4/`，
模型文件名为该目录内的明确文件；不要用“最新匹配”代替已核对的父检查点。

```bash
python scripts/rsl_rl/train.py \
  --task BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0 \
  --resume --load_run REPLACE_SOURCE_RUN_NAME --checkpoint REPLACE_MODEL_FILE.pt \
  --num_envs REPLACE_ENV_COUNT --max_iterations REPLACE_ADDITIONAL_ITERATIONS \
  --seed REPLACE_SEED --headless --logger tensorboard \
  --log_dir REPLACE_NEW_RESUME_DIR \
  env.tactile_policy_enabled=false
```

来源需要完整 `params/`、模型和契约，当前动作/物理/奖励语义须兼容；只传 `--resume` 不会
自动恢复源 run 的全部配置。示例仅说明恢复参数接口，复现实验须补齐来源的实际覆盖项。
`train.py` 将 `max_iterations` 传入 `runner.learn(num_learning_iterations=...)`；
历史 fork 中表示追加更新次数（[PEG-RESUME-20260919-01](../EXPERIMENTS.md#peg-resume-20260919-01)：800 起追加 14200，
计划末标号 14999）。换 Runner 版本后仍需核对实现，不能不加区分地写成总目标迭代数。
`--load_run` 是实验根下来源选择，`--log_dir` 是新写入目的地，两者不同。
历史清单使用的 `resume_source_20260919_170828` 软链接是当时服务器准备的，本地未确认存在。
续训验收至少核对父模型哈希、迭代、策略/归一化/优化器状态及实际更新；恢复启动不等于独立成功。

## 6. 独立评测：C 命令，B 历史记录另列

### D-Peg 深度保持协议

```bash
python scripts/rsl_rl/evaluate_d_peg.py \
  --task BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0 \
  --checkpoint REPLACE_ABSOLUTE_CHECKPOINT.pt --output REPLACE_NEW_EVAL_DIR \
  --num_envs 16 --seeds 20260919 --headless --device cuda:0
```

输入检查点及同目录 `params/env.yaml`、`params/agent.yaml`、观测契约；评测器读取保存配置，
调整环境数、seed、device、日志目录。`cuda:0` 是所选可见设备中的逻辑编号，不是资源预约。
每个 seed 对每个并行槽只取首个完整回合，此例恰为 16 回合；不支持 `--episodes-per-seed`。
输出 `report.json`、`episodes_seed_20260919.jsonl`、runtime/simulation 日志；目录须空或不存在。
报告应为 `complete=true`、样本数完整、权重前后不变，独立深度保持与任务标志一致。
其“独立”是独立保持计时，几何仍调用任务 `_geometry`，不构成第二套几何审计。
判据不要求 containment/entry history，不能拿来替代不同协议的完整插入验收。
该 seed 已用于开发历史评测；最终验收应使用冻结的新种子。
**B**：[PEG-EVAL-20260919-01](../EXPERIMENTS.md#peg-eval-20260919-01)，1/16；本次只重数记录，没有加载模型重跑。

### 旋出与桌面装配

```bash
python scripts/rsl_rl/evaluate_bulb_custom.py \
  --checkpoint REPLACE_ABSOLUTE_CHECKPOINT.pt --out REPLACE_NEW_EVAL_DIR \
  --physics-mode guided --num_envs 32 --episodes-per-seed 20 --seeds 4040 --headless

python scripts/rsl_rl/evaluate_bulb_assembly.py \
  --checkpoint REPLACE_ABSOLUTE_CHECKPOINT.pt --out REPLACE_NEW_EVAL_DIR \
  --num_envs 32 --episodes-per-seed REPLACE_EPISODES --seeds REPLACE_SEED --headless
```

两命令分别默认 Custom 旋出 / Assembly 注册 ID，不混用模型。使用新输出目录，保留模型、
保存配置及动作/触觉契约；确认模型与当前 `BulbActorCritic` 版本兼容。
会启动 GPU 仿真、按 seed 收集完整回合并保存报告/轨迹。旋出示例是开发评测，seed 4040 已用过；
`--physics-mode pure` 为另一个诊断口径，当前旋出评测器只将 guided 视为验收物理模式。
正式 seeds/样本数/门槛以脚本中的冻结协议和已获确认的实验方案为准。
Assembly 的 `--episode-length-s` 只供短测，不得缩短正式验收回合。
**B** 旋出旧版 v018 评测为 0/20，见 [BULB-EVAL-20260919-01](../EXPERIMENTS.md#bulb-eval-20260919-01)；
不证明当前 v3 的成功率。Assembly 当前策略完整任务成功率待验证。

## 7. C：回放、相机和导出（本次未执行）

```bash
python scripts/rsl_rl/play.py \
  --task BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0 \
  --checkpoint REPLACE_ABSOLUTE_CHECKPOINT.pt --num_envs 1 --headless \
  --video --video_length 300 --video_dir REPLACE_NEW_VIDEO_DIR \
  env.tactile_policy_enabled=false
```

输入模型、契约及与历史匹配的当前配置/覆盖；`play.py` 不自动完整恢复历史 YAML。
输出视频和预览；还会在检查点所在 run 的 `exported/` 写 `policy.pt`、`policy.onnx`，
所以回放不是只读操作，导出文件可能被更新。300 步结束只保证有限回放，不保证完整样本评测。
RTX 相机额外占显存；`--headless` 不代表不占 GPU。视频不能单独证明任务成功或“最佳模型”。
相机预览无需 checkpoint，但仍启动仿真并输出 PNG：

```bash
python scripts/rsl_rl/play.py \
  --task BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0 \
  --num_envs 1 --headless --camera-preview --screenshot REPLACE_NEW_PREVIEW.png
```

相机参数及坐标详见 [play_video.md](play_video.md)。其中历史小显存 PhysX 覆盖只适用对应短测，
不能搬到正式训练。旧灯泡 `--bulb-legacy-diagnostic` 是兼容诊断，不能据此声称恢复原动力学，
也不能假设能越过 v2/v3 契约不兼容。

## 8. C / D：触觉数据、对齐、蒸馏与部署

触觉采集/缓存/两种恢复网络/对齐/latent 可视化的完整命令沿用 [README](../README.md)，
避免第二处维护长参数表；本次均为 **C**，未执行、未验证数据集或权重完整性。
根目录执行；采集使用 Isaac/GPU，网络训练默认 CUDA，缓存会产生大量磁盘输出。
输入配对 key、模态单位与导出契约见 [Repose 接入说明](tactile_representation/repose_policy_integration.md)。
仅导出接口的例子（需真实原始 checkpoint）：

```bash
python scripts/tactile_representation/export_tactile_encoders.py \
  --sim-checkpoint runs/revo3_cross_modal_restoration_v1/best.pt \
  --unaligned-sim --output-dir REPLACE_NEW_ENCODER_DIR
```

生成 `sim_policy_encoder_unaligned.pt`，需要 PyTorch，会读取权重并写文件；本次未执行。
未对齐 `normalized_h` 和对齐后 `normalized_z` 的 PPO 不能混用。
真实配对数据及对齐效果、当前 Custom 蒸馏可运行性、端到端真机闭环均为 **D：待确认**；
未找到通用 Repose 独立完整回合评测入口，标为 **D：尚未实现/未发现**。
建议补齐冻结 checkpoint、多 seed、失败全计数及原始状态判据的 Repose 评测能力；
现有张量部署示例和 JIT/ONNX 导出不等于真机验收。

### 8.1 TactSpace RGB/Depth/Marker 适配版（2026-10-07）

仍使用 `scripts/tactile_representation/train_tri_modal_cross_autoencoder.py`。
无新增参数时仍为 `tri_modal`，旧 denoising、四种验证模式、优化器和调度保持。
`--model-type tactspace` 切换为同一步联合重建与模态 InfoNCE；无需先训练/冻结恢复网络。
这是本项目适配版，未声明官方完整复现：复用本地 CNN 与 Marker 点序列 Transformer，
各 domain × modality 分支独立；共享 latent MLP 后接四通道 FiLM 图像头及固定查询 Marker 头。
继承配置中的 `cross_layers/cross_summary_tokens` 不参与新结构，显式指定这些选项会报错。
解码路径没有目标观测、RGB reference 或动态空间 skip。

新增必要选项：`--model-type {tri_modal,tactspace}`（默认 tri_modal）、
`--input-modalities rgb depth marker`（默认三种，选择来源编码分支，仍重建所有有效目标）、
`--reconstruction-weight 1`、`--alignment-weight 1`、`--temperature 0.1`。
三种目标权重复用 `--rgb-loss-weight / --depth-loss-weight / --marker-loss-weight`。
两项联合权重可为零；若均无有效梯度则不执行优化器更新。
新模型不使用旧模型的降质和专用前景/方向损失；验证按联合损失选择 checkpoint。

传感器归一化复用原实现；导出 latent 契约为：**各模态 L2 → 有效模态均值 → L2**。
单模态得到实际训练的 latent；每帧独立编码、外部历史 oldest-to-newest，五指顺序保持。
冻结接口返回 `[B,5,d_model]`；可只传 `rgb`、`depth_m` 或 `marker`，
可用 `modality_valid[B,5,3]` 和 `marker_valid[B,5,K]` 标记无效观测。
任一指没有有效输入时报错。编码维度、域/模态、归一化、聚合和历史契约写入 checkpoint/bundle；
没有 projection 训练或随机 CLIP head。真实分支必须有成功优化器更新的记录才允许导出。

根目录运行，以下 `python` 可替换为已验证的 `/home/liuxinyu/miniconda3/envs/brainco/bin/python`。
数据路径待替换；输出必须使用新的目录。这些是运行模板，未对真实数据执行长期训练。

```bash
# 原模型：不加模型参数
python scripts/tactile_representation/train_tri_modal_cross_autoencoder.py \
  --dataset REPLACE_SIM_DATASET --output REPLACE_NEW_TRI_MODAL_RUN

# 新模型：联合训练
python scripts/tactile_representation/train_tri_modal_cross_autoencoder.py \
  --dataset REPLACE_SIM_DATASET --output REPLACE_NEW_TACTSPACE_RUN \
  --model-type tactspace --input-modalities rgb depth marker \
  --reconstruction-weight 1 --alignment-weight 1 --temperature 0.1

# 自动从 checkpoint 恢复模型结构；复用已有 sim-only 导出开关
python scripts/tactile_representation/export_tactile_encoders.py \
  --sim-checkpoint REPLACE_NEW_TACTSPACE_RUN/best.pt \
  --unaligned-sim --output-dir REPLACE_NEW_EXPORT_DIR
```

输出 `sim_policy_encoder_tactspace.pt`；这里 `--unaligned-sim` 表示没有跨域 projection，
不表示禁用了同域模态对齐。`--sim-model-type tactspace` 可显式核对，但不必填写；冲突类型报错。
恢复训练 `--resume CHECKPOINT` 会恢复结构与归一化，显式冲突参数报错。
旧 checkpoint 仅按已知 state/config 签名兼容识别，未知格式拒绝。

当前只支持原同步仿真采集格式（旧 collector 元数据或显式 sim 域）。
同域正样本来自同一采集帧，episode/frame ID 用于重复采样的多正样本屏蔽。
无 ID、有效样本不足或仅一种物理刺激时跳过 InfoNCE，并记录跳过数；不以数组下标跨域配对。
`training_ready=false` 拒绝；若 manifest 声明旧 loader 无法保留的
`pair_valid/training_ready/modality_valid/rgb_valid/depth_valid/pair_id` 列，也明确拒绝，不丢弃标记。
模型/损失张量接口支持有效性 mask；本轮未扩展采集或数据格式。
`train_latent_alignment.py` 遇 TactSpace 明确拒绝旧冻结 head 路径和未经认证的跨域索引连接。

短验证（A，CPU，仅合成接线）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/tactile_representation/check_tactspace.py
```

12 项检查含原入口 2 步、新模型 2 步及恢复 1 步、Marker 单模态 2 步、
梯度/损失均衡/无效值/无泄漏/多正样本/空更新/未训练分支拒绝/导出冻结一致性。
脚本使用临时目录并清理产物；不会覆盖现有模型或启动 Isaac/PPO。
本次日志 `logs/diagnostics/tactspace_20261007/cpu_checks.log`。
现有两个 `runs/revo3_*_v1/best.pt` 另经只读严格加载及 `[1,256]` 有限前向核验。
CUDA/AMP、真实数据训练和跨域迁移效果未验证；几步合成更新不说明模型有效。

## 9. A：Rotate-Bulb 速度实验（2026-09-20/21）

结构、固定条件和自然/同步计时定义分别见
[物理分析](rotate_bulb_physics_analysis.md)、[速度基线](rotate_bulb_speed_baseline.md)。
每个候选在独立工作区/进程执行；生产源码来自冻结快照，资产引用也逐项核验哈希。
`scripts/diagnostics/run_rotate_bulb_benchmarks.py` 的 experiment JSON 指定环境数和单因素 Hydra 覆盖。
该脚本会启动真实 GPU 仿真、短 PPO 更新或固定策略评测，并生成新的目录；不是轻量检查。
完整原始命令记录在各 case 的 `launch.json`，通用复现形式为：

```bash
/home/xinyu/envs/brainco/bin/python scripts/diagnostics/run_rotate_bulb_benchmarks.py \
  benchmarks/rotate_bulb/solver_experiments.json \
  /home/xinyu/training_runs/20260920_134756_rotate_bulb_segments90_armsteady_6144_gpu4 \
  /home/xinyu/training_runs/REPLACE_NEW_SOLVER_DIRECTORY
```

执行前必须有已授权的空闲 GPU 4；原正式训练恢复后不能直接同时运行此命令。
baseline/solver/contact 使用原 snapshot；触觉候选须通过 `--source-tar` 指向对应已保存快照。
launcher 与 `profile_bulb_sampling.py`、`evaluate_rotate_bulb_physics.py`、
`evaluate_rotate_bulb_release.py` 放在同一目录。

- throughput：8 轮，0–1 预热、2–4 同步分项计时、5–7 自然吞吐。
- policy quality：固定 checkpoint，64 个首个完整 20 s episode，加独立零动作 cohort；没有 PPO 更新。
- release probe：4 个注入状态，用于检查角度/行程门槛、释放、下落接触和局部 reset；不得计入 RL 成功率。
- launcher 检查 artifact 完整性、非有限值、PhysX 错误、溢出与超时；Hydra 异常可能仍返回进程码 0，
  因此必须同时检查 `results.json` 和 `evaluation.json.completed` / 8 轮 profile。
- `summarize_rotate_bulb_speed.py ROOT OUTPUT.json` 生成自然 FPS、分项时间、显存/GPU 利用率及评测结果；
  PhysX 错误记录会标记 `performance_valid=false`；其耗时只能用于故障诊断，不得列为有效 benchmark。
  该统计工具也保留不完整的 evaluation，不能跳过上面的完成性检查。
- 触觉 `fresh` 为新鲜位姿 + 完整 ContactSensor 参考，`pose_only` 使用相同位姿、省去内部无消费者的力归约。
  `pressure_pose_version=2` 和各 pad 采样周期写入契约；普通 train/play 拒绝将旧缓存语义模型当作兼容模型。
  诊断 JSON 的 `allow_pressure_pose_transfer=true` 仅用于不更新优化器的显式迁移检查。

实际采用配置为[recommended_profile.json](../benchmarks/rotate_bulb/recommended_profile.json)中的完整overrides：
TGS24/4、240Hz物理、30Hz控制/pressure、pose_only、6144env。所有min/max均显式设置。
新run为`20260920_183647_rotate_bulb_physx24_4_pressure30_6144_gpu4`，完整命令见该run的`launch_spec.json`；
配置与checkpoint输入为v2，新run从头训练，旧模型保留。回放/续训须携带相同overrides并通过契约检查。
[重启验证](../benchmarks/rotate_bulb/restart_verification.json)已确认实际PPO迭代0–4；
现在GPU4有正式任务，不应直接执行上面的benchmark示例与其争用。

此速度任务已重建 GitNexus 索引并在每次提交前检查完整 changes 图；下节为早先文档审计的历史限制。

### 9.1：Randomized 已训练策略性能诊断（2026-09-27）

`scripts/diagnostics/profile_bulb_sampling.py` 现支持 Randomized 的687维观测，以及
`BULB_PROFILE_CHECKPOINT=/绝对路径/model_N.pt`：经过触觉/动作契约核验后加载完整runner，
包括优化器和归一化状态。原673维任务入口仍保留。该环境变量只用于独立诊断包装。
计时使用相对更新编号：0–1预热、2–4同步计时、5–7自然吞吐；恢复model200时实际日志为200–207。
8次更新不是完整回合成功率评测。诊断有额外同步开销，分项总时长不能替代自然吞吐。

`BULB_PROFILE_POINT_VISUALIZE=0` 只关闭点云标记显示，保留64×3观测和实时位姿变换；
在Hydra转换后、场景构造前设置。不得改为`+env.observations.perception.object_point_cloud.params.visualize=false`，
原参数字典缺少该键时Isaac严格`from_dict`仍会拒绝。运行前应核实GPU归属并使用私有checkpoint和源码副本。

本次精确命令见[诊断证据](../logs/diagnostics/bulb_runtime_performance_20260927/)内各case的`launch_spec.json`；
`verify_cases.py ROOT`仅核验此批固定model200、8更新实验的契约、有限模型、优化器步数和计时。
旧`summarize_rotate_bulb_speed.py`假设从0开始及旧launcher字段，不能直接套用本次恢复训练结果。
GPU6旧作业已被用户暂停，性能测试结束不能自动恢复；GPU4＋7正式训练保持运行。
结果与失败尝试见[本次报告](bulb_runtime_optimization_2026-09-27.md)。

### 9.2：固定动作物理性能与质量对照（2026-09-27）

新增三个隔离入口：`benchmark_bulb_physics.py`（CUPTI/cProfile与场景审计）、
`replay_bulb_physics_factors.py`（初态/动作一致性、单因素与全回合质量）、
`run_bulb_physics_case.py`（按保存spec在空闲GPU的新目录复现）。
本次冻结接触1/跟踪40、model1300的Randomized/Custom物理版本；不能用另项新接触0.1续训替代。

```bash
R=/home/xinyu/training_runs/20260927_1345_physics_perf_gpu6
/home/xinyu/envs/brainco/bin/python "$R/run_bulb_physics_case.py" \
  "$R/replay1024/launch_spec.json" "$R/reproduce_replay_NEW" --gpu 6 --timeout 1200
```

在xinyu21执行，需现有冻结workspace/runtime/资产/checkpoint，以及当前授权的空闲GPU6。
不覆盖旧目录、不停止其他进程；此命令真实运行仿真，保留固定动作，不更新PPO。
6144基准、64回合质量、初态检查及其副作用见[完整复现说明](rotate_bulb_physics_performance_2026-09-27.md#7-复现命令)。
自然基准用相对iteration5–9；10含CUPTI必须排除。`physics.engine`是simulate+fetch_results外层，不能称纯solver。
所有关闭碰撞/约束/触觉的case均为诊断，不得搬到正式训练；本批无可采纳候选，保持TGS24/4。
结束已释放GPU6，未恢复旧暂停作业；GPU4/7由另项工作续训，不属于此性能实验。

### 9.3：collision_fast 独立碰撞对照与训练（2026-09-27）

本批使用当前接触权重 0.1 / 跟踪权重 40 的 Randomized 任务、model200；
与 9.2 的历史物理诊断分开。用户本轮允许接触近似及 Robot 自碰撞关闭；
仍保留原频率、求解设置、控制、观测、奖励、成功条件和 attachment。
实际候选、完整命令和结果见[collision_fast 记录](rotate_bulb_collision_fast_2026-09-27.md)。

```bash
# 在 xinyu21、已获授权且 GPU6 空闲时执行；输出目录必须不存在。
R=/home/xinyu/training_runs/20260927_1530_collision_fast_gpu6
/home/xinyu/envs/brainco/bin/python "$R/run_bulb_collision_case.py" \
  "$R/specs/train_selfoff6144.json" "$R/reproduce_selfoff_NEW" --gpu 6 --timeout 3600
```

此命令真实启动 6144 环境、从保存的 model200 恢复模型/归一化/优化器，执行 64 次 PPO 更新。
`specs/train_baseline6144_v2.json` 是同预算原碰撞对照；`specs/selfoff_asset1024.json`
是同初态/动作的 1024 环境 replay，末两步 CUPTI 所在计时窗必须排除。
`specs/eval_baseline128.json` / `eval_selfoff128.json` 是训练后冻结模型的首个完整回合评测，
不更新策略。每次复现都使用不同的新输出目录，禁止与正在运行的 collision_fast 抢占 GPU。

`BULB_COLLISION_FAST_CONFIG` 指向 case JSON，`collision_fast` 只接受
`lamp_asset`、`robot_asset`、`robot_self_collision`。包装入口在场景建立前应用配置；
仅把 USD 路径传给普通 `train.py` 不足以绕过原自碰撞强制开启逻辑。
必须检查 `physics.json` 的 live articulation 开关及 native shapes，而不是只看文件中的属性。
完整 case/spec 保存输入资产和父 checkpoint；恢复还需原动作/触觉契约及对应冻结 workspace/runtime。

检查 `status.json.completed` 和 `result.json.completed`，不能只看退出码；
本批第一次基准因诊断目录未创建而在 PPO 前失败，日志保留且不计入有效结果。
`metrics.jsonl` 的 collection+learn 是整轮 PPO 计算时间，`full_round_wall_s` 额外包含日志/保存开销。
比较排除前两轮，统一环境数、rollout 和更新预算；完整成功数/总数单独来自独立 eval。

## 10. GitNexus 只读核查的历史限制

本次未找到可调用 MCP 或 PATH 上的 `gitnexus`，从 `.gitnexus/meta.json` 的
`runnerIdentity.invokedArtifact.path` 定位**已有本地 CLI**，用 `node` 调用，未执行 npx/bunx 下载。
`list/status/query/context/impact` 已实际调用；状态显示索引 commit 与 HEAD 相同，但少 2 个录像 JSON。
`impact AGENTS.md` 未找到目标，根 `File:README.md` 无可解析调用方，均为 `UNKNOWN`；
已通过文档引用核对确认此次仅追加入口。`d_peg_success` 为 CRITICAL，详见审计记录。
没有重建索引；没有提交，因此没有进行提交前 `detect-changes`。未来提交仍必须遵守 AGENTS 的图检查规则。

## 2026-09-21：Assembly Curriculum

新任务 `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0`，原Custom任务不启用课程。
启动器 `scripts/rsl_rl/train_bulb_curriculum.py` 沿用 `train.py`，每阶段独立进程，失败/用尽预算保存后停止。
`--dry-run`只打印首条命令；实际命令、续训、冻结评测和原任务导出见[实施记录第0节](rotate_bulb_curriculum_implementation.md)。
GPU7 正式启动参数、停止旧作业、运行目录和 TensorBoard 标签见
[2026-09-21 启动记录](rotate_bulb_curriculum_gpu7_2026-09-21.md)。课程启动器支持 `--kit_args`，
同一物理 GPU 绑定同时传入训练与晋级评测；本次单 seed 不代表三训练 seed 对照已完成。
必须使用本仓库Python源码（启动器自动注入；直接train需设置PYTHONPATH）。输出目录须新建，不覆盖历史run。

A：本机已执行 `validate_bulb_curriculum.py --out 新目录 --num_envs 2 --steps 160 --headless` 的固定和`--free`变体；
2env短PPO/续训与4回合冻结检查通过，124相关测试通过4跳过。不是学习效果验证。
C：完整流水线默认每阶段最多2000轮、32env、每晋级3seed各100回合；只作预算示例，未执行正式长训练。
当前只支持单进程；改变课程源码/物理契约后旧成功状态库将拒绝加载。阶段运行和导出不改变原sim2real接口。

## 2026-09-28 关节求解短诊断与独立16/4配置

`scripts/diagnostics/benchmark_bulb_joints.py`复用train.py配置入口，以`BULB_JOINT_CASE`选择独立
physics-only场景。全部分组固定128env/240Hz/30Hz/TGS24/4；删除关节和无接触场景仅用于诊断。
`bulb_joint_runtime.py`导出native质量/惯量/驱动及USD拓扑，并读native睡眠/场景统计。
`summarize_bulb_joint_evidence.py`复用旧kernel分类汇总，导出逐关节/刚体CSV。

`scripts/rsl_rl/train_bulb_collision_fast.py`新增可选`joint_solver: {hz:240, position:16, velocity:4}`；
默认碰撞case保持24/4。只有独立case会改Scene上下限，完整动作/观测契约仍必须相同。
已验证256env各12更新、候选追加24更新、32完整回合评测；不代表6144或完整成功通过。

精确启动spec、参数、case与运行库位于`/home/xinyu/training_runs/20260928_joint_solver_gpu6`。
使用`run_bulb_joint_case.py 新spec.json --timeout 600`复现前，必须把spec/root、case.output、
log_dir、TMPDIR和case路径一致设置到新目录；入口会核查GPU空闲、限时及原始结果。
这是真实GPU执行，会创建独立日志/trace，训练case还会更新PPO并保存模型。
当前GPU6已释放；未来命令仍需当次适用的资源授权。详情见[本轮报告](rotate_bulb_joint_solver_2026-09-28.md)。

## P3当前Stage0 Storm GUI回放（2026-10-04）

远端`/home/brainco1/workspace/xinyu/assembly_play_support/fast90_model1100_gui_20261004_1916/launch_storm_mirror.sh`
启动冻结fast90训练源及独立model1100副本，DISPLAY=:1、1环境。已有play PID613765时不要重复启动。
`play_storm_mirror.py`用无物理API的显示副本解决Storm不读取GPU Fabric姿态的问题；只读真实GPU状态。
参数保留90Hz/4/1和严格契约；普通play以及仅update_transformations的版本显示静止，不应作为成功GUI入口。
训练与play共享GPU，单环境play仍额外占约3.5GiB，启动前查剩余显存；结束仅关闭play窗口或核实命令后终止play PID，勿停止训练。
入口针对本次checkpoint及冻结源，换模型需重新核对契约并复制到独立目录，避免play导出覆盖训练目录。
