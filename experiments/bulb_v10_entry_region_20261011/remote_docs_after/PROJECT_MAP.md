## 2026-10-11：当前灯泡教师v10入口

BulbGraspTeacher二/三阶段直接跟踪真实螺纹底端；二→三径向≤12 mm/深度0–10 mm/<30°即时晋级，目标仍5.5 mm。
`env_cfg.py:shallow_insertion_radial_tolerance_m`与最终装配径向0.5 mm独立；`curriculum.py:_update_metrics`采用新入口。
`screwing.py`有效旋入仍2 mm/10°/螺距一致/接触；`progress.py`显示真实完成条件。
`target_contract.py`严格核对版本；训练可显式`--migrate-v9-entry-window`续训，回放仍须同版本。
5帧MLP无LSTM、原物理和资产保持。[本次部署与验收](../experiments/bulb_v10_entry_region_20261011/README.md)。下文为保留的历史入口。

## 2026-10-07：TacSL 4096环境训练入口

独立工程新增 `scripts/train4096.sh` → `guard_run.py` → `run.py`（train完成后eval）。
`run.py`增加horizon/minibatches/保存间隔/PhysX容量/compact日志参数；`env.py`按批次统计完整环境分子分母，
compact模式仅缩减调试快照，不删训练环境。原奖励/动作/判据函数未改。
有效运行 `reports/train4096_20261007_v3`，同级v1/v2为保留的容量失败证据。
[运行结构与验收](../experiments/tacsl_train4096_20261007/README.md)。

## 2026-10-07：独立 TacSL 插入移植

相邻工程 `/home/liuxinyu/workspace/tacsl-insertion-lab`，服务器 `/home/brainco1/workspace/xinyu/tacsl_insertion_lab`。
包 `tacsl_insertion_lab` 自注册 `TacSL-Task-Insertion-Original-v0`；`env_cfg.py` → USD/PhysX/执行器，
`env.py` → 场景/重置/控制/双侧原生 TacSL，`task_math.py` → 10 个原任务数学方法，
`factory_control.py`/`torch_jit_utils.py` → 独立原控制副本。`scripts/run.py` 提供 smoke/tactile/train/eval，
`convert_assets.py` 由冻结原资产生成本工程 USD。未接入本仓库 Revo/Flexiv 任务注册或修改 Lab 内置任务。
原默认策略14维状态，触觉为独立输出；非AAC/full-state训练脚本覆盖配置。
[证据与范围](../experiments/tacsl_insertion_20261007/README.md)。

## 长方体接口方向纠正

当前 assets/head/tag_press_head 为20×20×40 mm主体，法兰尾柄在20×20 mm端面；四个20×40 mm面各2枚纸质tag（40–47）。布局契约v3替代v2，旧错误方向归档于archive/paper_v2_wrong_orientation。

## 当前长方体打印资产（2026-10-05 更新）

实际入口为 [assets/head/tag_press_head](../assets/head/tag_press_head/README.md)，含单色长方体、法兰尾柄和四面纸质Tag PDF；旧双色版本位于其 archive/inlay_v1。下方 calibration/assets/head 路径属于历史记录，不是本次输出目录。

## 2026-10-05：压头横向打印销

[calibration/assets/head/press_head_pin](../calibration/assets/head/press_head_pin/README.md) 保存三种试配杆径的STL／STEP、独立生成脚本及几何验证，不接入URDF。

## 2026-10-05：当前五面四标签压头

[calibration/assets/head/tag_cube_head](../calibration/assets/head/tag_cube_head/README.md)为当前方案，含Ø6×8尾柄及横销孔、五面共20枚ArUco、双色打印和角点坐标。`tag_press_head`保留为已替代的早期长方体方案。

## 2026-10-05：标签压头资产

[calibration/assets/head/tag_press_head](../calibration/assets/head/tag_press_head/README.md) 保存独立双色长方体压头、重建脚本、角点布局和离线视觉核验；不接入现有URDF或控制链。

## 2026-10-05：默认标定URDF的Tag唯一支架根

当前完整包root为tag_center，支架网格直接挂根；手通过tag_to_right_hand挂接。
删除lower_mount/calibration_world及3个辅助几何frame；压头XYZ链保留。
当前验证入口为calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_only/verify.py。
旧源包/导出器未更改，下文旧frame结构只对应历史版本。

## 2026-10-05：真机三模态采集链

`calibration/hardware/collect_real_trimodal.py` → `probe_revo3_readonly.py --worker camera`保存raw
→ `probe_revo3_trimodal.py`离线推理 → 每SN的samples.npz与session.json。
加载会话经`--reference-capture`复用空载raw第0帧；原推理默认行为不变。
该原生源数据契约与训练manifest有意分开，后者仍需物理标定和sim/real规范配对键转换。
见[入口说明](../calibration/hardware/README.md)。

## 2026-10-04 新增：独立灯泡抓取教师

`D/config/BulbGraspTeacher/`注册`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`和`...-Play-v0`。
2026-10-07本地v8：`curriculum.py:_set_stage_two_target`按当前姿态将螺纹端点浅插目标换算为中心命令，供原两项位置奖励共用；
`advance/_update_metrics`以端点5–6 mm/径向≤0.5 mm/倾角<30°即时晋级，不检查二阶段速度或保持；三阶段原稳定判据保持。
`env_cfg.py`独立配置`shallow_insertion_tilt_tolerance_deg`，回放HUD/trace记录端点径向。远端v7部署未更新。
可选`--lstm`：专用train/play在创建环境前调用`policy.py:configure_bulb_teacher_lstm`，
设历史1帧并选择`BulbGraspTeacherLSTM`；teacher_runner注册该类，沿用RSL循环采样/PPO。
Actor/Critic分别为380→单层LSTM256→MLP512/256/128→28/1；默认快照与5帧MLP保持。
局部类仅补探索噪声边界与bootstrap接口；回放按done清空状态，CPU检查入口`check_bulb_teacher_lstm.py`。
2026-10-07本地v7新增`curriculum.py:stage_two_success_bonus`，在原浅插晋级事件一次+10，48项CPU检查通过；远端仍运行v6。
v6：`curriculum.py:stage_two_position_orientation_reward`包装原success位置核，
仅阶段二乘螺纹朝下软系数，`env_cfg.py`参数`rewards.success.params.orientation_floor=0.2`（1关闭）。
阶段一/三与原函数保持；10月7日已在DSW新v6副本双卡各8192从v5/model9500续训。
专用train可选`--initial_adr_difficulty`匹配未保存在checkpoint中的ADR初值，本次0。
见`BULB-STAGE2-REWARD-20261007`及`BULB-V6-DSW-RESUME-20261007`。
`env_cfg.py`从typed snapshot重建教师配置；任务局部`mdp/`、`flexiv_rewards.py`隔离装配共享逻辑。
2026-10-06本地v5：`curriculum.py:BulbGraspCurriculumCommand`管理口上15cm→浅插5.5mm→实际三圈旋入，前两阶段8s、第三阶段单独6s。
`insertion.py`提供配对缩放、尖端深度和目标换算；`screwing.py:ScrewInsertionState`解包相对轴向角并验证螺距、计新高/回退。
同模块提供旋入奖励、第三阶段相对速度惩罚门控和最终一次性+20，原接触/位置奖励保持。
`reset_socket_from_target_ranges`按旧目标XY随机灯座；`curriculum_update`在timeout之前执行，metrics/ADR使用完整三阶段终态成功。
`build_socket.py`及`build_insertion_asset.py`生成独立灯座和bulb_insert.usd（源SDF碰撞保持）。
CPU检查`check_bulb_grasp_curriculum.py`；原生几何探针`probe_bulb_shallow_insertion.py`及`probe_bulb_screw_path.py`。
源snapshot/资产/观测宽度保持，但MDP版本5不同；2026-10-06已部署到DSW独立v5副本，
运行状态与验证见[记录](../experiments/bulb_v5_dsw_2x8192_20261006/README.md)。
`teacher_runner.py`保留源教师重力课程；专用入口为`scripts/rsl_rl/{play,train}_bulb_grasp_teacher.{py,sh}`。
train专用入口支持每rank PhysX容量CLI覆盖，默认不改变快照值；双卡仍通过split_gpu隔离Isaac实例并同步策略。
机器人复用assets/rotate_bulb_custom，灯泡从assets/bulb生成到assets/bulb_grasp_teacher。
详细契约/来源/证据：[任务说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md)。

## 2026-10-04：默认标定包Tag全局基准

`calibration/assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf`
新增`tag_center`，与根`calibration_world`重合；世界轴向保持，原点移到支架系(170,200,35.5)mm。
`world_to_fixture`与`world_to_press_stage`同步平移，原关节/网格/轨迹不改。
`coordinate_reference.json`登记新原点及平台世界Z=0.5mm；原始台显示值保留，外参仍未知。
包内`modifications/tag_origin/`保存变换、修改前备份、验证脚本/结果，manifest=v4。
旧导出器仍为v3源包流程；当前默认包的局部修订不能靠重新导出自动恢复。

## 2026-10-04：实体资产与任务数据分离

- [assets/README.md](../assets/README.md)：机器人、灯泡、插销、模具及传感器表面；保留自包含模型包结构。
- [task_data/README.md](../task_data/README.md)：D-Peg预抓/来源/历史验证、Custom抓姿、Randomized工作空间库、装配课程模板。
- [docs/tasks/rotate_bulb_rewards.md](tasks/rotate_bulb_rewards.md)：标准旋出任务奖励说明（非Custom）。
- `source/BrainCo_DexHand/BrainCo_DexHand/tasks/`：任务Python配置与注册保持原位置，代码未因本次迁移改动。
- `calibration/assets/`：已整理的机械标定资产保持原位置。

运行配置仍使用原assets路径，通过13个相对兼容链接读取task_data或奖励文档。
JSON内部来源路径及SHA256保持；下文历史资产数据路径继续有效，但新数据从task_data入口维护。
原包内生成器/验证脚本保持原位置，防止改变相对根路径。
具体逐项映射见[迁移清单](../task_data/organization_manifest.json)。

## 2026-10-04：M6×16沉孔派生资产

`calibration/simulation/build_m6x16_mount.py`读取原V4 BREP，仅加深四个安装沉孔。
当前完整包`calibration/assets/calibration_bench_visuotactile_urdf`已改引新STL，
CAD／打印件／验证／修改前备份在其`modifications/m6x16/`；旧V4-touch导出源仍保留。

## 2026-10-04：标定与数据采集目录统一

总入口为[calibration/README.md](../calibration/README.md)，完整映射见
[organization_manifest.json](../calibration/organization_manifest.json)。
原scripts/calibration_bench→calibration/simulation；scripts/force_map→calibration/pressure；
三模态采集／mmap／RL球探针→calibration/collection；标注／投影→calibration/tools；
原image_generator→calibration/image_generator；真机probe→calibration/hardware。
压头／支架／XYZ包→calibration/assets；历史ViTai图像→calibration/reference；
outputs/calibration_bench→calibration/outputs/simulation。原路径保留相对符号链接，无重复实体文件。
运行库仍在source/integrate/tacmap，通过calibration/shared索引；SDK与captures保留在calibration。
下方历史路径继续有效，但新操作使用总入口的新路径。

# 项目地图

2026-10-04新增`calibration/probe_revo3_trimodal.py`：读取只读采集目录的RGB、型号／序列号与时间，
调用ViTai SDK离线模式逐帧生成深度和光流marker，保存临时参考／模型哈希／warp／三模态NPZ。
独立于训练，不打开硬件；本次未确认空载，只用于诊断。见[机械标定说明7.6节](../calibration/机械标定台仿真搭建_README.md#76-五路三模态通路验证2026-10-04相机重新接通后)。

2026-10-04新增`calibration/probe_local.sh` → `probe_revo3_readonly.py`本机SDK启动入口，
复用源SDK虚拟环境与仓库原生包；不接入仿真／训练。
最新三模态接口与相机未枚举的复测结果见[机械标定说明7.5节](../calibration/机械标定台仿真搭建_README.md#75-本机连接配置与三模态复测2026-10-04)。

2026-10-03 新增 `assets/revo3_breadboard_mount/generated_calibration_v5_m6x20/`：在 V4 上只做底面
10 mm 平切并重设新底面为 Z=0，保留 M6 孔；含 STEP/BREP、STL/3MF、独立支架 URDF 和当前
视触觉手组合 URDF。生成入口为 `scripts/calibration_bench/build_m6x20_mount.py`，未替换历史 V4。

2026-10-03标定资产更新：`build_touch_fixture.py` 将 `assets/revo21_right_touch` 原视触觉手与
不变V4支架生成到 `assets/revo3_breadboard_mount/generated_calibration_v4_touch/`。
`geometry.hand_asset` 按config版本选择资产，`sensor_link` 映射五指名称；无版本字段保留旧手。
当前含XYZ交付为[calibration_bench_visuotactile_urdf](../assets/calibration_bench_visuotactile_urdf/README.md)，
`verify_touch_asset.py` 复核源元素、FK、任务rubber→DIP映射与支架抽样配合；下方旧URDF路径保留作历史。

2026-10-03 新增 [calibration/](../calibration/README.md)：Revo3 SDK源码及原生包副本、复制清单、
独立只读探测脚本与本机采样证据。主链路State/Touch与独立ViTai相机分开读取，未接入仿真控制或训练。
接口唯一维护入口为[机械标定说明第7节](../calibration/机械标定台仿真搭建_README.md#7-revo3-sdk与真机只读采集2026-10-03)。

2026-10-03 新增 [scripts/calibration_bench](../scripts/calibration_bench/README.md)：独立机械标定仿真。
`run.py` → `protocol.py`（轨迹／划分）→ `geometry.py`（URDF/FK/mesh）→ `model.py`
（几何深度／压力／既有 `integrate/curved_hydroshear_adapter.py`／诊断RGB）→ NPZ与`scene.py`生成USD。
`verify.py`核查数据，`preview_isaac.py`原生回放／截图；不接入训练注册、不连接SDK。
场景生成器现只输出手／V4支架、压头／法兰及查看所需相机／灯光，虚拟机械机构已按用户要求移除。
`export_urdf.py` 将该装配另存为[自包含URDF包](../assets/calibration_bench_urdf/README.md)，
新增无外观XYZ关节、`world_to_press_stage`可调安装frame及首回合CSV；原始手／工具资产保持。
本轮重新生成的有限证据在 `outputs/calibration_bench/`，不属于下段之前已清理的历史输出。

2026-10-03清理后：根目录 `tests/`、运行日志/录像/输出/benchmark 产物已移除；历史章节中的
相关路径只表示曾经的实现或证据位置。`assets/revo3_breadboard_mount/` 只保留自包含的
`generated_calibration_v4/`，不再包含 V1/V2/V3、生成器或包外说明，因而不能在本仓库重建。
`docs/` 下 52 份带日期后缀或 experiment(s) 命名的历史日志型 Markdown 同期移除；
STATUS、RUNBOOK、PROJECT_MAP 以及其余设计、实现、集成和使用说明继续保留。

2026-10-01当前默认更正：按用户要求，本地和服务器Randomized旋出已恢复GPU4高成功率profile3；`env_cfg.py`不再引用joint1_zero或initial_arm_pose，选用原机器人USD、原joint1初态/±160°和workspace_bank_high_z_v2，60Hz物理/30Hz控制/12秒。下文9月29日profile6描述为历史训练记录。新零位文件保留但不启用，装配课程保持。恢复证据见[STATUS](STATUS.md)。

核查：2026-09-20；适用 `main` / `4707074194091cd723ebd18dea7824c184e99c6d`
及当时未提交工作区。下文数值均为**当前源码定义**，不是某次运行的生效参数；
运行证据见 [实验索引](../EXPERIMENTS.md)，版本冲突见 [STATUS](STATUS.md)。

## 1. 目录与职责

本文用 `B = source/BrainCo_DexHand/BrainCo_DexHand`、
`D = B/tasks/manager_based/dexsuite` 缩写长路径；它们不是环境变量。

| 相对路径 | 职责 / 阅读入口 |
| --- | --- |
| [source/BrainCo_DexHand](../source/BrainCo_DexHand/) | 可安装扩展；`setup.py`、`config/extension.toml` 声明包与 Isaac Lab 扩展依赖 |
| `B/__init__.py`、`B/tasks/__init__.py` | 导入任务包、触发 Gym 注册 |
| [B/tasks/direct](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/) | Direct 环境；Repose 基础环境、视觉触觉派生环境与任务配置 |
| [D/config](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/) | ManagerBased 的 Revo3、InsertDPegCustom、RotateBulbCustom 配置与注册 |
| `D/dexsuite_env_cfg_grasp_tianji.py`、`D/mdp/` | 共享场景、观测、奖励、动作、事件；D-Peg 状态机/触觉/控制也在 `mdp/` |
| [scripts/rsl_rl](../scripts/rsl_rl/) | 训练、回放、任务评测、诊断、契约与分布式适配 |
| [scripts/diagnostics/benchmark_bulb_joints.py](../scripts/diagnostics/benchmark_bulb_joints.py)、`bulb_joint_runtime.py`、`run_bulb_joint_case.py`、`summarize_bulb_joint_evidence.py` | 独立灯泡关节分组、native拓扑/睡眠/CUPTI、空闲GPU执行和结果汇总；真实训练沿用`train_bulb_collision_fast.py`的显式`joint_solver`配置，原任务默认值保持 |
| [B/tactile_representation](../source/BrainCo_DexHand/BrainCo_DexHand/tactile_representation/) | 冻结策略编码器、恢复网络、数据/训练与对齐模块 |
| [scripts/tactile_representation](../scripts/tactile_representation/) | RGB/Depth/marker 采集、缓存、恢复训练、两塔对齐、编码器导出 |
| [tacmap](../tacmap/)、[integrate](../integrate/)、`B/force_map/` | TacMap 深度、HydroShear marker、Taxim RGB、WarpSDF 压力等传感器及演示适配 |
| [assets/d_peg_insertion](../assets/d_peg_insertion/)、[assets/rotate_bulb_custom](../assets/rotate_bulb_custom/)、[assets/bulb](../assets/bulb/) | 插销/孔/预抓姿、Flexiv 机器人/抓姿、灯泡/灯座资产 |
| [assets/mold_split](../assets/mold_split/README.md) | 独立模具资产；`build_mold_urdf.py` 从 `1.stp` 生成两link法向开合URDF、STEP/BREP、原始/参考几何与离线预览；未接入训练，碰撞网格待修复 |
| [1_stl/breadboard_mount_256](../1_stl/breadboard_mount_256/README.md) | 模具打印与光学台面固定版；从 `root_open_256` 增加25mm网格M6安装孔及打印定位配合，`build.py`独立重建CAD/STL并检查；原URDF网格不随之替换 |
| [assets/revo3_breadboard_mount/generated_calibration_v4](../assets/revo3_breadboard_mount/generated_calibration_v4/) | 当前唯一保留的 Revo3 标定支架包；入口 `urdf/revo3_right_calibration_lower_mount_v4.urdf`，自带 CAD、打印网格、手部网格、预览和验证 JSON。旧版本及生成器已清理，不可在本仓库重建；未接入训练 |
| [assets/head](../assets/head/) | 机械标定台三角、球形、锥形压头及连接法兰的 SLDPRT/STL 源资产；三份 `*_press_head_with_flange.urdf` 按 8 mm 插入深度和横向销孔组装，并提供 `press_head_tip` frame。质量/惯量和 Isaac Sim 动态导入待验证；坐标契约见[搭建说明](../calibration/机械标定台仿真搭建_README.md) |
| `calibration_bench_sim_package/`（历史路径，当前缺失） | 曾有双Y标定台CAD/Blender和四link URDF，当前不可使用；新系统仅用虚拟XYZ机构，不恢复历史资产或机械限位 |
| `tests/`（已清理） | 2026-10-03 按用户要求删除；历史测试命令不可在当前工作区执行 |
| [README](../README.md)、[docs/tactile_representation](tactile_representation/) | 触觉流程与各专题历史记录，命令验证等级由 RUNBOOK 补充 |

Isaac Sim、Isaac Lab 的 AppLauncher/环境基类/管理器，RSL-RL 的 PPO/Runner 是外部基础框架。
项目定制包括上述传感适配、任务 MDP、Flexiv 场景、观测/动作契约、Bulb 策略、Repose
版本选择、D-Peg PPO 诊断、分布式归一化修补。源码带上游版权不表示没有本地定制。

## 2. 主要任务与继承

以下 ID 在对应 `__init__.py` 中逐项核对；不将 `train.py` 兼容判断字符串当作注册证据。

| 完整任务 ID | 注册位置（相对 B） | 环境 / 配置关系 |
| --- | --- | --- |
| `BrainCo-Direct-Revo3-Repose-Cube-v0` | `tasks/direct/brainco/__init__.py` | `InHandManipulationEnv`；`BrainCoHandEnvCfg` |
| `BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0` | 同上 | `VisuotactileInHandManipulationEnv(InHandManipulationEnv)`；独立 `BrainCoVisuotactileHandEnvCfg(DirectRLEnvCfg)` |
| `BrainCo-Direct-Revo3-Reorient-Cylinder-v0` | 同上 | `tasks/direct/reorient.py:InHandManipulationEnv`；本次未深入调查 |
| `BrainCo-Dexsuite-Revo3-Right-Lift-v0` | `tasks/manager_based/dexsuite/config/Revo3/__init__.py` | `ManagerBasedRLEnv`；`DexsuiteRevo3LiftEnvCfg`，也是触觉采集示例任务 |
| `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-v0` | 同上 | 标准版 `DexsuiteRevo3RotateBulbEnvCfg`，与 Custom 是不同模块中的同名类 |
| `BrainCo-Dexsuite-Revo3-Right-Insert-D-Peg-v0` | 同上 | 标准 `DexsuiteRevo3InsertDPegEnvCfg`，配置文件 `dexsuite_revo3_env_cfg_insert_d_peg.py` |
| `BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0` | `tasks/manager_based/dexsuite/config/InsertDPegCustom/__init__.py` | `ManagerBasedRLEnv`；`DexsuiteFlexivInsertDPegEnvCfg(Revo3MixinCfg, DexsuiteReorientEnvCfg)` |
| `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0` | `tasks/manager_based/dexsuite/config/RotateBulbCustom/__init__.py` | `environment.py:make_rotate_bulb_env`；Custom `DexsuiteRevo3RotateBulbEnvCfg(Revo3MixinCfg, DexsuiteLiftEnvCfg)` |
| `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0` | `tasks/manager_based/dexsuite/config/RotateBulbRandomized/__init__.py` | `env_cfg.py:make_randomized_bulb_env`；固定臂/零手、独立采样与阶段奖励、687维契约 |
| `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0` | 同上 | 同一工厂；`assembly_env_cfg.py:DexsuiteRevo3BulbAssemblyEnvCfg` 派生自 Custom 旋出配置，替换 reset/奖励/终止/部分观测 |

Custom 灯泡虽然 ID 写 `Revo3`，实际机器人是 **Flexiv Rizon4 + Revo3 右手**。
Flexiv D-Peg 借用其 `robot_setup.py:Revo3MixinCfg` 和机器人 USD，插销资产、预抓姿、
PPO 与状态机由插销任务自有。不要把两者互作兼容检查点。

## 3. 启动、配置与训练链

```text
scripts/rsl_rl/train.py
  → argparse + cli_args.add_rsl_rl_args + AppLauncher（启动仿真）
  → import BrainCo_DexHand → Gym 注册
  → hydra_task_config(task, agent) → 配置类默认值 / __post_init__ / Hydra 覆盖
  → main 中再次应用 CLI（环境数、seed、max_iterations、device、resume 等）
  → configure_repose_checkpoint / configure_bulb_checkpoint（按任务和模型选择语义）
  → gym.make(task, cfg) → Direct 环境或 ManagerBased 工厂
  → prepare_policy_run / prepare_bulb_policy_run（保存或核验契约）
  → RslRlVecEnvWrapper → OnPolicyRunner 或 DistillationRunner
  → load（若恢复）→ 诊断 / 归一化适配 → params/*.yaml → runner.learn
```

最终事实应从新 run 的 `params/env.yaml`、`params/agent.yaml`、契约和启动命令确定。
`__post_init__` 派生字段可能早于 Hydra 覆盖生成；不能假设修改一个顶层键会自动重算所有字段。
`cli_args.update_rsl_rl_cfg` 会用 `--resume` 的布尔值覆盖 `agent.resume`，恢复必须显式传
`--resume`。`--log_dir` 控制新输出位置，恢复来源由实验根目录与 `--load_run` / `--checkpoint`
解析；详见 [RUNBOOK](RUNBOOK.md)。

`DurationLoggingOnPolicyRunner` 继承外部 Runner，并未在本仓库实现整套 PPO。
Custom D-Peg 的 MLP 默认 `[512,256,128]`、rollout 32、gamma 0.999、固定学习率 3e-5；
Custom 灯泡共用 `BulbActorCritic`、同宽 MLP、rollout 32、gamma 0.999、自适应学习率 1e-3；
Repose 视觉触觉新训练为基础 `ActorCritic`、`[1024,512,256,128]`、rollout 16、gamma 0.99。
来源：各注册目录 `agents/*rsl_rl_ppo_cfg.py`；这些值可被运行参数覆盖。

## 4. 主要 RL 接口与判据

### Repose-Cube 视觉触觉

- 配置：`B/tasks/direct/brainco/brainco_hand_visuotactile_env_cfg.py`。
  默认 `sim.dt=1/120`、`decimation=4`，控制 30 Hz，回合 10 s，32 环境，21 个手关节动作。
- `InHandManipulationEnv.compute_full_observations` 拼接：关节位置/速度 42、物体状态 13、
  目标位置/姿态与相对四元数 11、5 指尖位姿/速度 65、上次动作 21，共 152 维。
- 开启触觉：`VisuotactileInHandManipulationEnv._get_observations` 经
  `brainco/visuotactile_runtime.py:DirectVisuotactileRuntime.observe` 获得图像/marker，
  冻结 `FrozenTactilePolicyEncoder` 后拼接 `(152 + 5 × projection_dim) × history_length`。
  当前历史默认 1；64 维对齐包为 472，256 维未对齐包为 1432，实际从编码包读取。
  指序为 little/ring/middle/index/thumb；关闭 `tactile_policy_enabled` 后为 152 维，无 latent 占位。
- 当前 revision 1：动作经基类 `_apply_action` 从归一化值映射到关节范围、平滑并限位，
  是绝对位置目标。`compute_rewards` 计算位置、姿态、动作代价和姿态到达奖励；
  `success_tolerance=0.2` 的姿态目标到达会重新采样目标，不能直接解释为完整回合独立成功率。
  `_get_dones` 检查掉落距离 `fall_dist=0.24` 与超时；`_reset_idx` 重置初态。
- 历史 revision 2 走 `B/tasks/direct/repose_training_runtime.py` 与 `scripts/rsl_rl/repose_training.py`，
  使用增量动作、课程和保持判据。训练/回放会根据检查点标记选择版本，不能按新默认解释旧模型。
  `_reset_idx` 同时清理触觉缓存与历史；详见 [接入说明](tactile_representation/repose_policy_integration.md)。

### Flexiv D-Peg Custom

- 配置/重置：`D/config/InsertDPegCustom/env_cfg.py:configure_d_peg_insertion`、
  `reset_d_peg_pregrasp`；从已抓持插销起步，非桌面抓取全流程。
  `pregrasp_flexiv.json` 保存实际关节位姿及 drive target；两者不能互换。
- 默认 32 环境；`sim.dt=1/240`、`decimation=8`，控制 30 Hz，`task_episode_length_s=15`。
  孔 XY 随机范围 ±0.1 m、yaw ±90°，roll/pitch 为 0；仅说明当前配置，不冒充历史分布。
- 动作：`D/mdp/d_peg_actions.py:DPegPregraspResidualAction`，28 关节（臂 7 + 手 21）；
  `q_target = q_pregrasp_drive_target + scale × action`，不积分。
  当前 Custom `scale=1.0, clip=None`、PPO `clip_actions=None`、噪声上限 `None`，物理关节约束仍存在。
- 观测：`D/mdp/d_peg_tactile.py:initialize_d_peg_tactile_contract` 强制检查
  `policy=43`（物体四元数、目标 pose、动作、进度/阶段）、`proprio=434`（关节、指尖、接触/压力等）、
  `perception=192`（64×3 点云），关闭多模态共 669 维；开启每指 256 latent 后为 1949。
  当前默认 `tactile_policy_enabled=False`，仍保留压力观测，不能称作“所有触觉关闭”。
- 奖励/终止：`D/mdp/d_peg_insertion.py:_geometry`、`DPegRewardState.update`、`DPegInsertionState`、
  `d_peg_success/d_peg_dropped/d_peg_nonfinite`；孔坐标系计算几何、入口历史及进展。
  当前成功为有效数值状态下轴向深度 ≥27 mm 连续 0.5 s，锁存并一次给 +80；
  成功不再要求 containment / entry history。几何过程奖励仍有自己的门槛。
  `insertion_depth_m=0.028` 是配置的目标深度，不等于成功阈值 0.027。
  掉落、越界、非有限状态及超时分别记录；完成帧的成功/失败优先级见 `DPegRewardState.update`。
  此判据与旧几何验收不同，完整性疑问见 STATUS，不能自行替用户改判据。

### Custom 灯泡：旋出与桌面装配必须分开

2026-09-29已部署GPU6：Randomized的`joint1_zero.py`选择`assets/rotate_bulb_randomized/rizon4_joint1_zero.usda`及`workspace_bank_joint1_zero_v1.json`，将原joint1初始姿态定义为资产0位、有限±180度。几何初态及样本不变，原动作控制器自动使用新限位；旧checkpoint契约不兼容。已原生验证，GPU6 profile6采用新资产，GPU4保持原资产，见[证据](../logs/diagnostics/bulb_joint1_zero_20260929/README.md)。


2026-09-29当前Randomized reward profile6：`phase_rewards.py:initial_arm_pose_guidance`新增固定reset姿态弱引导，由`env_cfg.py`注册为`initial_arm_pose`。七臂关节、90度免罚、90度尺度、weight=-0.01、释放后0.2；profile5的-0.025减弱60%，结合新joint1零位资产在GPU6重新从零训练；GPU4仍profile3。

新增 Randomized 旋出任务位于 `config/RotateBulbRandomized/`，继承 Custom 物理但独立覆盖初态、
reset、目标和阶段奖励；`mdp.py` 加载 `assets/rotate_bulb_randomized/workspace_bank_high_z_v2.json` 的3456个合格位置/目标对。
臂固定、手21关节归零，支架X/Y ±10cm、目标相对支架X/Y ±15cm/Z +28–42cm；有限筛选分布。
`RandomizedPalmAboveSupport` 在旋出前阶段鼓励真实palm位于支架上方10–18cm，含有限保持/新增旋转奖励预算；
释放后的搬运不受此项约束。新随机任务契约v2包含奖励参数和目标高度，旧库保留用于历史回放。
`build_bulb_randomized_workspace.py` 负责离线几何/IK筛选；`validate_bulb_palm_high_z.py` 检查原生reset、奖励和契约。
`RotateBulbRandomized/phase_rewards.py`提供profile3的`ReleasedContactGuidance`与`prerelease_arm_motion`，
由Randomized配置及PLAY共用。接触辅助释放前0、释放后对向最高1/s且每回合最多4分；
臂速成本释放前无需先抓稳、权重−0.3，释放后豁免。关闭独立randomized_grasp payout，
保持原状态/观测、真实旋转接触门控及成功条件；profile版本单独写入env.yaml，动作契约不变。
该变更不改Custom/Assembly默认，旧回放需使用原冻结配置。CPU行为检查见`tests/test_rotate_bulb_phase_rewards.py`。
14维可观测阶段状态使总输入为687；`scripts/rsl_rl/bulb_training.py` 为独立task ID和采样库哈希校验契约。
不与673维预抓持旋出或669维装配混用检查点。实现与证据见[新任务执行记录](rotate_bulb_randomized_training_2026-09-21.md)。

- 旋出初始抓姿：`RotateBulbCustom/robot_contract.py:ROTATE_GRASP_REFERENCE_PATH` →
  `assets/rotate_bulb_custom/grasp_reference/dexonomy_20260920/simulation_state.json`。
  `configure_rotate_bulb_scene` 将其按关节名写入机器人默认初态；启动/reset 共用。
  `bulb_assembly_task=True` 时机械臂和场景保留旧参考；装配配置在父类初始化后将
  21 个 `right_*` 手部关节初值设为 0，启动/完整/局部 reset 与 PLAY 共用。
  D-Peg 的旧参考导入不变。

- `D/config/RotateBulbCustom/env_cfg.py:configure_rotate_bulb_scene`：默认 32 环境，
  `sim.dt=1/240`、`decimation=8`，旋出与装配控制均为 30 Hz，回合 20 s。
- `D/config/RotateBulbRandomized/env_cfg.py`：2026-09-28临时覆盖训练及PLAY为
  `sim.dt=1/60`、`decimation=2`、`render_interval=2`、`task_episode_length_s=12`；
  按用户修正要求保持控制30Hz，最多360控制步。GPU4已按此从头启动；GPU6冻结训练仍240Hz/8/20秒。
- 动作：`actions.py:BulbJointPositionAction`，28 维，版本 `BULB_ACTION_VERSION=3`。
  当前臂/手 scale 都为 1 rad；每控制步计算
  `q_target=clamp(q_measured + clamp(action,-1,1), joint_limits)`，子步保持同一目标。
  `scripts/rsl_rl/bulb_training.py:BulbActorCritic` 使用 tanh 高斯，std 为 `0.05+softplus(parameter)`，
  无 std 上限。旧 v2 的 0.025/0.05 rad 动作与噪声语义不可直接恢复为 v3。
- 观测继承 `D/dexsuite_env_cfg_grasp_tianji.py:ObservationsCfg`，由 Custom 配置替换触觉和阶段项；
  `policy/proprio/perception` 输入 actor/critic，关闭 latent 时当前旋出为673维（新增4维小目标），
  旧旋出与当前装配为669维。开启5×256 latent 的历史布局为1949维，当前新增引导布局未验证该组合。
  `tactile.py:initialize_rotate_bulb_tactile_contract` 记录实际分组/维度；
  `prepare_bulb_policy_run` 校验669/673与小目标契约，不允许只凭动作维数混用检查点。
- 压力位姿可选：`RotateBulbCustom/pressure_pose.py` 由 `environment.py:make_rotate_bulb_env`
  在 Hydra 覆盖完成后安装，仅作用于 guided 旋出；Assembly/纯碰撞提前返回。
  共享 `WarpSdfTactileSensor` 仅增加默认行为相同的 provider 工厂；
  `fresh` 使用完整 ContactSensor，`pose_only` 保留 PhysX 位姿读取并省去内部无消费者的力归约。
  父 pressure sensor 的更新周期独立保留；新语义/周期由 tactile 契约校验。
  原理与证据见 [物理分析](rotate_bulb_physics_analysis.md)，实验入口见 [RUNBOOK](RUNBOOK.md#9-a-rotate-bulb-速度实验2026-09-2021)。
- 隔离物理性能诊断：`scripts/diagnostics/benchmark_bulb_physics.py` 包装旧 profiler，增加
  CUPTI/cProfile与live stage审计；`replay_bulb_physics_factors.py`通过独立进程中的gym.make/runner钩子
  做固定初态/动作单因素试验和冻结策略质量检查；`run_bulb_physics_case.py`复用保存的launch spec。
  这些入口不由正式train/play自动调用，关闭碰撞/约束/压力的分支仅诊断使用。
  [2026-09-27证据与淘汰结果](rotate_bulb_physics_performance_2026-09-27.md)未产生正式配置修改。
- 旋出默认 `bulb_pure_collision=False`，螺旋关节导向；`True` 为自由刚体碰撞模式。
  `reset_lamp_on_table` 与 `reset_robot_to_grasp` 负责初态，
  `release_unscrewed_bulb` 处理释放；默认初始3圈。纯碰撞历史实现位于 `collision.py`，当前旋出只使用guided。
  奖励在 `DexsuiteRevo3RotateBulbRewardCfg`；当前系数见 [三圈分段引导](rotate_bulb_segmented_guidance_2026-09-20.md)。
  `RotateBulbGuidedRotation` 按单指接触/0.2秒记忆奖励连续新角度和12个90°里程碑，三圈合计最多6分；
  `rotate_bulb_arm_motion` 在释放前约束实际臂漂移和速度；`rotate_bulb_rotation_subgoal` 提供4维目标观测。
  搬运保留原双侧软抓持质量和0.1秒记忆，只给释放后新增最佳进度、最多4分；释放后抓稳/稳定预算各2分。
  成功由 `RotateBulbStableSuccess` / `RotateBulbSuccessTermination` 检查释放后目标距离 <3 cm、保持 1 s，
  不要求持续持握；掉落/越界/失握失败优先。失握需释放后持续无接触且远离，目标保持期间豁免；
  `non_finite` 终止目前仍关闭。上述为本地新默认，已运行进程不会热加载。
- Assembly 默认 `bulb_pure_collision=True`，`assembly.py:reset_bulb_assembly_on_table` 从桌面分离状态起步。
  `AssemblyState.update` 维护 DESKTOP→GRASPED→LIFTED→ALIGNED→ENGAGED→INSTALLED；
  `assembly_success` 要求有效啮合、当前深度约 18 mm、径向/轴线/螺旋误差、接触及速度稳定保持 1 s。
  `assembly_bulb_dropped/assembly_lost_grasp` 等处理失败；实际阈值集中在 `assembly.py`。
  `assembly_env_cfg.py:DexsuiteRevo3BulbAssemblyRewardCfg` 与 `assembly_guidance.py` 保留旧奖励（`OLD_REWARD=True`）；
  默认 `OLD_REWARD=False` 在工厂接入 `assembly_reward_v2.py` 有限预算账本，
  `assembly_reward_adapter.py` 只读取原物理状态/成功标志并导出分项及失败诊断。
  新项替换旧shaping与success reward，保留两项原动作成本；观测、动作和reset分布不变。
  数学与预算见[批准方案](rotate_bulb_reward_redesign_plan.md)，开关/标签/验证见[实施记录](rotate_bulb_reward_implementation.md)。
  装配本地悬停净空默认10cm，`descent_reward_length_m=0.04`将自由下降进度换算为奖励长度，
  保留入口以内的实际深度奖励尺度；已启动的旧4cm进程不会热加载。
  `assembly_lift.py` 提供按碰撞网格最低点计算的最佳抬起进展与一次性稳定离桌奖励，
  参数集中在 `env.assembly_lift`，各奖励历史由 RewardManager 随完整/部分 reset 清除。
  上方目标锁存是奖励状态，不等于物理啮合，也不是自动搬运。详见 [装配引导](tactile_representation/bulb_assembly_guidance_2026-09-19.md)。

- Randomized旋出专用回退/停滞成本：`D/config/RotateBulbRandomized/progress_penalty.py`
  提供独立Torch状态机与版本参数，`progress_penalty_terms.py`适配RewardManager并共享每步状态。
  仅在Randomized `env_cfg.py`接入两项，释放后免罚，不改Custom/Assembly配置或动作观测契约。
  [参数与生效边界](rotate_bulb_progress_penalty_2026-09-25.md)，原生检查入口`validate_bulb_progress_penalty.py`。

## 5. 评测、触觉学习、蒸馏与部署边界

- 独立入口：`evaluate_d_peg.py` 按深度保持计数；`evaluate_bulb_custom.py` 按冻结旋出协议；
  `evaluate_bulb_assembly.py` 检查完整装配。均位于 `scripts/rsl_rl/`。Repose 尚未找到同等独立完整回合验收入口，
  `scripts/diagnostics/validate_repose_training.py` 是 revision 2 诊断，不能替代通用独立评测。
- `play.py` 从当前注册配置 + CLI 创建环境，**不会自动还原整个历史 env.yaml**；契约校验通过也不证明
  每个奖励/物理参数都相同。其回放会导出 JIT/ONNX；视频/单次回放与独立样本统计分开。
- 数据路径：采集 → NPZ 分片 → `prepare_mmap_cache.py` → 恢复网络/三模态网络 →
  `train_latent_alignment.py` → `export_tactile_encoders.py` → 冻结编码器 + PPO。
  两塔对齐是 sim/real latent 对齐，与 RL 教师/学生蒸馏不同；同一仿真数据喂两塔不能证明真机对齐。
- TactSpace 适配分支：`models/tactspace.py`（位于上述 tactile_representation 包）实现
  独立域/模态编码、纯 latent 解码、联合损失及表示契约；原 `train_tri_modal_cross_autoencoder.py`
  通过 `--model-type tactspace` 调用，默认仍 tri_modal。`policy.py` 工厂/冻结包装器及原导出入口识别新类型。
  `scripts/tactile_representation/check_tactspace.py` 为短 CPU 接线检查；CLI 暂仅支持同步 sim 数据，
  旧 `train_latent_alignment.py` 不允许用 episode/frame 相等冒充可信跨域配对。
  [命令与契约](RUNBOOK.md#81-tactspace-rgbdepthmarker-适配版2026-10-07)。
- 部分 Revo3/灯泡注册有 `rsl_rl_distillation_cfg_entry_point`、`student_proprio` 与 DistillationRunner 分支。
  “存在入口”不等于当前 Custom 动作策略、教师检查点和学生部署已验证；本次未核查可用蒸馏基线。
- [Repose 接入说明](tactile_representation/repose_policy_integration.md) 提供真机张量推理接口；
  需要匹配编码包、关节/指序、标定和物体状态估计。没有据此确认端到端真机闭环控制入口或迁移成功。

## 6. 输出与资产来源

2026-09-27 碰撞优化使用独立入口，不修改共享任务注册或正在运行的源快照：

- `scripts/diagnostics/build_bulb_collision_fast.py` 生成 Lamp 减面/SDF/凸包候选；
  `build_bulb_robot_palm_proxy.py` 生成掌部代理或仅关闭自碰撞的 Robot 副本。
  `validate_bulb_collision_assets.py` 检查碰撞允许范围、网格和关节/质量等属性保留。
- `assets/rotate_bulb_collision_fast/20260927/` 保存基准及候选 USD、源哈希和构建清单。
- `scripts/rsl_rl/bulb_collision_fast.py` 在场景创建前应用独立配置并读取实际生效的碰撞设置。
  `train_bulb_collision_fast.py` → 原 `train.py`，严格恢复模型/归一化/优化器和完整契约；
  只添加只读阶段统计，支持短训练与冻结评测。
- `scripts/diagnostics/replay_bulb_collision_fast.py` → 原固定动作 replay；
  `run_bulb_collision_case.py` 以保存的 spec 在空闲 GPU、新输出目录运行并核验完成性。
- [实测记录](rotate_bulb_collision_fast_2026-09-27.md)区分固定动作初筛、6144 环境适应训练、
  完整回合评测与后续训练；证据位于 `benchmarks/rotate_bulb/collision_fast_20260927/`。

| 位置 / 配置键 | 内容与使用限制 |
| --- | --- |
| `logs/rsl_rl/<agent.experiment_name>/<时间>_<run_name>/` 或 `--log_dir` | `model_*.pt`、事件日志、`params/env.yaml` / `agent.yaml`、Git 状态记录及契约；只读抽样定位 |
| run 下 `tactile_policy_contract.json`、`bulb_action_contract.json` | 恢复/回放依赖；单独拷贝模型会缺失语义与资产信息 |
| `logs/diagnostics/`、`outputs/`、`assets/d_peg_insertion/validation/` | 诊断报告、配置与逐回合记录；按实验索引定点读取 |
| `recordings/`、`checkpoints/` | 本地视频、拉取的检查点；不按目录时间或文件名选“最佳” |
| `datasets/`（由 `--dataset/--output` 指定） | 原始多模态分片/缓存；本次未递归调查数据集 |
| `runs/revo3_cross_modal_restoration_v1/`、`runs/revo3_tri_modal_cross_v1/` | 原始编码器 `best.pt` 所在目录；本次仅检查目录存在，没有读取权重或确认 LFS 实体完整性 |
| `env.tactile_policy_checkpoint` | Repose 默认导出包在 `runs/revo3_encoder_exports/`；本次该目录未见，需显式准备或选择状态模式 |
| `assets/revo21_right_touch/`、`vitai_4Fingers-320*240/`、`urdf/`、`meshes/` | 标定、marker、相机参考、机器人几何；保持任务引用和来源，不递归加载 |

部分默认 USD 来自 `ISAAC_NUCLEUS_DIR`，本机能否离线解析未验证。
`logs/`、`outputs/`、`.autonomy/` 等被忽略；证据链接可能只在本工作区有效，跨机器应先核验存在性。

## Assembly Curriculum（2026-09-21实现）

2026-10-04性能诊断：`scripts/diagnostics/probe_bulb_floor.py`增加显式solver上下限、GPU分区、
独立灯泡资产和场景审计参数；保存固定动作序列的逐步耗时、原生solver范围与接触间隙。
诊断入口支持solver、GPU分区、独立灯泡资产；`core.py:CurriculumSettings`另增`fast_stage0_physics=False`，
仅显式Stage0可选90Hz；`configure_curriculum`按既有频率解析得到decimation3、30Hz控制。
默认物理与标准/Custom任务不变，Stage1–4拒绝快速标志；源码/物理契约继续阻止不兼容恢复。
训练通过train.py四项solver和两个课程时钟Hydra覆盖选择，不修改旧冻结工作区。


2026-10-04：课程`rewards.py:FloorBestReachReward`由`env_cfg.py:configure_curriculum`在账本配置后
仅对Stage0注册为`reach_best_progress`。复用Custom的五指最大壳体距离，按历史最小距离改善给100分/m；
保留所有原动作/机械臂代价，Stage1–4和独立任务不新增。课程契约哈希覆盖本次源码变化。
同日Stage0在账本替换后重新注册`any_finger_contact`，复用Custom的灯泡过滤接触函数，
任意单指>1N给0.7分/秒；原双侧稳定账本保持，其他阶段无此新增项。

2026-10-03 v2覆盖说明：当前默认首阶段为自由横躺，`floor_reset.py`按碰撞顶点放置；
`distributed.py`在PPO边界合并两卡回合结果／成功状态库，主卡检查点包含各rank随机状态；
`train_bulb_curriculum.py --num-processes 2 --num_envs 4096`代表双卡共训、每卡4096环境。
`floor_pregrasp_v2.json`为独立接近模板，`rewards.py`提供有符号接近进度、关节1路程／偏离成本和
训练聚合日志。无触觉模式移除压力／接触观测与全部触觉图像／压力传感器，输入369维；
`bulb_training.py`只为课程v2增加契约分支。物理120或240Hz、控制30Hz，通过
`bulb_curriculum.physics_hz`统一解析；源／状态库契约加入课程版本、decimation和新模块哈希。
启动器阶段为stage0→1→2→3→full，不再经过fixed／stage0_free。
新诊断入口为 `build_bulb_floor_pregrasp.py`、`check_bulb_floor_contract.py`、`probe_bulb_floor.py`；
旧 `validate_bulb_curriculum.py`／669维Custom导出只适用于旧冻结版本。
细节和边界见[课程说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum/README.md)。
下述2026-09-21/22条目为v1历史结构。

- 注册：dexsuite/config/RotateBulbAssemblyCurriculum/__init__.py，新ID为
  `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0`；原Custom ID保留。
- `core.py` 管阶段参数/调度/状态库，`runtime.py` 管reset/事件/奖励/日志，`environment.py` 管固定夹具与reset生命周期，
  `env_cfg.py` 管各阶段时长/奖励，`insertion.py` 在恢复验证后复用原插入项。
- 2026-09-22：`env_cfg.py:configure_curriculum` 在原阶段配置后读取 `OLD_REWARD`；新模式复用Custom装配账本，
  旧 `curriculum_progress` 不参与奖励。adapter另写 `assembly_reward_contract.json` 记录reward版本/开关/源码哈希，
  原物理状态库契约、成功、晋级、恢复逻辑保持原样，新旧reward可使用同一兼容状态库。
  `scripts/diagnostics/validate_bulb_reward_v2.py` 为有步数上限的原生检查入口。
- `assets/rotate_bulb_assembly_curriculum/pregrasp.json` 保存命名关节模板与资产来源；
  `scripts/diagnostics/build_bulb_curriculum_pregrasp.py`生成，`validate_bulb_curriculum.py`原生验证。
- `scripts/rsl_rl/train_bulb_curriculum.py`→原`train.py`→`bulb_curriculum_training.py`保存/晋级；
  `evaluate_bulb_curriculum.py`冻结评测，`export_bulb_curriculum.py`核对后导出原任务契约。
- 设计及限制见[实施记录](rotate_bulb_curriculum_implementation.md)，实验入口见EXPERIMENTS的ASM-CURRICULUM-IMPL-20260921-03。
