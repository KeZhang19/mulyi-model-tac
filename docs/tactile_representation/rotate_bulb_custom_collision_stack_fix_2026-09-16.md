# Custom 灯泡任务碰撞栈修复（2026-09-16）

目标服务器：`dsw-715zhrv7l6bux2ldbl`，项目根目录：
`/home/admin/workspace/xinyu/mulyi-model-tac`。

## 原因与修复

旧运行 `20260913T103730Z_nomodal_4096_seed42_dual_gpu` 每卡 4096 环境，
启动参数将碰撞栈固定为 1 GiB。9 月 16 日读取的最近约 10 MB 日志中，
PhysX 报告的所需容量最高为 1,251,133,104 字节，持续提示接触被丢弃。
这是碰撞缓冲容量不足；增加容量不会修复旧训练样本或证明成功率改善。

Custom 任务现在通过 `environment.make_rotate_bulb_env` 创建原有
`ManagerBasedRLEnv`。工厂在 CLI/Hydra 配置覆盖完成后、仿真分配前计算：

```text
容量 = max(256 MiB, 每进程环境数 × 512 KiB, 显式配置容量)
```

每卡 4096 环境得到 2 GiB，超过本次观察到的峰值约 72%；小规模回放保留
256 MiB 的基线。旧的 1 GiB 启动参数会被自动提升，更大的显式配置会被保留。
超过 PhysX uint32 容量范围的配置在创建仿真前报错。
这是根据已观察负载留出的余量，并非所有未来接触状态的容量保证。

新增 `scripts/rsl_rl/launch_rotate_bulb_custom_dual_gpu.sh`，提供双卡、每卡
4096 环境、2 GiB 碰撞栈及原运行参数；默认从零训练，并接受末尾的显式覆盖项。
任务奖励、动作尺度、观测、回合长度和优化器参数没有修改。

最初尝试固定提高所有场景默认容量，本地 8 GB 显卡出现分配失败，因此最终采用
上述按环境数量分配的实现。失败尝试未同步到服务器。

## 验证

- 本地与服务器各有 68 项相关检查通过，包括旧 1 GiB 参数覆盖、容量边界、
  任务注册、接触奖励、回合重置及触觉开关。
- 本地真实 GPU 仿真：2 环境、120 控制步、256 MiB 实际场景属性；完成强制
  超时重置，观测与奖励有限，检测到手指接触，日志无 PhysX/CUDA 错误。
- 服务器真实 GPU 仿真：2 环境、120 控制步，验证 2 GiB 实际场景属性与分配，
  同样通过接触、数值与强制重置检查，日志无 PhysX/CUDA 错误。验证于北京时间
  15:59 完成；这不替代重启后的双卡 4096 环境完整负载验收。
- 启动脚本语法及参数检查通过：双 worker、4096 环境、2 GiB、10 项共享手指网格。
- 广泛资产检查中的旧失败 `test_asset_swap_retains_original_actions_observation_definitions_and_rewards`
  已在未修改的 HEAD `5f6bb48` 上独立复现：Custom 的 `RotateBulbStableSuccess`
  早已不同于原版，但该测试仍要求二者相同。该既有失败未在本次修改中绕过或更改。

GitNexus 在本地仓库重建索引后，配置函数的上游为两个 Custom 配置入口，风险 LOW；
注册文件的导入影响为 3 个文件，风险 LOW。工厂函数通过 Gym 字符串入口动态调用，
图返回 UNKNOWN，已通过注册文本、注册回归和真实 `gym.make` 调用核实。
图变更检查为 LOW、未列出受影响执行流程；新文件另行核查，未将零流程视为无调用。
本次没有 Git 提交。

## 同步与生效

同步前校验服务器已有文件与本地修改前版本一致，随后备份并逐文件原子替换，
同步后逐文件 SHA-256 一致。备份目录：

```text
/home/admin/workspace/xinyu/.codex-sync-backups/bulb_collision_stack_20260916T075647Z
```

修复同步完成时（15:59），旧训练尚未重启。随后用户要求停止旧双卡任务，将两张卡
分别用于基础方块和 Custom 灯泡；现已按此安排从零启动。GPU 1 的灯泡为 4096
环境、2 GiB 碰撞栈，启动验收已通过。旧模型和日志保留，未用于新策略初始化。
详见[单卡独立训练记录](split_cube_bulb_single_gpu_2026-09-16.md)。

验证与同步记录：`logs/diagnostics/rotate_bulb_collision_stack_20260916/`。
容量单位与分配依据参见 [PhysX GPU 仿真文档](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/GPURigidBodies.html)。
