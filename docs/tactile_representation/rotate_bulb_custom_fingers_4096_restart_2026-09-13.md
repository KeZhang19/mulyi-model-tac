# Custom 逐指接触奖励：每卡 4096 环境从零训练

服务器：`dsw-715zhrv7l6bux2ldbl`，hostname：`dsw-949505-64bd8d4dc8-wpvz4`。
项目：`/home/admin/workspace/xinyu/mulyi-model-tac`。
任务：`BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0`。

北京时间 2026-09-13 02:43:43 启动修正后的运行。目标机原 Custom 训练的 supervisor、启动脚本、torchrun
和两个 worker，共 5 个进程已确认退出，旧检查点和日志保留。
另一台 `dsw-lal9ty3huw0odwjjz5` 的训练没有停止或重启。

| 配置 | 本次运行 |
| --- | --- |
| 并行环境 | 双 GPU，每卡 4096，共 8192 |
| RL 策略初始化 | 从零，`agent.resume=false`，不加载旧 RL 检查点 |
| 逐指接触辅助项 | 每指 `0.02 × clip(F / 0.5 N, 0, 1)` 分/秒，五指合计最高 0.1 分/秒 |
| 模态还原观测通路 | 关闭，显式 `env.tactile_policy_enabled=false` |
| 动作幅度 / 变化惩罚 | −0.0001 / −0.0002 |
| 纯物理 / 抓握加权圈数奖励 | 0.5 / 1.5 |
| gamma / lambda / rollout | 0.999 / 0.95 / 32 |
| 初始旋入 / 回合 / 控制频率 | 3 圈 / 20 秒 / 60 Hz |
| seed / 最大迭代 / 保存间隔 | 42 / 15000 / 25 |
| 碰撞栈 | `env.sim.physx.gpu_collision_stack_size=1073741824`，即 1 GiB |
| 手指表面射线网格 | 五指的 TacMap / marker 表面传感器共 10 项 `is_shared=true`，仍独立跟踪各环境姿态 |

停止旧训练前，169 个相关源码、资产、传感器依赖和测试文件已确认与本地哈希一致，
包括上一轮同步的逐指接触改动。此次无需再次覆盖相同源码。服务器相关测试 **184 passed**；
本地所选检查为 120 passed / 2 skipped，缺失依赖对应项目已在服务器测试中通过。
本次环境数量、碰撞栈和手指网格复用通过启动参数覆盖，没有改变共享任务默认值或其他任务源码。
网格复用还通过两环境真实 GPU 仿真检查：五根手指的顶点、面、相对姿态和缩放在环境间一致，
共享 Warp 网格 ID，观测与奖励有限，正常步进及超时重置通过。未更改灯泡目标网格的共享设置。

模型及 TensorBoard 目录：

```text
/home/admin/workspace/xinyu/mulyi-model-tac/logs/rsl_rl/dexsuite_revo3_rotate_bulb_custom/2026-09-12_18-42-34_custom_fingers_tactile_off_4096_scratch_seed42_dual_gpu_stack1g
```

运行控制目录：

```text
/home/admin/workspace/xinyu/mulyi-model-tac/remote_train_logs/rotate_bulb_custom_fingers_4096_restart_20260912T181638Z_retry_shared
```

其中 `train.log` 为训练终端日志，`state.json` 持续记录迭代和状态，`launch_manifest.json`
记录完整命令，`previous_training_stop.json` 记录旧进程退出结果，`preflight.json` 与
`pytest.txt` 保存同步和测试结果。新监督进程 PID 为 73147，首次初始化超时设为 3600 秒，
进入训练后仍为 600 秒无更新判定；数值与物理错误继续监测。

本地记录：`logs/diagnostics/rotate_bulb_custom_fingers_4096_restart_20260912T181638Z_retry_shared/`。

首个尝试在 02:19:19 启动，其首次初始化明显变慢。读取两个 worker 的调用栈确认耗时位于
`MultiMeshRayCaster._initialize_warp_meshes → create_trimesh_from_geom_mesh → trimesh.merge_vertices`。
手指目标网格没有开启 `RaycastTargetCfg.is_shared` 的提前复用，当前路径先逐环境解析，
再按顶点检查重复。这属于启动开销，不能当作每轮 PPO 耗时。
诊断工具仅下载到本次日志目录，没有安装到训练 Python 环境，也没有改动运行中的策略或传感器。

首个尝试在约 20 分钟时进入采样，PhysX 报告原 256 MiB 碰撞栈不足、需要至少 315966096 字节，
并已丢弃接触。错误监测触发停止，两个 GPU 已确认释放，未将该尝试作为有效训练结果。
此前仅为完成正在推进的初始化临时延长观察窗口，仍保持错误检测；检测到 PhysX 错误后立即恢复原
监督进程执行清理。记录在原控制目录 `startup_watchdog_extension.json` 中，该辅助监测进程已结束。

最终重试将碰撞栈增加到 1 GiB，并使用已经验证的手指网格提前复用，奖励及物理求解迭代设置保持原值。
GitNexus 对 Custom 网格配置生成入口的上游分析为 LOW：1 个直接调用、0 个已解析执行流程，
影响限于 Custom 配置模块；最终采用启动配置覆盖，没有编辑该函数。

启动验收已通过，详见控制目录的 `validation.json`：

- 两个 worker 的实际场景均为 4096 环境；观测契约以及 actor / critic 输入均为 669 维。
- 保存配置确认新辅助项权重 0.1、每指力阈值 0.5 N、动作惩罚与 gamma 正确，且 `resume=false`。
- 前三次 PPO 更新每次执行 20 次跨卡梯度同步，两卡参数最大差异均为 0；首次梯度均值独立核验通过。
- `model_0.pt` 的迭代编号为 0、优化器步数均为 20；检查点模型和优化器张量全部有限。
- `Episode_Reward/any_finger_contact` 已有非零记录，旋转圈数与旋转奖励指标均正常写入 TensorBoard。
- 修正后的日志未发现 PhysX / CUDA 错误、碰撞栈溢出或非有限数值。
- 对比失败尝试和重试实际保存的完整配置，差异仅为碰撞栈、10 个手指表面网格共享标志与运行目录/名称。

北京时间 **02:55:15** 最终检查时，训练处于 `running`，已完成 **9 次更新**；
最近 5 次更新的采样加学习耗时中位数为 **10.819 秒/轮**。GPU 显存为
18055 / 17998 MiB（约每卡 17.6 GiB），两卡总显存均为 46068 MiB。
这是启动阶段的测量；后续接触和重置频率变化会影响速度。
重试从进程启动到进入训练约 10 分钟，首次网格初始化开销未计入每轮耗时。
另外一台服务器上的训练仍为 `running`；169 个相关文件在最终检查时仍与本地哈希一致。
