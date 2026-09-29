# Dexonomy 灯泡初始抓姿（2026-09-20）

`simulation_state.json` 是编辑器经 Dexonomy 优化后的原样导出，来源和 SHA256 见
`provenance.json`，优化完整记录见 `optimization_report.json`。运行只读取项目内副本，
不需要编辑器或 Dexonomy 安装目录。

适用于 `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0` 及其 PLAY 配置。
`robot_contract.py:ROTATE_GRASP_REFERENCE_PATH` 指向本文件，
`configure_rotate_bulb_scene` 按名称写入机器人 `init_state.joint_pos`。
7 个机械臂关节和 21 个手指关节均以弧度表示；世界位置为米、四元数为 wxyz。
Isaac 的关节顺序与导出顺序不同，不能直接复制位置数组。

启动、完整 reset、局部 reset 均通过 `reset_robot_to_grasp` 恢复关节位置及驱动目标，
速度和力矩目标清零，环境原点只加一次。新旧机械臂根、腕、桌面、灯泡和灯座位姿相同，
优化改变手指角度。灯座位置约 `(-0.72, 0.04, 0.0350422)`，
灯泡位置约 `(-0.72, 0.04, 0.0473422)`；默认旋入 3 圈、螺距 0.006 m，
`screw_turn=-6π`、`screw_slide=-0.018 m`。改变圈数也会改变灯泡相对手的位置。

装配任务由 `bulb_assembly_task=True` 保留旧初始抓姿，D-Peg 仍使用原参考及自己的插销初态。
标准 Rotate-Bulb 不使用此配置。任务原有物理模式、碰撞、奖励、动作和成功判据均沿用。
导出内的 MuJoCo 物理参数和接触策略仅作来源记录，不覆盖 Isaac 物理设置。

在仓库根目录、Isaac Lab 环境中复查：

```bash
python assets/rotate_bulb_custom/validate_grasp_scene.py --headless --num_envs 2 --device cuda:0
```

报告默认写到本目录 `runtime_validation.json`，不会覆盖上一版验证记录。
默认只检查姿态/关节，保留场景和传感器，精简观测且不加载触觉编码器；
`--full-observations` 可额外检查完整观测初始化。

本次实际验证：本机 `brainco` 环境、CUDA 0、2 个环境，**440/440 状态检查通过**，
覆盖 35 个可对应机器人 link，最大位置误差 0.707 微米、最大旋转误差 6.65e-6 rad。
4 个导出参考 link 在 PhysX 刚体清单中不可对应，名称保留在报告中。
离线回归 **46 通过、2 跳过**（未启动 Isaac 的 Python 缺少 `pxr.Usd`）。
命令、代码哈希、边界见 `validation_summary.json`；本次代码改动见 `integration.patch`，
该 patch 基于用户已有未提交改动，不能当成完整仓库快照。

继续使用原 Custom 任务 ID 启动训练即可采用新初态，无需额外模板参数：

```bash
python scripts/rsl_rl/train.py --task BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0 --num_envs 32 --headless
```

这条训练命令是使用说明，本次不启动训练。旧训练进程不会热加载修改；
如用旧保存配置覆盖当前 `scene.robot.init_state`，需确认保存的关节角是否仍为旧模板。

来源模型静态检查：机器人穿透由约 3.857 mm 降到 0，5/5 指接触组通过，
QP 误差约 1.37e-5。原标注锚点最大偏差约 27.18 mm，未通过原锚点对齐。
这些是 MuJoCo/Dexonomy 的结果，不证明 PhysX 中接触无穿透、动态抓持稳定或旋出成功。
