# 抓灯泡场景参考

`simulation_state.json` 是抓取编辑器导出 `20260912_102202_ed803448` 的原样副本。
`provenance.json` 记录原文件路径与 SHA-256；任务运行只读取本目录的副本。

| 字段 | 用途与含义 |
| --- | --- |
| `robot.base` | 机器人底座相对每个环境原点的位置和朝向 |
| `robot.joint_names` / `joint_positions_rad` | 逐项对应的关节名称与弧度角；前 7 项为机械臂，后 21 项为手指 |
| `robot.links`、`wrist`、`flange` | 各连杆、手腕和法兰的正运动学参考，供校验模型对齐 |
| `object` | 灯泡刚体原点的位置与四元数，不是几何中心或质心 |
| `fixtures` 中的 `lamp_support` | 固定支架原点的位置与四元数 |
| `table` | 桌子完整尺寸、中心、旋转和桌面中心坐标 |
| `assembly` | 灯泡相对支架的变换，以及 3 圈、6 mm 螺距等来源记录 |
| `contact_pairs`、`validation` | 原标注和优化结果；保留未通过姿态质量检查的状态 |

位置为米、角度为弧度，四元数为 **wxyz**。任务将参考世界坐标视为环境局部坐标，写入仿真时加一次环境原点。关节按**名称**赋值；PhysX 动作/关节数组顺序与导出文件不同，不可直接整段复制数组。

目标任务读取机器人根位姿、全部关节角、桌面与支架位姿；灯泡位姿通过原生螺纹状态（默认 `screw_turn=-6π`、`screw_slide=-0.018 m`）恢复。`env_cfg.py` 的场景配置和启动/reset 事件完成应用。

此目录仅归档状态 JSON，未复制 MuJoCo `scene.mjb`。JSON 的 `model`、`state` 和 `physics` 是来源记录，不用来覆盖 Isaac Lab 的物理模型；需要 MuJoCo 复现时使用原完整导出目录。Isaac Lab 保留重力、驱动器、原生螺纹接触/约束及释放机制。

该参考的 `pose_candidate_pass`、`trajectory_validated` 和 `dynamic_lift_validated` 均为 false。场景对齐验证与抓取成功验证是两个独立检查。
