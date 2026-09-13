# 优化抓取仿真状态

这是接触优化后的单帧状态。读取 `simulation_state.json` 查看数据、顺序、单位和中文字段解释。
`scene.mjb` 内含机器人、手、物体、桌面、网格和实际模型参数；请整体移动此目录。
请使用导出时的 MuJoCo **3.10.1**；MJB 不保证跨版本兼容。

## 加载顺序

1. 读取 JSON，按相对路径加载 `scene.mjb`（附带加载器会检查模型 SHA-256 和版本）。
2. 创建 `mujoco.MjData(model)`，写入 `state.qpos/qvel/act/ctrl/time_s`。
3. 调用 `mujoco.mj_forward(model, data)`，得到机械臂、手和物体的世界位姿。
4. 需要物理执行时使用项目的轨迹生成/动态验证流程，配置重力、摩擦和关节控制器。

在项目根目录执行（将路径替换为本目录的实际路径）：

```bash
./run.sh inspect-state /path/to/simulation_state.json
```

其他已安装相同 MuJoCo 版本的 Python 环境：`python load_state.py`。代码中也可调用：

```python
from load_state import load_simulation_state
model, data, metadata = load_simulation_state("simulation_state.json")
```

## 数组顺序

所有索引从 0 开始。关节值单位为弧度，角度值仅辅助阅读。
机械臂 0–6；拇指 7–11；食指 12–15；中指 16–19；无名指 20–23；小指 24–27。
qpos / qvel 的实际下标见下表，跨模型时按名称映射。

| 顺序 | 关节名称 | 含义 | 角度 rad | 角度 deg | qpos 下标 | qvel 下标 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | joint1 | 机械臂第 1 关节 | -0.08229050997 | -4.7148989 | 0 | 0 |
| 1 | joint2 | 机械臂第 2 关节 | 0.8336003965 | 47.761785 | 1 | 1 |
| 2 | joint3 | 机械臂第 3 关节 | 0.09389161738 | 5.3795934 | 2 | 2 |
| 3 | joint4 | 机械臂第 4 关节 | -1.222653034 | -70.052859 | 3 | 3 |
| 4 | joint5 | 机械臂第 5 关节 | -1.168014088 | -66.922278 | 4 | 4 |
| 5 | joint6 | 机械臂第 6 关节 | 1.482473538 | 84.939477 | 5 | 5 |
| 6 | joint7 | 机械臂第 7 关节 | -2.100492007 | -120.34933 | 6 | 6 |
| 7 | right_thumb_CMP_joint | 右手拇指 CMP 关节（模型命名） | 1.641506404 | 94.051389 | 7 | 7 |
| 8 | right_thumb_CMR_joint | 右手拇指 CMR 关节（模型命名） | 1.427211562 | 81.773199 | 8 | 8 |
| 9 | right_thumb_MCP_joint | 右手拇指 MCP 关节（模型命名） | 0.2936687509 | 16.82598 | 9 | 9 |
| 10 | right_thumb_PIP_joint | 右手拇指 PIP 关节（模型命名） | 0.2509249068 | 14.376938 | 10 | 10 |
| 11 | right_thumb_DIP_joint | 右手拇指 DIP 关节（模型命名） | 0.1166308881 | 6.6824576 | 11 | 11 |
| 12 | right_index_MPR_joint | 右手食指 MPR 关节（模型命名） | 0.1401472655 | 8.0298468 | 12 | 12 |
| 13 | right_index_MCP_joint | 右手食指 MCP 关节（模型命名） | 1.14052069 | 65.347022 | 13 | 13 |
| 14 | right_index_PIP_joint | 右手食指 PIP 关节（模型命名） | 0.3458134657 | 19.813652 | 14 | 14 |
| 15 | right_index_DIP_joint | 右手食指 DIP 关节（模型命名） | 0.3362548448 | 19.265983 | 15 | 15 |
| 16 | right_middle_MPR_joint | 右手中指 MPR 关节（模型命名） | -0.03225299279 | -1.8479604 | 16 | 16 |
| 17 | right_middle_MCP_joint | 右手中指 MCP 关节（模型命名） | 1.05430835 | 60.407419 | 17 | 17 |
| 18 | right_middle_PIP_joint | 右手中指 PIP 关节（模型命名） | 0.2946112677 | 16.879982 | 18 | 18 |
| 19 | right_middle_DIP_joint | 右手中指 DIP 关节（模型命名） | 0.311321964 | 17.837435 | 19 | 19 |
| 20 | right_ring_MPR_joint | 右手无名指 MPR 关节（模型命名） | -0.09928886033 | -5.6888326 | 20 | 20 |
| 21 | right_ring_MCP_joint | 右手无名指 MCP 关节（模型命名） | 1.042836281 | 59.750118 | 21 | 21 |
| 22 | right_ring_PIP_joint | 右手无名指 PIP 关节（模型命名） | 0.3402522507 | 19.495018 | 22 | 22 |
| 23 | right_ring_DIP_joint | 右手无名指 DIP 关节（模型命名） | 0.2814556979 | 16.126224 | 23 | 23 |
| 24 | right_little_MPR_joint | 右手小指 MPR 关节（模型命名） | -0.1034478638 | -5.927126 | 24 | 24 |
| 25 | right_little_MCP_joint | 右手小指 MCP 关节（模型命名） | 1.233662373 | 70.683647 | 25 | 25 |
| 26 | right_little_PIP_joint | 右手小指 PIP 关节（模型命名） | 0.3173953561 | 18.185414 | 26 | 26 |
| 27 | right_little_DIP_joint | 右手小指 DIP 关节（模型命名） | 0.2431312817 | 13.930396 | 27 | 27 |

物体 qpos 下标：[28, 29, 30, 31, 32, 33, 34]，顺序为 `[x,y,z,qw,qx,qy,qz]`（位置为米）。
物体 qvel 下标：[28, 29, 30, 31, 32, 33]，顺序为 `[vx,vy,vz,wx,wy,wz]`；线速度为世界系 m/s，角速度为物体系 rad/s。

## 字段含义

| 字段 | 含义 |
| --- | --- |
| model | scene.mjb 是含机器人、桌面、物体及碰撞网格的完整编译模型；相对本 JSON 定位，无需原始资源路径。 |
| robot.joint_names / joint_positions_rad | 两个数组按相同下标一一对应；0–6 为机械臂，7–27 为手指。 |
| robot.joints | 逐关节给出顺序、名称、中文含义、弧度/角度值、关节限位和 MuJoCo 数组索引。 |
| robot.joints[].axis_in_body | 关节所在 body 局部坐标系的转轴单位向量；正角度遵循右手规则。 |
| robot.base / flange / wrist / links | 机械臂固定底座、法兰、手根及各连杆的世界位姿，由当前关节正向运动学计算。 |
| *.position_m / quaternion_wxyz | body 或 geom 原点在世界系的位置，和把局部向量旋转到世界系的单位四元数。 |
| *.rpy_xyz_rad / rpy_xyz_deg | 辅助阅读的固定轴 XYZ 欧拉角 [roll,pitch,yaw]，R=Rz(yaw) Ry(pitch) Rx(roll)；加载以四元数为准。 |
| object | 物体根 body 原点的世界位姿（不是质心），自由关节位置/速度索引、网格缩放和质量。 |
| object.source_assets | 原始资源路径仅用于追溯，包内路径相对项目根目录；重载 scene.mjb 不使用这些路径。 |
| object.mesh_scale_xyz | 原始物体网格沿 XYZ 的缩放；scene.mjb 中已应用，不要重复缩放。 |
| table | 桌面长方体中心的世界位姿、完整 XYZ 尺寸及顶面中心；size_m 不是 MuJoCo 半尺寸。 |
| fixtures | 固定支架/障碍物的名称、世界位姿和来源；已写入 scene.mjb，无自由关节，不占 qpos/qvel。 |
| assembly | 源任务、旋入圈数、螺距等装配记录；source_configuration 保留导入时数据，object_pose_in_fixture 为当前实际相对位姿。 |
| contact_pairs | 按标注顺序编号，从 0 开始；hand/object 为两端，局部点/法向属于各自 body。 |
| contact_pairs[].*.point_local_m / normal_local | 已缩放 body 局部坐标的表面锚点（米）与单位法向；不是原始未缩放网格坐标。 |
| contact_pairs[].*.point_world_m / normal_world | 世界系锚点和法向；p_world=R_body p_local+t_body，n_world=R_body n_local。 |
| contact_pairs[].distance_m | 优化后两个世界锚点的欧氏距离；标注为目标接触，不等于物理接触力。 |
| state | 单帧静态初始化：qpos 为优化姿态，time_s=0，qvel/act/ctrl=0，其余 MjData 保持模型默认值；不是动态运行中断点。 |
| state.qpos_layout / qvel_layout | 与 MuJoCo qpos/qvel 完整数组逐元素对齐；index 从 0 开始，name 和 unit 解释每一项。 |
| physics | 实际编辑/优化模型参数；重力开关、接触维度等原样记录，不自动切换成动态抓取模型。 |
| physics.bodies | body 质量、主惯量及惯性系相对 body 的位姿；惯量分量位于该惯性坐标系。 |
| physics.geoms | 碰撞参数按 geom_id 顺序记录；friction 顺序为滑动、扭转、滚动，condim=1 时只有法向无摩擦。 |
| validation | 保存实际优化报告；pose_candidate_pass 仅为姿态候选检查，轨迹/动态 lift 未验证。 |

## 本次结果

求解收敛：`True`；姿态候选通过：`False`。
最大标注接触距离：`0.018426749207259528` m；最大穿透：`0.0037981803498850584` m。
轨迹和动态抬升尚未验证；失败/未收敛的优化结果也会保留原状态供检查。
导出的编辑模型关闭重力，接触维度为 1（只有法向），没有关节执行器。
此包可复现优化姿态；直接 mj_step 不代表机器人会保持该姿态或抓起物体。

## 物体与固定支架

固定支架（d_socket）的世界位姿见 `fixtures`；peg 的世界位姿见 `object`，实际相对位姿见 `assembly.object_pose_in_fixture_xyz_wxyz`。
支架固定，peg 的 7 个 qpos 仍表示自由体位姿；装配约束和旋转/轴向关节没有加入本 MuJoCo 抓取姿态模型。
机器人与插销、机器人与支架、插销与插孔均启用碰撞；仅用于静态抓取姿态编辑。
源任务的装配参数记录在 `assembly.source_configuration`，不属于 MuJoCo qpos 数组。
