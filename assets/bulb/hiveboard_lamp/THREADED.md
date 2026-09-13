# HiveBoard 可旋入台灯

入口文件：`hiveboard_lamp_threaded.usd`。

这是新增的可旋入 / 可旋出版本，原来的 `hiveboard_lamp_locked.usd` 不作修改。
USD 自包含，加载不需要原来的 Dexonomy 路径，也不需要加载外部 OBJ/STL。

## 实现与边界

- 内螺纹、灯泡外螺纹的可见网格均保留：逐顶点、逐面索引对照原始 USD 验证。
  原始装配的 `0.001` 缩放被计入，顶点统一烘焙为米，Z-up。
- 灯座 / 大底座固定；灯泡及其螺口、两片外壳属于同一个刚体，彼此不会散开。
- 旋转关节与直线关节通过 USD 内置的 PhysX Mimic 约束联动。
  只驱动灯泡的旋转，不逐帧设置平移或传送刚体。
- 导程为 **6 mm / 圈**，完整行程 **24 mm / 4 圈**，支持反转退出。
  6 mm 来自模型螺纹截面估计，不代表标准 E27 或经过认证的机械尺寸。
- 内螺纹使用静态三角网格碰撞；外螺纹碰撞使用包内完整、闭合的
  `source/STL/Lamp_Screw.stl`，SDF 分辨率 512。没有用凸包填平螺口。
- 这是**轴线已经对齐、预啮合的导向旋入版本**。进给由原生约束保证，
  不是仅靠摩擦接触自然求解导程；不支持完全脱离后自由找孔、找牙。

碰撞细节：原始入口有局部干涉，故只在**内螺纹碰撞副本的入口 1 mm**
增加导入倒角，入口最小半径 13.85 mm；原始可见网格不变，中段螺纹不变。
仅调整 120 个碰撞顶点，具体最大调整量记录在 `threaded_metadata.json`。
外螺纹没有缩小；双方 `restOffset=0`，`contactOffset=0.1 mm`。
完全旋入位置距原始 CAD 零位保留 0.3 mm 轴向间隙，避免端面干涉。
灯泡初始绕 X 轴转过 180°，使外螺纹牙顶对准内螺纹牙槽。

## 运行

从 `/home/liuxinyu/workspace/mulyi-model-tac` 运行，使用已安装的
`brainco` 环境（Isaac Sim 5.1 / Isaac Lab，CUDA GPU）。

```bash
# 可视演示：完整旋入、旋出，重复三次
conda run --no-capture-output -n brainco python assets/bulb/tools/demo_threaded_lamp.py --cycles 3

# 无窗口验证两轮旋入、旋出
conda run --no-capture-output -n brainco python assets/bulb/tools/demo_threaded_lamp.py --headless --validate --cycles 2 --report assets/bulb/hiveboard_lamp/threaded_validation.json

# 独立验证螺纹碰撞确实参与物理计算
conda run --no-capture-output -n brainco python assets/bulb/tools/demo_threaded_lamp.py --headless --collision-probe --report assets/bulb/hiveboard_lamp/threaded_contact_validation.json
```

碰撞测试仅在临时场景里故意错开螺纹相位、关闭外壳碰撞，只留下内外螺纹碰撞。
通过条件是测到接触力且旋入被阻挡；它不会覆盖正常 USD 的啮合姿态。
正常对齐时允许有间隙，接触力可以为零，不能把零接触力误认为关闭了碰撞。
两个 JSON 报告包含被测试 USD 的 SHA-256，可核对是否对应当前文件。

## 接入 Isaac Lab

按 `tools/demo_threaded_lamp.py` 的 `ArticulationCfg` 加载。
**不要使用锁定版示例里的 `RigidObjectCfg`。**

| 参数 | 含义 |
| --- | --- |
| `Joints/screw_turn` | 灯泡相对灯座旋转，X 轴 |
| `Joints/screw_slide` | 被联动的轴向移动，不需要独立驱动 |
| USD 旋转范围 | `[-1440, 0]` 度；`0` 为预啮合、伸出状态 |
| Isaac Lab 旋转范围 | `[-8*pi, 0]` 弧度 |
| 直线关节范围 | `[-0.024, 0]` 米 |
| 约束关系 | `slide = angle_rad * 0.006 / (2*pi)` |
| 方向 | `+X` 从底座指向灯泡；负角旋入，正向回转退出 |

只对 `screw_turn` 下发平滑目标；示例用 10 秒完成 4 圈。
默认只配置这一个驱动，因此 Isaac Lab 可能提示 `1 != 2`，
这是被动直线关节未配置执行器的提示，不是关节缺失。
需启用 GPU dynamics（SDF 接触需要）、建议步长 `1/240 s`、TGS 求解器，
位置迭代 32、速度迭代 8；完整设置见演示脚本。
USD 不嵌入全局 PhysicsScene，以免引用到现有场景时重复创建物理场景。
默认关闭重力、底座固定；质量、摩擦与驱动值是仿真参数，未做实物标定。

## 重建

```bash
conda run --no-capture-output -n brainco python assets/bulb/tools/build_threaded_lamp.py
```

重建后需重跑上述两项验证。源文件、许可与原锁定版不受影响。
普通 URDF 无法完整表达本版本的跨旋转 / 平移 PhysX 联动与 SDF 碰撞；
本次不提供会丢失这些行为的 URDF 替代品。

原生固定底座与 Mimic 约束配置参考
[NVIDIA Omni Physics articulation 文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html)。
本模型的运动与接触行为以随包验收报告为准。
