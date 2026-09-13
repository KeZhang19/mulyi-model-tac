# D 形插销与盲孔座

本项目自行建立的参数化 CAD，未使用外部模型。`peg.usd` 与 `socket.usd` 均自包含、米制、Z 轴朝上；每个文件只有默认根节点一个刚体，没有关节。插销是质量 0.1 kg 的自由刚体，孔座是可在 reset 时移动的运动学刚体。

| 参数 | 数值 |
| --- | --- |
| 插入段 | 半径 12 mm，截面 `x²+y²≤12²` 且 `x≤6`，长 35 mm |
| 柱尖倒角 | 1 mm 等距内缩；z=0 截面半径 11 mm、平面 x=5 mm |
| 握持段 | 直径 50 mm、高 50 mm，z=35…85 mm |
| 孔座 | 80×80×50 mm，原点为底面中心 |
| 盲孔 | 深 30 mm，底面 z=20 mm，孔口 z=50 mm |
| 孔截面 | 半径 12.75 mm、平面 x=6.75 mm，圆弧和平面单侧间隙均 0.75 mm |
| 孔口倒角 | z=49…50 mm 等距扩大 1 mm |
| 默认目标深度 | 28 mm，柱尖距孔底 2 mm；握持段肩部距孔口 7 mm |
| 摩擦系数 | 握持段静/动 0.9/0.8；插入段和孔座 0.5/0.4 |
| 接触参数 | contact offset 0.1 mm；rest offset 0 |

`peg.usd` 默认根是 `/DPeg`，引用到 `/Object` 后，完整的单一闭合网格为 `/Object/VisualMesh`；`/Object/TactileSurface` 使用相同顶点和三角面，隐藏且不带碰撞 API，供触觉 SDF 直接绑定。孔座同理。握持段顶面的箭头仅用于显示，指向 D 平面的外法向 +X，排除在触觉和碰撞网格之外；打印件可按 D 平面方向标记顶面。

插销使用 Shaft、Grip 两个明确凸体。孔座使用一个 Bottom 和 132 个 Wall 凸块，每块独立使用 `convexHull`，孔口和盲孔由这些凸块之间的空腔形成。禁止对孔座整体设置凸包，否则孔会被填实。曲面由细分网格逼近，静态验证会检查凸体性、孔腔及 STL/USD/触觉表面一致性。

`cad/peg.step`、`cad/socket.step` 是可编辑实体；`meshes/peg.stl`、`meshes/socket.stl` 是闭合打印网格，**STL 导入单位必须选择毫米**。打印时将握持段顶面朝打印平台、D 段朝上；孔座底面朝平台。0.75 mm 是设计几何间隙，不是打印机补偿值，实物装配前应通过打印样件校准。

从仓库根目录重新生成和检查：

```bash
# 系统 Python 已有 CadQuery 2.7、numpy、trimesh、matplotlib。
python assets/d_peg_insertion/tools/build_assets.py

# 只使用 CPU/本地 USD 库，不启动 Isaac Sim/Kit。
conda run -n brainco python assets/d_peg_insertion/tools/validate_assets.py \
  --report assets/d_peg_insertion/static_validation.json
```

`metadata.json` 给出精确原点、网格相对路径、尺寸、摩擦和网格哈希；`preview.png` 展示形状与配合截面。静态检查不等于已验证稳定预抓或物理插入，任务的 Isaac 物理仿真验收由任务验证脚本单独完成。

独立的三工位接触验收不生成机器人，比较对准、横向偏移 5 mm、偏航 180°。它先检查自由落体，再通过有界外力和姿态 PD 驱动自由插销，不以写入 pose 的方式推进插入。请与其他 GPU 仿真进程串行运行：

```bash
conda run -n brainco python assets/d_peg_insertion/tools/validate_insertion_physics.py \
  --headless --device cuda:0 \
  --report assets/d_peg_insertion/insertion_physics_validation.json
```
