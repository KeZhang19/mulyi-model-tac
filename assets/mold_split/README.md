# 上下模具：保留 CAD 坐标的两部件 URDF

来源为仓库根目录 `1.stp`，本目录仅新增独立资产，不接入已有训练任务。
本版交付几何与单自由度开合关系；未标定材料、质量、驱动器或接触动力学。

## 使用入口

- [mold.urdf](generated/mold.urdf)：`lower_mold` → `mold_opening` → `upper_mold`。
- [交互预览](generated/preview.html)：直接用浏览器打开，离线可用；拖动旋转、滑块开合。
- [精确下模](generated/cad/lower_mold.step)、[精确上模](generated/cad/upper_mold.step)：毫米，保留原 CAD 坐标。
- [下模 BREP](generated/cad/lower_mold.brep)、[上模 BREP](generated/cad/upper_mold.brep)：直接保存导入后的解析实体，避免 STEP 重导出变换。
- [附加参考几何](generated/cad/unassigned_reference.step)：三个微小孤立实体及十五个额外曲面壳。
- [原文件归档](generated/source/original.stp)：原始文件逐字节副本。
- [来源与网格检查](generated/manifest.json)：源文件/资产哈希、实体清单、平面证据、图层、调色板与网格指标。

复制或移动时保留整个 `generated/` 目录：URDF 使用相对路径引用 `meshes/`。

## 划分与坐标

导入后五个实体中的两个大实体为上下模：零起始实体编号 0 为上模、2 为下模。
不执行重新建模、布尔融合、自动修补、封孔、简化或重新居中。
余下三个微小实体没有可信的装配归属；与十五个曲面壳一并保留在参考 STEP，
不随意绑定到任一运动 link。原始层归属仍完整保留于原文件及 manifest 的源语句中。

主要贴合面来自下模 face 2 和上模 face 24（OCCT 零起始面序号）：

```text
开模单位法向 n = (-0.0192123072704608, -0.00527192991744921, 0.999801527306441)
平面方程（毫米）：n · p = -6.866377852027004
相对于原 CAD +Z 倾角：约 1.14°
共面重合面积：约 20343.2 mm²
```

下模 link 原点和方向保留原 CAD 坐标系；上模/关节原点选在贴合面上离 CAD 原点最近的点，
轴方向仍与 CAD 坐标平行。上模 visual/collision 使用等量反向平移抵消关节原点偏移。
因此 `q=0` 时，每个网格顶点均位于原 STEP 的闭合位置；`q>0` 时，上模平移 `q*n` 米，
下模不动。没有把倾斜贴合面强行改成水平面。URDF 自身不添加 world link；导入仿真时固定下模基座。

`mold_opening` 为 `prismatic`，默认范围 **0–0.15 m**。150 mm 是可修改的建模行程，
不是原文件提供的机械行程；`effort=100 N`、`velocity=0.05 m/s` 同样只是满足 URDF 定义的
可修改参数，不表示真实执行器能力。需要独立移动上下模时，应另行修改场景约束。

## 信息保留和物理边界

原始 STEP 是最高保真来源。拆出的 STEP 保留解析曲面和实体几何，可能由导出器重编号/重写文件头，
不能替代原文件的图层、原始实体编号等元数据。所有原始信息仍可从逐字节归档恢复。
本次 STEP 重读发现下模/上模体积分别约差 -2.50/-1.84 mm³（约 2.06/1.30 ppm），
因此没有把重导出 STEP 宣称为数值无损副本；另提供直接保存原始导入实体的 BREP 分件。
STL 为三角网格近似：使用 **0.02 mm 绝对线性离散参数、0.08 rad 角度参数**，无二次减面；
这属于网格生成设置，不是对全曲面最大误差的独立测量。STL 顶点已换算成米，URDF `scale=1`。

原文件有调色板，但没有 `STYLED_ITEM` 的颜色绑定；XDE 导入也未发现实体/面颜色。
URDF/预览中的蓝色上模、灰色下模只是识别色，不能视为原材料颜色。

URDF 的 collision 引用完整主体网格，保留型腔和孔；没有生成会填平型腔的整体凸包。
本次上模网格闭合，下模网格有 **7 条开边**，朝向一致且无非流形共边；未做补洞或修模。
减小离散参数、改用 Delabella 和调整最小单元均未消除缺口，记录保留于
[meshing_attempts.json](diagnostics/meshing_attempts.json)。因此 collision 当前**未通过闭合验收**，
仅可把本交付作为几何/运动版；后续应对照保留的 CAD 定位修复后再验收碰撞。
**URDF 无法指定 PhysX 的凹形碰撞实现。** 在 Isaac Sim 中使用前需检查导入结果，
为上模选择适用的 SDF 或经验证的凸分解等碰撞表示，并检查型腔、导向间隙和接触。
不要把默认导入的 convex hull 当作已保留了凹槽。

材料、密度和实测质量未知，因此本版不写 `<inertial>`；导入器可能自行生成默认惯性，
这些默认值并未得到本资产验证。当前只验证几何、URDF 结构及离散开模位置，未进行 Isaac/PhysX
动力学、全行程连续碰撞或真实模具开合验证。

## 复现

在仓库根目录，用已安装 CadQuery 2.7 / NumPy / trimesh 的系统 Python 运行（CPU）：

```bash
python3 assets/mold_split/build_mold_urdf.py --output /tmp/mold_split_new
```

输出目录必须不存在，脚本拒绝覆盖。可通过 `--source` 指定归档 STEP，
通过 `--stroke-m`、`--effort-n`、`--velocity-m-s` 修改建模参数。
`--linear-mm` 和 `--angular-rad` 控制网格生成参数。脚本针对本次已检查的 5 实体/20 壳/860 面来源，
不声称是任意 STEP 装配的自动 URDF 转换器。

离散开模距离检查使用原始 BRep 主体，对上模施加 `q*n` 后调用 OCCT 最短距离；
记录见 `generated/validation.json`。距离为正表示该采样位置两个主体分离，不代表已覆盖连续行程，
也不包含未归属的参考几何。
