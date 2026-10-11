## 2026-10-07 标定台接口

标定台 SDK 已在 `ros_ws/src/py_force_ws` 核对；实际 ROS 2 接口、单位和停止语义见
[真机接口说明](hardware/README.md#标定台-ros-2-接口)。旧文档“SDK待提供”已过时；联合采集与坐标／时钟对齐尚未验收。

## 2026-10-05 当前支架坐标系

默认完整URDF仅保留 `tag_center` 作为支架世界根，已移除lower_mount和平台/球槽等辅助link。
手和压头结构/世界位置保持；坐标换算统一见[坐标转换.yaml](坐标转换.yaml)。
下文旧lower_mount/calibration_world关系为历史模型说明；当前验证见[包说明](assets/calibration_bench_visuotactile_urdf/README.md)。

# 标定与数据采集总入口

标定方案、机械台仿真、真机采集、压力拟合、相机参考和已有采样证据统一从本目录进入。
2026-10-04完成目录归类；迁移不代表新增物理标定结果。五路三模态仍只有诊断短测，尚无受控标定结论。

先读[机械标定框架](机械标定台仿真搭建_README.md)，再按用途选择入口。

**后续坐标关系的默认基准已由用户指定：**
`assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf`，
对应本目录的[实际URDF](assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf)。
用户确认右侧平台的**平面基面（避开圆形槽和贴码槽）**的机械台显示Z为 **−308.5605 mm**，作为平台高度基准，不能当作手指接触零点。
当前以Tag槽中心为世界原点、沿用原世界轴向，`calibration_plane`世界Z为 **0.0005 m**（支架局部仍为0.036 m）。
新增`tag_center`与`calibration_world`重合，码纸厚度待实测；控制器轴正方向及TCP对应核对后使用换算公式。
统一记录见[coordinate_reference.json](coordinate_reference.json)及主说明的“默认坐标基准”段。
登记文件不会直接写入机械台。当前完整URDF已改用Tag原点；历史NPZ/仿真输出仍保留原世界坐标，读取时需按版本转换。

五指首轮采样预定位点见 [fingertip_sampling_points.yaml](fingertip_sampling_points.yaml)：
当前Tag系、单位mm、全零手关节姿态，每个DIP触觉面中央附近1点，压头尖端统一Z=40mm。
先抬高并移动XY，再沿-Z下压；文件中的模型表面高度仅供参考，不是球压头接触零点或真机轴指令。

[压力机原始流程](docs/标定流程.md)保留原文；硬件能力、坐标、时钟和验收条件以框架第8–14节为准。

## 目录分工

| 目录／文件 | 内容 | 性质 |
| --- | --- | --- |
| [simulation/](simulation/README.md) | 原 `scripts/calibration_bench/`：配置、轨迹、触觉模型、资产生成、URDF导出、验证与Isaac回放 | 独立标定仿真源码；非已标定动力学 |
| [hardware/](hardware/README.md) | Revo3只读探测、本机启动器、离线RGB／Depth／Marker推理 | 真机短采与离线诊断入口 |
| [collection/](collection/README.md) | 三模态NPZ采集、mmap缓存、RL球探针压力采集 | 采集与数据转换；部分入口需要Isaac／GPU |
| [pressure/](pressure/README.md) | 压力trace生成／对齐／验证、响应拟合、网格与布局工具、官方回放参考 | 原 `scripts/force_map/`，保留内部模块关系 |
| [tools/](tools/README.md) | ViTai marker人工标注、相机点到曲面投影 | 几何预处理；不发送真机运动 |
| [image_generator/](image_generator/) | TacMap深度／RGB样本生成及配套BResNet训练 | 合成图像数据工具；配套网络随工具一起保留 |
| [assets/](assets/README.md) | 压头、V4／V5支架、完整XYZ URDF包 | 本地CAD／mesh／URDF，未改几何 |
| [reference/](reference/README.md) | `vitai_4Fingers-320*240/`的历史图像、内参与marker标注 | 历史相机参考，不是本次同步采样 |
| [captures/](captures/) | Revo3原始读取与三模态诊断历史数据 | 原始证据，保持文件字节与目录名 |
| [outputs/simulation/](outputs/simulation/) | 原 `outputs/calibration_bench/`的场景、回合、验证和失败记录 | 历史仿真证据，未重新生成 |
| [datasets/](datasets/README.md) | 后续三模态NPZ分片及mmap缓存 | 新采集默认输出；当前不表示已有正式数据集 |
| [docs/](docs/) | 压力机流程、压力trace／参考层级说明 | 专题文档 |
| [brainco-revo3-sdk/](brainco-revo3-sdk/README.md) | SDK源码、原生运行包、模型和许可证副本 | 保持厂商内部目录，不重装／改写SDK |
| [sdk_copy_manifest.json](sdk_copy_manifest.json) | 原SDK复制来源及SHA256 | 原始溯源清单，内容不变 |
| [shared/](shared/README.md) | 指向训练包、传感器运行库、共享手资产和可视化器的索引 | 共享实现留在原包；这里不复制或拆分训练代码 |
| [organization_manifest.json](organization_manifest.json) | 旧路径→新路径及共享索引清单 | 目录迁移契约 |

## 常用入口

以下命令从**仓库根目录**执行；读命令旁的副作用说明后再使用，整理目录本身没有执行硬件采集。
新输出目录必须不存在。SDK环境仍依赖原 `/home/liuxinyu/workspace/brainco-revo3-sdk/.venv`，
仿真环境仍使用已有 `brainco` Conda；没有因迁移变成独立安装包。

```bash
# 离线预览仿真规模：不初始化Isaac/GPU，不写数据
/home/liuxinyu/miniconda3/envs/brainco/bin/python calibration/simulation/run.py --plan-only

# CPU生成一个仿真回合：写NPZ、场景和图，不连接硬件
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/liuxinyu/miniconda3/envs/brainco/bin/python \
  calibration/simulation/run.py --episodes 1 --output calibration/outputs/simulation/my_new_run

# 检查上述输出；Isaac回放入口及副作用见simulation/README.md
/home/liuxinyu/miniconda3/envs/brainco/bin/python \
  calibration/simulation/verify.py calibration/outputs/simulation/my_new_run

# 真机只读短采：会打开手部串口和相机，无运动命令
bash calibration/hardware/probe_local.sh --output calibration/captures/my_new_probe

# 对上述图像离线推理：不打开设备；空载未确认时必须用诊断标签
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python \
  calibration/hardware/probe_revo3_trimodal.py \
  --capture calibration/captures/my_new_probe \
  --output calibration/captures/my_new_trimodal --reference-state unverified-current
```

三模态训练数据采集入口为 `calibration/collection/collect_index_cross_modal.py`，
不是机械台 `simulation/run.py` 的同一schema；mmap入口为 `calibration/collection/prepare_mmap_cache.py`。
原训练器仍在 `scripts/tactile_representation/`；本次没有调整训练、任务物理、奖励、模型或成功判据。

## 兼容路径、原始证据与迁移验证

旧 `scripts/calibration_bench`、`scripts/force_map`、原采集脚本、压头／支架资产目录、
`outputs/calibration_bench`及历史相机目录现在是**相对符号链接**，实际文件只有calibration下这一份。
根目录机械README也仅为兼容链接。历史日志中保存的旧路径、任务中的旧引用继续可解析；
新操作优先使用本目录入口。未修改历史NPZ／JSON／USD中的命令、源哈希或时间戳。

训练包内 `source/BrainCo_DexHand`、共享 `integrate`／`tacmap`仍维持导入布局，
相关标定与采集组件在shared中集中索引；任务专属录像、RL评测、训练日志与检查点不混入标定原始数据。
这是同一仓库内的目录整理，`calibration/`仍依赖这些共享库，不能单独复制它就宣称环境完整。
跨机器携带时保留符号链接，并携带所需共享目录、SDK运行环境和忽略的大文件。

`assets/`、`captures/`、`outputs/`和新生成的datasets继续保留本地忽略策略；历史相机参考中原已跟踪的
图像／标注迁移后仍可纳入版本管理。本次不自动git add／提交，也不上传数据。
迁移核验报告见[organization_verification.json](outputs/organization_20261004/organization_verification.json)；
它证明文件和入口检查范围，不代表硬件、完整Isaac动力学或任务成功率验收。

当前限制：压力L0工具在现有解释器下缺 `isaaclab.utils`，迁移前源码也同样失败，未为整理目录安装环境。
资产包中4份历史marker报告有12个原已失效的相对链接，保留原报告作为证据；新导航不依赖这些链接。
