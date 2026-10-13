# 本地标定与数据采集

2026-10-12 当前主标定为视触觉 RGB／Depth／Marker、台架位姿与独立标准六维力。
压阻 ADC 暂归入[其他标定](../../ros_ws/src/py_force_ws/doc/other_calibration/README.md)，不参与主流程或默认环境检查。
ROS 操作顺序见[当前主流程](../../ros_ws/src/py_force_ws/doc/标定流程/标定流程.md)。

先读[本地标定流程](docs/local_workflow.md)，再进入对应脚本目录。
当前已具备机械几何、ROS 2 接口说明、Revo3短采与离线三模态处理入口；
台架／相机／测力的联合记录、完整外参和时钟验收仍待完成。
SDK通路正常或仿真可运行不代表受控物理标定完成。

仿真—真机对齐训练的数据接入详见[真机采集数据接口](docs/real_data_interface.md)，
涵盖原始采集、SDK／米制派生、训练字段与配对约定，并区分已有实现和待实现转换。

## 按工作选择入口

| 工作 | 入口 | 输出／边界 |
| --- | --- | --- |
| 确认装配、坐标与预定位点 | [config/](config/README.md)、[当前URDF包](assets/calibration_bench_visuotactile_urdf/README.md) | Tag系／mm；预定位点不是接触零点或可直接下发的轴指令 |
| 核对台架、转台与力传感器接口 | [hardware/README.md](hardware/README.md#标定台-ros-2-接口) | 复用本机ROS工作区；联合采集尚未验收 |
| Revo3短采、空载参考与离线推理 | [hardware/](hardware/README.md#revo3-读取与离线推理) | 新会话写入 `captures/`；采集入口会打开设备 |
| 机械台几何／触觉模型仿真 | [simulation/](simulation/README.md) | 新回合写入 `outputs/simulation/`；独立CPU模型，Isaac回放另行运行 |
| Tag系可视化仿真采集 | [界面、启动与数据说明](docs/visual_collection.md) | 当前M6×16资产／ROS配置行程；Isaac交互或有限CPU采集，模型压力与RGB未标定 |
| 压力trace、对齐、拟合与复验 | [pressure/](pressure/README.md) | 区分L0–L3证据，仿真参考不是实测力真值 |
| 相机Marker标注与曲面投影 | [tools/](tools/README.md)、[reference/](reference/README.md) | 历史参考图与相机参数保持原位 |
| Isaac三模态训练数据采集、mmap | [collection/](collection/README.md) | 数据写入 `datasets/`；与机械台回合及真机原生schema不同 |
| TacMap合成图像与配套网络 | [image_generator/](image_generator/) | 合成工具；正式表示训练器仍在 `scripts/tactile_representation/` |

## 目录职责

```text
calibration/
├── README.md          # 总入口
├── config/            # 坐标、五指预定位点、simulation配置
├── docs/              # 当前流程、机械设计／历史、压力trace说明
├── hardware/          # 本机SDK入口、短采、离线推理与深度单位转换
├── simulation/        # 机械台轨迹、触觉模型、资产生成和验证
├── pressure/          # 压力trace、响应拟合、诊断及官方回放参考
├── collection/        # Isaac训练数据采集与缓存
├── tools/             # 相机／Marker预处理
├── image_generator/   # 合成图像及配套BResNet
├── vendor/            # 厂商SDK副本，内部源码／模型／许可证保持
├── provenance/        # SDK复制和两次目录整理的溯源清单
├── assets/            # 本地机械CAD／mesh／URDF
├── reference/         # 历史相机图像、内参与标注
├── captures/          # 原始真机短采和派生三模态会话
├── outputs/           # 仿真回合、诊断与失败证据
├── datasets/          # 训练NPZ分片和mmap缓存
└── shared/            # 指向仓库共享实现的索引
```

根目录剩余的SDK、配置、机械说明、probe及清单旧名字均为兼容链接，实际文件只有一份。
新文档和操作使用上表入口；不依据根目录旧名字继续新增配置或文档。

## 当前坐标基准

默认完整URDF为
`calibration/assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf`，
支架唯一世界根为 `tag_center`。当前人工换算以
[config/坐标转换.yaml](config/坐标转换.yaml)为阅读入口；Tag原点设备Z记录为 **−308.2 mm**。
模型平台平面为Tag系 **0.5 mm**；当前换算推算其设备Z为 **−307.7 mm**，
与历史平台实测 **−308.5605 mm** 不一致，待实测复核。

[coordinate_reference.json](config/coordinate_reference.json)保留指定URDF哈希、旧平台测量和
未启用的条件公式；本轮不改其内容。[五指采样点](config/fingertip_sampling_points.yaml)
是全零手关节姿态的模型预定位点，模型表面高度不代表真实接触零点。
历史仿真回合可能使用 `lower_mount` 世界系，读取时核对回合自己的 `frames/config/manifest`。

## 离线检查与运行条件

从仓库根目录执行，以下仅预览计划，不初始化Isaac／GPU、连接硬件或生成数据：

```bash
/home/liuxinyu/miniconda3/envs/brainco/bin/python \
  calibration/simulation/run.py --plan-only \
  --config calibration/config/simulation/config.json
```

CPU回合生成、Isaac回放、真机短采和空载／加载命令分别在
[仿真说明](simulation/README.md)、[硬件说明](hardware/README.md)和
[运行手册](../docs/RUNBOOK.md)维护。需要写输出的命令使用不存在的新目录；不要批量运行整个目录。
仿真仍依赖已有 `brainco` Conda；SDK依赖原工作区的 `.venv`，可用 `REVO3_PYTHON`选择兼容解释器。
[厂商副本](vendor/README.md)不是独立安装环境；共享库仍由[shared/](shared/README.md)索引。

## 历史与迁移

[机械设计和历史验证](docs/机械标定台仿真搭建_README.md)保留全部原章节；
[压力机原始流程](docs/标定流程.md)保留原文。当前实现状态以本地流程和硬件接口说明为准。
文件移动和兼容路径见[本次布局清单](provenance/layout_20261011.json)，
早期记录见[溯源入口](provenance/README.md)。

2026-10-11整理不删除历史原始数据、失败记录、CAD、SDK模型或检查点，不改标定数值、训练任务或判据。
`assets/captures/outputs/datasets`沿用本地忽略策略；跨检出需携带必要的本地资产、SDK原生包／模型与运行环境，
并保留符号链接。仅复制 `calibration/`不能保证完整运行。
