## 2026-10-07 标定台接口

标定台 SDK 已在 `ros_ws/src/py_force_ws` 核对；实际 ROS 2 接口、单位和停止语义见
[真机接口说明](hardware/README.md#标定台-ros-2-接口)。旧文档“SDK待提供”已过时；联合采集与坐标／时钟对齐尚未验收。

# Revo3标定硬件入口

完整接口、实测结果、采集字段和注意事项统一维护在
[机械标定台仿真搭建说明，第7节](../机械标定台仿真搭建_README.md#7-revo3-sdk与真机只读采集2026-10-03)。

- `brainco-revo3-sdk/`：SDK源码／示例副本；`runtime/python/`含bc-revo3-sdk 2.1.0和pyvitaisdk 1.0.17原生包、模型和许可证。
- `sdk_copy_manifest.json`：676文件的来源、排除项、字节数、SHA256；不把原目录历史图像当本次证据。
- `probe_revo3_readonly.py`：有限次只读探测，无手部运动或设备配置写入；相机设置请求的视频格式。
- `captures/revo3_readonly_20261003_final/`：本次最终报告、21关节与阵列快照、5路40帧图像、独立核查。

从仓库根目录执行（目录必须全新）：

```bash
/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python \
  calibration/probe_revo3_readonly.py --output calibration/captures/new_probe
```

优先加载复制的原生包，其余依赖仍来自源SDK已有Python环境；没有复制或新建虚拟环境。
`--help`为离线帮助，实际执行会打开指定串口和ViTai相机。
默认从站127、5 Mbps、第一路BrainCo稳定串口路径，`--port`和`--slave-id`可显式指定。
默认5路相机、每路8帧；`--frames`范围2–60。不要与其它设备控制／相机程序抢占同一端口。
退出码0只表示读取完成；仍需查看模块状态（本次拇指指腹Disabled）和健康原值。

首次`revo3_readonly_20261003_1604/`保留SDK枚举值序列化递归失败；
`..._1606/`为中间修复验证，最终目录补齐枚举名称与有效范围说明；三次均无运动。
所有相机采样均为逐路短测；没有跨相机同步、深度／marker推理验收或真机坐标对齐。
