# 标定配置

| 文件 | 职责 |
| --- | --- |
| [坐标转换.yaml](坐标转换.yaml) | 当前人工设备XYZ↔Tag换算说明，mm；完整外参／TCP与接触零点待验收 |
| [coordinate_reference.json](coordinate_reference.json) | URDF版本／哈希、原始平台高度、历史变换及未启用的条件公式；不是自动控制配置 |
| [fingertip_sampling_points.yaml](fingertip_sampling_points.yaml) | 当前Tag系、全零手关节的五指模型预定位点，mm；不是实测接触零点 |
| [simulation/config.json](simulation/config.json) | 机械台CPU模型默认采集矩阵及参数 |
| [simulation/smoke_config.json](simulation/smoke_config.json) | 仿真短测配置；不能作为完整采集矩阵 |

本轮仅归类，五份配置内容与迁移前逐字节相同。特别保留较新Tag原点与历史平台测量的差异，
不把JSON中保留的旧条件公式当作当前已经验证的设备换算。
完整URDF与CPU生成器的默认源装配也可能不是同一几何／世界系版本，按各自manifest核对。

旧 `calibration/` 下三个配置名字与 `simulation/{config,smoke_config}.json` 保留相对符号链接。
仿真脚本无参数时仍可读取默认配置；新命令建议显式传 `--config calibration/config/simulation/config.json`。
