# 压力标定、参考与数据验证

本目录从 `scripts/force_map/`整体迁入，模块相对关系保持；旧路径为兼容链接。
字段与参考层级的详细说明见[压力trace说明](../docs/pressure_sdf_tactile.md)。

| 文件组 | 用途 |
| --- | --- |
| `generate_pressure_l0_trace.py`、`verify_pressure_trace.py` | 解析参考trace生成、schema和数据验证 |
| `align_pressure_reference_geometry.py`、`apply_pressure_*reference.py`、`apply_pressure_geometry_normal_ray.py` | 显式几何／法向参考对齐 |
| `fit_pressure_*.py`、`pressure_fit_verify_command.py` | deadband、面积、mask/depth、sample support及spatial footprint拟合与复验命令 |
| `diagnose_pressure_*.py` | 参考和采样边界诊断 |
| `inspect_pressure_urdf.py`、`visualize_pressure_pad_layout.py` | 压力片布局检查和查看 |
| `generate_pressure_watertight_mesh.py`、`add_default_pressure_pads_to_urdf.py`、`color_pressure_pad_usd.py` | 网格／URDF／USD输出工具；执行会写资产，须看具体输出参数 |
| `pressure_calibration_setup.py` | integrated runner共享的压力标定辅助函数，原integrate导入位置保留链接 |
| `official_replay/`、`run_aloha_force_map.py` | 既有回放参考、数据、资产与运行入口；需要Isaac环境，不能当Revo3真实参考力 |

大部分工具为离线处理，部分会启动Isaac／写资产；不要批量执行整个目录或默认所有help均无副作用。
`pressure_raw_n`等历史字段的单位和L0/L1/L2/L3证据等级保持原契约。
