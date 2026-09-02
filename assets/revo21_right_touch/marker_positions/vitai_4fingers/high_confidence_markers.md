# Vitai 高可信 marker 筛选

## 运行时规则

RL 与集成可视化只保留同时满足以下条件的 marker：

1. 三维点和表面法线均为有限值，且法线长度非零；
2. `distortion_valid == true`：Brown–Conrady 反解位于已标定的单调有效域内，没有使用 `atan` 外推；
3. `method == "ray_hit"`：相机射线真实命中 rubber 外层表面，没有使用最近表面点回退。

对应布尔表达式为：

```python
high_confidence = finite & distortion_valid & (method == "ray_hit")
```

`marker_positions.npz` 和 `marker_positions.csv` 仍保留全部反解结果，便于复核；筛选发生在运行时可视化入口，不会删除原始记录。

筛选后的 343 个点另存为 [`marker_positions_high_confidence.csv`](marker_positions_high_confidence.csv)。

## 数量

| 手指 | 原始点 | 保留 | 排除 |
|---|---:|---:|---:|
| middle | 100 | 86 | 14 |
| index | 100 | 86 | 14 |
| ring | 100 | 86 | 14 |
| pinky | 100 | 85 | 15 |
| 合计 | 400 | 343 | 57 |

57 个被排除结果中包含：

- 52 个 `atan` 外推结果：同一组 13 个边缘 marker 在四根手指上的投影；
- 6 个最近表面点回退结果；
- 其中 1 个同时属于上述两类，因此去重后为 57 个。

## 运行时入口

- RL 可视化：`scripts/rsl_rl/visualize_rl_tactile_obs.py::calibrated_marker_local_cache`
- 集成点可视化：`integrate/run_integrated_tactile.py::TacMapFingerMarkerPointViz`
- 集成法线可视化：`integrate/run_integrated_tactile.py::TacMapFingerMarkerNormalViz`

高可信点显示为蓝色。原先表示低可信结果的洋红点及其黄色法线不会再创建。

## URDF 可视化

- 六视角：[`high_confidence_markers_on_urdf_multiview.png`](../../../../output/marker_diagnostics/high_confidence_markers_on_urdf_multiview.png)
- 可自由旋转的 PLY：[`high_confidence_markers_on_urdf.ply`](../../../../output/marker_diagnostics/high_confidence_markers_on_urdf.ply)
- `base_link` 坐标：[`high_confidence_markers_on_urdf.csv`](../../../../output/marker_diagnostics/high_confidence_markers_on_urdf.csv)

PLY 中蓝色球的半径为 `0.6 mm`，仅用于提高可见性；球心才是 marker 坐标，没有法向显示偏移。

## 数据来源

- 逐点反解说明：[`inverse_projection_report.md`](inverse_projection_report.md)
- 全量坐标：[`marker_positions.csv`](marker_positions.csv)
- 高可信坐标：[`marker_positions_high_confidence.csv`](marker_positions_high_confidence.csv)
- 机器可读布局：[`marker_positions.npz`](marker_positions.npz)
