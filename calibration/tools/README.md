# 相机与Marker预处理

- `annotate_vitai_markers.py`：历史ViTai参考图的人工标注工具，运行会打开交互界面并按操作保存标注。
- `project_vitai_markers_to_mesh.py`：将相机／Marker几何映射到传感器曲面；共享可视化器也调用其中函数。
- `marker_roi_live.py`／`marker_roi_live.html`：本机单路实时相机与多边形识别区域标定。仅打开指定 UVC 相机，默认预览约 5 Hz，30 分钟后关闭；不导入手部串口或 ROS。
- `marker_roi.py`：离线 ROI 检测适配器，供 SDK `MarkerTracking(..., find_marker_func=...)` 使用。依据检测连通域的中心筛选并保留完整连通域，不裁切 RGB、不改 SDK。
- `marker_template.py`：独立的原图参考点编号与保守匹配，固定导出 100×5；复用 SDK 检测结果，不调用原生 DFS 跟踪器。
- `marker_training.py`：两仓库共用的真实点固定选取／100行打包；ROS直接传入已保存的SDK DFS坐标与matched，不重新检测或插值。

ROS当前采集到训练的入口见[DFS训练适配](../../../ros_ws/src/py_force_ws/doc/sdk_marker_tracking.md#dfs-到训练数据2026-10-13-更新)。
它复用下述固定编号的输出规则；本页ROI工具的原图近邻匹配器仍是独立工具，不替代ROS原生DFS。
旧400点光流插值转换已禁用；缺少明确DFS来源时转换报错。

## 实时框选

从仓库根目录运行，`--output` 必须是不存在的新目录。此命令会打开食指相机；`--help` 仅解析参数。

```bash
/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python \
  calibration/tools/marker_roi_live.py \
  --serial WTUVL3197X260010B --port 8766 --duration 1800 \
  --output calibration/captures/NEW_MARKER_ROI
```

浏览器打开 `http://127.0.0.1:8766`。冻结当前画面后拖矩形或逐点画包含多边形；
另画排除多边形去掉反光／杂点。顶点可拖动，最后一个绘图动作可撤销。
框选坐标使用原始图的 `[x,y]` 像素，图像保持 640×480；页面缩放不改变坐标。
给真实黑点的运动留出边界余量，再切回实时预览检查漏点。保存前需再次冻结当前帧。

“保存选区并核验”写入独立 `selection_NNN/`：`roi.json`、选定帧原始 PNG、
检测连通域掩码、RGB 叠加图、近期最多 60 帧 RGB NPZ 及候选统计／SHA 报告。
各模态来自同一选定帧；近期帧时间为宿主读取完成时间，接触状态保持未确认。
预览期间只保留有限内存帧，未点击保存不会写入用户选区。
“结束预览并关闭相机”释放设备；自动期限或采集失败后需要用新输出目录重启。

## 固定编号与 100 行输出

可在上述命令中加 `--roi path/to/selection_001/roi.json` 复用同相机的选区。
参考点每次会话重新标定，避免相机／胶面移动后沿用旧零位。依次操作：

1. 完成 ROI，保持无接触并点击“从冻结帧生成参考点”。当前帧冻结；上方菜单可删误点或补实际可见漏点（靠近已有检测中心时吸附），支持撤销。
2. 核对参考点；确实无接触时勾选对应复选框，点击“固定编号并启用”。未勾选仍可诊断，但参考接触状态保持未确认。生成独立 `template_NNN/`，保存 `template.json`、`roi.json` 与原始 `reference.png`。
3. 切回实时预览，看匹配／缺失／未匹配候选与有效行数，再冻结并保存。参考启用后 ROI 和 ID 锁定；需要修改时点“重新标定参考点”，旧文件保留。

启用模板后，`selection_NNN/` 额外包含模板、参考原图和 `markers_100.npz`：

| 字段 | 含义 |
| --- | --- |
| `marker[T,100,5]` | 原图像素 `[x0,y0,dx,dy,valid]`，无效行五项全零；未归一化 |
| `reference_xy[N,2]`、`output_ids[100]` | 所有参考原点及固定输出映射；补齐 ID 为 -1 |
| `reference_valid[T,N]`、`current_xy[T,N,2]` | 全部 N 个参考点的观测状态／当前位置；无效位置为零，须读取 valid |
| `matched_candidate_indices[T,N]` | 匹配到本帧原始候选的局部下标，未匹配为 -1 |
| `candidate_xy[sum(K),2]`、`candidate_offsets[T+1]` | 完整可变数量检测候选；第 t 帧取 `[offsets[t]:offsets[t+1]]` |
| `candidate_in_roi[sum(K)]` | 同长度 ROI 判定，区域外候选也保留 |
| `sequence[T]`、`host_read_end_unix_ns[T]` | 与 `recent_frames.npz` 同序的帧号和宿主读取完成时间 |

参考 N<100 时补无效行，不补造位移；N>100 时在建模板时做一次最远点覆盖选取100个，
全部 N 个参考点及候选仍保存。ID 仅在建模板时编号，后续不按坐标或候选顺序重排。
每个参考点的允许位移为 `min(12px, 0.4×最近邻间距)`；互不相交的参考范围保证一对一。
范围内恰好一个候选才有效，零个／多个均置无效；缺失后重新进入范围可恢复原 ID。
这是小位移的保守参考关联，大变形／强遮挡／进入另一点范围的运动未验证，不能作为身份保证。
人工补参考不会生成观测；SDK 后续仍未检测到该点时该行保持无效。

本工具没有改动现有采集器、SDK或模型。新输出是原图真实黑点语义，不能仅凭100×5形状
直接替代既有SDK规则光流点训练数据；与RGB/Depth的warp、尺度和训练契约需另行对齐。
导出使用近期最多60帧的同一批原图与检测；不会插值补点或复制邻点。

离线使用（不打开相机）：

```python
import json
from calibration.tools.marker_roi import load_marker_roi
from calibration.tools.marker_template import FixedMarkerMatcher

roi = load_marker_roi("path/to/selection_001/roi.json")
template = json.load(open("path/to/selection_001/template.json"))
matcher = FixedMarkerMatcher(template, roi)
result = matcher.match(raw_detection_centers, actual_camera_serial, [480, 640])
marker_100 = result["marker"]  # numpy float32 [100,5]
```

CPU回归（无相机／SDK导入／机器人）：

```bash
/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python \
  -m unittest discover -s calibration/tools -p test_marker_template.py -v
```

2026-10-12离线60帧证据：[verification.json](../outputs/marker_template_20261012_validation/verification.json)。
97参考点、3行补齐、有效96–97，原始候选保留且匹配唯一；不代表已验证大变形或实际物理ID。

## 接入检测前筛选

```python
from calibration.tools.marker_roi import load_marker_roi, make_roi_detector
from vitai_marker import MarkerTracking

roi = load_marker_roi("path/to/selection_001/roi.json")
detector = make_roi_detector(roi, camera_serial=actual_camera_serial)
tracker = MarkerTracking(rows, columns, fps, init_image=raw_bgr,
                         find_marker_func=detector)
```

`rows/columns` 必须依据实际选中的规则点阵确认；不从候选数量相等推断正确行列或追踪成功。
`get_marker_vector(area_mask=...)` 是区域结果接口，本次通过 `find_marker_func` 在初始化／匹配之前筛选。
这里的 ROI 属于原始图；若先执行 SDK warp、缩放或局部校正，需把区域映射到对应图像坐标，
不能把原图 ROI 直接交给变换后的检测器。适配器会拒绝相机 SN、尺寸或输入类型不一致。
本次没有把适配器自动接入既有采集器或训练流程；曲面点阵顺序及较大变形追踪仍需另验。

连通域中心和多边形判定参照 [OpenCV 连通域／形状接口](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)。

默认参考数据位于[reference/vitai_4Fingers-320*240](../reference/vitai_4Fingers-320*240/)。
源图、原标注与原相机参数保持；本轮没有重新标注或重新标定。
旧 `scripts/`中的两个入口是兼容链接，共享调用方保持原导入路径。
