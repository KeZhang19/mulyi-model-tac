# 相机与Marker预处理

- `annotate_vitai_markers.py`：历史ViTai参考图的人工标注工具，运行会打开交互界面并按操作保存标注。
- `project_vitai_markers_to_mesh.py`：将相机／Marker几何映射到传感器曲面；共享可视化器也调用其中函数。

默认参考数据位于[reference/vitai_4Fingers-320*240](../reference/vitai_4Fingers-320*240/)。
源图、原标注与原相机参数保持；本轮没有重新标注或重新标定。
旧 `scripts/`中的两个入口是兼容链接，共享调用方保持原导入路径。
