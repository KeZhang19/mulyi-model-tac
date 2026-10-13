# 标定文档

- [Tag系可视化仿真采集](visual_collection.md)：坐标和ROS软件行程来源、Isaac交互／有限CPU入口、故障保存与新数据schema。
- [本地标定流程](local_workflow.md)：当前工作顺序、可用入口与未完成项，开始工作先读这里。
- [真机采集数据接口](real_data_interface.md)：仿真—真机对齐的数据字段、CLI、参考会话、单位、训练manifest、规范配对及待实现转换契约。
- [机械标定台仿真搭建](机械标定台仿真搭建_README.md)：完整设计、几何推导、SDK读取历史与标定数据契约；按章节日期阅读。
- [压力机原始流程](标定流程.md)：原流程文本，保留历史，不代表联合控制已经实现。
- [压力trace与参考层级](pressure_sdf_tactile.md)：L0–L3证据、trace字段和拟合历史。

ROS 2字段及停止／时钟语义只在[硬件接口说明](../hardware/README.md)维护；
可执行命令与依赖分别见[仿真说明](../simulation/README.md)、[硬件说明](../hardware/README.md)和
[项目运行手册](../../docs/RUNBOOK.md)。
