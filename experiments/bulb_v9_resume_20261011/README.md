# c3g 同目录 v9 从早期 v8/model900 续训

2026-10-11用户明确授权在原对应目录同步并挑选早期检查点续训。目标服务器dsw-c3g3e0ieowszisvolv；
目录仍是`/home/admin/workspace/xinyu/mulyi-model-tac-v8-20261007`，目录名保留，但本次运行源码为v9。
同步仅本次任务相关代码/证据，不全量覆盖其他任务；旧v8源码在远端替换前备份，旧日志与检查点保留。

选定原v8五帧MLP `rank0/model_900.pt`：801–900轮stage1完成率逐轮均值15.810991%，
701–800/model800仅2.937382%；900是最近100轮一阶段均值达到10%的最早已保存检查点。
截至900轮stage2晋级指标全部0；首个晋级记录1593。零晋级不能证明没有浅插经历/旧目标偏置。
选择为了减少后期旧目标学习，同时保留刚起量的抓取搬运。来源为rank0重置批次统计，非独立评测。
原模型SHA256 `b761ac46f46831c5a244d4819863546aa2b0cbae990dd834dd8a03359b1f2bf9`；optimizer步骤18020。

`launch.sh`沿用双卡每卡8192/共16384、5帧1900维MLP、28动作、30Hz、原PhysX容量和ADR0。
明确`--migrate-v8-tip-command`跨MDP续训，恢复policy/optimizer/normalizers/gravity，迭代900起追加29100，目标日志编号30000。
不是同协议训练连续性或公平消融。启动前两张GPU空闲、无该目录训练进程。

运行路径：`logs/rsl_rl/bulb_grasp_teacher_v9_mlp/2026-10-11_resume_model900_2x8192`。
已于13:32:39启动launcher4515；[启动配置](launch.json)、[检查点选择](checkpoint_selection.json)、[33项同步/备份](sync_verified.json)。
本地/远端CPU58/58通过。远端首轮缺pxr搜索路径，未通过即阻止启动；补入现有USD库路径后通过，无依赖安装，失败日志保留。
[24项启动验收](startup_verified.json)全部通过：900–911共12次PPO更新、真实DDP且参数SHA一致、第一份新model900 optimizer18040，
相对父模型18020继续更新；actor/critic归一化计数恢复后增长，模型/优化器有限；checkpoint重力状态恢复。
外层及2份Kit日志无Error或CUDA/PhysX容量溢出；机器人已有视觉引用Warning存在。显存34447/35891MiB（验收瞬时）。
原70项checkpoint尺寸/mtime保持，所选model900及最新model6900 SHA不变。训练持续进行，尚无独立成功率或新版录像。
v8备份已下载本地并核对SHA；`verify_startup.py`只读验收，不停止作业。

远端CPU复现需ISAACLAB_ROOT=/home/admin/workspace/xinyu/IsaacLab-d94504bc，已有USD库：
/home/admin/workspace/xinyu/brainco_isaaclab_2_3_2_env/.pixi/envs/default/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311；
将其加入PYTHONPATH，其bin和环境lib加入LD_LIBRARY_PATH，再用该环境Python执行check_bulb_grasp_curriculum.py。
