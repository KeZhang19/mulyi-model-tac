## BULB-MLP-LSTM-COMPARISON-20261007

- 22:26只读wpmt共享存储中的v5 MLP历史5帧/v7单帧LSTM原始rank0事件、保存agent/env配置与外层训练日志；分别读到12158/1015轮，对齐0–1015。两者从零、seed42/43、2×8192环境、rollout32、相同PPO超参数。
- 第一阶段700–799/800–899/900–999均值：MLP为0.4767%/8.4342%/49.9439%，LSTM为4.5404%/11.2141%/19.2279%。共同末100轮916–1015为56.4421%/21.3613%。LSTM更早起量，MLP后续上升更快。
- 900–999采样+学习平均16.1869/17.3905秒；0–1015累计15712.83/17395.06秒。异常终止900–999均值38.5599%/39.6995%，当前未见LSTM明显稳定性优势。
- 两者所有已读stage2/3成功指标最大值0。按轮不加权均值为rank0训练重置批次统计，非独立成功率。奖励版本不同（v7新增二阶段朝向门控和一次+10）、不同机器、单种子对，不能将差异严格归因LSTM；v6续训不纳入从零比较。
- GitNexus文档impact为UNKNOWN：当前文件不在图内，模糊命中旧快照（一个CRITICAL候选与当前文件不符）；已补查README/PROJECT_MAP引用，仅追加分析证据和文档，未改训练代码、未重启、未新开训练。
- [完整统计、远端路径及保存配置](experiments/bulb_mlp_lstm_comparison_20261007/comparison.json)。

## BULB-STAGE2-TIP-ENTRY-20261007

- 用户要求的局部任务修改，当前本地MDP v8；二阶段以实际螺纹端点引导，保留5–6 mm浅插/端点径向≤0.5 mm，倾角改为<30°，删除速度与保持门槛。
- 两项位置奖励通过随当前姿态换算的中心命令衡量尖端误差；保持原核、权重和一次+10；二阶段不要求旋转相位，三阶段原计圈及稳定验收保持。
- CPU课程检查52/52通过，新增高速即时晋级、严格30°边界、端点奖励/倍率、无预旋转和三阶段稳定隔离验证。原生物理/学习效果未验证，未部署远端。
- 保留首次旧断言失败日志及修改前文件；[证据、差异和复现命令](experiments/bulb_stage2_tip_entry_20261007/README.md)。

## BULB-V6-VIDEO-20261007-2030

- 旧v6最终日志10750，最后已保存model10700；按用户要求远端连续32秒回放传回本地，960帧/1280×960/30FPS。
- 原v6 MLP/1900观测/28动作/30Hz/三阶段配置，ADR0与重力−1匹配原训练。独立录像脚本只覆盖ADR初值；运行代码部署SHA均匹配，README有历史变化。
- 首次wpmt GPU0 Vulkan/CUBLAS显存分配失败退出，证据保留；重试c3g GPU0共享同一v6源码/模型，未停止训练或放宽条件。
- 第一阶段548帧、第二阶段412帧；二阶段倾角52.0270°–79.0081°、均值64.7067°，无≤5°帧或第三阶段。7个完整回合最终成功0，第8截断，不作独立成功率结论。
- 10文件SHA一致、960帧完整解码、模型SHA未变；20:30录像进程已退出，两台原训练均在。
- [视频、失败记录索引和验证](recordings/bulb_v6_model10700_20261007_2023_c3g/README.md)。

## TACTSPACE-COMPARISON-REVIEW-20261007

- 类型：用户请求的对比设计/既有证据分析；未启动新实验，未改运行代码。
- 结论：现实现保留 TactSpace 核心联合目标，适合明确标注的 RGB/Depth/Marker 适配表示基线；缺跨域训练及真实下游结果，不支持完整复现或迁移优劣主张。
- 公平性重点：旧模型解码有空间旁路，训练增强/选择损失也不同；应统一最终评测，不能直接比较各自 total loss。
- 推荐补齐：冻结表示公共下游头、相同输入/数据/预算、多种子独立任务评测；sim–real 主张需物理可信配对和跨域联合训练。
- [源码与论文依据、未实施建议、图分析限制](experiments/tactspace_comparison_review_20261007/README.md)。

## BULB-ANGLE-TREND-20261007-2005

- 当前TensorBoard已切换v7 LSTM：0–578，三个相邻100轮混合末态倾角80.2159°→80.4629°→81.0076°，没有持续下降。
- 旧v6原始事件9500–10750，停止前最后三个100轮59.7696°→61.2759°→61.8443°，最后时间17:20:48。
- 当前v7近100轮≤5°比例0.124022%，进入二阶段0.130727%；v6/v7阶段二完成标量均全0。均为rank0混合阶段批次按轮统计，不是二阶段专属/独立评测，不跨版本解释为连续退化。
- 本次只读远端与TensorBoard；仅前置文档记录，沿用本会话文档impact UNKNOWN及引用补查结论，不改运行代码。
- [证据](experiments/bulb_v7_lstm_dsw_2x8192_20261007/angle_trend_20261007_2005.json)。

## BULB-V7-LSTM-SLOWDOWN-20261007：只读耗时归因

- 保存配置确认物理/环境/动作/PPO次数保持；runtime50、灯泡8、机器人55文件SHA一致。
- v6末100轮12.7048s，对当前v7 459–558为17.0629s；学习更新只增加0.39953s，其余在采样侧。
- v5/v7同为400–499轮的历史时间14.963/16.2816s，约+8.8%，不是严格同轨迹消融。
- v7 480–499→540–559采样+2.6168s，学习基本不变；rank0关节超速异常终止26.378%→35.1715%，双rank事件729.75→1072.85/轮。
- 相关性支持排查轨迹/接触/reset；逐步LSTM推理/记忆复制也在collection内，未做分项profiling，不声称根因已完全定位。
- GPU/P0频率和RAM快照无明显资源异常；未干预训练。详见[审计](experiments/bulb_v7_lstm_dsw_2x8192_20261007/slowdown_audit_20261007.json)。

## BULB-MLP-LSTM-ONSET-20261007：旧曲线上升轮数与当前耗时

- 从原始事件核对v5 0–11496、v6 9500–10750、v7 LSTM0–553；无训练干预。
- v5 stage1_success100轮均值700–799=0.4767%、800–899=8.4342%、900–999=49.9439%、1000–1099=71.9462%。
- 明显上升约800轮；滚动100轮均值≥1%/10%/50%并连续30个窗口维持，首次结束轮824/906/1000。
- v5/v6二阶段/三阶段/完整成功所有事件为0；不能把进入二阶段当完整成功。
- 当前LSTM553轮，近100轮stage1_success0.10294%；每轮近100均值16.908s（采样15.88933、学习1.01865），近20均值18.4905s、最新19.47s。
- 网络/奖励版本不同，只作历史参照，不是公平消融或收敛时间保证；指标非独立评测。
- [统计证据](experiments/bulb_v7_lstm_dsw_2x8192_20261007/mlp_lstm_onset_comparison_20261007.json)。

## BULB-V7-LSTM-PROGRESS-20261007-1953：进度与TensorBoard

- 原双卡LSTM作业活动，模型到500、事件到540，近期16.5866秒/轮。
- rank0近100轮stage1_success0.106419%，最初100轮0.104730%；stage2/3、旋入圈数/位置成功完整541点全0。
- mean_reward首100轮6.56278→近100轮8.16275；异常机器人终止27.4303%，非独立评测，不能据奖励上涨声称学会抓取。
- 本机6006→wpmt loopback6008，run Bulb_v7_LSTM_2x8192/.，HTTP200/139标签、浏览器可见核验通过。
- [统计](experiments/bulb_v7_lstm_dsw_2x8192_20261007/progress_20261007_1952.json)、[TB验证](experiments/bulb_v7_lstm_dsw_2x8192_20261007/tensorboard_verified.json)、同目录launch/tunnel记录；未干预训练。

<a id="bulb-v7-lstm-dsw-2x8192-20261007"></a>
## BULB-V7-LSTM-DSW-2X8192-20261007：用户改为双卡各8192

- 用户确认停止旧v6后，17:20 launcher及两rank退出；13份检查点保留，最新model10700 SHA未变。
- 单卡尝试初始化期间依用户新指令取消；17:22:17启动双卡，launcher14243/rank14259、14260。
- GPU0/1各8192、总16384、seed42/43、从零30000更新；单帧380维LSTM256/v7完整三阶段。
- 1733项同步校验、DSW LSTM CPU7/7及课程48/48通过；17:29原生0–10共11次PPO更新，17项验收通过。
- 两rank真实all-reduce/首次参数SHA一致，LSTM380→256及v7配置正确；model0/optimizer有限，step20。
- 外层及两份Kit日志无Error/容量溢出；显存39705/43499MiB。后期容量峰值及独立完整任务成功率未知。
- [运行记录、脚本与验收报告](experiments/bulb_v7_lstm_dsw_2x8192_20261007/README.md)。

<a id="bulb-v7-lstm-dsw-1x8192-20261007"></a>
## BULB-V7-LSTM-DSW-1X8192-20261007：部署/CPU检查，训练尚未启动

- 指定wpmt DSW独立v7-lstm副本，1733/1733项SHA/链接匹配；当前本地修改保留，旧远端项目未覆盖。
- 现有环境CPU LSTM 7/7、完整课程48/48通过；课程检查前两次缺LAB/USD路径失败，补环境后通过，失败日志保留。
- 准备单GPU0/8192、单帧380维/LSTM256、v7三阶段、从零30000更新；尚未执行，无原生/GPU或学习效果结论。
- 两卡均被旧v6双卡作业占用；停止旧作业的确认待用户回复。未改变旧作业。
- [记录/启动脚本/证据](experiments/bulb_v7_lstm_dsw_1x8192_20261007/README.md)。

<a id="bulb-teacher-lstm-20261007"></a>
## BULB-TEACHER-LSTM-20261007：可选单帧循环策略CPU接线

- 用户授权最小化增加命令行参数；本地main/167fc3c，保护已有修改。专用train/play新增`--lstm`，默认MLP保持。
- 输入由5帧1900改为1帧380；Actor/Critic各单层LSTM256接原MLP。共享runtime、任务快照/奖励/物理不改。
- 任务局部适配探索std边界、PPO bootstrap不消耗记忆，回放按环境done清空隐藏状态；旧MLP检查点不兼容。
- CPU PyTorch2.7.0、任务runtime：`scripts/diagnostics/check_bulb_teacher_lstm.py` 7/7通过。
  真runner工厂/循环PPO，8环境×32步×2轮合成观测共512样本，原5epoch×4minibatch共40次优化；
  Actor/Critic LSTM和MLP梯度/权重更新、局部reset、噪声、严格保存加载和CLI检查通过。
- [日志](experiments/bulb_teacher_lstm_20261007/cpu_checks.log)、[修改前文件](experiments/bulb_teacher_lstm_20261007/before/)。
  无Isaac/GPU/远端/真机操作；原生接线、DDP与学习效果未验证，无成功率结论。
- GitNexus入口/新类UNKNOWN，补查shell/runner/回放动态引用；文档精确UID未收录，同名历史快照不作为当前文档影响证据。未提交。

<a id="tactspace-wiring-20261007"></a>
## TACTSPACE-WIRING-20261007：表示模型 CPU 接线验证

- 范围：当前 main `167fc3c` 上局部未提交修改；不触及机器人环境、PPO、资产、采集或数据标记。
- 实现：TactSpace RGB/Depth/Marker 适配版，独立域×模态分支与纯 latent 解码；默认旧三模态模型。
- 解释器：本机 `/home/liuxinyu/miniconda3/envs/brainco/bin/python`，PyTorch 2.7.0+cu128，实际使用 CPU。
- 命令：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/tactile_representation/check_tactspace.py`。
- 数据/预算：固定种子7、16×16图像/4点合成同步仿真格式；真实入口原模型2次、新模型2次+恢复1次、Marker单模态2次更新；临时产物清理。
- 结果：12项通过，覆盖单/多模态输出、各编码器/解码器梯度与参数更新、模态均衡MSE、目标无泄漏、
  无效点/缺失模态/无配对、重复ID多正样本、空更新、零权重、域分支独立、未训练real拒绝、恢复冲突、
  导出拒绝覆盖及冻结数值一致（直接保存加载要求完全一致，分块/单位变换容差rtol=2e-5、atol=2e-6）。
- 日志：`logs/diagnostics/tactspace_20261007/cpu_checks.log`；完整检查脚本已加入原表示脚本目录。
- 旧权重另验：`runs/revo3_cross_modal_restoration_v1/best.pt` 与 `runs/revo3_tri_modal_cross_v1/best.pt`
  按当前严格加载，输出均为有限 `[1,256]`；前者原 unaligned_sim 冻结接口维度256。
- 限制：无本地合格真实训练集；无CUDA/AMP、实际学习质量或sim/real迁移验证，不报告成功率。
  旧跨域loader仅episode/frame键连接，TactSpace请求明确报错；不把同索引视为物理刺激证据。
- 图分析：当前检出已执行影响分析，公共冻结链路CRITICAL，部分UNKNOWN及符号异常；引用补核与CPU回归通过。
  本次未提交，未把图结果视为低风险证明。下一步需可信物理配对数据后再单独接跨域训练。

## TACSL-TRAIN4096-20261007

- 用户新授权4096并行环境训练；有效run `train4096_20261007_v3`，4096×128×1500计划更新，从零训练默认14维状态MLP。
- 前5次更新2621440 samples、8192完整训练回合0成功；model0优化器128步，actor/critic权重L1变化29.6532/66.9937，所有模型/optimizer/标量有限。
- v1/v2碰撞栈溢出失败；扩到4293918720 bytes并增加guard后，v3验收无Error/接触丢弃。显存28274/32607MiB，作业继续运行；不声称后续峰值已完全覆盖。
- 状态策略不计算触觉图像，碰撞/compliant contact/奖励/重置/成功口径保持；只做记录和容量适配。完成后独立seed1042、1024回合评测尚待执行。
- [说明与复现](experiments/tacsl_train4096_20261007/README.md)、[启动验证](experiments/tacsl_train4096_20261007/startup_verified.json)。

## TACSL-INSERTION-LAB-5090-20261007

- 目标 RTX5090 主机的独立工程 `workspace/xinyu/tacsl_insertion_lab`，Sim 6.0.1 / Lab 3 beta2 patch1；源 tacsl SHA `f769e39fe5605324e0c6b3eb04ca7b8a7681f4e0`。
- 单环境 260 步及双侧触觉接触/释放通过；原任务 10 方法、控制库 14 函数/49 工具符号 AST 对照通过。
- PPO 8 环境 ×256 步 ×4 次更新，8192 transitions、64 optimizer steps；训练0/32，独立 seed43 评测 **0/16**。actor/critic 权重变化且全部有限；不声称学会插入。
- 触觉壳体可视网格的 RTX 遮挡适配、RSL MLP 与原 LSTM 差异、资产公开镜像来源和未验证的跨引擎动力学均已记录；保留失败日志，不修改成功判据或删除失败样本。
- [完整实验说明](experiments/tacsl_insertion_20261007/README.md)、[触觉证据](experiments/tacsl_insertion_20261007/tactile_validation_01.json)、[检查点审计](experiments/tacsl_insertion_20261007/checkpoint_audit.json)、[独立评测](experiments/tacsl_insertion_20261007/eval_01.json)。

## BULB-V6-COMPONENT-SUCCESS-20261007-1621

- 用户看到的两条曲线为orientation_success/position_success。本机6006服务run为v6；与c3g实例v5区分。
- 9500–10480完整981点：位置首次10388非零，共4个非零记录点，近100轮均值0.003223%；角度单项0.303642%，此前100轮0.195833%。stage2_success全0。
- 远端v6源码核实两项独立，阶段二仍需位置/角度/低速同时满足并连续保持；非零单项不证明同一回合联合达标。rank0混合阶段批次统计，非独立评测。
- GitNexus查询返回历史/当前候选和异常元数据，未用图作准确语义依据，以远端实际v6源码为准；本次未修改训练。
- [证据](experiments/bulb_v6_dsw_resume_20261007/component_success_20261007_1621.json)。

## BULB-V5-PROGRESS-20261007-1537

- 明确核查`dsw-c3g3e0ieowszisvolv`的v5作业；另一台wpmt实例的v6不可混用。PID948/1009/1010活动，两rank工作目录核实，最新model10300。
- 原始事件0–10338共10339点：rank0阶段二/三成功、浅插候选/保持、第三阶段时长、旋入进度及最终奖励全0。
- 近100轮进入二阶段87.5420%，混合阶段末态倾角74.8781°，≤5°比例0.0760832%；前100轮76.9542°、最近500轮76.9959°。尚未稳定对齐，不可称所有样本从未角度达标。
- rank1无阶段标量，训练统计非阶段二专属、非独立评测；没有双卡浅插成功数/总数。未修改远端训练。
- 初次按脚本名筛进程漏掉重命名的pt_elastic/python，后以PID、工作目录及日志增长纠正。文档impact UNKNOWN（仅索引历史快照），补查引用后只前置记录。
- [统计证据](experiments/bulb_v5_dsw_2x8192_20261006/progress_20261007_1537.json)。

## BULB-V6-PROGRESS-20261007-1447

## BULB-STAGE2-BONUS-20261007

- 本地v7新增浅插晋级一次+10；第三阶段一次+20保留，未放宽判据或改变原奖励；新项先除dt。
- 48项CPU回归通过；初次缺OpenUSD路径的5个导入错误修正环境后通过，日志保留。未做新版本原生训练/评测。
- 原始事件连续9500–10264共765点rank0阶段二/三成功0；验证二阶段晋级锁存到reset日志，不会因后来失败抹去。
- rank1无阶段日志，不能断言两卡历史浅插事件绝对为0；两卡汇总完整终态成功指标全0。修正此前过宽措辞。
- 活动远端v6保持运行，新奖金未在该进程生效。
- [实现、影响和证据](experiments/bulb_stage2_bonus_20261007/README.md)。

## BULB-V6-PROGRESS-20261007-1528

- 指定DSW v6至10238轮，TensorBoard9500–10238共739点；进入第三阶段、最终旋入成功、旋入圈数及最终奖励全0。
- 最近100轮混合阶段末态倾角63.1831°、≤5°比例0.117725%、进入第二阶段87.2918%；非阶段二专属或独立评测。
- 核对实际保存env.yaml及实现：无二阶段独立完成奖金；`success`是连续shaping；三阶段完整成功一次+20，物理失败同帧不给。
- 没有修改或重启训练。[指标与奖励核对证据](experiments/bulb_v6_dsw_resume_20261007/progress_20261007_1528.json)。

- 只读查看指定DSW v6作业：PID1446/1468/1469仍活动；至10050，共551次续训更新，model10000。
- rank0按轮平均：初100轮倾角74.5026°→前100轮68.2360°→最近100轮66.7958°；
  最近进入二阶段87.8432%，≤5°比例0.08832%，中心目标误差7.0383cm。
- 全v6历史551点中进入三阶段、浅插候选/保持、旋入/最终成功指标严格为0；粗略朝向有所改善，精确重定向未突破。
- 统计为混合阶段重置批次指标，不是二阶段专属分布或独立成功率；v5附带比较使用可用尾部10011–10110，非父checkpoint边界或公平消融。
- 作业/配置/检查点未修改；仅扫描外层日志无错误，未重新扫描原生日志。
- 文档更新前GitNexus对STATUS/EXPERIMENTS返回UNKNOWN（当前文件未索引，仅历史快照）；补查README、AGENTS、PROJECT_MAP、RUNBOOK引用，只前置本次记录，不改既有锚点或运行代码。
- [完整统计](experiments/bulb_v6_dsw_resume_20261007/progress_20261007_1447.json)。

## BULB-V6-DSW-RESUME-20261007

- 用户授权同步到`dsw-wpmt7kzoh4b53jpmov:/home/admin/workspace/xinyu/`，双卡各8192，评估容量和恢复方式。
  两GPU启动前空闲、各46068MiB，未停止其他作业。独立新副本`mulyi-model-tac-v6-20261007`，1735/1735项哈希/链接一致。
- 从v5保存的model9500续训：SDF/三阶段/控制/动作/观测一致，已具抓取搬运能力；旧v1/29999语义差异更大未选。
  SHA `a7c1862d74bf39a3fe9303a0b0085adf8c2806091e3d69b83a64e32fd693ad2b`；原优化器190020，模型全有限。
  不恢复逐环境物理/RNG/ADR状态；新增可选initial_adr_difficulty参数显式0，匹配父run。原默认行为保持，AST9分支检查通过。
- 启动2026-10-07 12:38:40（Asia/Shanghai），torchrun1446、rank1468/1469；seed42/43，完整三阶段、v6、14s。
  追加20500轮，runner计数9500–29999；run为副本下`logs/rsl_rl/bulb_grasp_teacher_v6/2026-10-07_resume9500_2x8192`。
- 容量：每rank collision_stack=4026531840，contact=16777216，patch=3000000，found_lost_pairs=4194304，
  found_lost_aggregate=67108864，total_aggregate=4194304，heap=134217728，temp=33554432。
  沿用v5已运行的8192配置，未再翻倍；物理/碰撞/SDF/判据保持。
- 服务器CPU44/44通过；启动验收观察9500–9511共12次PPO更新，真实gradient all-reduce、两rank参数哈希相同。
  新model9500优化器190040、actor权重已更新、所有模型张量有限、归一化计数保留；父模型SHA不变。
  实际env与父配置比较：actions/observations/commands/PhysX一致，除success以外奖励一致，floor0.2/版本6正确。
- 外层日志及两份Kit原生日志扫描未发现Error或CUDA/PhysX/容量溢出匹配，显存34403/35687MiB，GPU95%/97%。
  保留原visual引用警告；冷启动约6分钟后首轮更新。只验证初期容量，不证明后期密集接触峰值或学会重定向/完整装配。
- [说明与复现](experiments/bulb_v6_dsw_resume_20261007/README.md)、
  [启动报告](experiments/bulb_v6_dsw_resume_20261007/startup_verified.json)、
  [恢复依据](experiments/bulb_v6_dsw_resume_20261007/resume_source.json)、
  [同步验证](experiments/bulb_v6_dsw_resume_20261007/sync_verified.json)。
- GitNexus训练入口UNKNOWN，已补核shell→split_gpu/runpy动态引用；此前奖励改动impact与验证沿用本条下方记录。
  无安装/真机/其他主机操作、无独立成功率评测、无新监督循环、未提交。后续先核实作业身份，不重复启动。

## BULB-STAGE2-REWARD-20261007

- 范围：当前教师v5的三个阶段奖励提取、已有轨迹分析及本地单项最小修改；无远端/GPU操作。
- 源数据：保存的`orientation_remote_env.yaml`（20项奖励）和model8800/seed42的960帧连续trace。
  阶段二728帧、642帧中心误差≤1 cm；倾角均值69.65°、最小39.11°、≤30°为0/728帧。
  最小中心误差帧416为2.72 mm/77.66°/深度−43.07 mm。5完整回合加截断第6回合，非独立成功率。
- 主要假设：两项位置收益未耦合朝向，进入目标附近后保持歪斜仍高收益，暂离中心去重定向可能损失回报。
  构造C=1且仅比较position_tracking/success/thread_down/bulb_down_orientation的静态例子：
  d=0/θ=75°→d=2 cm/θ=0°，v5为17.84255→15.94372分/s，v6为12.56477→15.94372分/s。
  这是数值比较，不是实测回报/动力学/因果消融；trace缺指尖力/手掌速度，无法确认换指断门或滑移惩罚贡献。
- 修改：`curriculum.py:stage_two_position_orientation_reward`包装原success函数，只在stage_two且非stage_three
  乘`floor+(1−floor)*(1−θ/π)^2`；`env_cfg.py`接入floor=0.2、版本6。阶段一/三/其余19项、判据/物理均保持。
  floor=1恢复原奖励值；精确v5源码在before保存。未将完整四元数rot_std启用，保持轴向相位自由。
- 验证：[CPU日志](experiments/bulb_stage2_reward_20261007/cpu_checks.log)44/44通过；
  新增阶段隔离/局部reset、单调和上下界、参数关闭/非法值、无接触/自转不变及静态收益方向检查。
  [verification](experiments/bulb_stage2_reward_20261007/verification.json)记录既有课程函数AST和5项保护文件SHA未改。
  无原生仿真/新训练/独立评测，不能承诺重定向已解决。
- [完整奖励设置](experiments/bulb_stage2_reward_20261007/reward_settings.json)、
  [轨迹及静态分析](experiments/bulb_stage2_reward_20261007/analysis.json)、
  [精确源码差异](experiments/bulb_stage2_reward_20261007/source_delta.patch)。
- GitNexus刷新main/167fc3c后配置类LOW/3直接依赖/0解析流程；其中benchmark导入是同名误连，经源码核实。
  原success函数、测试及部分文档UNKNOWN，已补查snapshot/注册/诊断入口/文档导航；未以空调用链作为无影响证明。
  全局图有流程截断，不能据此声称完整影响覆盖。无本机相关训练进程，未提交。
- 下一步（未执行）：同一v5模型/冻结分布开展旧奖励与v6对照，统计阶段二专属倾角中位数/P90、≤30/10/5°占比、
  接触/失握、进入第三阶段和完整成功数/总数。若朝向停滞再单独测试thread_down软接触或权重；
  若朝向已达标而浅插仍失败，再分析深度/径向/速度，保持原验收阈值。

## BULB-V5-VIDEO-20261007-1013

- 指定DSW独立单环境seed42，选取时最新model8800，960步/32秒、1280×960/30FPS，连续回放未筛选。
- 录像脚本副本仅将ADR初值对齐训练的0，checkpoint恢复重力−1；生产训练/play文件未改。GitNexus UNKNOWN经入口引用补核。不同于完全恢复训练逐环境ADR/RNG状态。
- [视频与说明](recordings/bulb_v5_20261007_1013/README.md)、[验证](recordings/bulb_v5_20261007_1013/validation.json)：7文件哈希、checkpoint不变、960帧完整解码通过，抽帧目视通过。
- 2.9秒首次阶段二；阶段一232帧、阶段二728帧，无阶段三/成功帧；完整5回合最终0成功，第6回合截断。不是独立成功率评测。
- 录像71253已退出，训练948/1009/1010持续；原有视觉/RTX警告保留。HUD大倾角Radial为轴线外推误差，不是实际位移。

## BULB-V5-PROGRESS-20261007-1010

- 只读核查TensorBoard至8858，最新model8800，原双卡作业948/1009/1010存活；近期100轮平均12.5352秒。
- 近100轮对比昨日2867–2966：stage1_success83.1108%→87.4108%，混合末态倾角78.6031°→74.3373°，≤5°比例0.0777800%→0.0971001%，中心误差8.9316→6.9028cm。
- 全历史0–8858未四舍五入的stage2_success、stage3_success、screw_turns/best、stage3_elapsed_s、最终成功奖励全部0。ADR0和重力−1未晋级。此前100轮倾角72.5855°，最近74.3373°；不声称单调改善。
- GPU35921/35653MiB，88%/100%利用率；近期外层日志无致命错误，非全原生日志扫描。未新开GPU评测、未改代码/配置/作业。
- [完整统计](experiments/bulb_v5_dsw_2x8192_20261006/progress_20261007_101023.json)。上述为rank0重置批次训练标量按轮均值，非双rank独立成功数/总数；全阶段混合角度不是阶段二专属角度。

## BULB-V5-ORIENTATION-ANALYSIS-20261006

- 只读检查实际env.yaml及远端配置/课程源码哈希，与本地一致。stage2使用中心距离5×tanh核（std0.2，接触门控）及success的10×平方tanh核（std0.1，rot_std=null，无接触门控）；该success奖励不是严格浅插/完整成功。
- 朝向为灯泡−X朝世界−Z：thread_down=5×(1−θ/π)×stage_two×双侧指尖>1N门控；全阶段惩罚−0.2×(1−cosθ)/2，无门控。保持自由轴向自转。5–6mm深度/0.5mm径向/5°/低速0.2s是晋级条件，未增加独立逐项连续浅插奖励。
- TensorBoard近100轮2867–2966：stage1_success83.1108%，末态混合倾角78.6031°，≤5°指标0.0777800%，stage2_success0，thread_down日志奖励0.990109。初期0–99倾角80.5271°，2400–2499为76.7456°。不支持已学会可靠朝下；日志未提供阶段二专属角度分布。
- 推断：中心位置/稳定接触可在姿态未达标时提供较多收益；thread_down增大也可能由更多接触/进入阶段二导致。未做因果消融，不断言奖励权重是唯一原因。训练仍运行，未改代码或配置，未新开GPU评测。
- [配置/趋势与哈希](experiments/bulb_v5_dsw_2x8192_20261006/orientation_analysis_20261006.json)、[高精度近期标量](experiments/bulb_v5_dsw_2x8192_20261006/orientation_exact_20261006.json)、[实际保存配置](experiments/bulb_v5_dsw_2x8192_20261006/orientation_remote_env.yaml)。指标均为rank0训练重置批次按轮均值，非独立评测。

## BULB-V5-DSW-PROGRESS-20261006-1234

- 只读核查v5双卡8192/卡训练：进程948/1009/1010活动，TensorBoard至2952轮，最新model2900。近100轮进入阶段二指标83.1967%，最新84.1175%；来自rank0重置批次标量按轮平均，不是独立评测成功数/总数。
- 读取TensorBoard原始浮点值，0–2952全部2953点：stage2_success（进入阶段三）、stage3_success、旋入圈数/最佳圈数/阶段三时长/最终奖励全部严格为0，排除文本四位小数舍入掩盖非零的情况。不能据此声称未记录的rank1每一步都无偶发进入。
- 最近100轮平均12.6689秒；混合阶段末态倾角78.543°，非阶段二专属统计。重力课程仍−1m/s²、ADR0。近期日志无Traceback/CUDA/PhysX Error；本次未作全量原生日志扫描。
- [统计证据](experiments/bulb_v5_dsw_2x8192_20261006/progress_20261006_123400.json)、[全历史阶段核验](experiments/bulb_v5_dsw_2x8192_20261006/stage_history_20261006_1234.json)。未改代码/配置/作业，未做新评测或录像。

## BULB-V5-DSW-2X8192-20261006

- 用户授权同步指定新DSW并启动2卡，每卡8192；服务器重启为两卡后继续。目标`dsw-c3g3e0ieowszisvolv`，独立副本`/home/admin/workspace/xinyu/mulyi-model-tac-v5-20261006`，1729项SHA256/链接一致，保留旧项目/模型/日志。
- v5抓取抬起→浅插5–6mm→实际旋入3圈；从零、seed42/43、总16384环境、30000轮，torchrun PID948、rank1009/1010。run为副本下`logs/rsl_rl/bulb_grasp_teacher_v5/2026-10-06_scratch_2x8192`。
- 仅新增可选启动容量覆盖：每rank碰撞栈64MiB→3.75GiB，七项contact/patch/pair/heap/temp翻倍；默认保持教师快照。SDF、物理求解器、奖励、阶段、成功判据不改。未做小容量对照，不声称找到最低需求。
- 本地CPU40项通过；00:40–00:41核验到9/30000（10次更新）；真实all-reduce且两rank更新后参数SHA一致。model0的25个状态张量有限，优化器step20。配置v5、8192/卡、14s、120Hz/30Hz、最终+20均正确。
- 外层与两份Kit完整日志扫描未发现Traceback、CUDA OOM、PhysX Error、碰撞容量溢出或丢接触。GPU显存35427/35131MiB，利用率99%/100%；旧机器人visual引用警告保留。
- [证据与命令](experiments/bulb_v5_dsw_2x8192_20261006/README.md)、[启动核验](experiments/bulb_v5_dsw_2x8192_20261006/startup_verified.json)。原重力课程从−1m/s²、ADR0开始；独立成功数/总数未知，初期容量通过不证明后期密集接触峰值或任务成功。未创建监督循环，未提交。

## 2026-10-06 灯泡课程v5第三阶段三圈旋入及最终奖励

- 用户选定第三阶段单独6 s，并追加最终大成功奖励；前两阶段仍共8 s，配置最大回合14 s。stage2完成原浅插0.2 s后晋级，第三阶段关闭carry_relative_lin_vel_l2/carry_relative_ang_vel_l2，前两阶段和原接触项保持。
- 自由刚体实际相对姿态计圈，旋入方向为从口上方看顺时针。最终净3圈、入口深度基线+3×6 mm×倍率（±1 mm×倍率），末态对中≤0.5 mm/倾角≤5°、低速稳定0.2 s。目标根据有效物理进度超前1/8圈；没有接入导向关节、强制力、关闭碰撞或跳过阶段。
- 奖励借鉴RotateBulbCustom的RotateBulbGuidedRotation及Randomized的最大回退成本：带力新高+9/圈、首次1/4圈里程碑+0.75（合计+3/圈），回退最大幅度增量−14/圈、0.01圈死区。最强指尖过滤力0.5 N饱和，换指记忆0.2 s，奖励信用限1圈/s。新进度三圈最多36分，相比6 s接触上限约28.2分高28%。不存在额外接触预算包装器。
- 最终大成功权重20、函数除dt，首次完整成功只给一次；同帧物理失败不给，退出后重入不重复；原连续位置success奖励保持。终态成功仍不锁存，若后续丢失会失败；不提前成功终止。
- CPU40项通过：[日志](experiments/bulb_grasp_screw_stage3_20261005/cpu_success_bonus2.log)。39项阶段3检查曾通过；新增+20测试初轮误用浮点精确相等，19.999998被误判失败，改为Torch标准浮点比较，保留失败日志cpu_success_bonus.log，未改奖励/成功阈值。
- [最终原生报告](experiments/bulb_grasp_screw_stage3_20261005/native_success_bonus.json)：2环境×245步，2个完成回合，strict原model_22200加载、观测195/745/960及28动作、[-9.8]重力；日志确认curriculum_update先于time_out和final_screw_success权重20。只是运行接线，非三阶段策略成功率。
- [几何探针](experiments/bulb_grasp_screw_stage3_20261005/physics_path.json)：0.8/1/1.2倍，每1/4圈采样至3圈，39个制造位姿各零重力释放0.2 s。39/39满足预定误差，最大中心位移0.00000782 mm；不含机器人/策略，不保证任意初始相位的自主旋入。原始资产/点云/snapshot保护哈希通过。
- [完整证据与源码](experiments/bulb_grasp_screw_stage3_20261005/README.md)。未训练v5、未同步/停止/重启远端、未提交；已有其他工作区改动保留。

## TAG-PRESS-PAPER-20261005-03

纠正02对用户意图的误解：法兰接头应在正方形端面，长方形侧面每面2枚tag。当前主体20×20×40 mm，尾柄Ø6×8沿+Z，销孔Ø3.2沿X且Z=44，总高48 mm；无嵌件或凹槽。四张20×40 mm贴纸，12 mm标记ID40–47，每面对中心Z=30/10 mm。CAD单实体、STL闭合/绕序/尺寸、3MF XML回读通过；PDF 600 DPI识别8/8，配对上下顺序及20 mm中心间距通过，版面目视通过。错误版本归档，未做实物打印/安装/承载验证。 [说明](assets/head/tag_press_head/README.md)。

## TAG-PRESS-PAPER-20261005-02

最新用户要求：长方体顶部保留与其它压头一致的法兰接口，取消嵌入，四面tag改为纸质PDF。完成40×20×20 mm主体、Ø6×8尾柄、Ø3.2 X轴横销孔（肩部上4 mm）；四竖直面各一枚12 mm DICT_4X4_50 ID40–43。 [资产与复现](assets/head/tag_press_head/README.md)。CAD有效单实体，主体缺失体积0，STL闭合/绕序/40×20×28包围盒及3MF XML回读通过；PDF渲染600 DPI后4/4标签检出，排版目视通过。首轮3MF自动加载缺networkx，改为标准XML回读网格；未放宽几何断言。旧双色文件归档保留，未改原法兰或其它压头，未执行硬件/训练；实物安装、载荷和识别精度待确认。

## 2026-10-05 灯泡课程v4浅插实现与本地验证

- 用户选定入口5–6 mm、不旋紧；stage1保持口上15 cm和<5 cm晋级。stage2按尖端插深5.5 mm生成中心目标；完整成功要求深度5–6 mm、尖端/口沿径向偏差≤0.5 mm、倾角≤5°、线速≤0.02 m/s/角速≤0.5 rad/s连续6步（0.2 s）。保持8 s、原奖励权重及动作/观测，版本4。
- 新insertion.py管理坐标换算和prestartup匹配缩放，保留原灯泡0.8–1.2分布、灯座同倍率、reset XY旧分布；初始灯泡Z+1 cm保净空。bulb_insert.usd复制源闭合外螺纹SDF512和两壳SDF256，保留原内螺纹三角碰撞，仍为自由刚体，无关节/引导力；原始文件SHA保护通过。
- CPU30/30通过；首次完整原生启动暴露旧reset_socket的env_ids默认值不满足Isaac静态参数检查，已去默认值并以真实验证器补测试。失败报告native_smoke.json保留；native_smoke2.json通过：2环境×245步、2个完成回合，strict原模型加载，195/745/960和28维、-9.8重力。不是插入成功率评测。
- 物理隔离探针3种倍率0.8/1/1.2，5.5 mm目标零重力释放1 s未被挤出；从口上1 mm按参考相位无约束自由落下2 s，末态深度12.19/9.06/15.62 mm，全部失败于浅插判据。探针无机器人、先制造初态，不是完整任务成功证据；不声称随意相位/松手即能完成浅插。
- [证据包](experiments/bulb_grasp_insertion_20261005/README.md)、[CPU](experiments/bulb_grasp_insertion_20261005/cpu_checks2.log)、[原生任务](experiments/bulb_grasp_insertion_20261005/native_smoke2.json)、[碰撞探针](experiments/bulb_grasp_insertion_20261005/physics_probe3.json)。日志含已有机器人visual引用及marker警告，有限状态检查通过；未做视频渲染验收。
- 隔离探针前两次物理已完成但Kit收尾未退出，仅终止本次探针进程；最终补清理callbacks/SimulationContext单例，第三次物理完成并正常退出。失败报告保留。
- 未训练或独立评测v4；新成功率未知，8192环境SDF容量/性能未知。远端旧训练未同步/停止/重启。GitNexus两个配置/命令类LOW，新增符号/脚本/文档UNKNOWN经显式引用补核；未提交。

## BULB-GRASP-BULB-ORIENTATION-20261005

- 用户要求将−0.2手掌朝向惩罚改为灯泡朝向偏离向下的惩罚。本地env_cfg禁用palm_down_orientation、新增bulb_down_orientation；curriculum.py新增同名惩罚函数，以灯泡本地−X对世界−Z的倾角θ计算(1−cosθ)/2，保留原余弦形状和−0.2权重，两阶段均启用、无接触门控、自由轴向自转。原手掌函数/snapshot不改。
- 阶段二thread_down仍是权重5的线性角度正奖励并有双侧接触门控；其余奖励、随机灯座/两阶段位置/成功判据/时间保持。MDP版本2→3，观测/动作宽度保持，旧模型不代表新奖励效果。
- [CPU24项通过](experiments/bulb_grasp_bulb_orientation_20261005/cpu_checks.log)，新增0/30/90/180°、两阶段、有无接触、机器人朝向独立、轴向自转验证，配置验证仅替换目标惩罚/权重保持/其余奖励相同。未运行原生物理/策略/训练，未同步或重启远端。
- 同目录before保存三份代码备份。GitNexus当前仓库main/167fc3c配置impact LOW（含Play派生类），测试UNKNOWN，已核查任务注册及诊断入口引用；沿用此前文档影响分析与导航补核，仅局部修改，未提交。

## BULB-GRASP-CURRICULUM-V2-20261005

- 用户要求阶段一目标为灯座口上方15cm、灯座XY继承旧目标随机化，阶段二为口上方3cm或插入。根据实际资产采用明确的中心高3cm目标，不加入精确插入/旋紧假设；本次仅本地实现和CPU验证。
- `BulbGraspTeacher/curriculum.py`新增`reset_socket_from_target_ranges`，由env_cfg中的reset事件在原机器人/桌面/灯泡reset之后执行。直接读取原目标基座系X[−0.85,−0.55]m/Y[−0.25,0.25]m，经实际机器人根位姿转世界XY；灯座Z按当前桌面和源USD底部计算，保持直立/零速度，每回合一次。保留全部随机样本；局部reset不影响其他环境，定时重采样/阶段切换不移动灯座。
- 阶段一目标中心为灯座口上方15cm，位置误差<5cm即进入阶段二；新目标在同一灯座口上方3cm，下降12cm。仍以二阶段位置<5cm且螺纹倾角≤30°判定成功；thread_down权重5及接触门控、原15项奖励、8s总时长、28动作/1900观测/30Hz/120Hz保持，版本1→2。未增加保持、插深或旋紧判据。
- 几何依据：标称螺纹尖端距灯泡中心5.376cm，中心高3cm时尖端低于口沿2.376cm；0.8–1.2尺度下为1.301–3.451cm。但现有三凸包灯泡及较大缩放不保证能进入原灯座孔径，原5cm位置容差也不能证明插入。灯泡/灯座碰撞、几何、点云及teacher_snapshot SHA保持。
- 验证：更新`scripts/diagnostics/check_bulb_grasp_curriculum.py`，CPU23/23通过，执行生产函数/Isaac真实数学/CommandTerm生命周期，场景与写入API为替身；覆盖随机范围/均值/机器人yaw/环境坐标、reset顺序配置、单回合固定、局部/空/全reset、同座两高度、切换无瞬移、成功30°/严格5cm、末步成功缓存、旧奖励/动作/观测保持。保留原生RigidObject写入API签名核对。
- 真实USD检查及最小桌面1.1×1.1m/实际yaw/XY极值组合：灯座外接半径5.827cm，保守边缘余量≥5.7325cm；1.2倍灯泡任意姿态初态在口沿上方≥4.8395mm，未新增初始穿模。仅几何上界，非接触动力学认证。
- [验证JSON](experiments/bulb_grasp_curriculum_v2_20261005/validation.json)、[CPU日志](experiments/bulb_grasp_curriculum_v2_20261005/cpu_checks.log)、同目录before备份/差异；CPU命令见RUNBOOK。未运行v2原生仿真、策略回放或训练，学习效果未知；未同步DSW/重启旧作业，22:38原PID62901/62917/62918仍活动。
- GitNexus绑定当前main/167fc3c仓库并更新索引；课程类/配置类/阶段二目标方法impact LOW，测试/文档UNKNOWN补引用。索引存在全局流程截断和跨模块同名误连，未以零流程作为无影响证明；测试入口和注册/训练/Play调用已文本核对。保留大量既有未提交工作，未提交。

## BULB-GRASP-ORIENTATION-PROGRESS-20261005-2227

- 用户询问重定向是否仍在优化；只读核查DSW双卡作业及完整train.log，截至22:27:30已到14872轮，模型最新14800，原PID62901/62917/62918活动。解析2873个完整且唯一迭代块，无重复；未启动额外仿真/训练或修改远端。
- 视频模型附近13001–13100与当前14773–14872各100轮均值：末态thread_down_error_deg为55.1216°→50.0572°；orientation_success为23.9147%→32.1297%；stage2_success为20.4951%→28.4185%；thread_down奖励1.6584→1.8975。当前阶段一完成80.4087%。
- 最近相邻100轮14673–14772→14773–14872：倾角50.3651°→50.0572°，姿态达标30.7309%→32.1297%，完整成功26.7817%→28.4185%。当前500轮倾角48.0209°、完整成功28.2458%，说明近期有波动，不能声称持续单调改善或收敛。
- [统计证据及最新100轮原始标量](experiments/bulb_grasp_teacher_curriculum_resume_20261005/progress_20261005_2227.json)。均为rank0重置批次训练标量按轮均值，非双卡按回合加权或独立评测；倾角/朝向达标混合两个阶段，ADR难度分布也在变化。
- 当前代码thread_down权重5、二阶段/双侧接触门控、1−θ/π连续奖励；30°是成功判据，不是奖励停止点。仅新增证据/STATUS/EXPERIMENTS记录；沿用此前文档impact UNKNOWN及导航引用补核，未提交。

## BULB-GRASP-VIDEO-20261005-1942

- 用户授权服务器headless录制并拉回本地。固定选择时最新的续训rank0/model_13100.pt；19:55启动第二轮，单环境seed42、GPU0、−9.8重力、原两阶段任务/奖励/失败判据、strict模型与归一化加载，1900观测/28动作。Play采用配置ADR初始难度0.7317，checkpoint不含逐环境ADR状态，非训练分布完全复现或独立评测。
- 专用回放脚本增加可选`--video`、分辨率和相机参数；要求1环境、有限steps、新MP4路径。30FPS原始连续960帧/32秒、1280×960，失败/reset均保留；画面HUD显示当前阶段、位置误差、朝下倾角和瞬时成功，附逐帧JSON。仅上传至独立recordings目录执行，未替换远端训练源码。
- 实际结果：首次进入二阶段为frame77（2.57秒），688/960帧处于阶段二。前四个已完成回合终态成功0/4（一个提前终止、三个超时）；第五回合尚未结束，frame865–959（28.83–31.97秒）连续95帧满足完整成功谓词。单条seed42回放不能作为总体成功率。
- [完整视频](recordings/bulb_grasp_curriculum_20261005_1942/bulb_curriculum_model_13100_complete.mp4)、[实际回放报告](recordings/bulb_grasp_curriculum_20261005_1942/report_complete.json)、[命令/模型哈希](recordings/bulb_grasp_curriculum_20261005_1942/launch_complete.json)、[验证](recordings/bulb_grasp_curriculum_20261005_1942/validation.json)。22:21本地H264完整解码960帧通过，14项下载/源码SHA一致，原checkpoint前后SHA不变；抽帧确认场景、HUD、灯泡与灯座可见。Kit曾报告地面纹理PNG加载错误，实际抽帧网格地面仍可见。
- 首轮渲染等待约8分钟后自行恢复；此前误判停滞，随后仍执行了已准备的SIGTERM并最终清理该回放进程，产物不完整。保留`record.log`、`attempt1_stalled.json`及纠正记录`attempt1_correction.json`，不将其称为已证实死锁。第二轮复用缓存正常完成并退出。
- 22:20训练PID62901/62917/62918仍活动；未停止/重启训练，未筛选成功片段，未更改任务物理。GitNexus脚本及文档impact UNKNOWN，补查shell调用与README/AGENTS/PROJECT_MAP导航，仅局部可选录制和证据文档；未提交。

## BULB-GRASP-DOWNLOAD-20261005-2219

- 用户要求下载DSW最新checkpoint并报告成功率。既有torchrun62901/rank62917/62918仍活动；两阶段教师，8192环境。
- 22:18日志最新14777；近100轮14678–14777阶段一81.0656%、完整两阶段26.8963%，最新完整25.79%。统计为rank0训练重置批次均值再按轮平均，非独立评测或按回合加权总体成功率。
- 选择时最新model14700及agent/env配置、launch_contract、distributed_verified共五文件下载并核对远端前后与本地SHA256。模型哈希000f05a381618de354c3853533a072e03ef422b507677900dafd2467b1ccf259。此前model13300副本保留。
- [清单](checkpoints/bulb_grasp_teacher_curriculum_20261005/model_14700/download_manifest.json)及同目录training_log_tail_20261005_2218.log保留证据；未运行仿真、未停止或重启远端作业。
- STATUS/EXPERIMENTS文档impact UNKNOWN（索引未解析目标）；补查README/AGENTS/PROJECT_MAP导航引用，仅追加记录，无运行代码修改或提交。

## BULB-GRASP-CURRICULUM-PROGRESS-20261005-1914

- 用户询问二阶段触发和训练现状；只读SSH核查既有DSW作业的日志、进程和GPU，无部署/重启/训练配置修改。
- 19:14:26（Asia/Shanghai）最新12889，首12000，共890条完整轮日志；日志距检查3.82s，torchrun62901和rank62917/62918活动，两rank在GPU占用约9GiB。最新保存model12800。没有匹配到Traceback、OutOfMemoryError、CUDA error、ChildFailedError，非全面健康认证。
- 启动首100轮12000–12099→近100轮12790–12889：进入二阶段65.9833%→79.9247%，完整成功2.2682%→16.5714%；上一个100轮12690–12789完整成功16.0226%。最新单轮进入80.43%、完整17.78%。原始标量为rank0重置批次均值，经日志四位小数后再按轮平均，非按回合加权的两rank总体成功率，也非独立评测。
- 同窗口全阶段末态倾角72.7941°→57.2990°，姿态≤30°指标10.9620%→20.2265%，关节异常终止35.9388%→23.2756%，重定向奖励0.8834→1.6352。支持“已大量触发且学习在改善”，未达到稳定完成。角度/位置指标混合两个阶段，不能直接归因于阶段二姿态或时间预算；缺阶段切换耗时/剩余时间轨迹。
- [证据JSON](experiments/bulb_grasp_teacher_curriculum_resume_20261005/progress_20261005_191426.json)记录来源、范围、数值和统计限制。业务代码、模型、作业保持。STATUS/EXPERIMENTS文档impact UNKNOWN，经README/AGENTS/PROJECT_MAP导航引用补核，仅追加记录，未提交。

## BULB-GRASP-CURRICULUM-RESUME-20261005

用户授权同步本地修改并接最新checkpoint续训；17:44停止旧单阶段torchrun，17:45新torchrun62901启动。
同步7文件，93文件SHA匹配；备份旧版本。从旧run最新model12000恢复，双卡各4096，18000追加轮、总目标30000。
本地CPU16/16及原生4env×300步通过；远端两rank参数同步一致、优化器step240020→240040、新model12000保存、重力trigger5730/progress1/−9.8验证通过。
17:48到12012，stage1=60.9218%、stage2=0.53267%，为训练统计非独立评测。新版MDP与成功口径已改变；ADR张量未存盘，初始0.7317。
GitNexus env_cfg impact LOW、3直接引用、0流程；文档UNKNOWN，已补查为导航引用；没有修改训练源码或无关任务。
[完整部署记录](experiments/bulb_grasp_teacher_curriculum_resume_20261005/README.md)、[启动验证](experiments/bulb_grasp_teacher_curriculum_resume_20261005/startup_verified.json)、[同步清单](experiments/bulb_grasp_teacher_curriculum_resume_20261005/sync_manifest.json)。

## TAG-CUBE-ISAAC-SURFACE-20261005-03

按用户请求在本机Isaac原生RTX检验标签压头并排查侧面不均。
[报告](calibration/assets/head/tag_cube_head/surface_inspection/README.md)、[汇总](calibration/assets/head/tag_cube_head/surface_inspection/summary.json)。
五面射线5万点组合表面误差0，STEP黑白重叠／目标双向差体积0。7/7相机视角、40/40可见tag观测通过，覆盖20个ID；最大平移0.4923 mm、角度0.2738°、重投影0.8457 px。
诊断复现平滑法线导致假凹凸和单白色件0.6 mm槽；首两轮诊断的插值缓存／可见性问题已修正，完整过程保留，以isaac_run_03为准。
未改源资产、未训练或连接真机；静态理想场景不证明打印、安装、载荷及真实视觉精度。

## HEAD-PIN-20261005-01

依据现有法兰和三种压头STL／URDF，新增Ø2.8／2.9／3.0×12 mm带头打印销，默认2.9。孔壁面距、源SHA256、有效单实体／STL闭合及尺寸检查见[validation.json](calibration/assets/head/press_head_pin/validation.json)。最小单侧模型间隙分别0.14548／0.09548／0.04548 mm；完成截面预览核对，未打印／切片／承载验证。滑配销无自锁。开发期间过严间隙阈值与SciPy绘图依赖失败及修正保留于[README](calibration/assets/head/press_head_pin/README.md)。未改源网格或URDF，未运行硬件。

## TAG-CUBE-HEAD-20261005-02

用户将早期长方体要求改为顶面圆柱法兰接口、其余5面各4枚tag。按已告知的默认边长30 mm制作（尺寸尚无单独确认），尾柄沿用原资产Ø6×8、横销孔Ø3.2且距肩部4 mm。
[当前资产与复现](calibration/assets/head/tag_cube_head/README.md)、[几何证据](calibration/assets/head/tag_cube_head/geometry_validation.json)、[视觉证据](calibration/assets/head/tag_cube_head/vision_validation.json)。
CAD／网格闭合／3MF回读通过；5面正视20/20，底视12/12、顶视8/8；合成PnP最大平移误差0.0874 mm、最大旋转误差0.0562°。
未细化角点的初次底视RMSE为1.0008 px，改为标准亚像素角点细化后分别0.573／0.437 px，通过原有1 px门槛。
仅离线合成验证，实物精度／安装／载荷待验收；原URDF、法兰与历史压头均未改。早期长方体资产保留并标记已替代。

## TAG-PRESS-HEAD-20261005-01

用户确认40×20×20 mm，新增[独立标签压头](calibration/assets/head/tag_press_head/README.md)。沿用wuji-mjlab的DICT_4X4_50，侧面40/41、底面42，12 mm标签。
本次CAD有效，黑白网格闭合，3MF回读体积16000 mm³；STL表面正视图和单张透视合成图识别全部对应ID。
合成PnP误差0.199 mm／0.083°、重投影RMSE 0.631 px，仅验证图案和几何／坐标契约，不是实物相机精度。
[几何报告](calibration/assets/head/tag_press_head/geometry_validation.json)、[视觉报告](calibration/assets/head/tag_press_head/vision_validation.json)。
初次构建报告依赖缺失（trimesh无graph engine）改用CAD实体计数；初次透视渲染受内部接触面遮挡排序影响，修正为组合体外表面后通过，未改变标签或网格来放宽判据。
未运行硬件／Isaac／训练，未验证安装、承载或实际打印；未替换原有资产。重建命令见资产README及RUNBOOK。

## TAG-ONLY-20261005-01

- 授权：默认标定URDF中除手外支架仅保留Tag世界系，消除多余坐标系。
- 当前唯一根tag_center；删除calibration_world/lower_mount/calibration_plane/calibration_sphere_bottom/apriltag_pocket_floor。
  合并支架visual/collision并补偿origin，重接手安装；手内结构和压头运动链保留。
- GitNexus URDF影响LOW，3个直接文档引用、9个上游；未发现活动Isaac/采集进程。
- CPU验证142姿态（零位、20随机、121轨迹）：8662 link变换与14768视觉/碰撞变换误差0；104文件哈希不变。
- 当前61 link/60 joint、24活动关节。manifest、coordinate_reference、采样点源哈希及简洁YAML同步。
- [验证脚本](calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_only/verify.py)、
  [报告](calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_only/validation.json)、
  [备份](calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_only/before/)。
- 未运行Isaac/真机；当前导出器仍为旧源包，重建需保留本修订；历史验证不代表当前包验证。

## CALIB-TRIMODAL-BASELINE-20261005-02

- 用户授权：按三模态sim/real CLIP数据需求修改真机采集并采当前空载；其他指加载留待后续请求。
- 修改：新增`calibration/hardware/collect_real_trimodal.py`；原`probe_revo3_trimodal.py`新增外部参考参数。
  新会话保存RGB/SDK深度/400点五维Marker、时间、源样本键、基准/模型/源码哈希；未知量不冒充训练字段。
- 实际命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python calibration/hardware/collect_real_trimodal.py --phase baseline --confirm-unloaded --frames 16 --output calibration/captures/revo3_trimodal_unloaded_20261005_01`，退出0。
- 5/5相机各16帧640×480，5张空载参考＋75组有效三模态。GFBCT深度180×320，四路GFBCI240×320；Marker均400×5。
- 独立检查75组：RGB精确回溯原帧、depth原值、marker起点/差分、有限值、时戳递增、键唯一、SHA256全部通过。
  缺空载确认/缺加载参考的CLI拒绝检查通过。另用本批第1–3帧的离线fixture验证外部第0帧参考保持，3/3当前帧全部处理。
  最初外部参考测试使用先前目录时文件已不存在，未影响本批；该失败及修正保存在validation.json。
- GitNexus索引2026-10-04、HEAD匹配；两推理符号消歧后CRITICAL，返回110影响/81直接和跨任务关系。
  已事先告知；源码引用核查只见脚本内调用，新增入口UNKNOWN并补查无原调用；未将图风险当低风险。
  无活动同名采集/对齐训练进程；未改训练器、共享网络、SDK、运动代码。
- 边界：空载由用户确认，无外力实测；SN指名未确认，逐路非多指同步；depth_m尺度及100点几何适配未建立。
  session完整不等于训练manifest完整；training_ready=false/pair_valid=false，未伪造仿真正样本或目标力。
- [会话清单](calibration/captures/revo3_trimodal_unloaded_20261005_01/session.json)、[核验](calibration/captures/revo3_trimodal_unloaded_20261005_01/validation.json)、[用法与契约](calibration/hardware/README.md)。

## CALIB-UNLOADED-20261005-01

2026-10-05 15:05（Asia/Shanghai），用户明确确认当前空载不受力并授权保存五指图像。
执行 `bash calibration/hardware/probe_local.sh --output calibration/captures/revo3_unloaded_20261005_01 --frames 8`。
五路相机各预热5帧后保存8帧640×480 RGB，共40帧；逐路采集，非同步。
原始NPZ及时间戳保留，另导出40张PNG，逐张解码与NPZ像素完全一致；每路时间单调递增。
空载状态来源为用户确认，登记为`unloaded-confirmed`；每路frame_000.png登记为参考候选，未运行SDK推理。
相机SN→具体手指映射未确认，按SN分目录，未猜测手指名称。
手部串口读取失败：`Expected one endpoint, discovered 0`，故原report总体passed=false、退出2；
五路camera.passed均true。没有取得本次配套关节／阵列读数，没有运动、清零或芯片写参。
证据：[空载清单](calibration/captures/revo3_unloaded_20261005_01/unloaded_reference_manifest.json)、
[原始报告](calibration/captures/revo3_unloaded_20261005_01/report.json)。
仅验证图像完整性，不代表深度／力物理标定。未改运行源码；文档impact为UNKNOWN（未索引），补查为文档导航引用，本次仅追加记录。

## BULB-GRASP-CURRICULUM-20261005

- 用户要求在新灯泡抓取任务增加二阶段：沿用一阶段奖励，将位置目标切到灯座上方，新增重要的螺纹朝下奖励，成功口径增加30°姿态约束。本次仅本地实现/CPU验证，main/167fc3c，保留原有未提交工作。
- 实现：`BulbGraspTeacher/curriculum.py`与`env_cfg.py`；逐环境回合内阶段一原位置误差<0.05m即进入二阶段，不重置物理状态、不回退。二阶段目标为灯座口世界上方0.15m；原15项启用奖励全部保持函数/参数/权重，两个位置奖励跟随同一命令。新增`thread_down`权重5，原双侧接触门控乘`1−θ/π`；本地−X螺纹轴对世界−Z倾角θ，不限制自转角。
- 完整成功：已进入二阶段且当前目标位置误差<5cm、θ≤30°；无额外接触、低速或保持硬门槛，不因成功提前终止。终态离开目标不保留成功；阶段一完成独立记录。ADR和原重力runner读取完整成功。非终止更新钩子在终止计算阶段（奖励/reset前）同步目标与末步结果。
- 灯座：`build_socket.py`提取源USD的Support/Materials，生成`assets/bulb_grasp_teacher/socket.usd`和`.json`。原网格、碰撞/材质保持，kinematic固定底座，不带灯泡/关节。XY=(-0.30,0.32)m，底面在桌面，开口朝上；目标预留悬停间隙。源灯泡USD、point cloud、snapshot、模型不改。
- 契约：28动作、1900观测/5帧、30Hz/120Hz、原动作缩放/归一化与失败判据不改；回合4→8s，`bulb_grasp_curriculum_version=1`。旧模型仅可作为初始化候选，不声称完全恢复原MDP或新旧成功率可公平比较。
- 验证：`scripts/diagnostics/check_bulb_grasp_curriculum.py`，CPU **16/16通过**（Torch生产函数/Isaac真实数学和CommandTerm生命周期；场景、配置对象、visualizer为替身）。覆盖0/30/90/180°、自由自转、严格5cm、切换无瞬移、环境坐标、一次晋级及重采样锁定、局部/全reset、末步成功/失败、旧位置奖励切换、接触门控、ADR、观测/动作/旧奖励不变。真实USD核验网格点/面、碰撞API和开关、材质关系、唯一kinematic刚体、无关节/外部引用以及螺纹方向/悬停间隙。早期acos在极点放大float32舍入已改用atan2，未放宽角度阈值。
- [检查日志](logs/diagnostics/bulb_grasp_curriculum_20261005/cpu_checks.log)、[修改前配置/说明](logs/diagnostics/bulb_grasp_curriculum_20261005/before/)、[当前任务说明](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md)。CPU执行命令见RUNBOOK。
- 边界：本机未发现活动Isaac/训练/回放，未访问远端、启动GPU仿真、训练或修改检查点。原生场景加载、动态抓持/转向效果、8s预算是否足够、新任务独立成功数/总数均未验证。
- GitNexus显式绑定当前工作区并重建索引（12620节点/24953边/450流程）。配置类impact CRITICAL：463关联/128流程，修改前已告知；包含明显不相干映射及索引流程预算截断，不能据图保证隔离。方法/新增文件和文档UNKNOWN后补直接引用核查：本任务注册及Play继承，新增命令/ADR/奖励仅由本任务配置引用；原任务局部MDP与其他任务代码未改。未提交。

## BULB-GRASP-VS-ASSEMBLY-20261005

- 范围：用户要求比较新增机械臂抓灯泡与原装配课程。main/167fc3c，保护已有未提交修改；仅静态阅读和既有结果分析，无仿真、训练、远端或真机操作。
- 教师证据：[快照](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/teacher_snapshot.json)、[成功指标](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/mdp/commands/pose_commands.py)、[奖励](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/mdp/rewards.py)。position_only=true，success为目标位置误差<0.05m，不检查接触、速度、保持或姿态；success_reward本身为连续tanh核，不是二值成功。双侧接触+2/s、接触门控位置跟踪权重5、连续位置核权重10；搬运奖励允许3步接触宽限，hold按45步渐增，速度成本针对物体与手掌相对速度。4s回合；重力从−1开始，记录成功率严格>0.6触发3000迭代过渡至−9.8；ADR另以位置误差<5cm调整难度。
- 装配证据：[课程说明](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum/README.md)、[物理判据](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum/runtime.py)、[奖励账本](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/assembly_reward_v2.py)。Stage0为自由横躺、固定近距、5s；双侧>1N、世界线速度<0.05m/s、角速度<0.5rad/s、锚点误差≤5mm/5°连续>1s；之后抬起/对准/旋入/全流程。Stage0稳定相关预算多为每回合0.5，完成+8；与教师持续给分不可直接比较权重。当前已移植历史靠近和任意指接触项，不等于移植完整教师奖励。
- 接口差异：教师1900维/5帧且有接触，装配无触觉配置369维/1帧且无接触输入；教师相对动作乘0.1并裁剪臂±0.25rad、手±0.6rad，每物理子步重算目标；装配28维有界增量当前scale=1，每控制步计算一次目标并保持。两者checkpoint不可直接替换。教师PPO快照gamma=.99、LR=.0002；装配默认gamma=.999、LR=.001，均adaptive，不能作为全部历史run的实际学习率。
- 既有结果：[1243轮奖励审计](logs/diagnostics/assembly_5090_contact_20261004/reward_audit_1243.md)、[19:31统计](logs/diagnostics/assembly_5090_contact_20261004/summary_1931.json)。1211–1230轮严格接触统计步26.832%，其中约99.48%速度不合格，最近4000回合成功0；1224–1243轮每回合历史接近约10.946、持续/进度接近3.494、任意指接触2.733，稳定质量/保持/完成打印为0。不同窗口分别引用，不混作一次测量。
- 推断：全重力起步、稳定奖励乘积衰减与预算饱和、先静止抓稳后抬起的阶段门槛，是装配早期受阻的候选原因；动作、观测、资产碰撞和训练预算亦不同，未做消融，不能因果归因。教师为三个凸包的自由灯泡，原装配保留灯座/螺纹场景；不把任务简化自动视为物理错误。
- 验证边界：用户报告“接近成功”保留为用户观察，本地model22200只有接线回放/续训证据，未取得本次独立成功数/总数，未核实远端最新进度。建议固定全重力、记录位置成功、真实离桌稳定搬运及装配Stage0/1各自成功数/总数，再设计接口适配/蒸馏；未执行或启动。
- GitNexus显式绑定本工作区；索引commit相同但status报告3 changed/68 added，query/context出现不相干映射，结论以直接源码为准；未重建索引。文档impact均UNKNOWN/target not found，补查README/PROJECT_MAP/RUNBOOK引用后仅追加记录，既有锚点保留。无代码修改或提交。

## BULB-GRASP-DDP-SCRATCH-20261004

- 用户指定dsw-wpmt7kzoh4b53jpmov，停止现有作业、同步最新抓取教师、从零2卡×4096环境。旧assembly guard/controller/torchrun/workers共5进程已停；15个checkpoint保留。
- 目录/home/admin/workspace/xinyu/mulyi-model-tac；覆盖文件备份至.sync-backups/bulb_grasp_teacher_20261004_2155。部署87文件校验SHA256。源任务配置保留；fresh-run将ADR init_difficulty归零；初始重力−1m/s²，成功触发后3000迭代过渡到−9.8。
- 任务独立runtime固定原教师RSL-RL fork；torchrun world_size2，每rank物理GPU隔离、逻辑cuda0、种子42/43和独立Kit目录。无checkpoint/optimizer恢复，30000迭代；PID21435（仅为启动时记录）。
- 实测首次PPO optimizer step前gradient all-reduce执行；两rank全部策略参数SHA256均为f7c0b78479fc36458e229e7188ad09b48f4c695bcc0118b6d5833186b3d1f02e。各rank有distributed_verified.json。
- 22:00核查到iteration6，步数1835008（每迭代262144），两GPU约8.4GiB/75–85%利用率。初始训练成功率0，不构成独立评测。
- [启动记录](experiments/bulb_grasp_teacher_scratch_2x4096_20261004/launch.json)；远端日志experiments/bulb_grasp_teacher_scratch_2x4096_20261004/train.log；rank0模型在logs/rsl_rl/bulb_grasp_teacher_v2test/2026-10-04_scratch_2x4096/rank0/。
- 本地GitNexus新训练入口impact UNKNOWN，显式shell引用核查仅当前新任务；远端无GitNexus runner，只对已检查注册文件追加新任务导入，未修改现有任务函数。Python AST/compile和shell检查通过，无提交。

## BULB-GRASP-PORT-20261004

- 用户授权移植教师为本地独立任务并使用本仓库资产；main/167fc3c，已有未提交工作保留。
- 注册BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0及Play任务；39个任务函数/类AST保持原教师代码，完整runner保留重力课程。资产源与点云SHA一致；非整棵机器人引用树动力学认证。
- [回放](logs/diagnostics/bulb_grasp_teacher_port_20261004/smoke2.json)：4环境180步，strict模型加载，观测195/745/960、动作28、30Hz、−9.8重力通过，7个完成回合；没有测成功率。
- [配置核查](logs/diagnostics/bulb_grasp_teacher_port_20261004/semantic_audit.json)：仅本地资产/点云路径、log_dir及未使用recorder枚举表示不同。
- [续训核查](logs/diagnostics/bulb_grasp_teacher_port_20261004/training_audit.json)：4环境1次PPO更新，loss有限，优化器状态存在、trigger11502、gravity−9.8。小批量KL偏大，输出仅为接线检查，不作为优良模型；源checkpoint未变。
- 初次回放事件名错误，训练接线先后遇到本机新增optimizer字段和不支持share_cnn_encoders=False；已修复，失败日志保留。当前源码采用源agent快照，仅按接口移除无效的False CNN选项。
- Python编译与shell语法检查通过；GitNexus注册文件impact LOW，新增符号UNKNOWN补显式引用；未提交、未进行长期训练或远端操作。专用脚本和已验证范围见[任务说明](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md)。

## CALIB-POINTS-20261004-06

- 用户请求当前Tag系下5个指尖中心附近采样点，Z须在模型上方供先移动后下压。
- 输出[五指YAML](calibration/fingertip_sampling_points.yaml)，依据当前默认URDF和全零手关节姿态；文件内记录源URDF及初态SHA256。每个DIP rubber网格取世界XY包围盒中心，坐标取到0.001mm，向-Z射线求模型表面参考高度。
- 全部点使用尖端Z=40mm；当前手/支架最高29.85384mm（不含移动的压头/法兰），间隙约10.14616mm。压入方向[0,0,-1]，先抬高再移XY，换点前回到预定位高度。
- 5点射线均先命中自身触觉面、法向Z>0.9；对应URDF XYZ关节值均在名义限位内；YAML读取和5点字段检查通过。GitNexus新文件impact=UNKNOWN，补查无既有消费者，不接线控制器。
- [几何验证](calibration/outputs/fingertip_sampling_points_20261004/verification.json)保存完整计算、模型高度、射线法向及关节值。仅CPU几何，不证明有限工具碰撞安全或真实接触零点；未改URDF、未启动Isaac/真机运动。

## CALIB-TAG-20261004-05

- 用户要求以Tag中心作为全局原点，随后更正方向为原世界坐标系；最终未采用base_link轴向。
- 修改默认完整URDF并新增tag_center固定frame；名义点为lower_mount系(170,200,35.5)mm贴码槽底，码纸厚度未知。新根保持calibration_world，两条根关节只改平移。
- [变换和SHA256](calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_origin/transform.json)、[备份](calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_origin/before/)、[验证结果](calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_origin/validation.json)。
- 142组状态（零位、20组随机非零关节、121帧CSV）共9088个link变换通过，误差上限5.56e-17；104处mesh引用存在，104个网格/初态/轨迹文件SHA256保持；66 link/65 joint通过urdfdom解析。
- [只读复核脚本](calibration/assets/calibration_bench_visuotactile_urdf/modifications/tag_origin/verify.py)使用现有brainco Python的NumPy/SciPy；不启动Isaac。系统check_urdf包装器缺cmeel，直接使用brainco内urdfdom二进制并指定其lib路径成功，未安装依赖。
- manifest=v4，coordinate_reference=v2；保留原始平台测量-308.5605mm及旧公式历史，新平台世界Z=0.5mm。新条件公式未启用，未实测台架外参/TCP/码纸厚度。
- GitNexus相关frame impact为UNKNOWN，补查export_urdf/verify_touch_asset和引用后限定修改当前包。旧v3导出/验证仍按源包工作，当前派生包使用本次复核脚本；历史报告未覆盖。
- 未改手/支架/压头局部几何、关节轴/限位或真机设置，未运行Isaac/训练/远端操作，未提交。

## ASSET-REORG-20261004-01

- 目标：按用户要求分离实体模型与任务设置数据；main/167fc3c79fbf0c882fe85e3e3d8bb3f7d180f77e及已有未提交工作区。
- 13项迁移、68文件、26,814,304字节；[清单](task_data/organization_manifest.json)记录每个原路径、新路径及迁移前SHA256；旧位置保留相对链接。
- [验证](task_data/organization_verification.json)：67个迁移文件原字节保持，奖励文档仅修复1处链接；原544个资产文件均可经旧路径读取，543个保持哈希。source/scripts的215个Python文件与4个标定旧链接未变。
- Custom robot_contract实际加载、标准/Custom D-Peg及装配课程的真实加载函数（AST隔离Isaac依赖）通过；4份模板41项来源哈希通过。57个JSON（包括新迁移清单，不含后来生成的验证报告）解析通过。
- CPU命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_floor_contract.py`，12/12通过。
- GitNexus路径常量/数据目录impact为UNKNOWN；源码与生成器补查后维持旧路径和来源哈希，未改任务定义。全工作区detect-changes为CRITICAL，169文件/761符号/215流程，包含先前未提交改动；不是本次迁移的独立风险量化或干净图检查。
- 未运行Isaac/训练、未访问远端或真机、未删除失败候选、未提交。数据延续原本地忽略策略；跨机器携带task_data实体及兼容链接，不能只复制assets。

# CALIB-M6X16-20261004-01：默认URDF局部沉孔适配

- 用户授权修改默认URDF及STL以适配最长M6×16；本次仅CPU几何生成与校验，无硬件／训练／远端操作。
- 基线main/167fc3c；原工作区迁移及训练修改保留。原V4 BREP派生四个Ø12圆柱切削，Z=10以上；实际新增切除为Z=10–24环形材料，共4299.709369 mm³。总高／坐标不变。
- 可复现：`python3 calibration/simulation/build_m6x16_mount.py --output NEW_DIR`。默认包集成记录见下方integration_validation.json，原文件备份于同目录before。
- CAD与STEP有效单实体；STEP体积相对差1.319e-6（容差1e-5）；STL封闭、朝向一致、单连通，体积相对CAD差7.331e-5。四处原曲面微短边收拢最多0.001326 mm，两处补面最大抽样CAD距离0.001807 mm。
- 4×192 CAD孔位探针通过；四处头部／杆部包络与插入通道交叠体积0；STL射线确认底面0与承压面10 mm。默认URDF仅两处mesh引用变化，104引用均存在；坐标登记哈希同步。
- 头型假设Ø10×6 mm圆柱头、头下长16 mm；突出量6 mm（1 mm垫片则5 mm），实际螺纹啮合／承载未知，未运行Isaac或真机。
- GitNexus lower_mount、URDF及新函数查询UNKNOWN，引用核查限定到当前完整包。失败导出检查保存在calibration/outputs/m6x16_development，最终结果见[几何验证](calibration/assets/calibration_bench_visuotactile_urdf/modifications/m6x16/validation.json)及[集成验证](calibration/assets/calibration_bench_visuotactile_urdf/modifications/m6x16/integration_validation.json)。

# 实验索引与证据

## CALIB-COORD-20261004-04

- 用户指定默认建系资产：`assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf`，解析为calibration/assets下同一文件；文件哈希保存在[坐标登记](calibration/coordinate_reference.json)。
- 用户报告并进一步确认：右侧平台平面基面（避开圆形槽和贴码槽）的机械台显示Z为−308.5605 mm。非本助手采样，非某根手指的接触零点；XY、测力／接触阈值、重复性和测点位置未知。
- XML核对：calibration_world与lower_mount重合，calibration_plane世界Z=0.036 m，XYZ零关节尖端Z=0.0480797829510025 m。尖端触平台时的Z关节参考为−0.0120797829510025 m。
- 在Z同向且读数代表尖端触平台的前提下，z_world_m=0.036+(z_display_mm+308.5605)*0.001；轴方向／TCP待核对，未启用自动转换，更不构成完整6DoF外参。
- 已检查路径别名、URDF哈希及零点代入算术；未改URDF／历史轨迹、未控制硬件、未启动仿真。建系默认与测量原值已写入总入口／主说明／STATUS。

## CALIB-REORG-20261004-03

- 用户要求将标定及获取数据相关内容集中归类到calibration。main/167fc3c79fbf0c882fe85e3e3d8bb3f7d180f77e及原未提交工作区；无远端／真机／训练操作。
- 20项迁移与共享依赖索引见[组织清单](calibration/organization_manifest.json)，总入口为[calibration/README.md](calibration/README.md)。保留旧路径相对符号链接，主机械说明沿用用户已经移动的文件。
- 迁移1349文件、444093930字节；移动当时全部哈希一致；后续仅修订路径／文档和运行缓存。最终1272个非代码／非文档文件仍字节一致，旧NPZ／JSON／USD／CAD／网格等未改写。
- 48个Python文件AST、函数／类签名保持、12个URDF的606处mesh引用检查通过。原有用户训练修改哈希保持。
- 新旧run.py --plan-only均为270回合／55890帧。原checks.py 5项通过；新旧硬件／采集器离线help和新trimodal help通过，未打开串口或相机。旧scripts.project_vitai_markers_to_mesh、image_generator及integrate压力辅助导入通过。
- CPU迁移短测：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python calibration/simulation/run.py --episodes 1 --output calibration/outputs/organization_20261004/smoke`；随后同解释器verify.py及export_urdf.py --scene-dir该目录 --output calibration/outputs/organization_20261004/exported_urdf。
- 1/1回合完整、115帧、15/15数据检查通过，URDF另存完成；仍有原开口网格诊断，不认证物理／连续净空。未启动Isaac或GPU，不是物理标定实验。
- 压力L0入口尝试4×4网格／4步，但在输出前因当前解释器缺isaaclab.utils导入失败；用迁移前备份源码复现同一异常，属于已有环境约束，未改共享包或安装依赖。
- GitNexus按本仓库做迁移前文件影响分析，最高MEDIUM，UNKNOWN补引用查证。全工作区detect-changes --scope all后提高limit重跑，仍CRITICAL，包含路径迁移与既有用户训练差异；不作干净图验收或提交。
- 旧复制marker报告12个缺目标链接保留为历史限制；新导航、原始数据与资产检查见[完整核验](calibration/outputs/organization_20261004/organization_verification.json)。该输出目录也保存迁移前逐文件哈希和图分析记录。

## CALIB-SDK-20261004-02

- 授权延续本机SDK检查；用户重接相机后要求复测，并明确指尖有接触或状态不确定。
- 原始采集：`bash calibration/probe_local.sh --output calibration/captures/revo3_reconnected_20261004`，
  退出0；五路各8帧640×480 RGB，同时读取21关节和244有效点，拇指指腹禁用状态保持。
- 新增`calibration/probe_revo3_trimodal.py`，读取该批源帧，不重新打开硬件；各第0帧临时参考，
  第1–7帧SDK逐帧推理，lowcost缓存关闭，45秒/相机超时。`--reference-state unverified-current`，
  所有输出`physical_calibration_validated=false`。没有假定空载、运动手部或修改设备参数。
- 失败保留：默认SDK320×240与采集640×480不符，首次`InvalidFrameSizeError`存于
  原始目录`inference_trial.log`。后续显式匹配stream_format并保留工厂model/output/marker设置。
- 本机13:55:07–13:55:12批量推理，5/5路、35组成功。Depth：GFBCT180×320、GFBCI240×320，float32；
  Marker：20×20×2 int64光流采样网格，非400颗物理marker。GFBCT位移全0，四路GFBCI在−1～1；
  全部offset=current−origin，不据短窗变化认定物理响应正确。SDK深度原单位未标定。
- 独立60/60核查通过：源／模型哈希、帧索引与时间戳保持、RGB/BGR对应、参考图、有限值、维度、
  Marker差分及诊断标签。原始40帧与35组三模态可追溯；未使用旧图像替代本次输入。
- GitNexus使用当前仓库，新增符号UNKNOWN并补全文本引用核查，无现有调用者；README影响LOW。
  未修改SDK原生包、原采集函数或训练代码，无提交；未进行五路并发、受控加载或物理标定验收。
- [源采集报告](calibration/captures/revo3_reconnected_20261004/report.json)、
  [三模态报告](calibration/captures/revo3_trimodal_20261004_diagnostic/report.json)、
  [独立核查](calibration/captures/revo3_trimodal_20261004_diagnostic/validation.json)、
  [预览](calibration/captures/revo3_trimodal_20261004_diagnostic/preview.png)、
  [接口与命令](calibration/机械标定台仿真搭建_README.md#76-五路三模态通路验证2026-10-04相机重新接通后)。

## CALIB-SDK-20261004-01

- 授权：用户要求配置本机已连接视触觉手SDK、检查三模态，并补充机械标定README。
  工作区main/167fc3c79fbf0c882fe85e3e3d8bb3f7d180f77e，保留原6个文件的未提交改动。
- 新增`calibration/probe_local.sh`复用源SDK Python3.11环境和仓库内原生包；未安装／升级包。
  GitNexus绑定本仓库绝对路径，刷新索引后原探测入口及README影响LOW，无受影响流程；
  新脚本／维护文档UNKNOWN，补充文本引用核查，未修改现有采集函数、SDK或训练入口。
- 实测：`bash calibration/probe_local.sh --output calibration/captures/revo3_readonly_20261004_local_entry`，
  北京时间13:44:49–13:44:50；退出码2。21关节×3次、244有效点×3次成功，拇指57点禁用；
  独立维度／有限值与禁用状态核查18/18通过。SDK和系统均未枚举ViTai相机，0/5路，三模态未读通。
- 额外验证：shell语法和离线帮助通过；深度／Marker模块可导入，ONNX Runtime1.20.1 CPU
  加载GFBCI/GFBCT深度模型成功，路径、形状与SHA保存；无当前图像推理、空载标定或同步验收。
  两次探测均无运动／归零／使能修改，健康异常原值保留。未接触机械台或远端设备。
- [最终报告](calibration/captures/revo3_readonly_20261004_local_entry/report.json)、
  [独立核查](calibration/captures/revo3_readonly_20261004_local_entry/validation.json)、
  [首次探测及SDK预检](calibration/captures/revo3_readonly_20261004_connection/)、
  [接口与后续前提](calibration/机械标定台仿真搭建_README.md#75-本机连接配置与三模态复测2026-10-04)。

## ASM-REWARD-DDP-20261004-04

- 当前授权：用户明确指定SSH主机`dsw-wpmt7kzoh4b53jpmov`、项目目录
  `/home/admin/workspace/xinyu/mulyi-model-tac`，两张卡各4096环境开启课程训练。
  按本轮已讨论/修改任务，执行横躺装配课程，不是独立Randomized旋出。
- 前置：旧`assembly_floor_v2_20261003/ddp4096_curriculum`已Stage0预算结束，守卫finished/exit2，
  model1999保留。两卡均空闲、无活动训练/回放，不需要停止其他作业。
- 部署：复制旧完整冻结副本的source/scripts/assets到`.workspaces/assembly_reward_ddp4096_20261004`，
  只覆盖课程env_cfg/rewards/README和CPU检查入口4项。旧基线19项匹配，候选28项源码/资产匹配本地；
  原工作区仍匹配旧哈希。根目录缺完整课程模块，初次暂存的新4文件在确认此前不存在且哈希匹配后撤回，
  避免留下不完整课程包；实际运行完整新冻结副本。`sync_verification.json`保留这一过程。
- 预检：服务器brainco/Isaac5.1环境的11项CPU检查通过，启动器dry-run命令核对通过；未迁移旧检查点。
- 作业：`experiments/assembly_reward_ddp4096_20261004/train`，2026-10-04 13:35:22 Asia/Shanghai启动。
  守卫2408（start_ticks186895）、launcher2410、torchrun2411、rank0/1为2417/2418。
  CUDA_VISIBLE_DEVICES=0,1，NCCL双进程共同训练，每rank4096，共8192；seed42/43。
  物理120Hz、控制30Hz、Stage0回合5秒，每阶段2000轮，独立晋级seed401/402/403各100完整回合。
- 实际配置：reach_best_progress weight40/std0.4，any_finger_contact weight0.7/threshold1N；
  action_l2−0.02、action_rate_l2−0.01、arm_path−0.08、base_excursion−0.5，resume=false。
  每rank碰撞栈4026531840字节，配对容量三项各2^24、接触点2^24、接触片2^20；成功和失败规则保持。
- 13:43:36验收：迭代0/1/2完成、model0保存；首三次PPO各20次梯度均值同步、参数最大差0，
  first_gradient_mean_verified=true，两rank backend=nccl、device cuda0/1。前三轮23.08/16.62/14.98秒，
  即时显存24811/24772MiB。新增两项已产生非零日志奖励；无PhysX/CUDA/丢接触/溢出/Traceback，guard仍running。
- 启动阶段Kit与物理初始化较慢，但日志持续推进；读取完整41MB日志时一个临时正则诊断消耗CPU，
  按PID/父PID/命令确认后只停止该只读诊断PID3163，训练未中断。验收脚本改为逐行固定模式扫描并通过。
- [本地证据](logs/diagnostics/assembly_reward_ddp4096_20261004/)：精确4文件patch、前后哈希、launch.sh、
  startup_audit.py；server子目录含实际env/agent配置、动作/观测契约、DDP审计、启动/同步身份和验收JSON。
  远端同名experiment目录含完整train.log、supervisor.log、guard状态和逐rank课程日志；旧日志/模型未覆盖。
- 边界：只证明训练正常开始、奖励接线和DDP生效；未完成独立成功评测，不宣称学会抓持或晋级。
  当前训练继续运行，不因本次交接自动停止或重启。未提交Git，已有本地用户改动保留。

## ASM-ANY-CONTACT-20261004-03

- 授权：用户要求Stage0也奖励any_contact，参考本机dextac-Depth-main的接触引导。
- 源码：参考库`dextac-Depth/source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/`
  下`mdp/rewards.py:581`的`any_finger_contact`与`config/flexiv/dexsuite_flexiv_env_cfg_grasp.py:1110`
  的weight0.7/threshold1N。当前库已有适配函数`RotateBulbCustom/assembly_env_cfg.py:assembly_any_finger_contact`，
  但课程`configure_assembly_rewards`会将父配置同名项清空；此次在账本替换后仅Stage0重新注册。
- 语义：任意一指对灯泡的过滤力>1N即+0.7分/秒，30Hz每步+0.02333，5秒最多3.5分；
  不需双侧、低速或稳定时间，不按手指数加倍，按源实现无额外失败/抬起门控；不奖励桌面/灯座总接触力。
  原接触质量确实排除单指，但双侧>0.1N可有部分质量分；不能说所有接触信用都要求>1N的优秀接触。
- 范围：仅课程env_cfg接线与诊断/文档；共用接触读取/判据函数未改。
  保留历史接近+1分/cm、所有动作/臂路程/偏离成本、原双侧稳定奖励、+8稳定完成和reset规则；
  未另加参考good_contact +2分/秒。Stage1–4与独立任务未启用新增项。
- 验证：`/home/liuxinyu/miniconda3/envs/brainco/bin/python scripts/diagnostics/check_bulb_floor_contract.py`
  11/11通过；执行真实过滤力读取函数和接触函数，覆盖每根单指、恰等于阈值、多指不叠加、
  NaN/Inf/None、无灯泡力但总力很大、30/60/120Hz积分。实际两层配置函数检查仅Stage0启用、原成本保留。
  `git diff --check`通过。无GPU仿真、远端同步或训练，未验证接触传感器动态真实性或学习效果。
- GitNexus：当前仓库configure_curriculum影响LOW，1调用者/1流程；共用接触函数UNKNOWN且不修改函数体，
  文本引用核查确认父配置注册与课程覆盖路径；新测试方法未索引，按CPU诊断入口补查。未提交。
  本机未发现活动训练/回放进程，已有用户和上一轮奖励改动均保留。旧检查点仍需匹配冻结源码契约。

## ASM-BEST-REACH-20261004-02

- 用户授权：将`/home/liuxinyu/workspace/dextac-Depth-main`的历史最佳接近奖励移植到第一阶段，
  在保留机械臂动作惩罚的基础上调整量级。第一阶段沿本会话指Stage0抓持。
- 原实现：`dextac-Depth/source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/rewards.py:465`
  的`object_ee_distance_delta`，以及`dexsuite_env_cfg_flexiv.py:432`的weight200/std0.4。
  原项返回历史最小最大末端距离的改善量/std，经RewardManager乘dt；不是tanh势能之差。
- 新实现：课程`rewards.py:FloorBestReachReward`复用已有`assembly_hand_shell_distance`，
  对五指尖取最大距离，目标适配灯泡局部+X 4cm的壳体中心。只在课程env_cfg的Stage0注册新项，
  原有FloorReachReward与所有动作/机械臂代价保留；独立Custom和Stage1–4不新增该项。
- 量级：weight40/std0.4，函数除dt，最终100分/m，即每新靠近1cm给+1分；3–5cm改善给+3～5。
  同30Hz下原参考项每cm约+0.1667，新项为6倍。q1单向30°路程成本−0.04189为量级示例，
  并非抓取实际轨迹/总代价；完成抓持仍+8，未以奖励替代成功判据。
- 生命周期：首个有效奖励样本建基准；局部reset只清对应环境；远离再返回不重复付分；
  非有限距离之后重新建基准；失败终止步/已抬起后不支付。无隐藏物理步或额外reset。
- 验证：brainco解释器运行`python scripts/diagnostics/check_bulb_floor_contract.py`，10/10通过。
  新增4项覆盖dt等积分/往返、局部reset/NaN/Inf、失败/抬起门控，以及实际两层配置函数在0–4阶段接线与代价。
  配置测试首轮fixture缺Stage3的insertion_tracking而失败，补齐fixture后通过；未因此改生产Stage3。
  `git diff --check`通过。原生仿真、训练收益、成功率均未验证；未使用GPU或连接远端。
- 兼容：原动作/观测/物理未改，但课程契约记录env_cfg/rewards源码哈希，因此旧检查点严格恢复会拒绝。
  未篡改旧契约或状态库；旧模型保留用于原冻结代码回放。本机进程检查未发现活动训练/回放。
- GitNexus：明确绑定当前工作区，索引commit167fc3c；configure_curriculum影响LOW（1调用者、1流程），
  复用距离函数LOW（5项上游依赖，函数未改）。新类/诊断类/文档UNKNOWN，补查引用与实际配置路径后限定新增；
  未把UNKNOWN作为无影响证明。两份用户已有noise兼容改动保留。未提交。
- `detect-changes --scope all`返回MEDIUM、1条课程配置流程，包含已有用户改动；
  索引行号将新类插入区间关联到旧FloorReachReward等符号，源码diff确认旧函数体未改。
  [离线量级/来源哈希报告](logs/diagnostics/assembly_stage0_best_reach_20261004/report.json)保存
  8→7→7.5→7→5cm序列在30/60/120Hz下均为+1/0/0/+2、合计+3分的实际函数计算。

## ASM-GRASP-DIAG-20261004-01

- 范围：用户询问横躺Stage0不下降/接触不稳是reset还是奖励原因；仅本地源码、已有轨迹与检查点分析。
- 来源：`checkpoints/assembly_floor_v2_20261003/stage0/model_1999.pt`、同目录`curriculum_episodes.jsonl`；
  `recordings/assembly_floor_v2_20261003/model400_seed401/trajectory.npz`及`summary.json`。
  用户补充末轮速度54.92%、无双侧接触42.11%、slip37.76%、pose_error1.13%，未取得对应原始日志。
- model1999用CPU/weights_only读取：Stage0、iter1999，记录524288000转移/3517685完整回合，
  scheduler.last_rate=0、ready=false；不是全部历史回合成功率。契约9份Custom+8份课程源码哈希均匹配本地。
  回放JSONL的3回合各150步，全部timeout、无grasp/稳定成功、无失握/掉落/越界/reset恢复失败；
  该文件没有逐项奖励，不将此小样本代替训练总体或三seed评测。
- model400单轨迹取frame0..149，排除frame150的自动reset：掌z初始0.101338m、最低0.095061m、
  最高0.138576m；7臂关节各变化5.19°–11.00°。按当前FloorReach公式重算149段约+1.7239，
  按动作序列重算action_l2=-0.00781、action_rate_l2=-0.01219；记录总reward约+1.6816。
  重算不是完整逐项奖励审计（缺接触力/实际关节速度，状态/奖励采样相位亦需核验），
  不能用位置差分冒充arm_path实际速度积分，也不能外推到model1999。
- 源码：runtime的HoldCounter在`grasp & slow & stable_relative`失败时归零，并不直接reset。
  环境reset仍由timeout、掉落、失握、越界、非有限状态、阶段成功等终止项决定。
  reward_v2的no_contact=`~strict`、velocity=`strict & ~slow`；同一统计口径下，
  54.92/(100-42.11)=94.87%的双侧接触步速度不合格，接触且低速约2.97%，不能据此推31步成功概率。
  slip为相对线速>0.01m/s或角速>0.2rad/s并带抓持记忆门控；pose_error为5mm/5°偏差。
- 奖励疑点：持续接近信用仍在接触后生效至lifted；接触质量取双侧最小力并经0.1秒最小值过滤；
  grasp_hold仅奖本回合历史最长保持的新纪录，计时中断后重走旧长度不再获该项正信用。
  滑移/速度惩罚持续发生，但没有逐项实测证明总惩罚压过接触奖励。动作每步关节目标增量scale=1rad，
  接触后无专门减速控制，控制尺度/探索也是待验证因素。
- 判断：优先查稳定控制与奖励信用，不能认定硬性无法下降或环境reset过早。下一步建议记录连续接触段、
  hold中断原因、真实termination、动作目标/实际速度/接触力和奖励分项，再做保持严格验收的单因素对照；未实施。
- GitNexus绑定当前仓库，索引落后2提交；context用于导航，事实以源码及契约哈希为准。
  文档impact为UNKNOWN/目标未索引，补查README/AGENTS引用，仅新增记录，不改入口结构或运行代码。

## P3-ENV-20261003-02

- 范围：在同一 RTX5090 主机建立不污染既有环境的 Isaac Sim 6.0.1／Isaac Lab 3 beta 环境，验证训练与 RTX play，并探测原 stage0 checkpoint。
- 环境：`~/workspace/xinyu/assembly_play_sim6`；Isaac Sim 6.0.1.0、Isaac Lab `v3.0.0-beta2.patch1`、RSL-RL 5.0.1、Python 3.12、torch 2.11.0；驱动 595.91.07。
- 训练通过：`Isaac-Cartpole-v0`，PhysX，64 环境，8 次 PPO 更新，8192 步，保存 `model_7.pt`；进程正常退出。
- 回放通过：加载上述模型，RTX 连续渲染 120 帧，H.264 1280×720、60fps、2秒；SHA256 `00ada0eb9e0d03e1a18a46eb325b49cff025f7310b6f5671f9ba792bc80662f8`。首次相机启动约 80 秒用于 shader／streaming 初始化。
- 原模型：5.1 环境已加载 `stage0/model_1000.pt` 并完成 3 回合／450 transition，无 reset 失败、0/3 成功；这是链路证据，不是效果通过。
- 6.0.1 阻塞：任务注册可见 13 个 assembly task；首个 API 差异为 `AdditiveUniformNoiseCfg`→`UniformNoiseCfg`，已加入双版本回退。继续使用 Lab 3 官方入口时，项目模块在 Kit 启动前导入 `isaacsim.core`；旧项目入口又不符合 Lab 3 的新启动生命周期。因此原 checkpoint 尚未在 6.0.1 中进入策略推理，不得声称兼容。
- 依赖限制：`pip check` 保留上游 beta 的 `packaging`／`coverage` 声明冲突；采用 Sim 6 固定版本后，CUDA、官方训练和 RTX 回放实际通过。
- 远端证据：`~/workspace/xinyu/assembly_play_support/` 下 `sim6_gui_smoke.log`、`sim6_cartpole_train.log`、`sim6_cartpole_play_video.log`、`sim6_original_checkpoint_official_play*.log` 和 `sim6_project_*probe.log`。

## P3-ENV-20261003-01

- 范围：用户授权5090主机同步、环境配置和指定模型GUI回放；要求隔离其他环境。
- 项目：main/167fc3c79fbf0c882fe85e3e3d8bb3f7d180f77e；初始同步9330文件/4275539838字节，校验和无差异。
- 目标：brainco1@192.168.10.252，`~/workspace/xinyu/assembly_play_conda`，conda-forge Python3.11。
- 版本：IsaacSim5.1.0.0、IsaacLab2.3.2.post1、torch2.7.0+cu128、torchvision0.22.0+cu128、RSL-RL3.1.1、warp1.13.0。
- 通过：pip check；5090 CUDA 64×64全1矩阵乘积均值64；最终sys.path独立，不引用其他项目环境。
- 模型：assembly_floor_v2_20261003/stage0/model_1000.pt，SHA256
  `5efb857c8e009924cb3c479b4c6ffdf49df7834f0752bedcf04a4172991b38ff`；v2/stage0/369观测/28动作/120Hz。
  内嵌bank契约的17份任务源码哈希与当前一致；原目录缺sidecar，已在独立回放副本旁从内嵌契约与源码重建，非原训练目录下载。
- 失败：GUI进程552140在2026-10-03 22:37（上海）app ready后于librtx.scenedb.plugin.so崩溃，未到模型加载。
  系统驱动595.91.07，与[NVIDIA已知问题](https://forums.developer.nvidia.com/t/isaac-sim-5-1-crashes-on-startup-with-rtx-5060-ti-blackwell-sm-120-rtx-scenedb-plugin-crash/366252/2)相符；没有驱动对照，未确诊唯一根因。
- 证据：远端 `~/workspace/xinyu/assembly_play_support/` 的README.txt、environment.yml、requirements.freeze.txt、
  conda-pip-install.log、project-install.log、play_stage0.sh、play.log及stage0_model1000/preparation_provenance.json。
- 未验证：模型运行、场景物理、成功率。未改系统驱动/其他环境、未启动训练；后续需解决图形兼容并重新验证。
- 后续更新：同一 5.1 隔离环境已通过禁用 RTX 的无界面评测加载模型并运行 3 回合；GUI 崩溃结论仍成立。详见 `P3-ENV-20261003-02`。

本文件维护实验索引、复现缺口及新记录模板；实现详情沿用已有专题文件。
核查日期 2026-09-20。原始历史记录为**读取已有证据**；新增灯泡初态验证为本次实际运行，单独标明。
历史 commit 缺失时不得补填当前 HEAD；本次代码版本见 [STATUS](docs/STATUS.md)。
2026-10-03 按用户要求清理了 `docs/` 下带日期后缀及 experiment(s) 命名的 52 份历史日志；
索引内指向这些专题文件的旧链接仅作为路径记录，不再保证可打开，汇总结论仍保留在本文件中。

## 索引

| 唯一 ID | 日期 | 类型 / 状态 | 可支持的结论 |
| --- | --- | --- | --- |
| [REVO3-CALIBRATION-MOUNT-20261003-06](#revo3-calibration-mount-20261003-06) | 2026-10-03 | 底面截短以适配 M6×20 | 底面去除10mm；名义伸出6mm，CAD/STEP/STL/3MF/URDF检查通过，未实装 |
| [CALIB-TOUCH-20261003-04](#calib-touch-20261003-04) | 2026-10-03 | 视触觉手替换、任务坐标对齐与装配 | URDF/FK、8回合、五指与Isaac检查通过；拇指/支架干涉待复核 |
| [ASM-FLOOR-20261003-01](#asm-floor-20261003-01) | 2026-10-03 | 横躺课程v2、原生与性能对照、试训 | CPU与双进程检查通过；256基准物理步减半吞吐+68.5%；1024溢出后扩容通过；双卡PPO/恢复/独立评测链路通过，每卡4096已启动 |
| [CALIB-SDK-20261003-04](#calib-sdk-20261003-04) | 2026-10-03 | SDK副本与真机只读短测 | 21关节、244有效触觉点、5路40帧图像；拇指指腹禁用，深度／marker与对齐未验收 |
| [CALIB-URDF-20261003-03](#calib-urdf-20261003-03) | 2026-10-03 | 精简场景可动URDF交付 | 68link/67joint，压头XYZ＋原手21关节；XML/urdfdom/mesh/121帧FK通过，未验收动力学 |
| [CALIB-SCENE-20261003-02](#calib-scene-20261003-02) | 2026-10-03 | 按用户要求移除虚拟机构 | 仅保留压头／法兰与手／V4支架；保留几何和动画逐字节相同，Isaac回放／截图通过 |
| [CALIB-SIM-20261003-01](#calib-sim-20261003-01) | 2026-10-03 | 独立机械标定模型与原生场景短测 | 球头8/8回合1574帧、99项检查；三角/锥头各1/1；USD54网格与回放通过，不是独立物理标定 |
| [REVO3-CALIBRATION-MOUNT-20261001-05](#revo3-calibration-mount-20261001-05) | 2026-10-01 | V4仅扩大两槽，不拆根部/挡板 | 40mm球槽/55mm方槽，整段腕部与原V3一致，CAD/STL/STEP/URDF通过 |
| [REVO3-CALIBRATION-MOUNT-20261001-04](#revo3-calibration-mount-20261001-04) | 2026-10-01 | 矩形V3、球槽/贴码槽及URDF | CAD/封闭STL/STEP回读/URDF通过，保留V2配合；未做实际打印标定 |
| [REVO3-STABLE-MOUNT-20261001-03](#revo3-stable-mount-20261001-03) | 2026-10-01 | 新增固定腕部下支架V2及装配 | CAD/闭合STL/URDF通过；腕部穿入同批采样201/5000→0/5000；需绑带和实物验收 |
| [REVO3-BREADBOARD-URDF-20261001-02](#revo3-breadboard-urdf-20261001-02) | 2026-10-01 | 指定外部Revo3右手＋下支架URDF装配 | 两版解析/源资产保留通过；最终移除上支架，局部干涉待复核 |
| [CALIBRATION-BENCH-URDF-20261001-01](#calibration-bench-urdf-20261001-01) | 2026-10-01 | 双Y标定台Blender分组导出URDF与CPU验证 | base→Y→X→Z四link、两套限位解析及确定性重建通过；未验证动力学或连续碰撞 |
| [BULB-RESTORE-GPU4-20261001-20](#bulb-restore-gpu4-20261001-20) | 2026-10-01 | 按用户要求恢复两端Randomized旋出代码 | 三文件与GPU4快照SHA一致，两端各114项CPU通过；无新训练/原生评测 |
| [BULB-TRAINING-COMPLETION-20261001-19](#bulb-training-completion-20261001-19) | 2026-10-01 | 服务器全程事件及完成状态只读核查 | GPU6跑满15000轮但成功全0；旧GPU4末100轮训练成功94.23%，无新独立评测 |
| [BULB-REACHABILITY-AUDIT-20260930-18](#bulb-reachability-audit-20260930-18) | 2026-09-30 | 保存的IK见证及录像CPU正运动学复算 | 所有目标有无需160–180°后转的几何解；录像后转明显偏离位置区域，非动态成功/因果证明 |
| [BULB-ROOT-FRAME-AUDIT-20260930-17](#bulb-root-frame-audit-20260930-17) | 2026-09-30 | 冻结源码/组合USD/保存的原生姿态只读核查 | 根无翻转，任务在−X侧，目标/观测转换一致；无新实时服务器快照 |
| [BULB-JOINT1-ASSET-EXPLORATION-20260929-16](#bulb-joint1-asset-exploration-20260929-16) | 2026-09-29 | 冻结资产/策略噪声只读核查 | 无单侧硬限位/固定后转目标，joint1 std已接近下限；部分状态反向采样概率低，非因果消融 |
| [BULB-JOINT1-REWARD-AUDIT-20260929-15](#bulb-joint1-reward-audit-20260929-15) | 2026-09-29 | model1000后转奖励逐步复算/训练日志只读 | 后转阶段正奖励近0，姿态扣分生效但弱；速度少扣与远离约束缺口可疑，未作因果消融 |
| [BULB-JOINT1ZERO-VIDEO-20260929-14](#bulb-joint1zero-video-20260929-14) | 2026-09-29 | GPU6 profile6 model1000回放下载 | 36秒3回合超时，无释放/抓持，视频解码/下载校验通过；非独立评测 |
| [BULB-JOINT1ZERO-WEAKPOSE-GPU6-20260929-13](#bulb-joint1zero-weakpose-gpu6-20260929-13) | 2026-09-29 | 停GPU6，新零位/±180/较弱姿态项从零训练 | CPU57/原生12/pilot13轮/正式更新保存通过，效果待评估 |
| [BULB-JOINT1-ZERO-20260929-12](#bulb-joint1-zero-20260929-12) | 2026-09-29 | 本地零位重定义与±180限位，原生验证 | 原物理初态保持，CPU32/原生9项通过，端点约+180/-179.71度；未替换训练 |
| [BULB-POSE90-SCRATCH-20260929-11](#bulb-pose90-scratch-20260929-11) | 2026-09-29 | 停GPU6，90度免罚从零启动 | CPU53/原生12/32环境13轮及正式初始化更新保存通过，学习效果待验证 |
| [BULB-GPU6-VIDEO-20260929-10](#bulb-gpu6-video-20260929-10) | 2026-09-29 | GPU6 model1500短回放下载 | 36秒3回合全超时，无释放/抓持；joint1约-42至-7度，非独立成功率评测 |
| [BULB-GPU6-PLATEAU-AUDIT-20260929-09](#bulb-gpu6-plateau-audit-20260929-09) | 2026-09-29 | GPU6终端与同轮学习只读核查 | 未见训练期报错/非有限值；旋转继续进步，手掌接近和释放后抓取滞后，未证明因果 |
| [BULB-INITIAL-POSE-LEARNING-AUDIT-20260929-08](#bulb-initial-pose-learning-audit-20260929-08) | 2026-09-29 | GPU4/GPU6同轮日志与实际配置只读核查 | 仅新增姿态项；早期成本相对稀少正奖励大，GPU6进度回升，未证明因果或卡死 |
| [BULB-SUCCESS-VIDEO-20260929-07](#bulb-success-video-20260929-07) | 2026-09-29 | GPU4 model6600成功回放并下载 | 训练近100点约81.8%；48秒录像完整回合成功4/5，保留超时及末尾未完成回合，非独立成功率评测 |
| [BULB-INITIAL-POSE-SCRATCH-GPU6-20260929-06](#bulb-initial-pose-scratch-gpu6-20260929-06) | 2026-09-29 | 停GPU6旧作业，新奖励从零训练 | 原生10项/32环境13轮通过，6144从零初始化、更新/保存/新奖励验证通过；未评测学习收益 |
| [BULB-INITIAL-POSE-GUIDANCE-20260929-05](#bulb-initial-pose-guidance-20260929-05) | 2026-09-29 | 本地新增弱姿态成本、CPU与旧轨迹尺度验证 | 68项通过，新增成本约占旧回报3.68–4.15%；未部署或验证学习收益 |
| [BULB-JOINT1-TRACE-20260929-04](#bulb-joint1-trace-20260929-04) | 2026-09-29 | 同模型1440步动作轨迹诊断 | 四回合先转至joint1负限位再释放；姿态门控未覆盖绕转，训练保持 |
| [BULB-ARM-PENALTY-AUDIT-20260929-03](#bulb-arm-penalty-audit-20260929-03) | 2026-09-29 | 冻结源码与实际配置只读核查 | 姿态参考为抓持时刻且0.20rad封顶，释放后关闭；不是持续初态约束，未改训练 |
| [BULB-VIDEO-20260929-02](#bulb-video-20260929-02) | 2026-09-29 | GPU4最新保存点model5100回放并下载 | 约48秒/1439帧完整解码，两端与桌面哈希通过；训练保持，非独立成功率评测 |
| [BULB-GPU4-PROGRESS-20260929-01](#bulb-gpu4-progress-20260929-01) | 2026-09-29 | GPU4事件只读分析 | 释放日志99.91%、释放后接触0.799、搬运奖励非零；首次正成功3197轮，最近100点成功率日志0.113%，非独立评测 |
| [BULB-GPU4-PHYS60-SCRATCH-20260928-06](#bulb-gpu4-phys60-scratch-20260928-06) | 2026-09-28 | 停旧GPU4＋7，新GPU4从头训练 | CPU76、32环境13次PPO/原生10项/超时重置通过；6144正式到3轮，更新/保存/配置/契约及哈希通过 |
| [BULB-TIMING-CONFIG-20260928-05](#bulb-timing-config-20260928-05) | 2026-09-28 | 本地临时配置，未部署训练 | 用户修正为控制30Hz：物理60Hz/decimation2/12秒，CPU39项通过；原生物理及性能未测 |
| [BULB-PHASE-CONTACT-20260928-04](#bulb-phase-contact-20260928-04) | 2026-09-28 | 分阶段接触/臂速奖励，GPU6从model3500续训 | CPU60、原生10及pilot三次PPO通过；6144正式到3504、模型/优化器更新及契约通过；学习收益待评测 |
| BULB-EPISODE-TIMING-20260928-02 | 2026-09-28 | 当前日志只读核查＋历史profile重算 | 23.7033秒/轮、采样98.64%；历史物理84.35%，未新开GPU实验；[报告](docs/rotate_bulb_episode_timing_2026-09-28.md) |
| [BULB-VIDEO-20260928-03](#bulb-video-20260928-03) | 2026-09-28 | 当前接触0.1双卡run最新保存点回放并下载 | 选取时model3300；40秒/1200帧完整解码及两端哈希通过，原训练保持；非独立评测 |
| [BULB-TENSORBOARD-20260928-02](#bulb-tensorboard-20260928-02) | 2026-09-28 | 双卡旋出只读核查、本机TensorBoard接通 | 到3383轮/model3300；近100点释放99.976%，完整成功0，24.098秒/轮；非独立评测 |
| [BULB-JOINT-SOLVER-20260928-01](#bulb-joint-solver-20260928-01) | 2026-09-28 | native拓扑、128分组、256短训及16/4适应完成 | Robot/手指链主导无接触内部约束；16/4吞吐+19.6%，model234评测释放13/32、完整成功0/32；6144未验证 |
| [BULB-COLLISION-FAST-20260927-11](#bulb-collision-fast-20260927-11) | 2026-09-27 | 碰撞资产/自碰撞实改、固定动作筛选、适应训练进行中 | 1024自碰撞关闭名义吞吐+26.18%；螺纹/Shell无收益；6144与适应后任务进展待完成 |
| [BULB-CONTACT01-RESUME-DDP-20260927-09](#bulb-contact01-resume-ddp-20260927-09) | 2026-09-27 | 接触回退0.1，从指定双卡run内model100续训 | GPU4＋7共12288环境，完整恢复/正式更新/两卡同步通过；到110轮，新奖励学习收益待验证 |
| [BULB-CONTACT-RELEASE-DIAGNOSIS-20260927-08](#bulb-contact-release-diagnosis-20260927-08) | 2026-09-27 | 最新model1300录像、256回合及机械释放诊断 | 回放释放0/256、均最高2.879圈、拇指力0；人工解锁/自由落体通过，支持奖励收益冲突，未改训练 |
| [BULB-RELEASE-DECLINE-20260927-07](#bulb-release-decline-20260927-07) | 2026-09-27 | 释放下降趋势与奖励收益分析 | 释放及达到三圈下降、reward上升；支持释放前停留收益冲突，未做因果消融/改训练 |
| [BULB-REWARD-RANDOMIZATION-AUDIT-20260927-06](#bulb-reward-randomization-audit-20260927-06) | 2026-09-27 | 当前奖励/位置随机化只读核查 | 17个非零奖励；两rank启用3456组支架/目标配对随机化，含目标Z+28–42cm |
| [BULB-RELEASE-LIFT-20260927-05](#bulb-release-lift-20260927-05) | 2026-09-27 | 新训练释放与目标接近日志分析 | 到604轮/model600；释放约3.3%，位置奖励微增，抓稳/成功仍0；未证实稳定抬升 |
| [BULB-TENSORBOARD-20260927-04](#bulb-tensorboard-20260927-04) | 2026-09-27 | 新双卡训练日志分析、本机TensorBoard接通 | 到516轮/model500；最近100轮23.546秒；训练成功率仍0，非独立评测 |
| [BULB-RUNTIME-PERF-20260927-03](#bulb-runtime-perf-20260927-03) | 2026-09-27 | GPU6旧作业暂停、6144/4096性能对照 | 三个case各8轮通过；PhysX84.49%/压力5.07%；6144为25.434秒、4096为20.034秒；GPU6已释放 |
| [BULB-CONTACT-TRANSPORT-DDP-20260927-02](#bulb-contact-transport-ddp-20260927-02) | 2026-09-27 | 接触/搬运10倍、双卡12288从头训练 | CPU163通过；正式12288环境PPO/同步/保存通过，前1–5轮19.39秒，学习效果待验证 |
| [BULB-PROGRESS-VIDEO-20260927-01](#bulb-progress-video-20260927-01) | 2026-09-27 | GPU6新奖励模型录像并下载 | model6500完整40秒；两端解码、契约及下载哈希通过；训练继续到6603，非独立成功率评测 |
| [BULB-PROGRESS-PENALTY-TRAIN-20260925-05](#bulb-progress-penalty-train-20260925-05) | 2026-09-25/27 | 停装配、GPU6新奖励训练及状态核查 | 6144环境，6553轮；释放指标99.917%但完整成功0；最近100轮20.184秒 |
| [BULB-PROGRESS-PENALTY-20260925-04](#bulb-progress-penalty-20260925-04) | 2026-09-25 | 两端项目新增回退/停滞成本 | 本地/服务器各134CPU通过、原生212检查通过；当前训练未重启，学习收益未验证 |
| [BULB-RELEASE-DIAGNOSIS-20260925-03](#bulb-release-diagnosis-20260925-03) | 2026-09-25 | 冻结模型释放轨迹与奖励诊断 | model_3700释放1/64、成功0/64；接近三圈后回退而保留前段回报，非持续卡门槛；未改训练 |
| [BULB-LATEST-VIDEO-20260925-02](#bulb-latest-video-20260925-02) | 2026-09-25 | 晚间两项最新模型回放并下载 | model_3700/model_6900各40秒；训练到3817/6940、近100轮日志成功率均0%，装配回放0/8；解码及哈希通过 |
| [BULB-LATEST-VIDEO-20260925-01](#bulb-latest-video-20260925-01) | 2026-09-25 | 两项灯泡最新模型回放并下载 | model_2900/model_5900各40秒；训练仍运行、近100轮日志成功率均0%，装配回放0/8；解码及哈希通过 |
| [ASM-PHYSX-RESUME-20260924-04](#asm-physx-resume-20260924-04) | 2026-09-24 | 停插销、扩大装配容量并在GPU6恢复课程 | 7项容量翻倍、碰撞栈2→3.75GiB；8192首试OOM留存，4096固定复测271/300通过，自由阶段PPO3422–3424无原溢出 |
| [BULB-PALM-HIGH-Z-20260924-03](#bulb-palm-high-z-20260924-03) | 2026-09-24 | 手心上方奖励、高Z随机目标、GPU4从头重训 | CPU134通过/2跳过，原生96项和3轮PPO通过；正式6144环境PPO0–3/保存/新指标核验通过，学习效果待评测 |
| [PEG-BULB-VIDEO-20260924-02](#peg-bulb-video-20260924-02) | 2026-09-24 | 当前插销/随机旋出模型服务器录制并下载 | 各40秒/1200帧，完整解码及哈希通过；保留插销脱手画面，非独立成功率评测 |
| [TRAINING-STATUS-20260924-01](#training-status-20260924-01) | 2026-09-24 | xinyu21训练状态只读核查 | 装配固定Stage0冻结288/300通过，自由阶段碰撞缓冲区溢出停止；随机旋出/插销仍训练，近100轮日志均值98.79%/97.88% |
| [ASM-REWARD-V2-8192-20260923-03](#asm-reward-v2-8192-20260923-03) | 2026-09-23 | GPU1/8192环境/每阶段9000轮重启 | PPO 0–8轮、模型/奖励/标签核验通过；显存约20.4GiB，早期成功0/8470，继续训练 |
| [ASM-REWARD-V2-STATUS-20260923-02](#asm-reward-v2-status-20260923-02) | 2026-09-23 | 新奖励Stage0结果及容量只读核查 | 2000轮/65536000转移，成功0/442422；显存峰值约24%，双侧接触建立/保持仍是瓶颈 |
| [ASM-REWARD-V2-TRAIN-20260923-01](#asm-reward-v2-train-20260923-01) | 2026-09-23 | 服务器同步、旧装配停止、新课程正式启动 | GPU5/1024env，新reward PPO0–7、模型/标签核验通过；阶段成功率与学习收益未验证 |
| [ASM-REWARD-V2-20260922-03](#asm-reward-v2-20260922-03) | 2026-09-22 | 新reward实施与1000环境短测 | CPU161通过/2跳过，Stage0与Full原生通过并停止；无学习效果结论 |
| [ASM-CURRICULUM-REWARD-INVENTORY-20260922-02](#asm-curriculum-reward-inventory-20260922-02) | 2026-09-22 | 全阶段奖励静态提取与设计分析 | 24项提取核验通过；列明实际权重/预算/门控和成功差异，未修改奖励或运行新训练 |
| [ASM-CURRICULUM-REWARD-AUDIT-20260922-01](#asm-curriculum-reward-audit-20260922-01) | 2026-09-22 | 服务器Stage0奖励及训练日志只读核查 | 接近/接触合并到curriculum_progress；438574回合阶段成功0，已到2000轮预算停止，非无接触 |
| [ASM-CURRICULUM-VIDEO-20260922-01](#asm-curriculum-video-20260922-01) | 2026-09-22 | Stage0最新model_1999服务器回放并下载 | 40秒/1200帧完整解码与下载哈希通过；固定灯泡阶段回放0/8，非完整装配评测 |
| [ASM-CURRICULUM-VIDEO-20260922-01](#asm-curriculum-video-20260922-01) | 2026-09-22 | 课程初态场景视频 | 48秒/1152帧完整解码、17项检查通过；Stage2/3明确标注几何示意，无策略或成功率结论 |
| [MOLD-BREADBOARD-MOUNT-20260922-05](#mold-breadboard-mount-20260922-05) | 2026-09-22 | 25mm孔网M6固定及打印定位配合 | 四个藏头通孔、两根定位柱与圆/长圆配合孔；CAD/STL检查通过，实际台面尺寸、牙深与打印配合待核实 |
| [MOLD-ROOT-OPEN-20260922-04](#mold-root-open-20260922-04) | 2026-09-22 | 手根截面开口与CAD/STL检查 | 再去底部6mm，合模出口52.4×40.4mm；占地220×249mm，通道验证通过，未实际插线/打印 |
| [MOLD-PRINT-TRIM-20260922-03](#mold-print-trim-20260922-03) | 2026-09-22 | 256mm平台裁边CAD与STL验证 | 两模具220×255mm，494个型腔面保留，新打印STL闭合；未实打，Brim余量有限 |
| [MOLD-ENTITY-ANALYSIS-20260922-02](#mold-entity-analysis-20260922-02) | 2026-09-22 | STEP结构与实体位置分析 | 两模具加三个拇指近根部小实体；隐藏指尖状曲面重复三份，小实体的设计用途仍未知 |
| [MOLD-SPLIT-20260922-01](#mold-split-20260922-01) | 2026-09-22 | 模具几何转换与CPU检查 | 两link法向开合URDF；原始/分件/参考CAD保留；下模7条网格开边，未验证物理仿真 |
| [PEG-VIDEO-20260921-01](#peg-video-20260921-01) | 2026-09-21 | 指定D-Peg model_3000录像并下载 | 40秒/1200帧完整解码、下载哈希与669维契约通过；非独立成功率评测 |
| [ASM-CURRICULUM-GPU7-20260921-04](#asm-curriculum-gpu7-20260921-04) | 2026-09-21 | GPU7旧旋出停止，课程正式启动 | 原生169通过、短PPO6轮、正式0–3轮及模型保存通过；单seed Stage0，尚无学习效果结论 |
| [ASM-CURRICULUM-IMPL-20260921-03](#asm-curriculum-impl-20260921-03) | 2026-09-21 | 本地课程代码、原生检查、短PPO/续训/冻结评测 | 124通过/4跳过；固定/自由与联合恢复可运行，Stage0冻结0/4；学习收益未验证 |
| [ASM-CURRICULUM-DESIGN-20260921-02](#asm-curriculum-design-20260921-02) | 2026-09-21 | 当前reward/reset审计及四阶段实施设计，未实施 | 确认单侧零接触与hover门控；提供联合状态恢复、调度、风险和3训练seed对照方案 |
| [ASM-CURRICULUM-20260921-01](#asm-curriculum-20260921-01) | 2026-09-21 | 只读训练统计、192回合独立推理及课程方案 | 抓取0/96确定性、1/96随机；两者稳定抬起及后续阶段均0/96，未改代码或实施课程 |
| [BULB-RESUME-GPU7-20260921-01](#bulb-resume-gpu7-20260921-01) | 2026-09-21 | 原三圈Custom最新模型在GPU7续训 | model_1400完整恢复，1400–1403更新、优化器/归一化连续及新checkpoint通过 |
| [ASM-HOVER-20260921-01](#asm-hover-20260921-01) | 2026-09-21 | 本地装配10cm悬停及奖励尺度修正 | 62通过、2跳过；原有密集奖励保留，未同步/重启 |
| [BULB-RANDOMIZED-20260921-01](#bulb-randomized-20260921-01) | 2026-09-21 | 新随机旋出任务实现、同步及正式训练 | 3456个离散可行位置/目标对；零手初态、687维；原生检查与正式PPO更新/保存通过，学习效果待评测 |
| [BULB-RANDOMIZATION-DESIGN-20260921-01](#bulb-randomization-design-20260921-01) | 2026-09-21 | 新随机旋出任务方案、CPU 几何/IK 初筛 | ±10 cm 支架平移桌边余量 14.2 cm；扩大目标 207/243 有局部 IK 解；未实现任务或验证碰撞路径 |
| [BULB-VIDEO-20260921-01](#bulb-video-20260921-01) | 2026-09-21 | 当前日志入口修复、model_1100录像并下载 | 本地6006可见新run；40秒/1200帧完整解码与下载哈希通过；非独立成功率评测 |
| [BULB-PROFILE-20260920-01](#bulb-profile-20260920-01) | 2026-09-20 | 1024/3072 环境采样同步分项计时 | 物理求解占63.5%/70.4%，压力13.4%/7.3%；非6144原位计时，正式训练未改 |
| [BULB-SEGMENTS-20260920-01](#bulb-segments-20260920-01) | 2026-09-20 | 三圈小目标/臂稳定实现、原生验证与新训练 | 连续进度+12个90°目标、单指旋转信用、673维新契约；学习效果待验证 |
| [BULB-INITIAL-GOAL-20260920-01](#bulb-initial-goal-20260920-01) | 2026-09-20 | 初始手臂/灯泡/目标可视化视频 | 固定reset、仅相机移动；18秒，非策略回放或成功评测 |
| [BULB-CONTACT-LOG-20260920-01](#bulb-contact-log-20260920-01) | 2026-09-20 | 7个旋出run只读日志比较 | 接触改善、远离减少但当前圈数下降；旧奖励不可直接比较，无新干预 |
| [ASM-ZERO-HAND-20260920-01](#asm-zero-hand-20260920-01) | 2026-09-20 | 装配手部零位及远端重启 | 21 关节/reset 实测通过，奖励/探索未收紧；PPO 更新已确认 |
| [ASM-VIDEO-20260920-02](#asm-video-20260920-02) | 2026-09-20 | 新装配 checkpoint 服务器录像并下载 | model_200；40 秒/1200 帧解码通过；非独立评测 |
| [BULB-MEM-RESTART-20260920-01](#bulb-mem-restart-20260920-01) | 2026-09-20 | 远端同步、显存配置压缩及 6144 环境重启 | 新惩罚生效、并行数 +50%；已开始 PPO 更新，非学习效果评测 |
| [BULB-FAR-ARM40-20260920-01](#bulb-far-arm40-20260920-01) | 2026-09-20 | 本地远离代价与臂倍率修改 | 最高 −0.3/s、释放前臂倍率 40；252 通过，未同步/重启/训练验证 |
| [ASM-RESTART-20260920-01](#asm-restart-20260920-01) | 2026-09-20 | 远端同步并从头重启装配 | GPU 5；30 Hz/随机 reset/抬起奖励；PPO 更新及模型保存已确认 |
| [ASM-ANALYSIS-20260920-01](#asm-analysis-20260920-01) | 2026-09-20 | 装配奖励/曲线只读分析 | 桌面接触持续回报的停滞假设；未做干预或物理轨迹验证 |
| [BULB-M100-20260920-01](#bulb-m100-20260920-01) | 2026-09-20 | model_100 双回放及惩罚预算 | 当前0.527圈后远离；候选距离−0.3/s、臂倍率40仅离线重算，未修改训练 |
| [BULB-RESTART-20260920-02](#bulb-restart-20260920-02) | 2026-09-20 | GPU 4 重启 Custom 旋出，新奖励/30 Hz | PPO 更新和配置核验；非独立成功率评测 |
| [BULB-PHASE-ACTION-20260920-01](#bulb-phase-action-20260920-01) | 2026-09-20 | 本地分阶段动作惩罚 | 321 通过、3 跳过；未训练，抓持改善待验证 |
| [BULB-VIDEO-20260920-01](#bulb-video-20260920-01) | 2026-09-20 | Dexonomy 旋出服务器录像并下载 | model_300；约 30 秒、1799 帧完整解码通过；非独立评测 |
| [ASM-VIDEO-20260920-01](#asm-video-20260920-01) | 2026-09-20 | 服务器回放并下载 | model_1100；20 秒视频，1200 帧解码通过；非独立评测 |
| [BULB-SYNC-20260920-02](#bulb-sync-20260920-02) | 2026-09-20 | 同步并准备新卡从零训练 | 同步及 CPU 回归通过；GPU 共享选择待回复，未启动 |
| [BULB-REWARD-20260920-01](#bulb-reward-20260920-01) | 2026-09-20 | 本地软抓持奖励修改及 CPU 回归 | 289 通过、1 跳过；未同步、未训练、无新成功率 |
| [BULB-RESTART-20260920-01](#bulb-restart-20260920-01) | 2026-09-20 | 远端同步及新抓姿从头训练 | 迭代 0–4 更新、保存模型、新抓姿 28/28 匹配；不是成功率评测 |
| [BULB-GRASP-20260920-01](#bulb-grasp-20260920-01) | 2026-09-20 | Dexonomy 初态接入；本机状态验证通过 | 440/440 初态/reset 检查；不表示动态抓持成功 |
| [PEG-EVAL-20260919-01](#peg-eval-20260919-01) | 2026-09-19 | 固定模型开发评测；完成 | 深度保持协议 1/16；不是完整几何验收结论 |
| [PEG-RESUME-20260919-01](#peg-resume-20260919-01) | 2026-09-19 | 恢复训练首轮核查；有历史证据 | 发生 PPO 更新；训练样本 151/1299，非独立成功率 |
| [BULB-EVAL-20260919-01](#bulb-eval-20260919-01) | 2026-09-19 | 旧版旋出开发评测；未达标 | seed 4040，0/20，当前 v3 未验证 |
| [ASM-PROBE-20260919-01](#asm-probe-20260919-01) | 2026-09-19 | 装配引导接入探针；历史通过 | 2 环境/40 步；无策略成功结论 |
| [DOC-AUDIT-20260920-01](#doc-audit-20260920-01) | 2026-09-20 | 文档审计；非训练实验 | 本次检查与工作区保留记录 |

这些 ID 为此次索引分配，原始 run ID/文件名保持不变。没有补造旧实验的预算、commit、种子或成绩。

## REVO3-CALIBRATION-MOUNT-20261003-06

- 授权：按用户更正后的 M6×20 规格截短支架底面，至少可旋入光学平台 5 mm。
- 实现：从 V4 底面整体去除 10 mm；原四孔 Ø6.8 通孔、Ø12 藏头孔和 XY 坐标不变；新底面重设为 Z=0。
- 几何结果：承靠厚度 24→14 mm；M6×20 无垫片名义伸出 6 mm，加 1 mm 垫片名义伸出 5 mm。
- 验证：BREP/STEP 单有效实体；STL 87002 面、封闭、绕序一致、单连通；3MF 同拓扑且单位为毫米；
  四颗螺丝包络零干涉，四孔共 240 个孔径/承靠面探针通过；两份 URDF XML、102 个 mesh 引用通过。
- 边界：未打印或实装；名义伸出不等于已测有效啮合，平台倒角、盲孔深度和有效牙深仍待确认。
- 交付：[V5 M6×20 支架](assets/revo3_breadboard_mount/generated_calibration_v5_m6x20/README.md)。

## CALIB-SDK-20261003-04

- 授权：复制`/home/liuxinyu/workspace/brainco-revo3-sdk`至`calibration/`，读取本机已连接手及视触觉，完善机械README。
  未发送运动、使能、清零、归零、固件命令。保留源目录、现有用户修改和所有失败记录。
- 副本：源码版本`52e3fcb35bcbe91e17ab66bd8a0cbccc19d8e69f`，额外复制原`.venv`中bc-revo3-sdk2.1.0和pyvitaisdk1.0.17运行包／模型／许可证。
  排除虚拟环境本身、Git、缓存、日志等；676文件276,582,053字节，逐文件SHA见`calibration/sdk_copy_manifest.json`。
- 图分析：新入口`probe_revo3_readonly`及辅助新符号未索引，UNKNOWN；补引用核查仅新脚本本地调用，未修改共享任务／SDK源码。
- 实际入口：`/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python calibration/probe_revo3_readonly.py --output calibration/captures/revo3_readonly_20261003_final`。
  优先加载副本native模块，剩余依赖复用源环境；Python3.11.15、NumPy1.26.4、OpenCV4.11.0.86。无安装或驱动设置修改。
- 结果：2026-10-03 16:05:42–16:05:47，右手BCUVR1205J2600002／UltraVisionTouch／FW9.9.9.ML，ttyUSB0、5M、从站127。
  3次State各21关节，3次Touch均4×52指腹＋36掌部有效，模式PointArray／Force（mN）。
  拇指57点模块Disabled，声明301点不可说301点全部可采；enabled_mask=0x0551。未开启禁用模块。
  健康error_code=0、faulted_motor_count=0，但SafetyState.Unknown及240.84V字段回报待核实。
- 5相机由ViTai Finder和稳定by-id路径定位，逐路各8帧640×480 RGB，共40帧。
  请求30fps、驱动回报120fps、短时主机完成率58–62fps，不是并发／曝光时钟／长期吞吐验证。
  SDK主链路没有指尖图像；指尖SN→手指、具体ViTai型号／模型、深度／marker／力估计标定尚未验证。
- 失败保留：`..._1604`主链路SDK枚举序列化递归失败，全部相机成功；改为只读数据描述符后`..._1606`成功，
  最终补枚举名与passed范围后另存`..._final`。不将历史图片或中间失败计入最终有效数据。
- [报告](calibration/captures/revo3_readonly_20261003_final/report.json)、[独立核查](calibration/captures/revo3_readonly_20261003_final/validation.json)、
  [5路图像](calibration/captures/revo3_readonly_20261003_final/cameras_montage.png)、[接口与采集契约](calibration/机械标定台仿真搭建_README.md#7-revo3-sdk与真机只读采集2026-10-03)。
  只读获取信息已完成；未操作机械台、修改URDF坐标或联动训练。

## CALIB-TOUCH-20261003-04

统一尖端原点：用户明确要求法兰等坐标系的原点与轴向均统一到`press_head_tip`。
当前法兰／压头／tip三个frame重合，两个固定关节xyz/rpy全0；XYZ运动系零位也移至尖端，
world_to_press_stage z从0.0834975981590025m变为0.0480797829510025m。
各工具link使用不同的原frame→共同尖端frame变换重表达网格，不移动实体几何。
manifest v3记录完整`T_source_flange_shared_tip`，原法兰采集位姿与新frame比较须含平移。
本次三轴各1mm、20组非零坐标、121帧FK、完整资产检查、mesh SHA和urdfdom4通过；
与修订前比较工具frame重合／视觉碰撞／tip位姿最大元素误差均6.94e-18，CSV／初态／限位保持。
手／支架元素、源工具URDF／STL与采集记录保持。此前仅统一轴向的方案保留在下文作为历史。
备份`outputs/calibration_bench/shared_tip_frame_before/`；生成／复核日志`shared_tip_frame_fixed/`。
当前证据：[共同尖端坐标核查](assets/calibration_bench_visuotactile_urdf/coordinate_frame_validation.json)。

link坐标修复：用户发现压头link／joint朝向不一致。确认下述早期Z关节+π/2补偿使
`connection_flange`、`press_head`、`press_head_tip`相对stage错转90°；该历史方案已被替代。
当前XYZ关节origin旋转全0，六个stage/carriage/tool link轴向一致；导出器对工具link的
visual/collision/inertial origin用S⁻¹T表达，固定关节用S⁻¹TS表达，保持实际几何。
manifest升级v2并明确历史NPZ到新工具link的基变换；原始手、工具STL／URDF和采集记录保持。
本次三轴各1mm、20组非零坐标、121帧FK、完整资产检查及urdfdom4解析通过；
视觉／碰撞世界位姿最大元素误差3.45e-16，tip位置误差0，CSV／初态文件字节一致。
备份 `outputs/calibration_bench/tool_link_frame_before/`，生成和复核日志在`tool_link_frame_fixed/`。
未重新采集或运行动力学；[当前坐标验证](assets/calibration_bench_visuotactile_urdf/coordinate_frame_validation.json)。

后续坐标修订：用户要求压头X指向原−Y、Y指向原+X，按右手定则Z保持+Z。
当前URDF的stage yaw改−π/2，Z关节origin yaw补偿+π/2保持工具世界朝向；
导出器同步换算CSV及行程限位。原始文件备份于 `outputs/calibration_bench/press_frame_rotation_before/`。
三轴各1mm方向、20组非零坐标、手／支架／工具元素保持与urdfdom4解析通过；
完整资产检查重跑通过，121帧世界运动保持。未重新采集或进行动力学运行。
证据：[坐标核查](assets/calibration_bench_visuotactile_urdf/coordinate_frame_validation.json)。

- 授权：用任务中和 `assets/revo21_right_touch` 的视触觉手更新当前标定资产，先对齐再装配。
  `main/ecdb0d1292604ec1651c210551c8fd21b371cfde`＋当前工作区，保留用户既有改动；未提交。
- 原DV2手、V4支架与旧完整标定URDF保持；新增 `generated_calibration_v4_touch/` 手／支架包和
  `assets/calibration_bench_visuotactile_urdf/` 完整XYZ包。默认配置显式选新手，旧配置缺字段仍选旧手。
  全部活动关节、惯量、限位、视觉／碰撞和五路相机frame来自原DV2 URDF，未调整训练资产。
- 按原任务 `tactile_representation/direct_calibration.py` 验证五指rubber→DIP变换，保存变换矩阵
  与任务函数生成的marker布局；四指根部坐标验证旧／新一致，沿用腕部安装变换。
  没有认证各任务USD网格完全一致，也不把新虚拟UV采集声明为旧encoder兼容。
- 实际命令：本机brainco Python运行 `build_touch_fixture.py`，`run.py --config
  scripts/calibration_bench/smoke_config.json --output outputs/calibration_bench/visuotactile_v1`，
  `verify.py`，`export_urdf.py --output assets/calibration_bench_visuotactile_urdf` 和
  `verify_touch_asset.py`；CPU采集设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。
  原生入口 `preview_isaac.py outputs/calibration_bench/visuotactile_v1 --frames 30 --capture --headless`。
- 结果：食指8/8完整短回合、1574帧、99项数据检查通过；五指各6个载荷位姿探针有限且离面归零。
  5项原解析／协议检查通过。新URDF65link/64joint、104处mesh引用、101份mesh、24活动关节，
  原元素231项核对通过；零位误差2.78e-17、非零位误差2.22e-16，121帧XYZ误差1.67e-16。
  cmeel urdfdom4通过；Isaac52网格、121帧动画误差3.65e-10及30次查看更新／截图通过。
- 失败与修订：初版资产检查把源URDF无name的 `mujoco` 扩展误按命名元素查询，断言失败；
  修正检查器选择方式后通过，原始日志保留 `touch_update_audit/asset_validation.log`，
  复核日志 `asset_validation_r2.log`。源mujoco扩展本身原样保留。
- 剩余问题：每link500表面点、seed42的支架有符号距离检查显示新手拇指最大穿入2.383mm；
  旧手同一抽样方法最大0.599mm（拓扑不同，不是逐点对应）。不声称配合无干涉或不增加。
  保留零位和支架几何／碰撞，机械配合标记REVIEW_REQUIRED。指尖内部开口网格也未补面。
  未验动力学、连续碰撞、相机光学、真机装配或真实标定，未训练／远程执行。
- [完整资产与证据](assets/calibration_bench_visuotactile_urdf/README.md)、
  [资产验证](assets/calibration_bench_visuotactile_urdf/validation.json)、
  [任务坐标映射](assets/calibration_bench_visuotactile_urdf/reference/task_frame_alignment.json)、
  [原生截图](outputs/calibration_bench/visuotactile_v1/isaac_scene.png)。

## CALIB-URDF-20261003-03

- 用户要求当前场景输出URDF并包含压头运动，坐标系稍后自行调整。新增独立 `export_urdf.py`，
  合并原V4手／支架与球压头／法兰URDF，复制109份网格，加入无外观XYZ prismatic链。
  `world_to_press_stage` origin取当前场景首帧法兰位姿；q=0保持原始离面起始位置。
- 交付 `assets/calibration_bench_urdf/`，含URDF、mesh、121帧运动CSV、24活动关节初始位置、
  manifest、validation和调整说明。没有龙门架或滑块外观，工具旋转未加入三轴运动链。
- 本机brainco Python运行生成器；xmllint、单根无环68link/67joint检查、145处mesh路径和SHA、
  原手／工具元素保持、初始几何匹配均通过。独立FK逐帧对比法兰和tip，最大误差4.1633e-16；
  三个轴各移动1mm的方向核查通过。
- PATH的Python版check_urdf因cmeel模块不可用启动失败；直接使用已有cmeel urdfdom4二进制及
  对应lib路径解析通过，不安装／更改依赖。日志在包内 `urdfdom_check.txt`。
- 仅运动学交付；effort20N、velocity0.01m/s为明确占位，范围继承仿真包围盒；
  新运动链及压头质量／惯量未知，未填写，未做Isaac动力学导入、控制器或全行程碰撞验收。
  [URDF与坐标编辑说明](assets/calibration_bench_urdf/README.md)、[核查](assets/calibration_bench_urdf/validation.json)。

## CALIB-SCENE-20261003-02

- 用户指出压头机构并非龙门架，要求暂时移除其它结构。修改 `scene.py:export_scene`，
  去掉台面、导轨、立柱、横梁及虚拟滑台，只保留原始压头／法兰和手／V4支架。
- 复用 `validated_v1/episode_00000.npz` 的121帧位姿，未重新采集或训练；
  保留的 HandFixture 与 Tool USD段逐字节相同，不改变原资产、模型、接触或运动。
- 新目录 `outputs/calibration_bench/head_fixture_only/`：`scene_revision.json` 保存来源与SHA；
  Isaac54网格拓扑可读、逐帧位姿最大误差3.7073e-10、30次查看更新和RTX截图通过。
  运行命令：`preview_isaac.py outputs/calibration_bench/head_fixture_only --frames 30 --capture --headless`。
- [新场景](outputs/calibration_bench/head_fixture_only/scene.usda)、
  [截图](outputs/calibration_bench/head_fixture_only/isaac_scene.png)、
  [原生检查](outputs/calibration_bench/head_fixture_only/isaac_verification.json)。
  仅显示结构修改，未做新物理或采集质量评测；旧场景保留为历史证据。

## CALIB-SIM-20261003-01

- 授权：按现有论文与机械标定资产先搭建仿真系统，真机对齐后置；用户确认 HydroShear 主线，
  再要求根据仿真任务核对。源码反查确认实际 encoder 为 RGB/Depth/Marker，压力单独采集；
  本项目任务组合与 HydroShear 原论文不同，公式对照稿不足以确认完整论文身份。
- 版本：`main` / `ecdb0d1292604ec1651c210551c8fd21b371cfde` + 本轮新增独立源码，未提交；
  原训练改动与删除项保留。源代码／资产哈希在各输出 manifest，最终核对与当前文件一致。
  Python3.11.15、NumPy1.26.0、SciPy1.15.3、trimesh4.5.1、Torch2.7.0+cu128、
  Isaac Lab2.3.2.post1、Sim5.1.0.0；完整清单见最终目录 `runtime_versions.json`。
- 实现：`scripts/calibration_bench/run.py` 使用 V4 自含 URDF 和三种法兰压头，
  世界系为 lower_mount，实际 mesh 离散首接触求零，XYZ滑动随曲面修正Z。
  生成 L1 几何深度、临时壳层压力／面积力、现有曲面HydroShear、诊断Lambertian RGB；
  场景以规定轨迹播放，不是刚体／FEM动力学或独立物理真值。每回合完整阶段、失败原因、参数和哈希保留。
- 实际命令：在仓库根目录使用本机 `brainco/bin/python` 运行
  `scripts/calibration_bench/run.py --config scripts/calibration_bench/smoke_config.json --output outputs/calibration_bench/validated_v1`，
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`；随后 `verify.py` 检查该目录，
  `preview_isaac.py <目录> --headless --capture --frames 30` 做原生导入／回放。
- 球压头最终8/8完整回合，共1,574帧、CPU采集67.18秒，99项数据一致性检查通过。
  normal/hold/shear_x/shear_y/slip_x/slip_y/twist/roll各一次，中心0.3mm、30Hz、32×40网格、100marker；
  后两项标记虚拟旋转。此处8/8仅表示协议完成，不是RL任务成功率或物理标定成功率。
- 三角／锥头：分别 `--head triangle/conical --episodes 1 --no-scene`，各1/1完整回合、115帧，
  中心0.1mm，分别15项数据检查通过；没有原生动态碰撞测试。
- 故障保留反例：独立将模型力上限降到1e-6N，`force_abort_check/`在第52帧保留首次接触触发帧，
  写入 `model_force_limit`、`completed=false`，进程按预期返回2；未把它计入完整回合或删除失败数据。
- 解析和协议回归：`checks.py`5项通过。额外240×320、100marker七个位姿探针
  （离面3mm→0→0.1→0.3→0.5mm→0→离面3mm）有限值及接触前后归零通过；
  三个压入量模型总负载分别约0.00813/0.07611/0.21277N。是模型输出，不是真力；
  与粗网格数值不同，尚无分辨率收敛或高分辨率全轨迹验收。
- 原生检查：54个网格拓扑可读，首回合121个位姿与NPZ最大矩阵元素误差3.7073e-10；
  30次查看更新和1280×800 RTX截图通过。两次先前USD语法错误（灯光声明和prim间换行）已修复；
  `first_smoke/`、`sphere_normal_v2/`原始失败产物保留，不算通过。
- 证据：[最终目录](outputs/calibration_bench/validated_v1/)、[数据检查](outputs/calibration_bench/validated_v1/verification.json)、
  [原生检查](outputs/calibration_bench/validated_v1/isaac_verification.json)、
  [截图](outputs/calibration_bench/validated_v1/isaac_scene.png)、
  [三角](outputs/calibration_bench/triangle_smoke/verification.json)、
  [锥头](outputs/calibration_bench/conical_smoke/verification.json)。`outputs/`和现有资产未随Git提交传播。
- 限制／后续：原食指内部视觉mesh开口，几何守卫只有每15帧抽样，连续碰撞和全行程净空未认证。
  压力材料、RGB光学、真实相机/marker对应、硬件taxel布局和独立真值均待验证；
  默认270回合55,890帧全矩阵未执行，没有训练、真机／远端操作或SDK联调。
  采集字段、控制变量辨识顺序、分组防泄漏和论文验收边界见[系统说明](scripts/calibration_bench/README.md)。

## REVO3-CALIBRATION-MOUNT-20261001-05

- 用户最终取消挡板/低位凸块修整，只扩大圆槽和码槽；圆槽口径采用40mm、深度仍3mm，码槽净55×55mm、深0.5mm。
- 独立V4从保留V3实体只做两个切削，不改变固定结构/手姿态/参考frame。整段腕部Y≤80mm及原四孔/两定位区域CAD对称差0；切削范围外移除材料0、新增材料0。
- CAD有效单实体，移除2102.04096mm³；STEP回读有效单实体，体积相对差1.324ppm。STL62354面、封闭单连通/绕序一致、220×252×45mm，CAD/网格体积差0.00604%。
- 球面40解析点/16口径点、贴码槽底25点/边界8点通过；最终STL98条槽与外侧平台射线最大标高误差0.01061mm，非全表面误差保证。
- 两URDF的XML/urdfdom/相对路径通过，组合61link/60joint/21活动关节，除底座mesh文件名外所有元素和FK保持；原V3包全部文件哈希、复制手网格保持，外部手源仓库clean。
- seed42每visual400点，V3/V4局部>0.1mm穿入计数相同（拇指35、食指MCP橡胶2）；仍REVIEW_REQUIRED，未打印、未做绑带强度/精度/真机标定。
- GitNexus绑定本工作区`mulyi-model-tac`/ecdb0d1；两个新增生成器路径未索引（UNKNOWN），补核不存在现有调用或活动生成进程。只增加独立生成器和文档，不改训练代码或V3生成器/产物，未提交。
- [交付与重建](assets/revo3_breadboard_mount/CALIBRATION_V4.md)、[CAD报告](assets/revo3_breadboard_mount/generated_calibration_v4/cad_validation.json)、[装配/STL报告](assets/revo3_breadboard_mount/generated_calibration_v4/validation.json)。

## REVO3-CALIBRATION-MOUNT-20261001-04

- 用户要求补齐L形空缺为矩形，并确认40mm含白边贴片、口径20mm/深3mm球槽；新增平台Z36、贴码槽深0.5mm。
- 新包`assets/revo3_breadboard_mount/generated_calibration_v3/`含STL/STEP/BREP、两份URDF、预览及JSON检查记录；原V1/V2/手源保留。
- CAD有效单实体，未切除V2材料；四安装孔/两定位区域对称差0，接口/绑带通道点检通过。
- STL61634面，封闭单连通/绕序一致，220×252×45mm，CAD体积相对差0.00561%；STEP回读有效单实体，体积差1.325ppm。
- 球面解析底点/40点、贴码槽底25点通过；STL58条向下射线最大高度误差0.00866mm，非整表面误差保证。
- 组合URDF61link/60joint/21活动关节，新增3固定基准frame；原手元素/FK/网格哈希保持，两份urdfdom解析及相对路径通过。
- seed42每visual400表面点，V3与V2的>0.1mm穿入计数相同：拇指35点、食指MCP橡胶2点；不是新平台导致，无连续碰撞/打印验收，REVIEW_REQUIRED。
- 首次STL导出遇球面极点一个零面积三角形，候选保留`/tmp/revo3_calibration_v3_initial_mesh_failure_20261001/`；V3显式去重复索引零面积面，原BSpline有界面小缺口保留原容差修复。未丢弃有限面积曲面。
- GitNexus精确路径目标未索引（UNKNOWN），补核仅本独立资产生成器引用；同名export_mesh命中另一资产不作为放行证据。未改训练、未提交、未做真机标定。
- [尺寸、用法与重建](assets/revo3_breadboard_mount/CALIBRATION_V3.md)、[CAD证据](assets/revo3_breadboard_mount/generated_calibration_v3/cad_validation.json)、[装配/STL证据](assets/revo3_breadboard_mount/generated_calibration_v3/validation.json)。

## REVO3-STABLE-MOUNT-20261001-03

用户反馈装配松动，并确认目标是消除根部晃动。保留V1，在独立`generated_stable_v2/`
生成新下支架及右手组合；右手继续使用指定brainco-description b609f2f资产，姿态及21关节保持。
新建45mm高腕部鞍座、0.25mm截面余量、前端角挡、两处掌背支撑及8mm软绑带通道，
不恢复原上模、不增加活动夹紧机构。必须实际穿带收紧才提供向上保持，未提供实体绑带或力学证明。

CAD单实体有效，STL57446三角面/单连通/闭合/绕序一致；220×252×45mm，网格/CAD体积差0.00902%。
四安装孔/两定位柱局部CAD差为零，内部穿带通道、底板/挡边与中央接口开口点检通过。
单支架和组合URDF解析通过；手姿态、参数及手网格哈希保持。独立候选e与交付STL/组合URDF逐字节一致。
每visual400点、腕部5000点、seed42：腕部旧版201点穿入/最大3.108mm，新版0点；
旧V1记录的最大2.793mm来自更稀疏采样，不是此次相同采样。新版±X/−Y平移0.5mm出现阻挡，
+Z/+Y仍可移动，明确需要绑带保持；未用固定关节代替真实夹紧验证。
未改指槽仍有拇指干涉，最大抽样约0.689mm；无全表面/全关节扫掠、打印预紧、抗振或动力学结果。

失败候选b在同域面清理后失效；c/d新增支撑截断旧BSpline使网格化失败，保留/tmp候选证据。
限制第二条支撑带外缘至X≤113mm后e版有效，旧曲面离散缺口按原有界面验证恢复3三角面。
见[V2设计及使用边界](assets/revo3_breadboard_mount/STABLE_V2.md)、
[CAD验证](assets/revo3_breadboard_mount/generated_stable_v2/cad_validation.json)、
[配合验证](assets/revo3_breadboard_mount/generated_stable_v2/validation.json)。

## REVO3-BREADBOARD-URDF-20261001-02

用户要求用 `1_stl/breadboard_mount_256/lower_03_print.stl` 与右手生成新URDF，
先装配上下支架再去除上支架；随后指定右手来源为 `/home/liuxinyu/workspace/brainco-description`。
本次使用该仓库b609f2f的`revo3_system/urdf/revo3_right.urdf`；只在本仓库新增独立包。
最终58 link/57 joint、完整合模59 link/58 joint，两版均保留21活动关节及原手参数。
手到CAD变换为(y,z-19.2mm,x)，加下支架打印平移(60,19,40)mm；上支架先逆打印翻转再恢复合模。

CPU验证：ROS Humble urdfdom两版通过，单根连接树、相对路径、单位变换、手参数保留、
源件/复制网格SHA一致、最终模型仅去掉上支架与固定关节，均通过；临时独立构建两URDF逐字节一致。
零位每visual抽样300点/seed42，经最近点距离及射线内外判定，下支架内49样本，最大深度2.793mm；
上支架内218样本，最大2.591mm。主要涉及新版腕部及拇指/橡胶与原型腔差异，未宣称无碰撞。
保留原几何及碰撞，REVIEW_REQUIRED；支架惯性未知未填，手effort/velocity=0保持原值。
未做Isaac/动力学、全行程碰撞或打印装配，不接入已有任务。

证据：[交付说明](assets/revo3_breadboard_mount/README.md)、
[来源](assets/revo3_breadboard_mount/generated/manifest.json)、
[验证](assets/revo3_breadboard_mount/generated/validation.json)、
[预览](assets/revo3_breadboard_mount/generated/preview.png)。

## CALIBRATION-BENCH-URDF-20261001-01

用户要求把本目录内装配支架导出为能在自身行程内运动的URDF。本次以检查版
`calibration_bench_sim_package/assets/blender/dual_y_press_checked.blend`为源，在本机Blender
5.2.2 LTS后台CPU导出，不修改原Blend、STEP或既有网格。

按源场景控制空对象和父子关系拆为`bench_base → y_gantry_link → x_carriage_link →
z_carriage_link`，双Y作为单个同步自由度。四份米制二进制STL合计481268三角面；生成两份
URDF：概念CAD名义行程X/Y/Z ±250/±150/±75mm，以及`mechanical.yaml`软件限位
X[-340,0]/Y[0,170]/Z[-310,0]mm。软件限位版的Z范围超过候选CAD的150mm名义行程，未把它
声称为机械可达或碰撞安全。

两份文件均通过xmllint及ROS Humble `/opt/ros/humble/bin/check_urdf`解析；另核对四link、
三prismatic关节、base→Y→X→Z拓扑、轴向、上下限、相对网格路径、二进制STL长度和源Blend
SHA。再次导出到独立临时目录后与交付目录`diff -rq`无差异。未运行Isaac/Gazebo导入、动力学、
控制器或连续全行程碰撞测试；按主要零件包络生成的碰撞盒、质量/惯量、100N effort及0.05m/s
velocity均为显式占位，原装配`REVIEW_REQUIRED`和约0.5mm连接间隙仍保留。

交付：[说明](calibration_bench_sim_package/generated_urdf/README.md)、
[导出报告](calibration_bench_sim_package/generated_urdf/checks/export_report.json)、
[生成器](calibration_bench_sim_package/tools/export_urdf.py)。

## BULB-RESTORE-GPU4-20261001-20

恢复基准为高成功率GPU4 `20260928_140146_rotate_bulb_phase_phys60_30hz_12s_scratch_gpu4`冻结workspace，非按当前git HEAD推测旧版。
本地与`/home/xinyu/mulyi-model-tac`的Randomized env_cfg.py、phase_rewards.py、README逐字恢复，三SHA均匹配GPU4 source_manifest；profile3、无初态姿态惩罚、原joint1=-8.35275°/约±160°、旧高Z库、60Hz/2步/12秒。
分别21份旋出相关源码、10项资产来源hash通过，本地114 CPU/10.53秒、服务器114 CPU/11.67秒；原GPU4模型8400和两份契约保留。无新GPU仿真、恢复权重或训练。
保留共享入口中独立门控的装配课程新增逻辑，未回滚其他任务。未启用的新零位模块/资产仍保留；撤销功能的初态惩罚测试移入本地备份/retired_tests（非用删测试掩盖仍存在功能的失败）。
本地备份`logs/diagnostics/bulb_restore_gpu4_20261001/local_backup/`；服务器`/home/xinyu/project_updates/bulb_restore_gpu4_20261001/`。配置图谱LOW仅追到PLAY、动态注册补核；新增惩罚UNKNOWN补核实际注册和测试引用，文档导航补核；无提交。
证据：[local_restore.json](logs/diagnostics/bulb_restore_gpu4_20261001/local_restore.json)、[remote_restore.json](logs/diagnostics/bulb_restore_gpu4_20261001/remote_restore.json)、[validation.json](logs/diagnostics/bulb_restore_gpu4_20261001/validation.json)、[remote_validation.json](logs/diagnostics/bulb_restore_gpu4_20261001/remote_validation.json)。

## BULB-TRAINING-COMPLETION-20261001-19

2026-10-01 06:56 UTC实时读取xinyu21，重启既有本机SSH转发后连接恢复，训练未改动。GPU6新零位profile6在09-30 17:14 UTC正常完成15000轮、exit0、model14999。
全程15000个成功率/成功奖励/成功终止记录均0，release_grasp全0；末100轮旋出峰值2.960007圈、最终2.946296圈、释放指标0.0519996、超时0.976312、回报8.169879、搬运奖励1.16246e−5。
旧GPU4仍8467轮stopped/model8400，末100轮成功率0.9422898、末点0.9591492、释放指标0.9999557。为各run训练日志窗口均值，非汇总独立回合评测或公平因果消融。
保存两run全部相关tag的last/末100/末1000/全程最大/最佳100窗口及checkpoint路径：[training_snapshot.json](logs/diagnostics/bulb_training_completion_20261001/training_snapshot.json)。未录新视频或据此判断最终策略是否仍后转。

## BULB-REACHABILITY-AUDIT-20260930-18

4752姿态按新USD复算，见证保留旧坐标且已转换，源资产SHA全匹配；末端最大0.400毫米/0.377°，初态对原生掌心误差4.99e−7米。
432灯座接近/抓取/抬起joint1最大67.14°；3456搬运已有解仅8个超90°、最大111.42°，不是最小所需角。初掌心距灯座XY最大18.44厘米，搬运位移28–47厘米。
model1000三回合1080步FK重建，末掌心距整个灯座区域XY至少39.7–42.8厘米，未抓持/释放；不是每回合精确物体距离。未新做碰撞/物理/策略分层评测、未改训练。
见[程序、数据、图及限制](logs/diagnostics/bulb_reachability_audit_20260930/README.md)。

## BULB-ROOT-FRAME-AUDIT-20260930-17

组合USD顶层、底座及固定根关节均单位旋转，与冻结profile6配置和原生reset一致。原始抓取参考9刚体对原生最大位置误差3.93e−7米，joint1父帧约171.65°不代表根翻转。
3456目标全部在底座−X侧，与灯座和初始掌心一致；命令世界→根、奖励根→世界、几何观测逆变换对应，4份源码SHA匹配冻结回放。
服务器旧隧道拒绝连接，本次仅离线读取/复算，未新开仿真、修改训练或证明后转因果。文档图谱impact为UNKNOWN，补核导航引用；无运行符号编辑、无提交。
见[证据与限制](logs/diagnostics/bulb_root_frame_audit_20260930/README.md)。

## BULB-JOINT1-ASSET-EXPLORATION-20260929-16

当前GPU6组合资产8层hash一致，joint1 ±180/drive target0/velocity0，无发现固定后转外力矩设置。策略双向Normal+tanh；录像仅均值，无随机探索。
model1000 joint1 latent std0.0600，model1300 0.05455（下限0.05）。录像对应部分状态反向动作概率<1%，但各回合2秒后平均仍22–23%，不能称完全单侧。
未新作配对物理干预，不能完全排除碰撞/耦合不对称或确定方向偏好的最初成因。未改训练。见[核查与限制](logs/diagnostics/bulb_joint1_asset_exploration_20260929/README.md)。

## BULB-JOINT1-REWARD-AUDIT-20260929-15

补充参考系核查：1077步动作/变化率/速度惩罚复算均吻合；动作0对应当前位置目标，初态姿态项对应固定reset而非七臂全0。平滑项局部偏好延续已有动作，但示例反向成本仅1.29e-7/步，非主因证明。

三回合model1000逐步奖励/目标/姿态复算及当前训练日志审计；七相关源码hash匹配。姿态公式误差2.82e-10，后转阶段没有显著正奖励，接近期速度成本减小和未抓持远离惩罚缺失构成可疑组合。
单joint1累计仅约0.038–0.058分，90度死区使约前4.9–6.6秒不罚joint1。最新窗口奖励改善主要来自少扣分，不能直接归因后转或认定新checkpoint保持同动作。
未改训练/未新开GPU实验，结论不是消融因果证据。见[审计与限制](logs/diagnostics/bulb_joint1_reward_audit_20260929/README.md)。

## BULB-JOINT1ZERO-VIDEO-20260929-14

选取时最新GPU6 profile6保存点model1000，冻结原运行源码/库/资产/契约，单环境seed42连续36秒回放。
三完整回合均超时，成功0/3，无释放/抓持；joint1新坐标约-19.42至162.10度。完整1080帧解码、服务器/本地/桌面hash及实际配置核验通过。
GPU6继续训练；临时空闲GPU4录像结束。不是独立学习效果/成功率评测，不能从三条轨迹推断总体表现。
见[视频和证据](recordings/xinyu21_bulb_gpu6_joint1zero_20260929_2253/README.md)。

## BULB-JOINT1ZERO-WEAKPOSE-GPU6-20260929-13

profile6初态姿态weight -0.025→-0.01、90度deadband保持，部署上轮joint1零位/有限±180度资产。旧GPU6停在1267，model1200保留。
新6144环境独立从0/空优化器/LR0.001启动。CPU57、原生12项、32环境13轮及正式至2轮的配置/契约/hash/更新保存通过；GPU4保持。
同时改变了资产坐标/范围与姿态weight，不能将后续差异单因果归因系数。尚未开展成功率/学习效果评测。见[执行记录](logs/diagnostics/bulb_joint1_zero_weakpose_gpu6_20260929/README.md)。

## BULB-JOINT1-ZERO-20260929-12

最终用户方案：原joint1初始-8.35275度定义为资产0位，有限±180度。仅Randomized专用层改变joint1.localRot0及上下限，任务初态置0；原控制器、90度姿态项、碰撞及成功判据保留。
CPU32、USD逐属性等价检查及GPU7两环境100步/9项原生通过；全机器人reset最大位置差1.19e-7米，端点+179.99966/-179.70734度，旧契约正确拒绝。3456样本保持。
GPU4/GPU6原进程保持，未部署新训练、未评测学习效果；连续方案早期尝试被最终指令替代并保留证据。
见[最终实现与记录](logs/diagnostics/bulb_joint1_zero_20260929/README.md)。

## BULB-POSE90-SCRATCH-20260929-11

按用户要求把初态姿态项免罚范围15→90度，其余奖励/物理/PPO保持，版本profile5。原GPU6在1754轮停止并保留model1700，新6144环境独立从0/空优化器/LR0.001启动，未加载pilot或旧模型。
CPU53、原生12项、32环境13轮及正式至8轮的更新/保存/实际配置/契约/hash通过。GPU4保持。尚未评估学习效果。
见[执行与验证](logs/diagnostics/bulb_pose90_gpu6_20260929/README.md)。

## BULB-GPU6-VIDEO-20260929-10

当前GPU6新姿态引导run最新保存点model1500，原冻结profile4/60Hz物理/30Hz控制/12秒、seed42单环境确定性回放36秒；3个完整回合均超时，0次成功、全程步前无释放/抓持，joint1范围-42.34至-7.05度。
视频1080帧完整解码、两端/桌面hash、代码/契约/配置通过，回放正常退出，训练保持；未筛选，非独立评测。
见[录像、桌面链接及证据](recordings/xinyu21_bulb_gpu6_latest_20260929_1815/README.md)。

## BULB-GPU6-PLATEAU-AUDIT-20260929-09

本次GPU6完整终端告警均在首轮前，PPO更新持续，两个run的事件无NaN/Inf；1351–1450轮GPU6回报4.310且持续上升、旋转2.925圈，比同轮GPU4圈数更高。
同轮主要差异为palm_above_support 0.00081对GPU4的0.30962，非当前姿态成本直接扣分；约99.19%超时，释放/搬运未建立。不支持报错致坍缩，姿态项影响早期探索仍为待验证假设。
只读，未停训/重启/调参；证据和统计口径见[诊断记录](logs/diagnostics/bulb_gpu6_plateau_audit_20260929/README.md)。

## BULB-INITIAL-POSE-LEARNING-AUDIT-20260929-08

两run实际env仅profile与initial_arm_pose差异，agent仅run_name差异，冻结源码/运行库校验通过。629-728轮mean_reward为GPU4 -1.831、GPU6 -2.112；旋出峰值日志均值0.747/0.439，成功率均0，GPU6释放指标略高。
GPU4在900–999轮奖励均值已+2.653，GPU6尚未到该区间。GPU6新增姿态项在100–299轮相对旋转正奖励约30–48倍，说明成熟策略回报占比4%的原标定不足以覆盖从零探索；近期新增成本已下降、旋出进度回升。
单seed观察，无受控消融，不能作确定因果结论；未改奖励/停止/重启训练。[原始窗口、参数差异与图表](logs/diagnostics/bulb_initial_pose_learning_audit_20260929/README.md)。

## BULB-SUCCESS-VIDEO-20260929-07

GPU4最近100个Train/success_rate平均81.785%；选取其最新保存model6600，保持冻结profile3、物理60Hz/控制30Hz/12秒，单环境seed42确定性回放1440步。
原始视频约48秒，5个完整回合成功4/5（成功终止和+10奖励一致），另1个超时、末尾1个未完成；不筛选成功片段，不外推整体成功率。
1439帧全部解码、下载/桌面哈希、源码/运行库/实际奖励/契约核验通过；GPU4/GPU6训练保持，临时回放退出。GPU6为新增姿态奖励的早期训练，不能与此当作公平版本比较。
[录像、选择窗口及每步轨迹](recordings/xinyu21_bulb_gpu4_success_20260929_080026/README.md)。

## BULB-INITIAL-POSE-SCRATCH-GPU6-20260929-06

用户要求替换GPU6并明确从零。原GPU6停在6430、model6400保留；临时GPU4 model6300续训在用户纠正后停止，独立新目录从零初始化。
最终 `20260929_071719_rotate_bulb_initial_pose_phys60_scratch_gpu6`：6144环境、32步、seed42、LR0.001、15000轮、60Hz物理/30Hz控制/12秒，实际PPO参数与GPU4一致；只增加profile4姿态引导。
10项原生奖励检查、从零32环境13轮、正式6144到6轮、空优化器/禁止load、更新/保存、新负奖励、契约和哈希核验通过。未加载任何checkpoint/pilot权重。
GPU4保持；尚未做姿态改善或独立成功率评测。证据：[完整记录](logs/diagnostics/bulb_initial_pose_gpu6_launch_20260929/README.md)。

## BULB-INITIAL-POSE-GUIDANCE-20260929-05

Randomized profile4新增固定初始关节角引导：-0.025、15度免罚、90度尺度、joint1..7权重[2,1,1,1,.5,.5,.5]、释放后0.2。无抓持门控、无大偏移封顶，不改主奖励/物理/契约或成功条件。
CPU68项通过；真实函数与RewTerm参数对旧model5100四条轨迹的离线pre-step计分为-2.018/-1.856/-2.018/-1.924（旧回报的3.68–4.15%）。不是修改后的仿真或学习效果验证。
未同步服务器/重启GPU4或GPU6；原生和训练未测。证据：[参数、代码哈希、复算及测试](logs/diagnostics/bulb_initial_pose_guidance_20260929/README.md)。

## BULB-JOINT1-TRACE-20260929-04

GPU7临时重放model5100/seed42原冻结环境，独立只读行跟踪wrapper，1440步/4回合。joint1从-8.35度主动转到约-160度；第一回合2秒已接近限位、5.43秒释放，grasped始终false，位置成本没有生效。第3回合抓持在大幅反转后建立，参考也随之更新。
第一回合arm_motion累计-0.6217，对比position_tracking+37.266；逐项累计与返回奖励核验一致。支持当前奖励允许低成本大幅绕转，未做限制joint1的因果消融，不宣称该路线必然最优。
四回合均超时/最终success0，仅动作诊断；GPU4/GPU6进程及start_ticks保持，GPU7释放。下载轨迹哈希通过；原录像与诊断录像抽帧近似而非位级一致。
证据：[轨迹、图表、脚本及核验](logs/diagnostics/bulb_joint1_trace_20260929/README.md)。

## BULB-ARM-PENALTY-AUDIT-20260929-03

核对model5100录像冻结源码四文件与本地SHA256一致；实际play_config中action_l2/action_rate_l2/arm_motion权重分别为-0.05/-0.02/-0.3。
arm_motion释放前罚速度，仅grasped时增加姿态项；参考在对向接触连续0.12秒建立抓持时刷新，失联0.2秒退出抓持，重抓再次刷新，释放后整项关闭。
位置项每关节为clip((abs(q-reference)-0.05)/0.15,0,1)^2，再取七关节均值。0.20rad起封顶，不会继续增加对大幅转动的成本。
动作是当前实测q加最多±1rad目标增量，再裁剪关节限位；动作L2与动作差分L2均不是累计初态偏移。
支持“当前奖励未表达始终靠近初态”的代码结论；未提取视频关节轨迹，不能确认第一关节精确180度或因果唯一性。未改运行代码/训练。
证据：[录像配置](recordings/xinyu21_bulb_phys60_latest_20260929_031720/play_config.yaml)、[阶段奖励](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbRandomized/phase_rewards.py)、[参考状态](source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbRandomized/mdp.py)。

## BULB-VIDEO-20260929-02

用户要求录制当前已出现成功率的训练最新checkpoint并拉回本地。03:15:37 UTC 查询GPU4到5159轮，最新保存点为model5100；录制固定该模型，未改训练。
使用GPU4冻结391源码/52运行库、私有checkpoint及匹配两契约；空闲GPU7单环境、seed42、确定性策略，实际60Hz物理/30Hz控制/12秒回合和奖励配置与训练一致。
play退出0；视频1280×960、30FPS、1439帧、47.9667秒，原生请求1440控制步产生1439帧，差一帧原因未确定，全部声明帧解码通过，无剪辑。
下载与桌面副本哈希均匹配服务器；GPU4/GPU6四个训练进程PID/start_ticks保持，GPU7回放退出释放。画面可见离座握持抬升，未核验最终目标成功，也非独立成功率评测。
完整命令、模型哈希、配置、日志及验证见[录像证据](recordings/xinyu21_bulb_phys60_latest_20260929_031720/README.md)。

## BULB-GPU4-PROGRESS-20260929-01

- 来源为GPU4 `20260928_140146_rotate_bulb_phase_phys60_30hz_12s_scratch_gpu4`，检查时约4904轮/最新model4900。
- 固定原始窗口4705–4804与4805–4904：后窗口释放99.9099%、最高旋出2.99927圈、释放后接触质量0.79941；抓稳/稳定/搬运奖励持续非零。
- 搬运奖励2.85096→2.89734、目标误差27.876→27.396cm；成功率日志0.130320%→0.113137%，未随奖励同步改善。首次正成功日志3197轮。
- 日志有实质阶段进展和少量完整成功，但不能判断已接近稳定收敛；窗口均值不是独立成功数/总数，接触质量也不是抓取成功率。
- 未改配置/训练/预算，未新开回放或独立评测。[原始指标与口径](logs/diagnostics/bulb_gpu4_progress_20260929/README.md)。

## BULB-GPU4-PHYS60-SCRATCH-20260928-06

- 用户最终选择：物理60Hz/decimation2、控制30Hz、回合12秒、从头训练；奖励与GPU6保持一致。
- 原`20260927_141100_rotate_bulb_contact01_tracking40_resume100_ddp`于21:52停在3680轮；逐PID/start_ticks核对后SIGTERM监督，四进程退出，model3600及日志保留。
- 新`20260928_140146_rotate_bulb_phase_phys60_30hz_12s_scratch_gpu4`于22:03:56启动，GPU4/6144环境/32steps/seed42/15000轮，初始LR0.001，未加载任何checkpoint。
- GPU6保持原监督1174265/worker1174266及其冻结物理240Hz/decimation8/20秒；GPU7已释放。
- 391源码/52运行库以GPU6冻结副本为基线，仅Randomized配置改变运行行为，另更新README；全部rewards与GPU6实际配置一致，物理/回合变更不改成功判据或碰撞。
- 本地CPU76项通过；最终短测32环境13次PPO/416控制步、原生奖励10项、超时重置通过，optimizer20→260、8个actor参数更新、77张量有限，66种标量742值有限。
- 240Hz/12秒续训候选只做三次PPO后被用户选择取代，未正式启动；中间候选与审计修订保留，不用于最终新策略。
- 22:10:39正式到3轮，首个model0 optimizer step20且动量非零、77张量有限；66种标量263值有限，配置差异仅dt/decimation/render_interval/回合时间，两契约及391源码/52运行库哈希通过。
- GPU4约26636MiB，GPU6原进程到3709、GPU7为3MiB/0%；未作独立任务成功评测，不声称物理等价、吞吐收益或抓取改善。
- [方案、运行路径及原始证据](logs/diagnostics/bulb_gpu4_timing_launch_20260928/README.md)。

## BULB-TIMING-CONFIG-20260928-05

- 用户要求每动作物理更新与桌面抓取一致、最终回合时间12秒；后明确控制30Hz。最终本地Randomized配置为物理dt1/60、decimation2、render_interval2、task_episode_length_s12，控制30Hz、360控制步/720物理步上限。此前只改decimation、保留dt1/240导致120Hz控制的理解错误已修正，旧解析JSON留作历史。
- GitNexus本工作区main/4707074影响LOW，PLAY直接继承；动态注册另行核对。未修改Custom/Assembly及共享函数，保留奖励、碰撞、solver24/4和压力30Hz。
- CPU执行实际共享配置函数及完整Randomized训练/PLAY类，仿真对象/configclass用替身，解析通过；无Isaac原生启动。`brainco/bin/python -m pytest -q tests/test_rotate_bulb_randomized.py tests/test_rotate_bulb_phase_rewards.py tests/test_rotate_bulb_progress_penalty.py tests/test_bulb_bounded_policy.py::test_new_checkpoint_requires_matching_contract_and_reloads_exactly`：39 passed in 5.03s。
- GPU4＋7及GPU6冻结作业未重启/覆盖；旧checkpoint控制周期契约保护保留。未测量墙钟提速、显存及策略质量，不宣称性能收益。
- [配置解析与验证范围](logs/diagnostics/bulb_timing_config_20260928/resolved_config_cpu.json)。
- 修正后再次执行上述CPU命令，39项通过；[当前30Hz配置解析与验证范围](logs/diagnostics/bulb_timing_config_20260928/resolved_config_cpu_control30.json)。未验证物理dt1/60下的原生稳定性或任务质量。

## BULB-PHASE-CONTACT-20260928-04

- 用户授权释放前减接触奖励/臂运动、释放后引导抓取，GPU6训练；确认从当前双卡最新checkpoint续训。
- Randomized profile3：辅助接触释放前0，释放后单侧最高0.25/s、对向1/s、每回合最多4分；
  randomized_grasp权重1→0；arm_motion改为所有释放前状态约束臂速、权重−0.1→−0.3，
  位置漂移仍仅真实抓稳后约束，释放后该项0。其他奖励/物理/动作/观测/分布/成功判据保持。
- 来源为接触0.1/跟踪40双卡run自己的model3500，SHA256以`2f084913c739842d`开头；
  模型/8项归一化/优化器77张量精确恢复，step70040、有效LR0.00013647718878928572。
- 新run `20260928_123831_rotate_bulb_phase_contact_resume_gpu6`，20:45:18启动GPU6/6144，
  从3500追加11500，总上限15000。391源码只替换一个配置/新增一个奖励模块，52运行库保持。
- CPU60通过，32环境原生10项及pilot2三次PPO通过，step70040→70100；首个pilot的验证断言失败
  发生在加载前，旧日志/脚本保留。正式重新加载原model3500，不继承pilot权重。
- 20:53:41正式到3504：首个新model3500优化器step70040→70060，8个actor参数改变、77张量有限，
  两个契约相同，实际env.yaml无预期之外的差异；本机TensorBoard新run及阶段接触指标HTTP200通过。
- GPU4＋7原四PID/start_ticks保持，当时到3533；GPU6显存约26296MiB。单卡每次196608转移，
  原双卡393216，不能按迭代号当作等样本比较。本次未证明抓取/搬运成功率提升，未作独立评测。
- [方案、代码差异、失败修订及原始证据](logs/diagnostics/bulb_phase_rewards_20260928/README.md)。

## BULB-VIDEO-20260928-03

- 用户要求在服务器用当前最新checkpoint回放录像并拉回本地。19:59:48选取时训练到3397轮，
  最新保存点为当前接触0.1/跟踪40双卡run的model3300，SHA256以`153107cd08a83fc2`开头。
- GPU6空闲核查通过；独立模型/契约/源码/运行库副本，390源码及52运行库文件与正式run哈希一致。
  原生play、Randomized旋出、1环境、seed42、1200步；全部训练env.*覆盖原样保留，未改奖励/物理/判据。
- 视频40秒、1200帧、30FPS、1280×960；完整服务器解码和模型/契约核查通过，无指定致命错误模式。
  本地视频及中文桌面副本与服务器SHA256一致；本机无ffmpeg/cv2，未声称本地独立解码。
- 原训练四PID/start_ticks保持，录制后到3410轮；GPU6释放为3MiB。未作额外策略评测或成功片段筛选。
- [视频、精确命令及完整验证](recordings/xinyu21_bulb_contact01_latest_20260928_115948/README.md)。

## BULB-TENSORBOARD-20260928-02

- 当前GPU4＋7、12288环境、接触0.1/跟踪40续训保持，实际任务为Randomized旋出，继承Custom物理。
  19:54:06到3383/15000，最新model3300；原worker1100555/1100556、start_ticks78691089保持。
- 原始事件文件读取至3382；3283–3382共100点released99.9761%、最高圈数2.999079、reward10.95985。
  释放后抓稳奖励0、位置跟踪奖励2.4153e−6、目标误差36.319cm、超时94.651%、掉落5.354%。
  本次读取的100–3382共3283个Train/success_rate标量全0，尚无完整搬运成功证据。
- 近100轮采样23.8755/PPO0.2223秒，合计24.0978秒；相邻前100点释放99.9688%，已保持高释放状态指标。
  日志点等权平均，非跨卡回合合并计数、非固定策略独立评测；未新开训练或仿真。
- 72MB训练日志未检出Python Traceback、CUDA OOM/非法访存、指定碰撞栈及PhysX error模式。
- 复用TensorBoard66933，建立本机6006到远端6006的跳板SSH转发；页面、run及标量HTTP200通过。
- [快照、完整精度曲线及连接说明](logs/diagnostics/bulb_training_tensorboard_20260928_1954/README.md)。

## BULB-JOINT-SOLVER-20260928-01

- 本次实际执行，遵循用户后续“少环境、少轮数、尽快定位”要求。xinyu21/GPU6，
  `R=/home/xinyu/training_runs/20260928_joint_solver_gpu6`。旧GPU6碰撞队列停止并保留结果；GPU4/7未操作。
- 从当前接触0.1/跟踪40的Randomized冻结model200恢复，SHA以`c1ab41bb`开头；505文件复制核验，核心生产源码一致。
- 128环境统一无接触、不睡眠的Robot＋Lamp/Robot/Lamp/臂/手/无固定tip诊断，native数量逐项确认。
  Robot38links/28DOF、Lamp3/2；手部21DOF的内部约束26.31ms，臂7DOF9.16ms（16物理子步）。
  无重复父子关节对；Lamp驱动native stiffness/damping0，必要attachment保留。接触约束的逐资产份额仍未知。
- 正式独立case仅Scene position上下限24→16，保持velocity4、240Hz、30Hz、自碰撞、触觉、奖励与成功判据。
  baseline/candidate各256env×32step×12更新；共同初态最大差0，完整恢复模型/归一化/优化器，optimizer4040→4280。
  排除2轮预热及trace，9轮整轮8.3005→6.9404秒，986.93→1180.34 transitions/s（+19.6%）；
  旧trace相同kernel分类228.630→173.237ms/16子步，主要调用448→320，即每子步28→20次。
- 候选再24次短续训至`adapt256_16_4/train/model_234.pt`；267结束训练回合227曾释放，完整成功0，非有限/大范围异常0。
  固定模型32完整回合：两圈32/32、释放13/32、完整成功0/32，策略参数保持。未证明完整搬运能力或6144吞吐。
- 5个固定tip省略仅为结构诊断（33links/28DOF，轮时−4.3%），未将质量/观测不完整的删除作为修复。
  3个启用碰撞的静态诊断实际接触0，不用于接触归因；真实接触验证来自PPO。
- [完整报告](docs/rotate_bulb_joint_solver_2026-09-28.md)、[本地证据](benchmarks/rotate_bulb/joint_solver_20260928/README.md)。
  失败启动/诊断及原始trace保留；没有采用旧严格误差门槛，没有宣称零成功等效。

## BULB-CONTACT01-RESUME-DDP-20260927-09

- 用户要求只将接触权重1→0.1、保留position_tracking40，并停原双卡训练、从其日志内高释放保存点恢复。
- 唯一正确父run为`20260927_041844_rotate_bulb_contact_transport_12288_ddp`，标签Contact1_Tracking40_12288_DDP_GPU4_7。
  原run停在1425轮，model1400与日志保留，精确核验原四个进程退出；其他GPU作业未操作。
- 对现存保存点比较完整尾部50点released，model100为39.2601%、model200为30.0495%、model1400为3.5927%。
  选择该run的train/model_100.pt，SHA256为`1b4c9521d427b5f7c1066fb20d6585b4a47f797ea803d45a7d7f449f485ab024`。
  这是主rank历史训练窗口平均，不是固定checkpoint独立释放率；原训练完整成功仍未得到证明。
- 纠错记录：早先误用旧GPU6 model7300做32×2三次更新的隔离短测，未启动正式任务；澄清后标记SUPERSEDED。
  正确model100另做32×2三次更新核验，正式再次从原model100加载，未继承任一pilot的更新结果。
- 新run `20260927_141100_rotate_bulb_contact01_tracking40_resume100_ddp`，22:16:07正式启动。
  GPU4＋7每卡6144、共12288、32steps/env、seed42/43；从100追加14900次更新，总上限15000保持。
- 390个冻结项目文件仅Randomized/env_cfg.py变化，52运行库文件相同；本地/服务器/冻结配置SHA一致。
  实际env.yaml仅接触1→0.1及版本1→2；tracking40、failure−6、随机化、物理、观测/动作与成功判据保持。
- 两rank逐张量完整恢复模型、8个归一化状态、优化器77个有限张量，恢复step2020和有效LR0.00037290056934580207。
  命令同步恢复算法adaptive LR标量，避免RSL-RL只加载optimizer后首次更新覆盖为默认值。
- CPU64通过；正确pilot及正式前三次更新均验证梯度平均、每次20归约、最大跨卡参数差0。
  正式首个新model100优化器step2040、actor参数有更新、77张量有限、两rank契约与父run一致，原文件哈希不变。
- 22:25:42正式到110轮；106–110平均17.0748秒/轮，属于初始化后短窗，不能作为稳定提速结论。
  本机TensorBoard新Contact0.1_Tracking40_Resume100标签及HTTP200已验证，原标签保留。
- 未做新奖励独立释放/抬升/成功评测，尚不能宣称收益冲突缓解。仿真环境重新reset，checkpoint不含采样现场。
- [本地详细记录和41份下载核验文件](logs/diagnostics/bulb_contact01_resume_ddp_20260927/README.md)。

## BULB-RELEASE-DECLINE-20260927-07

- 18:46正式run到917轮；只读主rank全精度曲线、实际PPO配置及RewardManager/CommandTerm/runner聚合源码。
- 500–599→800–899：released3.2973%→1.5422%，达到三圈2.1223%→0.8549%，reward23.5648→24.2749，超时98.3343%→99.3888%。
- 接触0.757907与位置跟踪0.00009935为最近100轮同口径日志均值，释放抓稳/成功均0。
  持续接触最高+1/s、停滞最高-0.3/s，释放后可能丢接触并掉落-6；支持接近末端后保持接触的局部收益解释。
- 五指平均饱和接触可以由单侧四指获得0.8/s，不等价于释放后对向抓稳；无逐指力轨迹，不断言具体哪根指未接触。
- gamma实际0.999，std由约0.270降到0.253；未做单因素消融或冻结策略轨迹，本次未修改奖励/训练。
- [完整分析与50轮平滑趋势图](logs/diagnostics/bulb_release_decline_20260927_1845/README.md)。

## BULB-REWARD-RANDOMIZATION-AUDIT-20260927-06

- 当前GPU4＋7新run保存配置、两rank已初始化契约、5个源码文件、本地/远端bank哈希交叉核对，只读，未重启。
- 配置21项中17个非零；接触1按秒积分，位置跟踪40为释放后新进展累计预算，失败每回合一次-6。
- 支架XY各±10cm；目标相对支架沿世界XY各±15cm、Z+28–42cm。支架Z/姿态、臂初态固定，手零位、初始三圈。
- 441个支架网格点保留432，每点8个目标，共3456行配对离散分布，startup/reset均匀抽样；两rank种子42/43。
- 本次没有新的原生抽样/成功率实验；启用证据是实际配置、初始化契约与执行源码，几何筛选不等于动态成功。
- [全部奖励与证据](logs/diagnostics/bulb_rewards_randomization_20260927_1653/README.md)。

## BULB-RELEASE-LIFT-20260927-05

- 18:43只读跟进至909轮：released单点4.0104%，最近20点1.8463%、最近100点1.6108%，成功率仍0。
  原始曲线见`logs/diagnostics/bulb_release_rate_20260927/`；仍为主日志rank的标量窗口平均，不是独立评测释放率。
- 16:49正式新双卡run到604轮/model600，未修改或重启；TB快照0–601轮。
- 两个连续50轮窗口502–551/552–601，释放指标3.3101%→3.3040%，最高圈数2.93855→2.94259。
  position_tracking均值0.00010694→0.00013523，非零日志点42/50→47/50；release_grasp及成功率均0。
- 当前冻结源码与本地两配置文件哈希一致：位置奖励为释放后、软接触支持下三维目标接近增量；
  无独立正向Z位移条件，不能以微弱非零判定已形成抬升。目标误差仍约35cm，稳定抓持奖励近零。
- 两条辅助grasp日志最后在206，已标为过期；首次被其拖回旧窗口的无效统计保留，最终窗口以持续更新指标确定。
- 主日志rank训练指标，不是跨卡合并回合率或独立评测。[原始曲线及源码口径](logs/diagnostics/bulb_release_lift_20260927_1647/README.md)。

## BULB-TENSORBOARD-20260927-04

- 16:15:40正式GPU4/7、12288环境训练到516轮，最新model500；原worker start_ticks保持，未重启/改配置。
- 通过已有SSH增加本机127.0.0.1:6006转发，HTTP200、当前run和23条标量读取通过。
- TB快照截至512，最近413–512共100轮采样23.30351/PPO0.24228秒，合计23.54579秒；最近20轮23.13372秒。
- 总奖励均值从100–199窗口2.7221升至23.0728，接触奖励0.11054→0.73484，最高圈数2.75309→2.93669。
  最近100点释放指标3.847%、最终圈数2.89036；release_grasp为0，position_tracking为5.8242e-5（80点非零）。
  全部513个训练成功率日志点仍0，奖励上涨未证明释放后抓稳/搬运改善。
- 仅主日志rank训练指标、窗口对日志点等权平均，非跨卡全回合合并计数，非冻结checkpoint独立成功率评测。
- [曲线入口、重连步骤和原始数据](logs/diagnostics/bulb_training_tensorboard_20260927_1614/README.md)。

## BULB-RUNTIME-PERF-20260927-03

- 用户要求定位耗时并尝试优化，随后明确授权停止旧GPU6用于测试；旧run在7352轮正常退出，model7300及日志保留，USER_PAUSED。
- GPU4/7新12288环境训练继续，最近152–251共100轮采样25.00863/PPO0.30534秒。
  xinyu20实查GPU0有约30GB余量但被rerun可视化进程占用，其他卡有计算任务，未操作该机作业。
- 新run冻结390源码/52运行库、model200私有副本，在GPU6独立6144环境8轮基准通过；
  同步物理84.4867%、压力5.0722%、奖励/终止4.7824%、点云0.6592%。
  无同步窗口采样25.2556/PPO0.1783秒，两种窗口分开记录。契约一致、模型有限、最终优化器step4180。
- 失败记录保留：首次基准缺失地面资产链接，首次点云显示覆盖被Isaac严格from_dict拒绝，均0轮PPO。
  已修正诊断入口，退出码0不作为实验通过条件。
- 点云显示关闭8轮通过：点云组0.1716→0.0100秒，但整轮25.4339→25.5130秒，没有整体收益。
- 主线程绑核ABA各10轮：整轮25.6109→25.1372→25.3416秒，短窗约1.33%收益，不足以确认稳定加速；原affinity已恢复。
- 单卡4096同model200/原物理8轮通过：19.9072采样＋0.1270PPO＝20.0342秒，6542.4样本/秒。
  单卡吞吐较6144下降15.37%；三卡各4096保持12288，约21%轮时缩短仅是外推，未启动/实测三卡DDP。
- 15:04:53三个case模型有限、契约相同、优化器4020→4180；诊断退出，GPU6为2MiB/0%。
  原GPU6 model7300哈希保留；GPU4/7生产源码390/运行库52和进程start_ticks未变，到339轮/model300。
  最近20轮25.2016秒。本次没有生产代码优化部署，没有降低物理精度或缩短回合。
- [实验细节](docs/bulb_runtime_optimization_2026-09-27.md)、[本地证据](logs/diagnostics/bulb_runtime_performance_20260927/)。

## BULB-COLLISION-FAST-20260927-11

- 当前接触0.1/跟踪40生产快照、model200，390源码/52运行库哈希匹配；仅使用GPU6。
- 独立Lamp副本实现螺纹减面/SDF128、Shell单凸包；独立Robot副本将掌部28凸形状合一。
  原资产、关节/attachment、物理频率/solver/partitions、控制、观测、奖励和成功判据保留。
- 1024固定动作256步，前64步预热：baseline 10.3000s、threads_low 11.4300s、
  shell_hulls 11.0057s、selfoff 8.1632s/32步；初态最大差0、同动作哈希、抽样非有限状态0。
  不用旧轨迹偏离或接触误差淘汰；前两候选无速度收益，selfoff需适应训练验证。
- 首个6144对照在PPO前因输出目录缺失失败，已保留日志并修复；32环境3次PPO及完整恢复通过。
  6144等预算训练、独立完整回合评测、最终采用配置尚在执行，本记录不宣称任务完成。
- [实改与结果记录](docs/rotate_bulb_collision_fast_2026-09-27.md)、
  [资产验证及原始证据](benchmarks/rotate_bulb/collision_fast_20260927/)。

## BULB-CONTACT-TRANSPORT-DDP-20260927-02

- 用户要求增强抓握和搬运回报；仅Randomized配置覆盖any_finger_contact=1、position_tracking=40，
  均提高10倍，failure保持−6。保留接触门控、释放/成功判据、动作/观测、回退/停滞惩罚和物理配置。
- 本地主检出main/HEAD4707074带既有未提交修改；定向同步一个运行文件，新SHA256
  `57a0aeaa402243eefffa43dc1d080c8f0d39d2024a86e23195edb79aad17c75d`；原文件备份。
- CPU相关检查163通过；单卡32环境3轮及6144环境1–5轮启动检查通过，后者平均18.87秒。
  用户确认后停止旧GPU4和临时GPU7单卡，原checkpoint/日志保留，GPU6训练继续。
- NCCL默认协议小数据成功，但512KiB独立通信及原生模型同步失败；失败实验保留。
  新run限定NCCL_PROTO=LL后，通信到32MiB通过，32×2原生3轮PPO通过：
  每轮20次梯度归约、首轮均值正确、三次更新后两rank参数差0，优化器step60/25状态张量变化。
- 正式run `20260927_041844_rotate_bulb_contact_transport_12288_ddp`，4＋7每卡6144、共12288，
  seed42/43、393216样本/轮、最多15000轮，从头训练。正式13:00:31核验通过，
  实际12288环境、前三次更新参数差0，model0有限/优化器step20、390源码/52运行库/契约核对通过。
- 正式第1–5轮平均19.39秒（采样19.0658/PPO0.3234），平均20281条/秒；显存GPU4/7为25957/26311MiB。
  TensorBoard标签实际可见，正式证据下载哈希一致；刚启动窗口不是长期性能基准，无学习成功结论。
- [实现、资源与故障记录](docs/rotate_bulb_contact_transport_2026-09-27.md)、
  [本地证据](logs/diagnostics/bulb_contact_transport_20260927/)。

## BULB-PROGRESS-VIDEO-20260927-01

- 用户要求当前旋出模型视频；10:11:20选取新GPU6 run最新保存的model6500，seed42/单环境/1200步。
- 对应390源码文件、52运行库文件先核对后复制，使用私有checkpoint；含新增回退/停滞奖励。
  相机/渲染、线程及单环境容量预分配仅在录像副本覆盖，原任务成功条件与物理规则保持。
- 空闲GPU7录制40秒/30 FPS/1280×960，完整视频保留；两端OpenCV与本地FFmpeg解码通过，
  源模型及私有副本哈希一致，两份契约一致，下载与桌面视频SHA256一致。
- 10:19:46新训练6603轮、6144环境，GPU6/GPU4原PID与start_ticks保持，GPU7回到3MiB。
  原训练未暂停；装配/插销保持暂停。本次没有训练代码改动。
- 抽帧未见明显抬高搬运，仍主要在支架附近；灯泡被手遮挡，未据此判定约束是否释放。
  无独立完整回合成功数，不作为成功率评测。回放未检出Traceback、碰撞栈溢出或CUDA OOM。
- [录像、manifest与核验文件](recordings/xinyu21_bulb_progress_penalty_20260927_021120/README.md)。
  桌面副本：灯泡旋出_回退惩罚_model_6500_20260927.mp4。

## BULB-LATEST-VIDEO-20260925-02

- 用户再次要求查看两个训练并录制最新checkpoint。19:59:16固定选取旋出model_3700、
  装配stage0_free/model_6900；两项原run与下午相同，没有挑选成功片段。
- 20:07:23复核：旋出3817/15000、装配6940（自由阶段自3422新增3519次更新），最近100轮日志成功率均0%。
  旋出最大进度日志均值2.966117圈、释放比例0.016808；装配no_contact失败原因比例0.998237。
  20:00附近末10000完整训练回合严格抓持0/10000、reset失败0/10000；非独立评测。
- 原冻结项目385/498文件及RSL运行库52/45文件、共享资产链接全部核对后复制；
  装配完整恢复课程settings，保留stage0/free、transfer=false、evaluation=true和原契约。
- 在空闲GPU1/7同时单环境回放；仅录制专用渲染、线程及容量预分配调整，
  未修改物理、动作、观测、奖励或成功条件。原PID852306/871712及start_ticks保持，原checkpoint哈希保持。
- 两段各40秒/1200帧/30 FPS/1280×960；服务器和本地OpenCV、本地FFmpeg完整解码通过，
  服务器/归档/桌面SHA256一致。装配回放严格抓持0/8，非reset帧见手臂移开、灯泡留桌；
  旋出抽帧仍见灯泡位于底座。失败保留，不构成独立完整任务成功率评测。
- 两份完整训练日志定向扫描未检出原碰撞栈溢出、CUDA OOM、Python Traceback或PhysX error。
  录制结束GPU1/7各回到3MiB；插销保持暂停，本次没有训练干预。
- 交付：[晚间录像目录](recordings/xinyu21_bulb_latest_20260925_120000/README.md)、
  [核心核验快照](docs/evidence/xinyu21_bulb_video_20260925_2000.json)。
  桌面：灯泡旋出_手心上方_model_3700_20260925.mp4、灯泡装配_自由抓取_model_6900_20260925.mp4。

## BULB-LATEST-VIDEO-20260925-01

- 授权：查看当前两个训练，回放服务器各自最新checkpoint，录制并下载本地。
  当前两项为随机灯泡旋出和灯泡装配课程；插销保持用户暂停。
- 13:10:31固定选取旋出model_2900及装配Stage0-free/model_5900，不挑选成功模型或片段。
  原run分别为20260924_081649_rotate_bulb_palm_high_z_gpu4、
  20260924_100337_bulb_curriculum_buffers_4096_gpu6。
- 13:24:13训练复核：旋出2934/15000，装配6026（自由阶段自3422起新增2605次更新）；
  最近100轮日志成功率均为0%。旋出最大进度均值2.963圈、释放比例均值1.861%；
  装配no_contact原因比例均值0.998573，近期10000完整训练回合严格抓持0/10000。
  两份完整训练日志定向扫描未检出碰撞栈溢出、CUDA OOM、Python Traceback和PhysX error。
- 使用各自冻结快照：项目385/498文件、RSL运行库52/45文件哈希一致。
  装配完整恢复checkpoint内课程settings，stage0/fixed_bulb=false、transfer=false、evaluation=true；
  保留原动作、观测、奖励、物理及成功判据，未把模型作为完整装配任务回放。
- 空闲GPU1依次录制；仅录制进程改1环境、RTX相机、线程与容量预分配。
  原训练PID852306/871712及start_ticks保持，原checkpoint哈希保持，GPU1结束后释放。
- 两段各40秒/1200帧/30 FPS/1280×960；服务器及本地OpenCV、本地FFmpeg完整解码通过，
  服务器/归档/桌面SHA256一致。装配回放严格抓持0/8；失败完整保留，不是独立完整任务评测。
  旋出抽帧见手在灯泡上方但灯泡仍在底座；装配非reset帧见手臂离开、灯泡留桌。
- 首次旋出录制副本遗漏assets/usd嵌套共享链接，场景创建即退出；补原链接重试通过，
  失败日志保留，未改原训练或任务代码。PPO继续不代表当前策略有效。
- 交付与详细证据：[录像目录](recordings/xinyu21_bulb_latest_20260925_051200/README.md)、
  [核心核验快照](docs/evidence/xinyu21_bulb_video_20260925.json)。
  桌面文件：灯泡旋出_手心上方_model_2900_20260925.mp4、
  灯泡装配_自由抓取_model_5900_20260925.mp4。
- 未执行：新的训练干预或独立多seed完整任务成功率评测。

## ASM-PHYSX-RESUME-20260924-04

- 当前授权：扩大灯泡装配课程的PhysX碰撞缓冲区，最好两倍以上；停止插销，在服务器重新训练。
  操作对象为 `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0`，不是旋出任务。
- 原始原因已复核：9月23日8192环境run的固定Stage0模型3419通过288/300冻结评测，
  随后Stage0-free报 `collisionStackSize buffer overflow` 并丢失接触。两阶段实际env.yaml的碰撞栈均为2GiB；
  报错建议值为负，不能直接采用。本次假设增加预分配容量可消除该项瓶颈，不把启动成功当作长期稳定证明。
- 当前run `/home/xinyu/training_runs/20260924_100337_bulb_curriculum_buffers_4096_gpu6`。
  4096环境、seed42、新reward v2；7项相关容量翻倍，碰撞栈改为4026531840字节（3.75GiB、1.875倍）。
  碰撞栈为PxU32，4GiB已越过4294967295上限；故此项没有声称翻倍。
  原值/新值/单位与官方类型依据见[运行手册](docs/RUNBOOK.md#2026-09-24装配课程扩大-physx-容量并恢复gpu6)。
  未改变碰撞几何、接触约束、求解器、任务奖励、难度分布或成功条件。
- 实施采用已有课程启动器的 `env.sim.physx.*` 覆盖，传递到所有训练及独立评测阶段；
  没有改共享任务默认值和Python。新run冻结498份项目文件、45份RSL文件，与原装配运行快照逐项SHA256相同；
  六个共享依赖链接保留，不称完整隔离镜像。当前本地main/4707074与大量原有未提交改动未被重置。
- 检查点复制为新run `predecessor_stage0/model_3419.pt`，SHA256仍为
  `5447169ed79d84e38138274e0de54f4a156f239c9f4798f54b2154307b2098c9`，
  连同原契约、参数、阶段状态、冻结评测一起留存。CPU读取检查点全部张量有限，669输入、
  固定Stage0、scheduler.ready=true；只8项env覆盖与旧run不同，其余env覆盖保持。
- 恢复路径：已有启动器 `--start stage0 --checkpoint <model3419>`，继续策略/优化器/归一化/课程状态，
  固定阶段重新累计提名窗口并做401/402/403各100回合评测，合格后按原逻辑进入Stage0-free等后续阶段。
  没有把固定模型伪装成自由阶段checkpoint，也没有跳过晋级检查。
  每次阶段启动最多追加9000更新；固定恢复阶段日志迭代从3419接续，后续相邻transfer逻辑不变。
  首试 `20260924_095053_bulb_curriculum_buffers_gpu6` 于17:53:19启动8192环境，
  建场景86.34s、物理启动314.49s，模型加载后18:02:24首轮采样CUDA OOM；GPU已用31.21GiB，
  剩余约99.56MiB时再申请128MiB失败，监督停止该进程组。无PPO完整更新，失败原样保留。
  18:03:37以4096环境启动当前run（监督865565、课程865566；训练/评测子进程随阶段切换）；保留所有扩大后的容量。
  18:09核验已完成PPO3419–3421，实际env.yaml确认4096环境及8项扩容数值，其他PhysX参数与原配置相同。
  三份动作/触觉/奖励契约均与原检查点一致，127个最新标量有限，OLD_REWARD=0、reset_failure=0；
  采样显存28090MiB/32607MiB、GPU95%，当前未再出现OOM或碰撞栈错误。
  18:09时尚无新checkpoint；随后PPO3422获得提名并按原机制保存 `model_3422.pt`。
  该模型25个状态张量变化，优化器step从68400到68480，所有检查点张量有限；
  SHA256 `6c022bee293ee88c4aa48dde1c28e2b8e6bf078d5912bc90051e0a886990fa2e`。
  新三seed冻结评测为91/100、97/100、83/100，合计271/300（90.33%）；
  达到原平均≥90%、单seed≥80%门槛，不删除失败回合、不改门槛。该结果不表示优于旧288/300。
  18:18已按原逻辑加载model3422并启动Stage0-free，真实训练PID871712/start_ticks51341145。
  18:24核验自由阶段已完成PPO3422–3424（新增3轮），固定约束为false、transfer=true、4096环境，
  8项缓冲容量实际生效，三份契约与前序匹配，127个最新标量有限，6006实际可见stage0_free标签。
  显存采样29650MiB/32607MiB、GPU92%；首批更新未再出现原碰撞栈溢出或OOM。
  自由阶段尚未保存新checkpoint，沿用100轮/晋级保存机制。长期稳定和完整装配成功仍未验证。
  以环境数折算，碰撞栈为原8192配置的3.75倍，其余7项为4倍；这是容量比例，不是性能/学习效果倍率。
- 插销停止：核对监督242932/训练242935的UID、命令、start_ticks14607241/14607244后，
  只向监督发送SIGINT；17:50:55正常退出0、最后日志13738。
  `model_13700.pt`停止前后哈希一致，旧run所有日志/模型保留。GPU6由18868MiB降至3MiB后才启动装配。
  GPU4旋出PID852306、start_ticks50679487保持，不停止或重启其他用户GPU进程。
- 本地证据 `logs/diagnostics/bulb_curriculum_buffers_20260924/`；远端同run有
  `buffer_profile.json`、`training_verification.json`、`peg_stop_audit.json`、`launch_spec.json`；
  首试另保留 `configuration_verification.json`。核心数据见[核验快照](docs/evidence/bulb_curriculum_buffers_20260924.json)。
  TensorBoard6006入口 `Bulb_Assembly_Buffers375G_4096_GPU6_20260924/<phase>`。
  本次没有Python实现变化，因此未重复运行代码单测；验证重点为参数实际生效、检查点兼容、PPO及物理错误。
  GitNexus文档影响：EXPERIMENTS为LOW；STATUS/RUNBOOK未索引返回UNKNOWN，已用入口文档引用核实。
  未提交Git。

## BULB-PALM-HIGH-Z-20260924-03

- 用户授权：根据刚录制的随机旋出视频，鼓励手心位于支架上方，提高目标Z并随机化，同步xinyu21重新训练。
  仅修改 `RotateBulbRandomized`，不是独立桌面装配课程。假设：增加实际palm相对支架的软奖励，
  并提高可达目标库，能引导旋出姿态及搬运高度；是否学会须后续冻结模型评测/回放。
- 实现：世界XY径向误差2cm内满分，外侧6cm衰减尺度；世界Z相对支架root高10–18cm，带外4cm衰减。
  使用机器人真实 `palm` 刚体原点，不是抓取点/目标姿态。参考抓姿palm高13.36cm用于几何选值。
  接近保持奖励为满接触的20%，随对向手指接触增强，最高0.5分/秒；每回合保持预算2分，
  新旋出进度另有2分预算，反复正反旋转不重复给分。失败/释放后不再给新信用，搬运阶段豁免。
- 目标原本已有Z随机化+18–30cm，本次提高为相对支架+28–42cm并重新筛选。
  新库 `assets/rotate_bulb_randomized/workspace_bank_high_z_v2.json`：3456个位置/目标对，
  保留与旧库完全相同的432个支架位置，2646个不同高度；高度均值0.351889m。
  保存IK解的FK复核最大位置误差0.400mm/旋转误差0.378°；这是离散几何筛选，不是动态成功证明。
  SHA256 `1cc388f22d73219e30ae6fc5da4a024468229e3f50fcd3ed243132935af2b9c6`。
  原库、碰撞、固定臂/零手、三圈释放、3cm/1秒成功标准和20秒回合均保留。
- 契约：仍为687维观测/28动作，随机任务版本升至2，记录新库哈希、Z范围与奖励参数；不把旧v1模型当直接resume。
  源码本地main/4707074加已有未提交工作区及本次改动；远端原382项源码哈希核对，
  新快照385项（含新增文件），独立RSL runtime52项；没有把当前HEAD当作完整干净源码版本。
- GitNexus：刷新本地索引后，配置、workspace加载、契约、新奖励及生成器main影响分析均LOW，
  涉及PLAY继承、reset/command、启动契约和生成器入口。文档索引HIGH已事先告知；
  未找到的Markdown节点为UNKNOWN，补充核实文档引用；保留旧记录。未提交Git。
- 实际验证：CPU `test_rotate_bulb_randomized`、bounded policy、segmented guidance、failure、grasp reset
  合计134通过/2跳过。服务器8环境、16次reset与40物理步共96项原生检查通过，
  128次目标抽样有102个不同高度且覆盖0.28–0.42m；实际palm、释放豁免、无重复支付和契约均检查。
  GPU6用空余显存运行32环境/3轮PPO，退出0，25个模型状态张量发生变化，检查点及优化器张量均有限。
  此短测缩小PhysX预分配容量，正式6144环境保留旧大容量/24+4求解配置。
- 服务器主检出和新快照只定向同步本任务6个代码/测试/资产文件及任务README，旧文件备份在新run的
  `sync_backup/`，旧run源码未改。新run为
  `/home/xinyu/training_runs/20260924_081649_rotate_bulb_palm_high_z_gpu4`；
  `launch_spec.json`、`source_manifest.json`、`runtime_manifest.json`提供精确参数和源码证据。
  GPU4/6144环境/15000轮/seed42/从头训练，启动于16:27:32 CST，PID852306、监督852305。
  16:35 CST核验正式PPO0–3已完成，`model_0.pt`及优化器张量全有限，优化器step均20；
  新契约与短测一致，新奖励为非零且5个手心指标可回读，TensorBoard6006实际可见新run。
  GPU4采样25262MiB/90%使用率；这些证据验证更新/保存/配置生效，不代表新任务学会。
- 旧GPU4 run `20260921_050053_rotate_bulb_randomized_6144_gpu4` 在11243轮收到SIGINT，正常退出0。
  `model_11200.pt`原地保留，停止前后SHA256一致：
  `724244a9f5dff385c8b2216f2e60ccf0d2a15914e3472396e97f7ce1c01835d0`。
  只停止已核对UID/cwd/启动时钟的旋出监督进程；插销PID242935保持原启动时钟14607244。
- 日志入口：TensorBoard6006 `Rotate-Bulb_PalmAbove_HighZ_GPU4_20260924`；新增手心横向误差、相对高度、
  对齐得分、保持预算和有效阶段比例5个指标。新run `native_probe/report.json`、`pilot/verification.json`
  和 `formal_verification.json`、`old_stop_audit.json` 为远端证据；本地工作证据在
  `logs/diagnostics/bulb_palm_high_z_20260924/`，核心数据归档至[核验快照](docs/evidence/bulb_palm_high_z_20260924.json)。
  未执行新模型独立成功率评测；奖励变化和目标分布变化均影响结果，不与旧98.8%训练日志作公平比较。

## PEG-BULB-VIDEO-20260924-02

用户授权录制当前插销与灯泡旋出视频并拉回本地。经跳板在xinyu21录制，全部原始帧保留；
本次未修改训练源代码、配置、奖励、碰撞或成功门槛，未重启/停止训练。
本地main/`4707074194091cd723ebd18dea7824c184e99c6d`加已有改动，回放实际源码以各任务快照哈希为准。

- 交付：[录像目录及说明](recordings/xinyu21_peg_bulb_latest_20260924_074033/README.md)。
  远端`/home/xinyu/recordings/peg_bulb_latest_20260924_074033`保留独立源码、runtime、checkpoint副本、导出和原日志。
- 插销：原run`20260920_041251_task_revision_restart/insert_d_peg_xyyaw_gpu6`的model_13400，
  seed44、669维输入、28动作、原XY/yaw随机化及v4契约；源快照205项预核对，录制185项非测试文件核对。
  MP4为2110206字节，SHA256 `7dcd77937a47a0c54cb1eef19921aec48db5c125a8c985bda3fbe99d9455da61`。
- 随机旋出：原run`20260921_050053_rotate_bulb_randomized_6144_gpu4`的model_11100，
  seed42、687维输入、28动作、原随机工作空间及v3动作契约，原快照382项核对。
  MP4为2194265字节，SHA256 `f10adf498f550e78ef21e55ed747a9d095c42e3257bded89fe69b2dbc3e49a80`。
- 两个模型在准备时按实际已保存的最大迭代号选择；不是按文件名宣称最佳。携带params及契约复制后回放，
  原文件前后哈希不变。共享RSL运行库复制并记录52份Python源码哈希；不推断其与训练启动时逐字节一致，
  本次实际通过checkpoint载入与任务契约检查。
- 均为GPU6单环境、1200控制步、30Hz、dextac相机1280×960；仅回放覆盖PhysX预分配容量，
  collision stack=268435456、contact count=262144、patch/pair/aggregate容量=65536。
  原求解迭代数、物理模式及碰撞开关保持；详细命令、容量覆盖、非敏感环境白名单见各manifest。
- 验证：回放退出码均0；两端OpenCV完整解码1200/1200、本地FFmpeg退出0且无解码报错、下载哈希一致。
  已查看初态及第150/600帧，画面有效；插销含脱手样本，灯泡可见持物离开底座，未筛选或删除失败片段。
  本次未统计回合成功数/总数，不将该单seed确定性录像与训练日志百分比混用。
- 15:56（北京时间）收尾核验：原训练PID242935/489880及启动时间保持，分别running至13498/11168轮；
  录制PID843908/846011均已退出，所查Traceback/CUDA/OOM/PhysX溢出致命错误匹配为空。
  证据见目录内server_postflight.json和各server_verification/local_verification.json。
- 桌面副本：`/home/liuxinyu/桌面/D型插销_model_13400_20260924.mp4`与
  `/home/liuxinyu/桌面/随机灯泡旋出_model_11100_20260924.mp4`。未下载模型或整套训练数据。

## TRAINING-STATUS-20260924-01

用户授权经跳板连接xinyu21并核查当前训练；本次只读服务器，无新训练、推理评测、配置修改或进程干预。
核查时间2026-09-24 15:28–15:30（Asia/Shanghai），本地main/`4707074194091cd723ebd18dea7824c184e99c6d`加已有改动。
当前远端工作区版本未重新审核，本记录依据各run自身状态/日志/评测产物，不声明它们与本地源码一致。
机器hostname为`brainco`、登录用户`xinyu`；没有将认证信息写入项目记录。

- 证据：[只读JSON快照](docs/evidence/xinyu21_status_20260924_153033.json)，包含原始status、最后完整迭代日志、
  末100轮统计、课程进度、三个冻结summary及逐回合重数、Stage0模型SHA256。
- 装配run：`/home/xinyu/training_runs/20260923_104131_bulb_reward_v2_8192_9000_gpu1`。
  固定Stage0从零训练至3419，状态记录6086023回合、896532480转移，提前提名后进行既有冻结评测。
  `curriculum/stage0_validation_{401,402,403}/episodes.json`逐回合重数为92/100、99/100、97/100，
  与三个summary一致，总288/300；`policy_unchanged=true`，`end_to_end_evaluation=false`，reset失败均0。
  最后10000训练回合（ID6076023–6086022）成功8721，87.21%；没有扫描全程10.38GB课程日志和15.16GB奖励诊断日志。
  `curriculum/stage0/model_3419.pt`为12271157字节，SHA256
  `5447169ed79d84e38138274e0de54f4a156f239c9f4798f54b2154307b2098c9`。
- 装配退出：日志在加载上述模型进入`stage0_free`后，于2026-09-24 01:53:46 UTC报
  PhysX `collisionStackSize buffer overflow detected`、`Contacts have been dropped`；
  status标记`fatal_simulation_error`，exit_code=-9。原监督/课程进程已不存在；自由阶段无model文件，
  仅有88字节事件文件等初始化产物，后续阶段未出现。不能将固定阶段96%解释为自由抓持/完整装配成功率。
  报错建议容量为负数，不能直接将其作为合法缓冲区配置；此次未诊断数值来源或实施修复。
- 随机旋出run：`/home/xinyu/training_runs/20260921_050053_rotate_bulb_randomized_6144_gpu4`，
  GPU4/PID489880，6144环境，快照11107/15000轮；model_11100已保存。
  最近100轮打印成功率均值98.7891%，范围97.15%–99.83%；reward均值23.3701，迭代均时24.8243秒。
- 插销run：`/home/xinyu/training_runs/20260920_041251_task_revision_restart/insert_d_peg_xyyaw_gpu6`，
  GPU6/PID242935，8192环境，快照13450/14200轮；model_13400已保存。
  最近100轮打印成功率均值97.8783%，范围96.98%–98.52%；reward均值114.6599，迭代均时28.2796秒。
  两项任务的进程存在且多次读取间迭代/日志继续推进；上述均为每轮日志百分比的算术均值，
  不是按完整回合数加权的总体成功率，也不是冻结模型独立评测；本次没有新评测成功数/总数。
- GPU4进程显存27510MiB、GPU6进程显存18830MiB；原GPU1已被其他用户使用，不具备直接原卡重启条件。
  下一步建议：保留model_3419，诊断自由阶段碰撞缓冲区并按原物理/成功判据验证；
  恢复前核实资源与当前授权。此为建议，未启动恢复、未修改碰撞或放宽门槛。

## ASM-REWARD-V2-8192-20260923-03

用户明确要求8192环境重启，并接受把每阶段上限从2000改为9000；保留提前提名、
401/402/403三个冻结评测seed各100回合及原晋级阈值，不是无限训练。
达到9000轮仍未提名会保存后停止；独立评测失败、仿真致命错误也停止，不自动重启。
GPU5核查时已被其他用户的评测PID736245占用，因此选择空闲GPU1；未停止其他任务。

- 正式run：`/home/xinyu/training_runs/20260923_104131_bulb_reward_v2_8192_9000_gpu1`。
- 任务：`BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0`；8192环境、seed42、
  OLD_REWARD=False，从零Stage0，原动作/观测/reset/奖励/成功及物理约束保持。
- 重用前次1024环境run的498份项目文件和45份RSL运行库文件，启动前逐项SHA256一致。
  6个共享依赖链接保留，不是完整隔离镜像；本次没有Python修改。
- 10:41:31启动：监督737978（start_ticks39963355）、课程启动器737979、训练737980
  （start_ticks39963361）。场景创建89.35s、物理启动310.89s；10:53:16核验已完成PPO 0–8轮，继续运行。
- 实际env/agent YAML确认8192环境、9000轮、seed42、固定Stage0、从零训练、新reward。
  model_0含78个有限张量，actor/critic归一化计数各262144=8192×32；669输入/28动作、30Hz保持。
  TensorBoard 127个标量标签有限，六个指定reward项、Stage0–3失败原因均存在；6006 HTTP已发现新标签。
  8470完整回合reward组件全有限，成功0/8470；另有8548个共同回合的success与原runtime日志完全一致。
  正例成功和后续阶段学习效果尚未验证，不将启动通过当作学会装配。
- 显存20872/32607MiB（约20.4/31.8GiB），初期三轮吞吐12488/10401/12348转移每秒，
  采样20.78/24.99/21.01s、PPO学习约0.22s；这些是初期数据，不是稳态性能基准。
  完整启动日志未检出所查PhysX/CUDA/溢出/Python致命错误；原USD视觉引用和TGS警告保留。
  GPU4旋出489880和GPU6插销242935的start_ticks保持，没有向其他任务发信号。
- 每次PPO更新262144转移；每阶段9000轮最多2359296000转移。
  环境数和迭代预算均增加，不与旧1024×2000按迭代数作公平对比。
- TensorBoard既有6006新增标签`Bulb_Assembly_RewardV2_8192_9000_GPU1_20260923/stage0`。
- 首次8192/2000尝试`20260923_103916_bulb_reward_v2_8192_gpu1`于10:40:24主动停止，
  当时仍在场景初始化、没有PPO更新；日志保留，未覆盖旧检查点。

证据：[启动配置](logs/diagnostics/bulb_reward_8192_20260923/launch_spec.json)、
[快照核验](logs/diagnostics/bulb_reward_8192_20260923/snapshot_verified.json)、
[进程身份](logs/diagnostics/bulb_reward_8192_20260923/formal_started.json)、
[正式核验](logs/diagnostics/bulb_reward_8192_20260923/formal_verification.json)、
[交接状态](logs/diagnostics/bulb_reward_8192_20260923/handoff_health.json)。
GitNexus对实验索引报告HIGH（README直接引用及后续扩散关系），STATUS/RUNBOOK未入索引为UNKNOWN；
已补充文档引用核查，仅追加记录。曾分析无限预算改动，但用户接受9000后没有实施运行代码修改。

## ASM-REWARD-V2-STATUS-20260923-02

用户询问为何仅1024环境及当前训练效果。北京时间10:28读取当前服务器状态和日志，
没有启动新训练/容量测试、改reward/配置或恢复已结束作业。

- run为`/home/xinyu/training_runs/20260923_010600_bulb_reward_v2_gpu5`，仍只有Stage0目录。
  07:56:06完成2000轮，65536000物理转移，442422结束回合，阶段成功0/442422。
  model_1999保留，bank为空、promotion_nominated=false；exit code 2为预算结束未提名，非仿真崩溃。
  监督724505、启动器724506、训练724507均不存在，GPU5读数2MiB/0%利用率。
- 一次遍历全部442422行reward诊断，成功0、累计分项无非有限值；完整console未匹配所查致命模式。
  全部已结束回合步数65462339，小于总转移数，因为末尾仍有未结束回合。
  全程7735个回合曾有双侧各>1N，最长连续双侧接触37步；最长严格稳定抓持28步/0.933s，未到31步门槛。
- 最近10000回合：201个曾有严格双侧接触；严格稳定抓持最长11步/0.367s，p95为0步。
  99.9759%的采样步未满足双侧各>1N。no_contact表示这一联合力阈值失败，不能据此说所有手指都无接触。
  原runtime独立日志最近10000回合成功0、宽口径grasp事件203、stable_grasp事件0，与未成功结论一致。
  保持中断的no_contact/slip/pose_error共现计数分别228/223/214，不把多标签共现当作互斥根因。
- 最近10000回合平均reach_progress=0.270124，contact_stability=0.000350，
  grasp_hold=0.00002857，stage_completion=0；奖励主要推动接近，尚未学到稳定抓持。
  新方案成功率仍0，不能凭数值检查通过、PPO运行或较小shaping回报宣称学习改善。
- 训练显存峰值7790/32607MiB，GPU利用率采样中位81%；最近100轮吞吐中位2656转移/s，
  采样12.2265s、PPO学习0.1135s。1024是沿用旧课程的保守环境数，不是已测显存极限。
  4096可作下一次容量/吞吐测试起点，6144仍需实测；低显存占用不保证吞吐线性增长。
  增加环境数会同时改变每次PPO更新的样本量，后续按总转移数记录预算，不能只按相同迭代数宣称公平。

完整统计：[training_summary.json](logs/diagnostics/bulb_reward_status_20260923/training_summary.json)，
独立runtime核对：[original_runtime_last10000.json](logs/diagnostics/bulb_reward_status_20260923/original_runtime_last10000.json)。
本次只追加分析证据和状态说明，没有实施下一轮诊断/扩容。

## ASM-REWARD-V2-TRAIN-20260923-01

用户授权同步xinyu21并启动新装配训练，可停止旧装配。仅8个奖励相关Python和2份专题文档同步，
本地/远端相同HEAD上的已有工作区修改保留，目标文件覆盖前备份、后验SHA256一致。
新run `/home/xinyu/training_runs/20260923_010600_bulb_reward_v2_gpu5`，冻结498份项目/资产/测试、
45份RSL源码，6个共享依赖链接单独记录。

旧GPU5装配监督360228、训练360229身份核对后SIGINT，01:05:53正常退出于9328轮；
`model_9300.pt`停止前后哈希不变。GPU4旋出489880、GPU6插销242935的启动时间保持。
新任务为Assembly-Curriculum-v0，GPU5/1024环境/seed42/OLD_REWARD=False，从零Stage0，
每阶段最多2000轮，原窗口与401/402/403各100回合冻结评测通过才晋级。
01:11:13启动监督724505、启动器724506、训练724507，未加载pilot或旧策略。

远端CPU161通过/2跳过；首次因快照漏两份测试而没有执行，补齐后重跑，失败日志保留。
32env pilot六轮/6144转移/32回合，成功0/32；model_5有78个有限张量，25个模型张量相对model_0变化。
01:14:14正式核验0–7轮、model_0、归一化计数32768、669输入/28动作与实际配置；
127个TensorBoard标量全有限，六个奖励标签和四阶段失败原因有值，OLD_REWARD为0。
正式已读1094回合，成功0/1094，尚无晋级或完整任务学习收益结论。
未检出PhysX/CUDA/溢出/Python致命错误，保留已有USD视觉警告；显存约7660MiB。
训练继续，失败/阶段预算用尽会停止，不自动重启。

详见[部署记录](docs/rotate_bulb_reward_gpu5_2026-09-23.md)、
[正式核验](logs/diagnostics/bulb_reward_deploy_20260923/formal_verification.json)、
[确切参数](logs/diagnostics/bulb_reward_deploy_20260923/launch_spec.json)。
GitNexus本地接线分析为CRITICAL且存在异常边；远端缺Node、索引绑旧路径，未声称远端图检查有效。
同步使用已审计确切文件及相同基线哈希，未新增奖励实现或修改物理/动作/观测/reset。

## ASM-REWARD-V2-20260922-03

- 授权：按已确认reward方案实施，1000环境有限短测后停止；不做PPO/长程学习。
- 版本：main/4707074194091cd723ebd18dea7824c184e99c6d加既有工作区改动及本次reward修改。
  本次仅4个既有Python文件改动（3个生产接线、1个隔离测试），新增4个Python；
  基线295份Python中291份逐字节不变，无删除。会话专属patch与SHA256见
  [范围核验](logs/diagnostics/bulb_reward_v2_20260922/scope_verification.json)。
- 功能：OLD_REWARD布尔开关、27项有限预算reward/成本、原两项动作成本、各项TensorBoard、
  Stage0–3失败原因与最长保持/中断共现。成功函数、观察/动作、机器人和reset采样保留。
- CPU：161 passed / 2 skipped（standalone环境未载入pxr.Usd）。预算、瞬时/断续接触、重复目标、
  入口停驻、螺纹方向、频率、部分reset、原最终成功1秒及失败中断均有测试。
- 原生：本机RTX5070 Laptop 8GB，IsaacSim5.1、brainco环境、seed42、1000环境、30Hz；
  Stage0×180步=180000转移、1000完整回合，reward/分项/观测有限、success标志一致、
  TensorBoard六必需标签与四阶段失败标签回读通过。探针动作下成功0/1000。
  Full×620步=620000转移、1000完整回合，上述检查同样通过，成功0/1000；两进程均正常退出。
  最终复查恢复原物理契约函数，把reward元信息保留在独立运行文件，避免新旧开关阻断同库消融；
  此元信息调整晚于Stage0短测，未改已测奖励/成功/reset路径，新增CPU契约一致性检查通过。
- 拒绝/资源差异：首轮默认预分配池出现PhysX 524288000字节分配失败，虽张量有限仍判无效。
  重测只把检查脚本aggregate-pair预分配容量2^25限制到2^22，并在每步扫描原生错误；
  生产求解器/模型/传感器/碰撞未变。已有USD视觉警告保留，未伪造成功状态库。
- 边界：旧模式保留测试通过，未跑旧模式学习；Stage1–3依靠张量/原runtime测试，未用假库启动原生rollout。
  固定动作短测与合成成功轨迹不证明学会任务，学习收益及新旧reward公平消融待后续授权实验。
- 详见[实施与短测报告](docs/rotate_bulb_reward_implementation.md)，[原始证据](logs/diagnostics/bulb_reward_v2_20260922/)。
`.autonomy/`、`logs/` 链接是被 Git 忽略的本地证据，跨检出可能缺失；缺失时不能只凭本表复现实验。

## ASM-CURRICULUM-VIDEO-20260922-01

用户指定 `Bulb_Assembly_Curriculum_GPU7_20260921/stage0` 最新检查点，要求服务器play录像并拉回本地。
TensorBoard链接解析到 `20260921_113720_bulb_curriculum_gpu7/curriculum`，最新文件为
`/home/xinyu/training_runs/20260921_113720_bulb_curriculum_gpu7/curriculum/stage0/model_1999.pt`，SHA256：
`a658daba066e1a4689ff70bfad2ad9bdb84d81c2dfbf3a477e86923db682d988`。
该训练03:01已用完Stage0预算退出（exit 2），没有晋级；此次未续训。

- 服务器独立输出：`/home/xinyu/recordings/bulb_curriculum_stage0_model1999_20260922_092007`。
  489项源码/资产和45项RSL文件逐项哈希匹配训练快照，复制后运行原 `play.py`。
  meshes/tacmap等原共享依赖沿用原链接，未声明所有依赖完全冻结。
- GPU2共享、1环境、seed42、Stage0固定灯泡、evaluation=true，显式恢复检查点课程状态，
  保持原物理求解器/动作/观测/奖励及5秒回合。只降低单环境PhysX预分配容量并增加离屏相机，
  全部命令、容量覆盖及非敏感环境参数见manifest；课程兼容性校验通过。
- 17:20:07–17:34:23（Asia/Shanghai）运行1200控制步，正常退出0；输出40秒、30FPS、1280×960。
  服务器和本地OpenCV均完整解码1200帧，本地FFmpeg解码退出0且无错误；抽帧确认有策略动作。
  视频SHA256为 `43762e53146af8497bfc718d5de6277849188f36d9fa1e9638745d76f6da7c0a`，两端一致。
- `curriculum_episodes.jsonl`保留8个完整回合，阶段成功0/8，总1200步；这是单seed确定性录像，
  不代表多seed独立成功率，更不代表自由灯泡或完整装配任务表现。
- 原始及复制检查点哈希均未变，动作/触觉契约与训练逐项相同；GPU4/5/6原训练PID与start_ticks未变。
  未修改运行源码、重启训练或改动原判据；GitNexus文档影响分析为STATUS UNKNOWN、EXPERIMENTS LOW，
  补充检查README/AGENTS/RUNBOOK/PROJECT_MAP文档引用，只新增记录，不据图推断运行代码影响。

本地[视频及证据目录](recordings/xinyu21_bulb_curriculum_stage0_model1999_20260922_092007/README.md)含
manifest、服务器核验、本地解码/哈希核验、逐回合JSONL、源码哈希清单和原始日志。
桌面副本：`/home/liuxinyu/桌面/灯泡装配课程_Stage0_model_1999_20260922.mp4`。

## MOLD-BREADBOARD-MOUNT-20260922-05

用户要求下模固定至孔距/边距均25mm的光学面包板，并确认按M6内六角圆柱头、打印定位柱与配合孔直接开合。
本次基于root_open_256原生CAD，在本机CPU生成新文件，不覆盖用户指定的旧STL。
下模四孔的打印XY坐标为(18,235)、(93,235)、(143,10)、(193,10)mm，共同落在25mm孔网上；
孔径6.8、藏头孔径12、承靠面距底24mm。两根定位柱身Ø8、露出6mm，沿实际贴合面法向；
上模圆孔Ø8.6、长圆孔8.6×10.6、深7.2mm，长圆方向沿两柱连线。
上下模占地均220×249mm，高分别42.921/41.246mm。螺丝外包络参考单独保存，不纳入打印件。

原上模206个面中205个保持一致，仅贴合面开孔；原下模308个面中306个保持一致，仅贴合面和背面改变。
逐面面积、质心和包围盒签名最大分量差3.98e-13，型腔及此前缩短的根部通道保持原样。
STEP重读均有效单实体、装配为两个有效实体。闭合与开模0.5/2/5/6.2/10mm六个位姿无材料体积交叠；
四个螺丝外包络无干涉，各孔48个材料/空隙探针通过，手根通道343点为空。
打印STL回读为单连通、闭合、绕序一致、零边界/非流形边，分别38078/49614个三角面。
原下模网格继承两个小缺口，UV三角化补回三面；459点到有边界CAD面的最大距离0.002538mm。
严格UV内外分类及探索性0.002mm检查未通过，失败结果保留；最终按明确的0.02mm绝对离散参数核对局部采样通过，
不声称上述更严格检查通过，也不把局部采样当作全局误差证明。CAD/网格体积差约0.0084%/0.0117%。

台面实际长宽、有效螺纹深度未知。示例将下模左下角置于台面(7,15)mm，需至少250×275mm完整台面及相应孔网。
无垫片时M6×30会伸出下模底面6mm；螺丝长度仍应按实际有效牙深选择。未实打、未测定位配合松紧或承载。
源STEP与上一版文件哈希未变，实体02/04/05和URDF未改，未运行训练或远端作业。
新增独立生成脚本及STATUS未被GitNexus索引，UNKNOWN补以引用核查；未发现生成脚本运行调用方，STATUS仅为文档入口。
EXPERIMENTS影响为LOW、直接依赖README、无已解析运行流程；既有运行代码不改。

交付与复现：[README](1_stl/breadboard_mount_256/README.md)、[manifest](1_stl/breadboard_mount_256/manifest.json)、
[validation](1_stl/breadboard_mount_256/validation.json)、[孔位图](1_stl/breadboard_mount_256/mounting_layout.png)、
[网格尝试记录](1_stl/breadboard_mount_256/diagnostics/recovery_attempts.json)。

## MOLD-ROOT-OPEN-20260922-04

用户要求上下模在手根底部再截去一块，将完整根端截面露出以便插线。
基于上一版trimmed_256的CAD和已验证闭合STL，保留原坐标Y≥-19mm，较旧底边Y=-25再去6mm。
原根端面Y=-19.2，越过0.2mm以完整打开；出口X为[-26.2,26.2]、Z为[-20.2,20.2]，即52.4×40.4mm。
上下模均单实体，占地220×249mm；STL背面贴底、型腔朝上，CAD及合模STEP保留原装配坐标。

型腔逐面签名验证：上模192面、下模294面保持一致，各有4个根端面按授权移除或缩短；
最大面积/质心/边界分量差约5.46e-12。根端及通道343个内部点均无模具材料，四侧材料点存在。
两CAD的新切面材料总面积14339.04mm²，等于220×74.8减去52.4×40.4，证明根端完整开口。
在既有闭合网格上做平面裁切，只补齐剩余模具材料的截面；两个STL回读均单连通、闭合、绕序一致。
网格切面面积与CAD差小于0.0001mm²；上模原单精度顶点在出口边缘产生0.000099mm²数值覆盖，
向内缩0.0001mm后的出口没有网格覆盖。STEP/BREP重读有效，合模STEP包含两个有效实体。

证据：[文件说明](1_stl/root_open_256/README.md)、[装配与开口预览](1_stl/root_open_256/preview.png)、
[验证](1_stl/root_open_256/validation.json)、[清单](1_stl/root_open_256/manifest.json)、[ZIP](1_stl/root_open_256.zip)。
源STEP、原五个STL、上一版打印STL哈希均未变；实体02/04/05和URDF未修改。
仅本机CPU几何/网格处理与软件渲染，未实际插线、打印、切片软件验证或仿真。

## MOLD-PRINT-TRIM-20260922-03

用户要求只处理实体01/03，裁掉横竖多余边框、保留型腔，允许去除定位凹凸件；
补充确认打印范围256×256mm、上下模各一件、平放打印。本次本机CPU CAD布尔运算、软件渲染与网格检查。
源哈希仍为`ca100ee804fe35ccc022bc51b41307f2521a5d2d29a4e3e158a6726372998059`。

两半统一切至X≥-60、Y≥-25，左侧去10mm、底侧去15mm，外形由230×270缩为220×255mm。
上模三定位孔填回原倾斜贴合面，下模三定位柱切回该面；两者仍为有效单实体。
通过与手形内环连接的面图识别型腔，上模196面、下模298面的曲面类型/面积/质心/三维边界均与源匹配，
最大签名分量差约5.46e-12；新左/底边距型腔的轴向最小余量为4.984/5.8mm。
STL上模翻转180°后平移贴底，下模直接平移贴底；STEP/BREP保留原装配坐标。

下模直接0.005mm离散仍不闭合；以该网格直接补UV环会产生非流形边，已拒绝该结果。
最终下模以0.02mm/0.08rad重新离散，双精度保存，按原BSpline face101补回两处缺口共3个三角面，
459个补片采样点均落在原面内部/边界，最大采样距离0.001455mm；未改CAD型腔，非全局Hausdorff界。
最终两STL回读均单个连通体、闭合、绕序一致，边界边/非流形边为零；CAD/STEP回读检查见validation。
实体02/04/05、源STEP、原STL和既有URDF均未修改，旧URDF网格未自动替换。

交付[说明及文件](1_stl/trimmed_256/README.md)、[ZIP](1_stl/trimmed_256.zip)、
[型腔保留](1_stl/trimmed_256/cavity_preservation.json)、[网格恢复](1_stl/trimmed_256/mesh_recovery.json)、
[验证](1_stl/trimmed_256/validation.json)。诊断尝试留在该目录的diagnostics，未放入交付ZIP。
尺寸不含Brim/裙边，长边在256mm平台居中时两端各余0.5mm；没有实际切片、打印或材料强度试验。

## MOLD-ENTITY-ANALYSIS-20260922-02

用户询问五实体为何存在、是否来自手模型。本次本机CPU读取原STEP、测量BRep距离并软件渲染。
源SHA256：`ca100ee804fe35ccc022bc51b41307f2521a5d2d29a4e3e158a6726372998059`，与STL拆分来源一致。
源文件包含一个PRODUCT、五个MANIFOLD_SOLID_BREP、零个NEXT_ASSEMBLY_USAGE_OCCURRENCE。
STL编号01/03为上下模；02/04/05在拇指近根部，与上/下模最小距离分别为
2.915/3.786、4.697/4.471、4.284/4.252mm，均不接触。
额外15个曲面壳列入`.BLACK_HOLE`图层，#101091设该层为INVISIBILITY；
其中第7/8/9组为同位置指尖状曲面，各96面，同参数离散网格舍入1e-8mm后的哈希相同。
已确认手部模具的几何关系；“布尔残体/孤立型芯”仍是假设，STEP无原特征树，不能判定小实体材料归属或删除。
没有修改STEP、STL、URDF或运行代码，没有启动仿真或训练。
证据与完整测量：[分析](1_stl/analysis/README.md)、[数据](1_stl/analysis/findings.json)、
[原位图](1_stl/analysis/location_and_reference.png)、[参考曲面](1_stl/analysis/reference_shells.json)。

## MOLD-SPLIT-20260922-01

用户要求将根目录 `1.stp` 的上下模分开，沿贴合面法向建立两部分 URDF，并尽量保留信息。
本次仅本机CPU CAD处理、URDF解析和软件WebGL预览；未运行Isaac、训练、远程同步或真机。

交付 [assets/mold_split](assets/mold_split/README.md)。主体为OCCT实体0/2，主要贴合面为上模face24/下模face2，
法向 `(-0.0192123072704608, -0.00527192991744921, 0.999801527306441)`，相对原Z倾斜1.1415506754°。
两个link保留原闭合位置，单prismatic关节；150mm行程、100N effort、0.05m/s速度均为可修改的建模参数，非CAD事实。
3个孤立微小实体及15个额外曲面保留在参考STEP；原文件字节副本、图层/调色板语句及哈希完整保留。

已实际通过：urdfdom解析（根lower_mold、子upper_mold）、相对网格路径、单位与闭合零位坐标抵消，
浏览器软件WebGL预览；原BRep在0.1/1/5/20/100/150mm开度的主体最小距离均为正，闭合位距离为0。
这是离散检查，不是全行程连续无碰撞或成功率评测。

保留的未通过项：下模0.02mm网格44979三角面、7条开边，上模38224三角面闭合；
细化0.005/0.001mm、Delabella和最小单元调整均未消除下模开边，未自动补洞。
STEP重导出体积约差2.50/1.84mm³，未通过0.01mm³严格阈值，因此额外直接保存原始导入实体BREP。
几何验证与物理可用性分开；`collision_ready=false`，不填写未知质量/惯性，未对导入器默认凸包或惯性做认可。

证据：[manifest](assets/mold_split/generated/manifest.json)、[validation](assets/mold_split/generated/validation.json)、
[网格尝试](assets/mold_split/diagnostics/meshing_attempts.json)、[预览](assets/mold_split/generated/preview.png)。
下一步若需物理仿真，应先定位网格缺口并针对型腔配置碰撞，再补实测质量、材料和驱动参数。

## ASM-CURRICULUM-GPU7-20260921-04

用户授权停止GPU7旧训练并运行新同步的课程任务。
旧 `20260921_074610_rotate_bulb_3turns_resume1400_gpu7` 正常停止于1871轮，保存的model_1800哈希不变。
新 `20260921_113720_bulb_curriculum_gpu7` 于19:46:50启动，1024环境/seed42/从零Stage0，
任务 `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0`；监督529620、启动器529621、训练529622。
每阶段最多2000轮，窗口和3冻结评测seed通过才晋级，最终full仍为原20秒。

服务器2env/160步原生检查169/169通过；32env短PPO完成6轮/6144 transitions/33回合，成功0/33。
19:49:32正式训练已更新0–3轮，model_0保存，78个张量有限；669输入/28动作、归一化计数32768核验通过。
534项源码/资产/RSL快照哈希一致；共享目录依赖单独记录，未称作完全冻结环境。
GPU4/5/6原训练的PID和启动时间一致；6006课程标签实读53项标量。
失败包括快照缺tacmap链接、共享临时目录权限、无效旧显示覆盖项，日志保留，修复均限启动环境。
另给课程启动器补充训练/评测一致的Kit物理GPU传参，15项本地相关测试通过。

详见[启动记录](docs/rotate_bulb_curriculum_gpu7_2026-09-21.md)。原始证据位于新run根目录的
`stop_previous_verified.json`、`native_probe_v3/report.json`、`pilot_verification.json`、
`formal_verification.json`、`launch_spec.json` 和清单文件；本地备份
`logs/diagnostics/bulb_curriculum_gpu7_20260921/`。
只确认Stage0可训练/保存，尚未晋级、建立真实前缀库或验证课程优于完整任务；未完成三训练seed对照。

## ASM-CURRICULUM-DESIGN-20260921-02

类型：当前源码分析、历史数据复核、实施/实验设计；**不是新的训练或仿真实验**。
对象为 `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0`。
本地 `main` / `4707074194091cd723ebd18dea7824c184e99c6d` 加本次开始时已有未提交改动。

- 按用户要求只做Phase1–4分析设计，Phase5仅列实施文件、风险和四个阶段commit边界，未执行实现提交。
- 当前14项reward源码审计确认：grasp有1N阶跃；lift/transport虽有连续g，但g取拇指与另一指的较小力，
  任一侧为0就为0；hover有lift历史/满抓持/严格几何锁存；insertion需hover锁存及后续真实接触/螺旋关系。
  当前本地10cm和历史run4cm分开，未拿旧评测证明新目标优劣。
- 核对桌面/灯泡/灯座名义位置、原7臂关节与21零手、root/关节速度/drive target reset，以及28维v3动作。
  预抓IK/碰撞/闭合动态可行性未知，未执行物理探针。
- 重新读取 `logs/diagnostics/assembly_curriculum_20260921_0910/` 的6个逐回合JSON：
  确定性96回合grasp0，随机96回合grasp1；两者稳定grasp/lift/hover/alignment/engaged/insertion/final均0，
  与 `episode_summary.json`一致，192条原奖励累计核验标记均真。这是读取旧证据，不是重跑或重新计算物理奖励。
- 设计：固定Stage0＋free迁移（双侧>1s）→桌面Stage1→成功前缀状态库Stage2→真实hover起点Stage3；
  保留原669维及各观测语义，原28维动作/最终成功不变。联合reset包含控制target、速度及接触复建；
  不能在部分reset内部额外推进全局仿真，初始化控制需隔离PPO数据；不具备可靠collector时先用真实前缀继续路径。
- 实验计划为B/C各3个独立训练seeds42/43/44，同版本20秒完整评测；建议预算尚未授权执行。
  独立评测每训练seed用101/202/303各100回合，报告五项主指标及训练seed差异，额外记录状态库/reset全部成本。
- GitNexus执行query/context/impact及 `analyze --index-only`，本地索引刷新至9680节点/19351边/370流程。
  图报告流程截断；reset函数关联仍异常且CRITICAL、STATUS为CRITICAL，已告知后仅追加文档，
  配置/Guidance为LOW下界；新实施文档UNKNOWN补充引用搜索。未以该图认定共享代码可安全修改。

交付：[分析补充](docs/rotate_bulb_curriculum_analysis.md)、[实施设计](docs/rotate_bulb_curriculum_implementation.md)。
原分析第1–7节及历史实验记录保留；只新增本次读取说明、源码分析和方案。
本次未运行Isaac/训练/回放、未连接远端或操作旧PID、未改运行代码和原资产、未commit。
交付检查：`source/scripts/tests`下280个Python文件前后SHA256一致，无新增Python文件；
已有分析正文、STATUS历史和EXPERIMENTS原记录保留核验通过；两份课程文档的本地链接目标均存在，
`git diff --check`通过。没有以文档检查冒充CPU单元测试或原生物理验证。

## ASM-CURRICULUM-20260921-01

用户要求分析20秒全任务、逐episode阶段成功比例、奖励贡献与饥饿，以及Stage0–3 reset课程；明确不修改代码。
对象为RotateBulbCustom目录中的桌面`Right-Bulb-Assembly-Custom-v0`，不是三圈旋出任务。

- 正式run：`/home/xinyu/training_runs/20260920_124630_bulb_assembly_zero_hand_gpu5`；
  历史TensorBoard快照迭代0–3499，最近窗口3400–3499。原日志只有reset批次/迭代聚合，
  不能重建历史逐episode分母；奖励日志×20明确标为回合尺度代理。
- 独立冻结`model_3400.pt`，SHA256 `2040b83ba3cec8f28992c1d11b2ed3e034d2a063b5ee7d95b5bb95ee827aaa00`；
  使用原run的206文件源码快照，4cm hover、20秒/600步、纯碰撞、零手/原臂、原XY/yaw随机化。
  GPU2，seeds101/202/303，每seed32个首个完整episode，两种模式各96；无PPO学习。
  保留早失败；外部观察器在自动reset前记录末帧，原reset原样调用，不改项目源码。
- 确定性grasp0/96、超时96/96，平均回报24.084701；随机grasp1/96、超时92/96、越界3/96、
  失抓1/96，平均回报21.571532。两种模式稳定抓持/抬起、hover、alignment、insertion和最终成功均0/96。
  随机单指接触奖励34/96，双侧/位置奖励仅1/96，下降旋入0/96；历史稳定抬起和旋入奖励0/3500日志点。
  最近100点约99.924461%正奖励来自接近/朝向。支持早期抓取瓶颈与后期reward starvation，
  不证明20秒物理上不能完成，也不证明本地10cm新版本更好。
- 192/192回合累计奖励与RewardManager一致；策略/归一化未变；3个进程退出0。
  配置差异限环境数、seed、路径和预分配容量；缩小并行规模可能产生PhysX轨迹差异，非正式验收。
  初次观察器切换模式遇到推理张量reset上下文错误；保留已完整完成的seed101确定性32回合及错误日志，
  仅修复外部观察器上下文并重跑随机模式。未重启原训练，未丢弃失败episode。
- 17:35核验源码/所列直接资产/checkpoint哈希一致；正式PID360229及启动时间不变，轮数3499→3577。
  本地280份source/scripts/tests Python文件哈希不变。GitNexus文档impact：STATUS LOW、EXPERIMENTS MEDIUM；
  新报告尚未入图，按UNKNOWN补充引用核查，仅追加文档与独立证据，未重建索引或提交。

报告：[rotate_bulb_curriculum_analysis.md](docs/rotate_bulb_curriculum_analysis.md)。
本地证据：[assembly_curriculum_20260921_0910](logs/diagnostics/assembly_curriculum_20260921_0910/)，
含原始逐episode数据、训练标量、聚合JSON、外部观察器文本、运行参数、配置差异、错误日志和`final_audit.json`。
远端同名分析目录：`/home/xinyu/analysis_results/assembly_curriculum_20260921_0910/`。
课程仅为方案，Stage2/3需先取得真实成功前缀状态库；本次未实施或新增训练。

## BULB-RESUME-GPU7-20260921-01

用户授权占用xinyu21 GPU7，并从 `Rotate-Bulb_3Turns_PhysX24-4_GPU4_20260921` 最新checkpoint续训。
该标签实际指向 `/home/xinyu/training_runs/20260920_183647_rotate_bulb_physx24_4_pressure30_6144_gpu4/train`；
按数字遍历确认最新为 `model_1400.pt`，保存迭代1400，SHA256
`a2372b1cccd49ce16197ac72dd911c7df93081060f215eeb92e7694f2b8ff2c4`。

- 新run：`/home/xinyu/training_runs/20260921_074610_rotate_bulb_3turns_resume1400_gpu7`。
  临时显存占用在启动前释放；第二次启动为15:50:03（Asia/Shanghai），监督506696、训练506697。
  任务仍为 `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0`，GPU7/6144env/seed42。
  继承30Hz、PhysX24/4、三圈guided、673维观测、28维动作、关闭编码器及所有原物理/奖励/成功设置。
- 复制原冻结workspace，446项源清单全部匹配；另冻结17项公共USD资产、52项RSL-RL Python文件。
  环境YAML差异仅路径；agent只改变续训选择、run名称、迭代预算和恢复的学习率。
  原Runner会加载优化器但不自动将算法的学习率标量改为已保存值，因此显式设为
  checkpoint的 `9.203429591335062e-05`；未改Runner/业务源码。
  Runner从保存标号1400开始追加13600次更新，计划到14999；新目录中的model_1400是恢复后首轮保存，
  与只读父目录同名模型区分，未覆盖源文件。
- 15:58核验已完成1400–1403轮，首轮checkpoint优化器步数28020→28040；actor/critic归一化计数
  275447808→275644416（+196608=6144×32）；权重更新，全部77个张量有限，父模型哈希未变。
  动作/触觉契约逐字段完全一致；再次核查463个源/资产文件哈希通过，当前日志无所查致命错误。
- 首次启动因冻结清单未包含公共地面USD，在PPO前FileNotFoundError退出；既有监督器记录exit0，
  本次依据traceback及无PPO判为失败，完整保留 `attempt1_missing_ground/`。
  补齐公共USD目录后重新启动；原有视觉prim警告仍存在于新旧运行，未因此修改碰撞或物理。
- 新TensorBoard标签 `Rotate-Bulb_3Turns_PhysX24-4_Resume1400_GPU7_20260921` 已通过6006 API读取46项标量。
  核验时GPU7显存约25–27GiB；GPU4 Randomized489880、GPU5装配360229、GPU6 D-Peg242935继续存在，
  未对其停机/重启。本次没有独立成功率评测，不将PPO更新视为任务成功。

证据：[启动清单](benchmarks/rotate_bulb_resume_gpu7_20260921/launch_spec.json)、
[来源核查](benchmarks/rotate_bulb_resume_gpu7_20260921/preflight.json)、
[配置差异](benchmarks/rotate_bulb_resume_gpu7_20260921/config_comparison.json)、
[完整验证](benchmarks/rotate_bulb_resume_gpu7_20260921/resume_verification.json)。
本地只追加文档/证据；原有未提交业务改动保留。文档GitNexus impact：STATUS LOW（4个直接文档引用），
EXPERIMENTS MEDIUM（7个直接文档引用），无受影响执行流程；索引commit匹配4707074但工作区内容有漂移，
补充核查文档引用后只追加本记录，未重建索引或提交。

## BULB-RANDOMIZED-20260921-01

用户要求落实新任务、同步xinyu21既有仓库、停止旧旋出并启动新训练。
实现ID `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0`；本地main/4707074加已有工作区与本次改动，
不以旧优化分支提交代表本次源码，确切版本由同步清单和新run的382项冻结清单标识。

- 初态固定臂、手21关节归零；支架/灯泡一起X/Y各±10cm，目标相对支架X/Y各±15cm、Z+18–30cm。
  IK和离散路径碰撞检查保留432/441个支架点、每点8目标，共3456对；运行时均匀抽取这些对。
  完整支架在该平移范围内最小桌边余量14.2cm；9个剔除点、目标拒绝计数和无效首轮诊断均留档。
  序列化后全3456目标FK复核：最大位置残差0.398mm、朝向残差0.452°；不证明动态完整三圈成功。
- 接近/旋出/搬运臂幅度倍率1/5/1、抓持时锁存姿态参考，有界抓持塑形与14维阶段状态；总输入687。
  原Custom/Assembly实现未改；三圈guided、成功/失败、必要碰撞、20秒回合保留。
- 新行为7/7；原任务定向51通过/2跳过，新行为+bounded policy23通过/2跳过；
  扩大灯泡回归424通过/3失败/2跳过，3项均在未修改的既有实现/测试，详见报告，不称全套通过。
  本地与服务器2环境原生各121/121、40物理步；服务器32环境3轮PPO正常结束，模型权重更新且有限。
- 定向同步10个文件到 `/home/xinyu/mulyi-model-tac`，替换前备份且逐项哈希一致。
  旧run `20260920_183647_rotate_bulb_physx24_4_pressure30_6144_gpu4` 在13:02:42收到SIGINT，
  1417轮正常停止、退出0；`model_1400.pt`保留，SHA256前后均为
  `a2372b1cccd49ce16197ac72dd911c7df93081060f215eeb92e7694f2b8ff2c4`。
- 新run `/home/xinyu/training_runs/20260921_050053_rotate_bulb_randomized_6144_gpu4`：
  13:05:49启动，GPU4/6144环境/15000轮/seed42/from scratch；监督489879、训练489880。
  13:14验收迭代0–5和model_0，77张量有限；687维、固定臂/零手、30Hz、TGS24/4、全部冻结哈希通过。
  显存25255/32607MiB，未见致命日志；其他装配/D-Peg进程启动时间不变。
  TensorBoard6006标签 `Rotate-Bulb_Randomized_ZeroHand_GPU4_20260921` 已见57条标量。
- [执行报告](docs/rotate_bulb_randomized_training_2026-09-21.md)、
  [本地证据](logs/diagnostics/bulb_randomized_implementation_20260921/)、
  [正式验收](logs/diagnostics/bulb_randomized_implementation_20260921/remote/formal_verification.json)。
  服务器run保留launch_spec/source_manifest/sync_backup/native_probe/pilot及全部训练输出。
  本次只验证实现、启动、采样与PPO链路，新分布上的独立完整任务成功率尚未评测。

## BULB-RANDOMIZATION-DESIGN-20260921-01

用户要求支架与灯泡位置随机化、扩大目标范围，随后明确保持机械臂初态、21 手关节归零，
评估放宽臂惩罚与新建独立任务。已完成[设计](docs/rotate_bulb_randomization_design_2026-09-21.md)，未改运行代码。

- 证据：[CPU 报告](logs/diagnostics/bulb_randomization_design_20260921/kinematics.json)、
  [确切脚本](logs/diagnostics/bulb_randomization_design_20260921/probe_workspace.py)、同目录输入哈希/USD 警告。
  `main` / `4707074194091cd723ebd18dea7824c184e99c6d` 加已有工作区改动；未提交。
- 支架源包围盒在世界 X/Y 各 ±0.10 m 的四个极值位置转入桌面坐标，最小边距 0.142121647 m；
  台面有 yaw，已按实际旋转计算，不是把世界 X/Y 当作桌面局部 X/Y。
- USD 关节树 FK 与 Dexonomy 手掌参考误差 1.427e-7 m；有界最小二乘保留 5° 关节余量，
  合格门槛为 <1 mm/<1°。9 个支架网格点抓持与抬升 18 mm 分别 8/9 合格；
  9×27 个目标（相对支架 X/Y ±0.15 m、Z 0.18/0.24/0.30 m）207/243 合格。
- 初筛沿用原手掌朝向/抓持关系，仅用于找候选抓持和目标运动学见证；不是最终零手 reset 验收。
  局部求解失败不等于不可达。没有检查自碰撞、桌面/支架碰撞、连续路径、动态接触或完整任务成功率。
- 使用本机 brainco Python 与已有 Isaac USD CPU 库；未启动 Isaac/GPU/远端/训练。
  初次导入依赖路径不完整，随后补充现有 USD bin 与环境 lib 路径；未安装任何依赖。
  确切环境前缀：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`，
  `PYTHONPATH=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311`，
  `LD_LIBRARY_PATH=<同一 USD 库目录>/bin:/home/liuxinyu/miniconda3/envs/brainco/lib`，
  解释器 `/home/liuxinyu/miniconda3/envs/brainco/bin/python`；脚本输出 `/tmp/bulb_randomization_kinematics_20260921.json` 后归档。
- 建议：独立任务，臂幅度倍率接近/旋出/搬运先试 1/5/1；接近时不罚离开 reset 臂姿态，
  旋出参考抓持时姿态；筛选目标与路径，保持三圈和最终成功判据。所有参数与新任务均尚未实施。

## BULB-VIDEO-20260921-01

用户要求排查SSH转发6006后看不到当前训练日志，并用最新checkpoint录制视频拉回本地。
2026-09-21 02:46:05 UTC选择run
`/home/xinyu/training_runs/20260920_183647_rotate_bulb_physx24_4_pressure30_6144_gpu4`
最高已保存检查点 `train/model_1100.pt`，SHA256
`13e124cb8e0a5c05d0d00ab02d0fc1f03fadab9f2486e81cb5fd5d6939bbdde4`。

- TensorBoard PID66933监听127.0.0.1:6006，logdir为历史run下的
  `/home/xinyu/training_runs/20260919_115128_nomodal8192/tensorboard/active_runs`；该目录没有本次run。
  新增 `Rotate-Bulb_3Turns_PhysX24-4_GPU4_20260921` 指向当前 `train/`，保留其他入口。
  远端和本机现有SSH隧道HTTP200，新run可见并返回1126轮成功率标量。
  `ssh -N`正常连接后不输出日志，需保持隧道并在浏览器打开 `http://localhost:6006`。
- 原源码归档恢复到独立workspace并核验446项哈希；复制checkpoint、params和契约。
  三圈guided、240Hz物理、30Hz控制/压力、pose_only、TGS24/4、673维观测、28维动作及原规则不变。
  GPU3、单环境、seed42、auto相机、1200控制步；仅缩小PhysX预分配容量，实际碰撞栈钳制到256MiB。
  首轮只恢复归档中的部分资产目录，初始化结束且未产出视频；退出码0不足以证明成功。
  失败日志保存在远端 `attempt1/`；改为引用原run完整资产根并核验110项资产哈希后重试成功。
- 回放退出码0，原检查点哈希未变，回放动作/触觉契约与训练逐项相等；未发现致命物理/CUDA错误。
  OpenCV与FFmpeg独立完整解码通过：1200帧、40秒、30FPS、1280×960；初始和3/6/10秒抽帧画面有效。
  MP4为1,864,257字节，SHA256 `36bf93b0a9078c8545f0ebf20164bded3b0381224ded5f7207aaa92416992158`，
  本地工程目录和桌面副本均与服务器一致。
- 本地[视频](recordings/xinyu21_rotate_bulb_latest_20260921_024605/rotate_bulb_model_1100.mp4)、
  [验证](recordings/xinyu21_rotate_bulb_latest_20260921_024605/validation.json)、
  [TensorBoard证据](recordings/xinyu21_rotate_bulb_latest_20260921_024605/tensorboard_verification.json)，
  同目录保留manifest、source_hashes、server_verification、play.log与抽帧。
  桌面 `/home/liuxinyu/桌面/旋转灯泡_model_1100_20260921.mp4`；
  远端 `/home/xinyu/recordings/rotate_bulb_latest_20260921_024605/`。
- 02:54:05 UTC原训练PID459464仍运行至1138轮，未重启训练或TensorBoard。
  本次是录像而非独立成功率评测，未将视觉抽帧或训练99.5%当作回放成功回合统计。
  文档影响检查绑定本工作区/index4707074：STATUS为LOW、4个直接文档引用，EXPERIMENTS为MEDIUM、6个直接引用；
  图的文档执行流程归属存在杂项，已补查实际引用。本次无业务函数修改或git提交。

## ASM-HOVER-20260921-01

用户要求分析任务合理性、将悬停改到入口上方10cm，并在缺少密集奖励时补充。
核查当前已有14项奖励，接近/抬起/搬运/下降旋入均有密集项；接触阈值和悬停前置条件会造成
部分阶段信号为零，因此“有密集项”不等于任意初态都能获得后期奖励。
保留既有奖励，只修改 `assembly_guidance.py` 的悬停默认值及自由下降距离换算。

- 目标净空4→10cm，新增 `descent_reward_length_m=0.04`；下降剩余比例乘该长度，
  入口以内仍使用实际18mm深度与原旋转/接触/螺旋关系。旧目标设回4cm时恢复原曲线。
  不改成功判据、物理、动作/观测、奖励权重或20秒回合；没有把中间悬停当作成功。
- 生产函数数值检查：悬停处满门控下降奖励率0.52154/s；仅改高度为0.02732/s，
  本次换算恢复0.52154/s；入口2.89051/s和后续相同旋入轨迹奖励保持。
  数据：[reward_profile.json](logs/diagnostics/assembly_hover_10cm_20260921/reward_profile.json)。
- 本机brainco：`python -m pytest -q tests/test_bulb_assembly_guidance.py tests/test_bulb_assembly_lift.py
  tests/test_bulb_assembly_state.py tests/test_bulb_assembly_reset.py tests/test_bulb_assembly_contact_rewards.py`，
  **62 passed、2 skipped**，跳过为缺pxr.Usd的真实几何检查。旧失败防护、完整10cm下降、
  30/60/120Hz时间积分、局部reset、原4cm曲线兼容均通过。
  [日志](logs/diagnostics/assembly_hover_10cm_20260921/pytest.log)与修改前文件副本保留。
- 编辑前本机未发现活动训练/回放；服务器装配PID360229仍运行，15:49约3273轮。
  本次未同步远端、未重启、未执行Isaac或新增训练；10cm目标的IK/动态碰撞净空和成功率待验证。
  此前18秒初态视频对应旧4cm，不能当作新目标可视化。
- GitNexus已刷新当前main/4707074工作区；类影响LOW，两个方法和测试文件为UNKNOWN，
  引用核查确认独立装配的两个RewardTerm及抬起测试依赖。图包含动态分派边界、杂项流程和
  全局流程提取截断，未以空调用集认定无影响；未修改共享抬起函数或其他任务，未提交。
  全工作区detect-changes（扩大limit重跑）汇总46文件/108符号/352流程、critical，
  涵盖大量原有改动且CLI展示有省略；本次运行模块/测试原已未跟踪，不能靠git diff报告覆盖。
  因此另保存本任务修改前后`task.patch`和文件哈希，以直接引用核查及62项回归验证本次范围，
  不把全工作区检查称为干净通过。`git diff --check`与修改Python文件AST解析通过。

## ASM-INITIAL-GOAL-20260921-01

用户要求查看当前装配任务手、臂初态和物体目标位置。使用当前run
`/home/xinyu/training_runs/20260920_124630_bulb_assembly_zero_hand_gpu5` 的206文件源码快照，
逐项核验快照SHA256后建立独立预览目录；GPU7、单环境、seed43，冻结reset状态，只移动相机。

- [视频](recordings/xinyu21_assembly_initial_goal_20260921_0742/initial_pose_and_goal.mp4)：
  18秒，1280×960，24FPS，432帧；中文标注全景/手部近景/俯视。
  青色安装参考点world `(-0.720000, 0.040000, 0.047031)m`，
  灯泡初始参考点 `(-0.711490, -0.165857, 0.022331)m`，距离20.75cm。
  绿色悬停参考点 `(-0.720000, 0.040000, 0.105031)m` 对应螺纹尖端离入口4cm，
  青色线框仅为目标灯泡示意，非位置容差或物理物体。
- [核验](recordings/xinyu21_assembly_initial_goal_20260921_0742/verification.json)25项全通过：
  28关节与训练初态一致、21手关节为零；动作/奖励/终止/reset/引导/物理模式/控制周期与保存配置一致；
  录制前后关节与灯泡状态逐元素相同；完整视频解码及432帧计数通过。
  seed43展示单环境随机样本，不声称等于6144环境训练env0的采样。
- 仅预览使用小容量PhysX缓冲、单环境渲染和无碰撞标记，并隐藏观察点云调试标记。
  没有加载checkpoint、执行策略或推进物理；未改共享业务函数或训练配置。
  初稿已完成，随后准确化悬停文字并隐藏红色点云标记后重录；远端保留draft证据。
- SHA256 `8f8d4bc385c1e9f4e2eca684ef2f72182938b5c8a819f2d7f1793da0e1268af3`，本地下载一致；
  桌面 `/home/liuxinyu/桌面/灯泡装配_手臂初态与安装目标_20260921.mp4`。
  远端 `/home/xinyu/recordings/assembly_initial_goal_20260921_0742/` 保留隔离workspace与原日志。
  本地同名带xinyu21前缀目录保留job、录制/核验脚本、配置、scene、源码manifest与四张代表画面。
- 原装配PID360229和启动时间不变，核验时持续至3254轮。此次不构成任务成功或独立成功率证据。
  GitNexus绑定本仓库，索引存在滞后；预览接口save_camera_preview为LOW（play.main及一个流程），
  STATUS为LOW、EXPERIMENTS为MEDIUM（文档引用）；结合独立脚本与实际配置核对，无业务代码提交。

## BULB-INITIAL-GOAL-20260920-01

用户要求查看当前旋出任务手和臂的初始姿态及物体目标位置视频。
使用 `20260920_121222_rotate_bulb_far_arm40_6144_gpu4` 源码快照、启动覆盖与seed42，
GPU2单环境离屏预览，场景保持reset状态，不推进物理、不加载策略；全景/近景/俯视共18秒。

- 目标中心world坐标 `(-0.741961, 0.028669, 0.176963)m`；灯泡初始物体参考点
  `(-0.720000, 0.040000, 0.047342)m`；距离约13.2cm。青色圆环半径3cm为成功位置容差。
  每回合目标随机，此片只展示本次单环境采样，不声称与训练6144环境的env0随机样本相同。
- 初始关节配置、目标采样范围、奖励、终止、物理模式、控制周期与保存训练配置一致；
  录制前后机器人关节和灯泡状态逐元素相同，关节与默认初态误差<1e-5。
  仅在独立预览进程添加无碰撞可视标记及中文标注；未修改共享play入口或训练代码。
- 本地[视频](recordings/xinyu21_bulb_initial_goal_20260920_130439/initial_pose_and_goal.mp4)，
  1280×960、24FPS、432帧；同目录有四张代表画面、`scene.json`、`effective_env.yaml`、`verification.json`。
  远端 `/home/xinyu/recordings/bulb_initial_goal_20260920_130439/` 保留job/脚本/日志和失败尝试。
  本地诊断脚本及影响分析在 `logs/diagnostics/bulb_initial_goal_video_20260920/`。
- 初次因隔离目录缺少tacmap依赖退出，随后补齐只读依赖链接；首帧为空的问题通过渲染预热修正。
  退出码不能独立证明成功，最终额外检查视频、场景元数据、状态不变及训练进程启动时间。
  原旋出352175、装配360229、D-Peg242935均继续运行。本次不是训练效果或成功率验证。
- GitNexus共享预览入口CRITICAL、实验索引HIGH，已报告；新诊断函数UNKNOWN，经实际调用引用补核。
  只新增独立诊断包装和追加文档记录，保留既有源码及历史实验。

## BULB-PROFILE-20260920-01

用户要求计算采样中的机械臂、手指、灯泡碰撞与触觉开销。
8卡都有作业，GPU2顺序执行1024/3072环境的隔离短跑，来源为正式三圈guided run的源码快照。
每组445/445源码/资源哈希匹配；物理/控制/奖励/观测不变，环境数、训练预算、输出路径和碰撞栈调整。
各6次PPO更新，迭代0–1预热，2–4用CUDA同步墙钟计时，5记录未加分项计时的采样。

- 1024：每轮18.8095秒，其中物理11.9450秒（63.5%）、压力2.5170秒（13.4%）、接触读取0.3647秒（1.9%）。
- 3072：每轮25.8943秒，其中物理18.2192秒（70.4%）、压力1.8893秒（7.3%）、接触读取0.4098秒（1.6%）。
- 各测量轮核对256次物理步、32次压力观测、352次压力阵列更新；使用exclusive范围避免重复计时。
  范围合计覆盖采样的98%以上，剩余归入调度；returns属于learn_time，未加到采样。
- 所有诊断完成后退出，无碰撞容量/内存/非有限值错误；生产训练未重启，22:21仍在第59轮。
- 局限：GPU2共享负载、不同环境数与PPO轨迹、计时同步扰动、只覆盖初期短窗口；
  不能直接分解正式6144环境的25.83秒，也没有臂/手/灯泡可独立相加的求解耗时。
- 附带发现：压力内部pose传感器在无reset的测量轮内更新0次；周期1/30与零dt调用的CPU调度复现
  为96次调用/0次刷新，推进dt后96次刷新。尚未原生逐帧比较位姿或验证策略影响，未改实现。

远端run：`20260920_141037_bulb_sampling_profile_1024_gpu2`、
`20260920_141417_bulb_sampling_profile_3072_gpu2`（`/home/xinyu/training_runs/`下）。
本地[诊断目录](logs/diagnostics/bulb_sampling_profile_20260920/)含计时脚本、原始数据、日志、
配置/契约、差异和源码核验；[完整报告](docs/rotate_bulb_sampling_profile_2026-09-20.md)。
未提交代码，未调整生产性能参数或任务成功判据。

## BULB-SEGMENTS-20260920-01

用户要求重新检查三圈旋出的臂手分工、密集奖励与长程引导，并授权调研相关工作后实施。
依据 [Learning Dexterous In-Hand Manipulation](https://arxiv.org/html/1808.00177) 与
[In-Hand Object Rotation via Rapid Motor Adaptation](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)
设计局部进度/目标和允许换指的旋转信用；本项目系数为待训练验证的起点。
实现与完整量纲见 [专题说明](docs/rotate_bulb_segmented_guidance_2026-09-20.md)。

- 代码仅改Custom旋出配置/奖励及共享入口的观测契约校验；保留装配669维/动作v3。
  三圈默认不变：连续1.5分/圈+每首次90°段0.125分，总计最多6分。
  旋出以最强指尖接触/0.5N饱和、0.2秒换指记忆，奖励推进速率封顶1圈/秒；
  不支持无接触补领奖励、重复旋转领同段奖励或脱离后的空关节领奖励。
  额外臂姿态/速度代价最高−0.2分/秒，释放后关闭；原臂倍率40、远离−0.3保留。
  新增4维可观测小目标状态，旋出673维，契约记录引导版本1；从头训练。
- 本地最终CPU回归 **352 passed, 5 skipped in 13.72s**。
  覆盖旋出行为/预算/30–120Hz、边界/局部reset、装配隔离、动作与检查点契约；
  原生能力相关跳过不计为通过。初次旧预期失败及浮点累计误差保留在first/second报告，
  随新奖励语义更新测试预期，最终保留三圈6分与完整任务收益预算检查。
- 服务器独立目录 `/home/xinyu/training_runs/20260920_134350_rotation_guidance_validation`：
  GPU2、2环境、30个零动作物理步、局部重置、无副作用观测、v3动作映射检查通过；
  后续真实PPO更新0–2，model_0→model_2有25个状态张量变化，全部模型状态有限。
  测试工具早期出现kit_args转义、资产快照缺地面和InferenceMode上下文问题；
  对应失败日志保留，修正独立工具后通过，未据这些失败修改生产物理/奖励。
- 独立CPU运动学检查使用实际USD关节树和当前Dexonomy初态：保持掌朝向、上移3–18mm的6个点，
  都找到各臂关节变化≤0.05rad的解；18mm时最大0.03490rad（2.00°），最大位置残差约3.4e-8m。
  只验证局部运动学跟随空间，未验证无碰撞或动态抓持，不将解写入策略/仿真状态。
- 同步10份代码/测试/新说明，运行相关2个代码文件远端旧哈希与本地修改前完全一致，
  最终代码哈希与原生验证一致；新run有445文件代码快照、同步备份及完整启动参数。
  新run：`/home/xinyu/training_runs/20260920_134756_rotate_bulb_segments90_armsteady_6144_gpu4`，
  GPU4/6144环境/30Hz/seed42/15000迭代/from scratch；21:48:42启动，
  监督PID374551、训练PID374552。21:56正式迭代0–3及model_0保存已确认，
  保存的673维/引导契约v1、奖励/臂参数全部符合预期；配置差异只在两项旋转奖励、臂状态项、
  4维小目标观测、run路径与名称，PPO/物理/初始化/成功失败判据保持一致。
  初期显存25723MiB；未发现容量溢出、非有限动作、内存/PhysX致命错误，非长期容量保证。
  旧arm40 run在170轮停止（监督器SIGTERM，子进程退出−6），已有model_0/model_100哈希保持。
  装配PID360229和D-Peg PID242935启动时间不变。
- 风险：GitNexus重建后Custom奖励配置为CRITICAL（8个直接依赖、0条可解析流程、5模块），
  共享场景与契约等部分查询仍UNKNOWN/索引关联异常，已补充训练/PLAY/评测/装配文本引用核查。
  不将0条图流程或UNKNOWN当作不受影响；用CPU装配回归和原生新契约检查补足。
  未修改释放、成功、失败或物理关键碰撞，也未评测完整任务成功率。

证据：`logs/diagnostics/bulb_rotation_guidance_20260920/`，包含修改前备份、增量patch、
图分析、全部本地测试报告、远端验证/部署参数与返回记录。首次运行并非完成训练结论。

## BULB-CONTACT-LOG-20260920-01

21:30（Asia/Shanghai）补查同一arm40 run：136/15000，6144环境/GPU4，
保存配置确认guided模式（`bulb_pure_collision=false`）、初始3圈、远离−0.3、释放前臂倍率40；进程持续运行。
最新检查点model_100，显存27299MiB；成功相关日志历次最大值仍为0。
117–136轮日志点均值：最大旋出1.2515圈、最终0.8469圈、达到3圈指标0.04613、
超时0.99279、接触奖励0.012823、远离−0.002376、动作幅度−0.053455、
两项旋出合计0.022996、mean_reward −0.94743。
最新136轮最大旋出均值2.2179圈，但不以单点推断稳定改善。
这些是训练日志汇总，窗口未按回合数重新加权，不是独立评测；无新代码或训练干预。
原始快照：`logs/diagnostics/bulb_status_20260920_2130/metrics.json`。

用户询问当前是否鼓励手指接触及比旧训练差的原因，随后询问当前任务状态。
只读远端7个run保存配置、TensorBoard和旧Dexonomy源码快照；当前主快照迭代72，补查78。
没有修改奖励、生产代码或重启训练。原始标量、统计脚本和比较图保存在
`logs/diagnostics/bulb_contact_log_analysis_20260920/`，方法、指标口径、样本不等权局限和结论见
[专题分析](docs/rotate_bulb_contact_log_analysis_2026-09-20.md)。
当前接触鼓励存在，但双侧抓稳独立塑形禁用；近期接触改善而圈数下降，属于待验证的奖励权衡问题。
先前臂40建议只经固定轨迹预算校准，不是已证实最优。未执行新物理回放、多seed评测或奖励干预。
GitNexus查询/上下文存在异常关联，以实际源码与各run保存快照补核；文档影响STATUS LOW、索引MEDIUM。

## BULB-MEM-RESTART-20260920-01

用户明确要求同步修改、重启旋出训练，并减少显存占用以提高并行环境数量。
采用独立新 run，从零训练；本次未修改生产代码逻辑，只同步已验证奖励修改并调整启动覆盖。

- 新 run：`/home/xinyu/training_runs/20260920_121222_rotate_bulb_far_arm40_6144_gpu4`，
  2026-09-20 12:13:30 UTC 启动；训练/监督 PID `352175` / `352174`。
  GPU 4、6144 环境（旧 4096，+50%）、30 Hz、seed 42、15000 迭代、每100迭代保存、resume=false。
  TensorBoard 新增 `Rotate-Bulb_FarArm40_6144_GPU4_20260920`，旧 run 入口保留。
- 同步：9 个相关代码/测试/文档逐文件备份后原子替换，两端 SHA256 一致。
  443 个相关源码/资产已归档，逐项与本地哈希一致；运行配置与旧 run 比较只有奖励两项、环境数、
  四项 PhysX 容量及日志/run 名称发生变化。观测669、动作v3、guided、自碰撞、初态、成功协议保持。
- 显存设置：三项 `gpu_found_lost_pairs_capacity`、`gpu_found_lost_aggregate_pairs_capacity`、
  `gpu_total_aggregate_pairs_capacity` 从各134217728降为33554432条；碰撞栈2147483648→3221225472字节。
  未缩减接触判据、SDF几何、仿真频率或传感器观测。碰撞栈按原每环境512KiB经验留余量，
  没有利用工厂自动2GiB上限压缩其容量。后续若报接触丢弃/溢出立即停止，不能继续使用损坏仿真。
- 验证：服务器相关 CPU 回归 **252 passed in 14.24s**，测试范围同上一条本地252项。
  场景创建62.77秒、物理启动220.53秒；已观察迭代0–8的 PPO 更新并保存 `model_0.pt`，
  `hand_far.weight=-0.3`、释放前臂倍率40、释放后手指变化倍率2均在保存YAML核对。
  显存最近连续四个30秒采样为26143MiB（25.53GiB），旧4096环境27627MiB（26.98GiB），
  并行数增加50%时显存仍减少1484MiB（1.45GiB，约5.4%）；初期吞吐中位8266 steps/s。
  检查时未发现容量溢出、丢接触或PhysX/CUDA内存错误；日志仍有原有USD视觉引用警告。
  这是不同策略阶段的运行采样，不是严格性能基准或长期峰值保证；6144环境每迭代样本更多，
  不能按相同迭代数直接比较样本预算或学习成绩。保留容量余量，本次未进一步尝试8192。
- 原训练 `20260920_105800_rotate_bulb_custom_phase_30hz_gpu4`：核对PID/启动时间/命令后向监督器发SIGTERM，
  转发给原训练，停止时迭代254、status=stopped、exit_code=-6；原model_0/100/200均哈希不变。
  装配PID325605、D-Peg PID242935启动时间不变，未重启。
- 当前 torch2.7.0+cu128、Isaac Lab2.3.2.post1、Isaac Sim5.1.0.0、rsl-rl-lib metadata3.1.1；
  实际 `/home/xinyu/runtime/rsl_rl` fork文件哈希与旧run一致，已重新核对。
  本次 GitNexus 配置类 MEDIUM（8直接、0流程、1模块）；STATUS/RUNBOOK LOW，实验索引MEDIUM。
  奖励文档 UNKNOWN，通过实际引用补核；没有把零流程视为无调用。未提交 Git。
- 证据：远端 run 中 `launch_spec.json`、`sync_backup/`、`sync_manifest.json`、`source_snapshot.tar.gz`、
  `source_hashes.json`、`runtime_provenance.json`、`restart_stop_record.json`、`restart_verification.json`、
  `regression.txt`、`gpu_samples.jsonl`、`train.console.log`；本地
  [记录目录](logs/diagnostics/bulb_restart_far_arm40_20260920/) 保留同步前快照和核验副本。
  无独立成功数/总数，未声称训练改善。PhysX容量背景见
  [官方 GPU Memory 文档](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html#gpu-memory)。

## BULB-FAR-ARM40-20260920-01

用户在 model_100 分析后明确要求先修改远离最高 −0.3 分/秒、释放前机械臂倍率 10→40。
本次仅本地修改，不同步远端或重启现有作业。版本为 main / `4707074…` 加原有未提交工作区。

- 运行代码仅修改 `RotateBulbCustom/env_cfg.py` 的奖励配置：启用已有 `rotate_bulb_hand_far`，
  weight=−0.3、near=0.08m、far=0.2m、grace=0.5s；`arm_before_release_multiplier=40`。
  远离在释放前后均生效，目标保持/终止帧豁免；其余奖励、成功/失败、物理和动作观测契约未变。
- 数量级：距离 14cm 时 −0.075/s，≥20cm 时 −0.3/s；20秒扣除宽限后的积分上界 5.85分。
  未释放幅度代价为每秒 `−(1.0*A + 0.025*H)`，释放后仍 `−0.025*(A+H)`；A/H 为组内动作均方。
  连续20秒臂动作0.5、手为0时，释放前幅度扣5分，释放后扣0.125分。
- CPU 验证 **252 passed in 8.63s**：
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python
  -m pytest -q -rs -p no:cacheprovider tests/test_rotate_bulb_phase_action_penalties.py
  tests/test_rotate_bulb_custom_failure.py tests/test_rotate_bulb_custom_any_contact.py
  tests/test_rotate_bulb_reward_balance.py tests/test_rotate_bulb_soft_grasp_rewards.py
  tests/test_rotate_bulb_rewards.py tests/test_bulb_assembly_contact_rewards.py`。
  覆盖30/60/120Hz积分、远离阈值、释放前后、目标豁免、局部reset、阶段动作分组、完整奖励管线及装配兼容。
  初次命令误写不存在的装配测试文件名，收集退出码4，未运行测试；更正后上述用例全部通过。
- GitNexus 配置类影响为 CRITICAL（8直接/115流程/20模块），远离函数亦 CRITICAL，存在异常跨任务关联；
  已在编辑前报告并补查实际引用。部分新测试/文档为 UNKNOWN，未视为低风险。
  实际训练和 PLAY 共用 Custom 奖励类，标准旋出独立配置、装配独立奖励类；共享函数默认倍率未变。
- 证据：[本地目录](logs/diagnostics/bulb_far_arm40_20260920/) 含修改前相关文件、图分析输出、
  `regression.txt` 和本次增量 `implementation.patch`；首次收集错误保留在 `regression_collection_error.txt`。
  本次无物理回放、新训练或独立成功率结论；运行中的远端训练不会热加载本地修改。

## BULB-M100-20260920-01

用户要求录制当前 model_100 并分析较旧策略的退化，随后要求分析远离和机械臂动作惩罚的统一数量级。
实际完成两条原配置回放、日志/轨迹分析和固定轨迹重算；未实施奖励建议，未修改运行代码、未重启训练。

- 来源：当前 `20260920_105800_rotate_bulb_custom_phase_30hz_gpu4/train/model_100.pt`，
  SHA256 `c53e36f9e311af53f265e10efa696faaeed286a624200a4cb0bae1dd54207d9e`；
  对照 `20260920_092213_rotate_bulb_custom_dexonomy_gpu4/train/model_100.pt`，
  SHA256 `61191937996fe22a2adcad1f2e3658d610518c31fcc1b8207427d7186657709d`，均位于 `/home/xinyu/training_runs/`。
- GPU7 单环境、seed42，原源码快照/配置/动作观测契约；各21秒。当前630/5040控制/物理步，旧1260/5040，
  状态及观测有限、目标误差0，子步目标保持正确；反向接触校验误差0。回放退出码均0。
- 首个完整20秒回合：当前最大0.5273圈、0.725s后失去全部机器人接触、距离代理最大1.398m、回报+0.07354；
  旧最大2.5062圈、双侧>1N接触0秒、回报+5.02587。两者均未释放/未成功并超时；不同奖励下不公平比较回报。
  每模型只观察一个固定seed完整回合，不是多seed独立评测。
- 关键解释：未释放远离不触发已有失握失败；动作是每步相对当前关节位置的请求，中小动作可以累积大偏移。
  当前臂RMS=0.07137，20秒幅度代价仅−0.02547。软门控、30Hz、臂倍率同时变化，未做单因素训练归因。
- 候选（未实施）：启用既有距离平方代价，near=0.08m、far=0.20m、grace=0.5s、weight=−0.3/s，
  未释放也生效、保留目标保持豁免；释放前臂multiplier=40，对应每秒系数1.0。
  该轨迹离线距离扣5.715、臂扣0.102，回报由+0.074变−5.718；这不是实际训练结果或最优系数证明。
- 本地录像：[当前](recordings/xinyu21_bulb_model100_compare_20260920_113845/current_model_100.mp4)、
  [并排](recordings/xinyu21_bulb_model100_compare_20260920_113845/model_100_side_by_side.mp4)。
  原片均21秒、1280×960、全解码通过、两端哈希一致。
- 完整命令/配置/哈希/轨迹/分值表见[分析报告](docs/rotate_bulb_model100_penalty_analysis_2026-09-20.md)；
  服务器证据 `/home/xinyu/recordings/bulb_model100_compare_20260920_113845/`，本地同名 recordings 目录带 xinyu21_ 前缀。
  `server_postflight.json` 确认两个原模型未变、诊断退出、三个训练进程启动时间未变，旋出迭代188。

## BULB-RESTART-20260920-02

用户授权连接 xinyu21、同步本地项目改动并重启旋转灯泡 Custom。两端 main / `4707074…` 加已有工作区改动。
此次未修改运行代码、未提交 Git，未传录像/数据集/模型或执行删除同步。

- 同步核验：83 个改动/新增源码、配置、文档、测试和小型初态/证据文件在执行前已全部 SHA256 一致，
  因而无需覆盖；额外核对新 run 的 443 个相关源码/资产快照文件与本地全部一致。
- 新 run：`/home/xinyu/training_runs/20260920_105800_rotate_bulb_custom_phase_30hz_gpu4`。
  2026-09-20 10:58:53 UTC 启动，训练 PID `337170`、监督 PID `337169`；GPU 4、4096 环境、
  seed 42、15000 迭代、每 100 迭代保存；从头初始化策略/优化器，`resume=false`。
  沿用旧 run 已核实的资源/物理覆盖，WarpSDF 更新周期从 1/60 改为 1/30 以匹配新默认控制频率。
- 生效 YAML/契约：`sim.dt=1/240`、decimation=8、guided、自碰撞开启、初始 3 圈、
  Dexonomy 28 个初态关节与模板最大误差 0、动作 v3、观测 669、关闭触觉编码器；
  `arm_before_release_multiplier=10`、`hand_after_release_multiplier=2`。观察到迭代 0–4 的 PPO 更新及 `model_0.pt` 保存。
  这只是启动/配置验证，无独立成功数/总数，不能认定学习效果改善。
- 旧 run `20260920_092213_rotate_bulb_custom_dexonomy_gpu4` 停在迭代 504；向确认过启动时间和命令行的
  监督器 `275256` 发 SIGTERM，由其转发给训练 `275257`，最终状态 stopped、exit_code=-6。
  `model_500.pt` 前后 SHA256 一致；原日志/模型保留。装配 `325605` 与 D-Peg `242935` 的进程启动时间不变。
- 服务器 CPU 回归：`PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  /home/xinyu/envs/brainco/bin/python -m pytest -q -p no:cacheprovider tests/test_rotate_bulb*.py
  tests/test_bulb_bounded_actions.py tests/test_bulb_bounded_policy.py tests/test_bulb_assembly*.py`，
  **378 passed、11 skipped、1 failed**。失败是既有 `test_rotate_bulb_collision_stack.py` 将同一
  `(8192, 2**28)` 同时列入有效和应报错用例；本次 4096 配置及分阶段动作惩罚用例通过，未改测试掩盖失败。
- 环境：torch 2.7.0+cu128、Isaac Lab 2.3.2.post1、Isaac Sim 5.1.0.0、rsl-rl-lib metadata 3.1.1，
  实际使用 `/home/xinyu/runtime/rsl_rl` fork，源码哈希在 `runtime_provenance.json`。
- 证据位于新 run：`launch_spec.json`、`source_snapshot.tar.gz`、`source_hashes.json`、
  `sync_manifest.json`、`sync_verification.json`、`restart_stop_record.json`、`restart_verification.json`、
  `training_verified_tail.txt`、`regression_summary.json`、`sync_evidence/`。
  TensorBoard 添加 `Rotate-Bulb_Custom_Phase_30Hz_GPU4_20260920`，保留旧 run 入口。
- GitNexus 当前工作区/commit 已绑定，动作函数报 CRITICAL（跨任务关联异常），奖励类消歧仍 UNKNOWN；
  已向用户报告并补查实际引用，装配调用默认倍率 1，Custom 旋出传 10/2；未将图结果当完整影响证明。
  文档 impact UNKNOWN，以实际文档交叉引用核查补充。仅追加此次状态和证据，未更改历史记录。

## BULB-SYNC-20260920-02

用户授权同步当前项目并在另一张 GPU 从零训练 Custom 旋出。已连接 `xinyu21`；远端仓库
`/home/xinyu/mulyi-model-tac` 与本地均为 main / `4707074194091cd723ebd18dea7824c184e99c6d` 加未提交改动。

- 本次同步 79 个源码/配置/文档/测试/初态及小型证据文件，15 个新增/更新、64 个内容已一致；
  未传录像目录，未使用删除同步。逐文件 SHA256 79/79 一致。
- 证据目录 `/home/xinyu/training_runs/20260920_095303_rotate_bulb_custom_softgrasp_30hz`：
  `sync_evidence/manifest.json`、`remote_manifest_before.json`、`verification.json` 与 `sync_backup/`。
- 沿用上一条奖励记录中的 14 个 CPU 测试，本地和远端均 289 passed、1 skipped；
  跳过因独立 Python 未加载 `pxr.Usd`，不是物理 rollout 或成功率验证。
- 同步含当前软抓持奖励和 30 Hz 默认；亦包含同步取样时已存在的装配初态随机化修改。
  本次未编辑生产代码。检查期间本地仍有其他修改写入，新 run 以保存的源码快照为准。
- GitNexus 当前仓库索引 commit 匹配，场景/奖励分析为 UNKNOWN 或 CRITICAL，存在符号解析异常；
  已提示用户并补查本地/远端逐文件 diff、注册及继承引用。未将异常图结果当完整调用证明。
- GPU 0–7 均有进程；旧旋出 PID 275257/GPU 4、装配 PID 242934/GPU 5、D-Peg PID 242935/GPU 6
  未干预。候选新训练：GPU 1 共享、2048 环境、seed 42、15000 迭代、guided、关闭编码器、resume=false；
  用户共享选择待回复，当前未启动新训练。备份和候选启动配置保留在上述独立目录。

## BULB-REWARD-20260920-01

用户批准旋出/释放后奖励调整，特别要求搬运用原抓取质量作软门控，保留弱接触学习信号。
本次授权仅为本地修改与验证，用户自行同步到服务器。本地未发现相关训练/回放进程；
远端运行状态仅来自前一条记录，本次未连接或干预远端。

- 实现：仅修改生产文件 `RotateBulbCustom/env_cfg.py`，新增阶段无关连续抓取质量及衰减记忆 helper；
  两项旋出按软质量支付新增圈数；搬运以释放位置为基线、到 3 cm 窗口归一化，新增最佳进度预算 4 分。
  启用现有 `release_grasp` +2、`released_stability` 封顶 +2；失握改为释放后无接触与远离同时持续达标，
  并入一次 −6。原成功 +10/位置保持、物理、初态、动作和观测不变。
  公式、阈值、兼容字段和同步清单见 [奖励说明](docs/tactile_representation/rotate_bulb_custom_rewards_2026-09-20.md)。
- 基线：`main` / `4707074194091cd723ebd18dea7824c184e99c6d` 加开始时已有未提交修改；
  先保存本次相关文件快照再编辑。没有覆盖用户其他修改，没有创建 Git 提交。
  修改后 `env_cfg.py` SHA256：`fa9781889068ff933a1cd9a572e886ca7b8efbda32c83ae8f085fce599a72cfc`。
- 修改前定向 CPU 基线 **240 passed**；修改后扩大至下方 14 个相关测试文件，**289 passed, 1 skipped**。
  唯一跳过为 `test_rotate_bulb_custom_assets.py:31` 的 `pxr.Usd` 导入不可用。
  新增 [软门控行为回归](tests/test_rotate_bulb_soft_grasp_rewards.py) 验证弱/单侧接触连续得分、
  无接触记忆过期、无进度重复得分、重新抓握不补领、局部 reset、控制频率预算、释放/失握联合条件。
  保留并更新既有标准/Custom 共用测试中的版本预期；标准任务与装配状态/奖励回归通过。
- AST 对照确认 reset、机械释放、圈数几何、最终成功函数均未改变。未启动 Isaac、未做 GPU 探针或训练；
  CPU 张量轨迹的收益关系不能当作真实策略成功率。旧/新奖励的回报不可直接比较。
- GitNexus 已重建索引并执行逐目标 upstream impact。场景配置/失败汇总等报告 CRITICAL，已在修改前提示；
  部分类报告 MEDIUM，README 为 UNKNOWN 并补查文本引用。函数图存在过宽关联及动态分派边界，
  用源码引用和回归限定修改范围，不把缺失边当作无影响。已运行全工作区 detect-changes；
  其范围含用户原有改动，不能作为仅本次变更的风险结论。

复现离线验证（仓库根，已有 `brainco` 环境，不启动仿真）：

```bash
/home/liuxinyu/miniconda3/envs/brainco/bin/python -m pytest -q -rs \
  tests/test_rotate_bulb_rewards.py tests/test_rotate_bulb_reward_balance.py \
  tests/test_rotate_bulb_custom_turns.py tests/test_rotate_bulb_custom_failure.py \
  tests/test_rotate_bulb_soft_grasp_rewards.py tests/test_rotate_bulb_custom_any_contact.py \
  tests/test_rotate_bulb_grasp_reset.py tests/test_bulb_assembly_guidance.py \
  tests/test_rotate_bulb_custom_assets.py tests/test_rotate_bulb_pure_collision.py \
  tests/test_bulb_assembly_contact_rewards.py tests/test_bulb_assembly_state.py \
  tests/test_rotate_bulb_custom_task.py tests/test_revo3_rotate_bulb_task.py
```

下一步由用户同步并新建训练环境，检查实际保存的奖励与终止配置；尚未实施。

## BULB-RESTART-20260920-01

本次按用户明确授权连接 `xinyu21`、同步本地改动，用户确认使用新抓姿从头训练。
远端仓库 `/home/xinyu/mulyi-model-tac` 与本地同为 `main` / `4707074194091cd723ebd18dea7824c184e99c6d`，
均有未提交修改；实际版本以快照及哈希为准。

- 新 run：`/home/xinyu/training_runs/20260920_092213_rotate_bulb_custom_dexonomy_gpu4`。
  完整命令/cwd/非敏感环境在 `launch_spec.json`，训练日志 `train.console.log`，状态 `status.json`。
  GPU 4、4096 环境、seed 42、15000 迭代、每 100 迭代保存；guided 物理、v3 动作、关闭触觉编码器。
  新模型及优化器从头初始化，未加载父模型；仅更换 run/output/TMPDIR，沿用旧训练资源及覆盖参数。
- 2026-09-20 09:22:58 UTC 启动，监督 PID `275256`、训练 PID `275257`。
  已确认迭代 0–4 的 PPO 更新及 `train/model_0.pt` 保存，核查时状态 running、无 fatal_error 文件。
  `effective_config_verification.json` 核对保存 YAML：28/28 关节角与新模板完全一致（最大误差 0），
  resume=false、观测 669 维、动作契约 v3、初始 3 圈。`restart_verification.json` 及
  `training_verified_tail.txt` 保留运行证据；未执行独立成功率评测。
- 同步 112 个本地改动/未跟踪文件，54 个新增或更新、58 个原已一致；同步后 SHA256 112/112 一致。
  无删除同步。`sync_manifest.json`、`remote_manifest_before.json`、`sync_verification.json` 为逐文件证据；
  `source_backup/` 及 `rsync_backup/` 保留被替换文件，`source_snapshot.tar.gz` 保留相关源码与此次文件快照。
- 旧 run：`/home/xinyu/training_runs/20260920_041251_task_revision_restart/rotate_bulb_custom_v3_gpu4`。
  向监督 PID `242930` 发 SIGTERM 后监督器按已有 10 秒超时终止子进程 `242933`，状态为 stopped、exit_code=-9、
  末迭代 1704；旧日志和 `train/model_1700.pt` 保留。装配 PID `242934`、D-Peg PID `242935` 持续运行。
- 本地及远端均执行 `pytest -q -p no:cacheprovider`：`test_rotate_bulb_grasp_reset.py`、
  `test_bulb_bounded_actions.py`、`test_bulb_bounded_policy.py`、`test_rotate_bulb_custom_turns.py`，
  设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1`，均 **41 passed, 2 skipped**。
  跳过为未启动 Isaac 时缺少 `pxr.Usd`；不将 CPU 回归当作动态成功验证。
- 环境：`/home/xinyu/envs/brainco/bin/python`；torch 2.7.0+cu128、Isaac Lab 2.3.2.post1、
  Isaac Sim 5.1.0.0、rsl-rl-lib metadata 3.1.1；运行配置使用 `/home/xinyu/runtime/rsl_rl` fork。
  `runtime_provenance.json` 记录包版本及 26 个相关资产哈希，`rsl_rl_source_hashes.json` 记录 52 个 fork 源文件哈希。
- GitNexus 场景配置消歧后报 CRITICAL（229 影响项/18 过程），共享 robot_contract 报 MEDIUM（15 项）。
  图结果有异常 ID/不稳定计数，未将其当完整调用证明；补充远端实际 diff、引用和回归，确认运行改动仅接入新抓姿，
  装配保留旧姿态、D-Peg 旧常量保持。未修改奖励/成功判据，未创建 Git 提交。

## BULB-GRASP-20260920-01

本次实际运行：将 Dexonomy 新灯泡抓姿设置为 Custom 旋出任务及 PLAY 的默认初态。
来源、运行命令、代码 SHA256 和本次 patch 见
[validation_summary.json](assets/rotate_bulb_custom/grasp_reference/dexonomy_20260920/validation_summary.json)，
逐项结果见 [runtime_validation.json](assets/rotate_bulb_custom/grasp_reference/dexonomy_20260920/runtime_validation.json)。

- 版本：`main`，HEAD `4707074194091cd723ebd18dea7824c184e99c6d` 加原有工作区改动和本次局部修改；没有提交。
- 来源模板 SHA256：`e0e1f2cc303618d5198ae3fed00606d550827ead6b0d15d1b78c5fa5b18ef246`。
  机械臂/物体/灯座姿态相同，手指姿态更新；保留默认 3 圈及原物理/奖励/成功判据。
  装配通过原有 `bulb_assembly_task` 标记使用旧参考，D-Peg 的参考也保持原样。
- 影响分析：GitNexus 对场景配置返回 CRITICAL（229 个影响项、17 个过程）；部分返回 ID 异常，
  不将其视为可靠完整调用图。补充引用检查确认训练/PLAY 调用该配置、装配继承该配置，
  并以配置回归测试检查新旧姿态隔离。没有提交，因此未执行提交前图变更门禁。
- 本机 `brainco` Python、CUDA 0、2 环境，240 秒超时上限；状态检查完成后正常退出，退出码 0。
  启动、3 次完整 reset、扰动后的局部 reset 共 **440/440 通过**，检查 35 个匹配 link，
  最大位置误差 7.074e-7 m，最大旋转误差 6.643e-6 rad。4 个不可对应参考 link 已在报告列出。
- 离线回归 **46 通过、2 跳过**，跳过因未启动 Isaac 的解释器缺少 `pxr.Usd`。
  首轮新测试误将导出的接触策略说明文本作为数值几何比较，修正为比较实际源配置后通过。
- 范围：精简观测，未初始化完整触觉编码器、未执行动态 rollout；未训练、未操作远端作业。
  来源 MuJoCo 的零穿透/5 指静态接触不能直接推断 PhysX 动态稳定或旋出成功率。

## PEG-EVAL-20260919-01

历史详情：[深度判据评估](docs/d_peg_depth_success_evaluation_2026-09-19.md)。
直接证据：[report.json](assets/d_peg_insertion/validation/depth_success_model_800_20260919/report.json)、
[episodes JSONL](assets/d_peg_insertion/validation/depth_success_model_800_20260919/episodes_seed_20260919.jsonl)、
[manifest.json](assets/d_peg_insertion/validation/depth_success_model_800_20260919/manifest.json)。

| 字段 | 已知内容 / 缺口 |
| --- | --- |
| 日期 / 状态 | 2026-09-19T08:59:49Z 创建报告；16 回合完成，小样本开发评测 |
| 问题 / 假设 | 旧入口/包含门槛是否拒绝深度 ≥27 mm 保持 0.5 s 的样本；同时检查该模型深度表现 |
| 对照基线 / 改变 | 固定同一 model_800；成功按深度保持判据，去除成功中的 entry history/containment 门槛；不是重新训练对照 |
| 保持不变 | 报告记录模型/归一化状态哈希前后一致；基于保存配置，仅改 env 数/seed/device/log_dir |
| 任务 / 版本 | `BrainCo-Dexsuite-Flexiv-Right-Insert-D-Peg-Custom-v0`；`d_peg_depth_dwell_evaluation_v1`，不能仅以 task ID 区分旧判据 |
| 代码 / 分支 / 差异 | 历史 commit、分支与完整差异快照未知；评测器哈希 `8fa6e3ef3f0760f5bce23478a46056845e103b112248d57e59ee75b19e9039a0`；奖励源码哈希 `7a637297c86113d19b1d1c041fce2ff761cd31e7d4a0501f0136863c4983156b`，本次确认这两个当前文件匹配 |
| 环境 / 依赖 | manifest 指向历史 `brainco` Python；完整包版本未知，不从环境名推断 |
| 生效配置 | 源模型同目录 `params/env.yaml` / `agent.yaml`；报告保存其 SHA256 和环境覆盖；本次未读取远端 YAML |
| 命令 / 目录 | 完整 argv 在 manifest，运行 `scripts/rsl_rl/evaluate_d_peg.py --checkpoint … --output … --num_envs 16 --seeds 20260919 --headless --device cuda:0`；历史评测 cwd 未在该 manifest 单独记录，待确认；可改用的新目录命令见 RUNBOOK |
| seed / 数量 / 迭代 | seed 20260919；16 并行槽各 1 个完整回合；模型迭代 800，未做参数更新 |
| 预算 / 停止 | 每回合最多 15 s 仿真时间；每槽首回合完成后停止统计；历史 GPU 总预算未知，本次没有新的运行授权 |
| 模型 / 输出 | 历史模型 `/home/xinyu/training_runs/20260919_115128_nomodal8192/insert_d_peg_replicated/model_800.pt`；模型 SHA256 `e100e6c39ccffe514ea8f67d3cfe6739b0a4e1e85bb043173cf04367e43866f3`；服务器评测目录 `/home/xinyu/training_runs/20260919_165949_d_peg_depth_quick_eval/evaluation`；视频证据未知 |
| 评测协议 | 确定性动作，保存的初始状态分布；孔坐标轴向深度 ≥0.027 m 连续 0.5 s；不要求几何包含/入口历史；独立计时仍调用同一 `_geometry` |
| 指标 / 结果 | 本次 JSONL 重数 **1/16**，15 超时，0 掉落/越界/非有限终止；报告 Wilson 95% 区间约 1.11%–28.33%；成功样本两个入口标志均 false |
| 结论 / 强度 | 可支持这 16 个样本在该深度协议下的计数一致、旧入口标志会拒绝其中成功样本；不支持稳定成功率或完整几何插入已通过 |
| 其他解释 / 下一步 | 小样本、单 seed、几何计算共用；完整源码/环境快照缺口仍在。明确协议后收集新 seed 完整回合与几何轨迹，不混报旧新成功口径 |

## PEG-RESUME-20260919-01

详情：[续训记录](docs/d_peg_depth_success_resume_2026-09-19.md)。
证据：[启动清单](assets/d_peg_insertion/validation/depth_success_resume_20260919/resume_manifest.json)、
[恢复验证](assets/d_peg_insertion/validation/depth_success_resume_20260919/resume_verification.json)、
[来源核查](assets/d_peg_insertion/validation/depth_success_resume_20260919/source_preflight.json)。

| 字段 | 已知内容 / 缺口 |
| --- | --- |
| 日期 / 状态 | 2026-09-19 17:08 北京时间启动；17:11:52 验证；当前运行状态未核实 |
| 问题 / 假设 | 在新版深度成功规则下能否保留训练状态续训 |
| 基线 / 改变 | 上条 model_800 为父检查点；旧进程最后记录 862，但只恢复已保存 800；切换成功判据并新建 run，记录学习率覆盖与追加更新数 |
| 保持不变 | 验证文件称模型、归一化器、Adam 状态连续；env.yaml 仅日志路径不同，策略契约相同；不等于运行源码未变 |
| 任务 / 版本 | 同一 Flexiv Custom；深度 ≥27 mm、0.5 s、一次奖励 +80 |
| 代码 / 分支 / 差异 | 完整 commit/分支/未提交快照未知；奖励源码 SHA256 与 PEG-EVAL 一致，当前该文件匹配；全代码可复现性仍有缺口 |
| 环境 / 依赖 | manifest 有解释器、PYTHONPATH、外部 `/home/xinyu/runtime/rsl_rl` 线索；精确依赖/fork 版本未知 |
| 生效配置 | 新 run 的 `params/`；verification 列出与父 run 的 env/agent 差异，包括 LR、load_run、resume、max_iterations；本地不持有完整远端配置证明 |
| 命令 / cwd | 完整 argv/环境白名单在启动清单；cwd `/home/xinyu/mulyi-model-tac`；`train.py --resume --load_run resume_source_20260919_170828 --checkpoint model_800.pt --max_iterations 14200`，其余参数必须沿用清单 |
| seed / 环境 / 更新 | seed 44，8192 环境，rollout 32；从 800 追加 14200，计划最终标号 14999；只核对到归档的 800–801 |
| 预算 / 停止 | 历史配置的迭代上限如上；GPU 时间预算未知，不能视为本次获准续训 |
| 模型 / 日志 / 输出 | 历史新 run `/home/xinyu/training_runs/20260919_170828_d_peg_depth_success_resume_gpu6/train`；父模型哈希见上条；首轮同名 model_800 新哈希 `1308175c6d4852fd4010968743c26aace8e79cba4c9caf8f07694055226b9b3a`；新旧目录不能混淆 |
| 协议 / 指标 | 非独立评测：探索动作、首轮随机回合长度；**151/1299** 结束训练回合成功，不能当作稳定评测率 |
| 结果 / 强度 | 归档记录模型有限且更新；优化器累计 16020→16040，actor/critic normalizer 各增 262144；本次未加载模型重新核验张量 |
| 其他解释 / 下一步 | 首轮统计和判据变化可能影响比例；无长期趋势/新独立评测证据。后续先核对活动作业与完整快照，再决定是否评测，不因本条自动恢复训练 |

## BULB-EVAL-20260919-01

证据：[完整报告](.autonomy/bulb_custom/evidence/v018_model450_finaltarget_seed4040/report.json)、
[旧状态](.autonomy/bulb_custom/state.json)、[冻结协议](.autonomy/bulb_custom/eval_protocol.json)。
这些文件被 Git 忽略，本次没有迁移或修改它们。

| 字段 | 已知内容 / 缺口 |
| --- | --- |
| 日期 / 状态 | 报告创建于 2026-09-19T01:07:38Z；开发评测完成，未达标 |
| 问题 / 假设 | v018 bridge 模型是否能在最终目标位置分布完成释放后搬运/保持 |
| 基线 / 改变 / 不变 | 固定 v018 model_450，从训练 bridge 目标转到冻结最终分布；模型/归一化前后哈希一致；不是当前 v3 对照训练 |
| 任务 / 协议 | `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0`；`rotate-bulb-custom-v1-frozen-2026-09-18`，protocol SHA256 `a7c6f12e2b89cfa8b08945650220cb20553519de2fc1f95eab97622489ab532c` |
| 代码 / 分支 / 差异 | 完整历史 commit/分支/差异未知；报告 params 哈希可对照；当前动作 v3 与旧协议运行不可直接混用 |
| 环境 / 配置 | 精确依赖未知；guided 模式、初始 3 圈，32 环境；报告存 agent/env YAML 哈希，完整非默认项需结合旧源码快照核对 |
| 命令 / cwd | 完整启动 argv/cwd 未由本条报告确认；入口 `evaluate_bulb_custom.py`，参数包含 guided、32 环境、20 回合、seed 4040；这是已知参数归纳，不伪装成完整原命令 |
| seed / 样本 / 迭代 | 4040；20 个完整回合，固定模型标号 450；评测无训练更新 |
| 预算 / 停止 | 报告 max_steps=3600、实际 steps=1200；完整 20 样本后结束；总 GPU 预算未知 |
| 模型 / 输出 | 父目录见 report.checkpoint；SHA256 `40ccd006312809a4baf490dd17ebab2e5f5f29c00362e24e244feb970ef8b0ae`；本地候选位于 `.autonomy/bulb_custom/evidence/v018_model450/`，本次未读权重；逐回合状态内嵌报告 |
| 评测协议 | 最终目标 x/y ±0.04 m、z 0.12–0.16 m；原始 reset 前状态中 released + 距离 <0.03 m + 连续 1 s；失败/超时计入样本 |
| 指标 / 结果 | 本次重数报告 episodes：**0/20**；报告 complete=true、weights_unchanged=true；无当前 v3 成绩 |
| 结论 / 强度 | 仅支持旧模型在该开发 seed/目标分布未成功；已释放/转满圈不代表成功搬运 |
| 其他解释 / 下一步 | 单 seed、小样本、版本差异；不能把旧结果归因到当前奖励或控制。保留失败基线，先补齐版本映射；用户暂停记录不构成重启授权 |

## ASM-PROBE-20260919-01

详情：[装配引导专题](docs/tactile_representation/bulb_assembly_guidance_2026-09-19.md)。
原始报告：[report.json](logs/diagnostics/bulb_guidance_20260919/probe/report.json)。

| 字段 | 已知内容 / 缺口 |
| --- | --- |
| 日期 / 状态 | 2026-09-19T05:25:52Z；探针 passed=true，非训练实验 |
| 问题 / 假设 | 接触软门控、上方目标和下降旋入奖励能否接入实际 ManagerBased 环境 |
| 基线 / 改变 | 原 Assembly 增加三项引导奖励；不是单因素成功率对照 |
| 保持不变 | 专题记录声明动作/观测/初态/独立验收未改；适用于该历史版本，非当前 v3 |
| 任务 / 版本 | `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Custom-v0`；`bulb_assembly_physics_probe` protocol_version=1 |
| 代码 / 分支 / 差异 | 历史完整 commit/分支/差异未知；报告记录 environment/assembly_env_cfg/assembly 源码哈希；本次 environment.py、assembly.py 匹配，assembly_env_cfg.py **不匹配** |
| 环境 / 依赖 | 专题称本机 brainco / PyTorch 2.7.0；完整依赖锁定未知，未在本次解释器复验 |
| 生效配置 | 报告 config：2 环境，seed 42，纯碰撞、20 s 回合、1/60 s 控制周期；其余见报告 |
| 命令 / cwd | 入口 `scripts/rsl_rl/probe_bulb_assembly.py`；已知 2 环境/40 步/seed42，本地项目；完整 argv 未由报告证明，当前模板见 RUNBOOK |
| seed / 环境 / 更新 | seed 42；2 环境、40 步零动作探针，无 PPO 更新 |
| 预算 / 停止 | 40 步有限探针；历史总预算未知，无新算力授权 |
| 输出 / 模型 | 上述 probe/report.json，专题关联 smoke.log；无训练模型、无独立评测模型 |
| 协议 / 指标 / 结果 | 状态有限、reset/部分 reset、奖励接入等检查通过；成功数/总回合数不适用，不能将 passed=true 当作装配成功 |
| 结论 / 强度 | 历史接入/短仿真证据；不证明新奖励可学或物理全流程可行，当前配置需重新验证 |
| 其他解释 / 下一步 | 螺纹接触稳定性未排除；在获授权后先验证当前物理轨迹，再考虑训练，不用长训练替代物理检查 |

## 其他记录与证据缺口

- [Repose 接入说明](docs/tactile_representation/repose_policy_integration.md) 的 2026-09-11 章节记录
  真实 encoder 短训/恢复与远端 summary 路径；本次未读取对应远端原始记录，不提升为当前代码运行成功证据。
- [录像清单](recordings/xinyu21_20260920_110956/manifest.json) 和
  [v3 录像清单](recordings/xinyu21_v3_20260920_153650/recording_manifest.server.json)
  将文件关联到 task/checkpoint/hash，但没有独立成功样本统计。本次没有观看视频或加载模型。
- `.autonomy/bulb/experiments.jsonl`、`.autonomy/bulb_custom/experiments.jsonl` 是已有详细实验历史入口；
  本次仅围绕上述证据定点核查，没有全文整理两条历史或将所有状态数字都当作已验证结论。
- `best.pt` 命名、目录新旧、训练 reward 与一次回放均不构成“最佳策略”的评选证据。

## 实验记录与维护规则

- 调试围绕一个可解释问题；多项同时变化要列全，不将结果单独归因于其中一项。
- 训练 reward 与独立成功计数分开；简化任务/课程和完整任务分开；改变协议或分布后需同口径重评基线。
- 正式结论报告成功数/总数、种子范围与样本量；评测种子数不等于独立训练次数。
- 失败实验保留；修正旧结论时追加版本/证据说明，不抹掉原始记录。
- commit 加未提交代码不能完整复现；保存相关差异及未跟踪源码快照位置，未保存就写复现缺口。
- 不适用填“不适用”，缺证据填“未知”；预算/停止条件未由用户确定就写“尚未授权”。
- 实际实验或结果分析后更新此索引，当前结论只在 STATUS 维护。记录增多再拆入 `docs/experiments/`。

## 新实验模板

```text
实验 ID / 日期 / 状态：
研究问题与假设：
对照基线：
本次主要改变：
保持不变的条件：
任务名及版本（含动作/观测/奖励/成功协议版本）：
代码 commit、分支与未提交差异说明（含未跟踪源码和快照位置）：
环境/依赖版本（解释器、Isaac、torch、RSL-RL fork/版本及 import 来源）：
实际生效配置或配置快照位置：
启动命令及工作目录：
随机种子、环境数、训练步数/迭代数（新增/累计分别写）：
算力预算与停止条件（未确定则“尚未授权”）：
训练日志、模型、回放和评测输出位置（含 parent checkpoint/哈希）：
评测协议：初始条件、样本数、成功/失败定义、是否冻结策略与归一化：
主要指标：成功数/总数；训练 reward 另列：
结果：
结论及证据强度（代码/历史/本次执行）：
仍无法排除的解释：
下一步（建议与已授权行动分开）：
```

## DOC-AUDIT-20260920-01

类型：文档审计与已有结果核对，**不属于训练实验**。
问题：新会话能否从项目入口定位代码、当前版本、证据和下一步，而不重新穷举仓库。
基线：既有 GitNexus AGENTS、README、专题文档和本地证据；缺统一状态/地图/手册/实验索引。
改变：仅补充本次 Markdown，AGENTS 的 GitNexus 块和 README 原文保留。
保持不变：用户原有业务实现、测试、配置、资产、历史记录及作业。
任务/随机种子/训练迭代/算力预算/模型更新：不适用；没有启动训练、仿真、真机或远程连接。
代码：`main` / `4707074194091cd723ebd18dea7824c184e99c6d` + 下列已有改动；
没有保存完整业务差异快照，因此本文不作为训练可复现代码包。

### 本次实际检查

- Git 根/分支/HEAD/status/ignore；适用祖先指令、代码/文档范围的局部指令及已涉及资产祖先定点检查。
  未发现适用 `AGENTS.override.md`；没验证 Codex 自动加载；没修改全局配置。
- 系统 Python 3.12.3、pytest 7.4.4；只读取包 metadata，未导入 torch/Isaac。
- 相机测试命令见 RUNBOOK：15/15 通过（0.02 s）；两条双卡脚本 `bash -n` 返回 0。
- `ast.parse` 12/12：`scripts/rsl_rl/{train,play,play_camera,cli_args,evaluate_d_peg,evaluate_bulb_custom,evaluate_bulb_assembly,probe_bulb_assembly}.py`，
  以及 `B/tasks/direct/brainco/__init__.py`、`D/config/{Revo3,InsertDPegCustom,RotateBulbCustom}/__init__.py`
  （B/D 定义在 PROJECT_MAP）；仅解析、不导入、不写 pyc。
- 重数 D-Peg JSONL 得 1/16；旧旋出报告内 episodes 得 0/20；核对源码 SHA256，未读取权重。
- GitNexus 已有缓存 CLI 1.6.12：list/status/query/context/impact；未安装、重建或下载。
  绑定当前 Git 根；索引时间 2026-09-20T04:07:17.072Z、同 HEAD，status 为 stale：0 changed、2 added、0 deleted，
  两个 added 是 v3 录像清单 JSON。原 AGENTS 统计是旧生成值，本次保留其管理块。
- Markdown impact：AGENTS 不在图中、根 `File:README.md` 无可解析调用方/流程，均为 UNKNOWN。
  补查文档引用后只做追加导航；没有据零调用方判断可删除或安全改代码。
- 为理解成功路径，`d_peg_success` impact 报 CRITICAL，213 个受影响项、136 直接依赖、18 流程、15 模块，
  输出含 train/play/evaluate 等文件和多条触觉流程。部分 context/impact 关联宽泛或与源码引用不符，
  视为需复核的图结果；已在编辑前向用户提示，不修改该函数。图不能替代运行验证。
- 未提交，因此未执行提交前 detect-changes；不声称全工作区图检查通过。

### 证据强度与后续

本次证明文档索引、静态入口和上述历史样本计数可对应；没有验证策略收敛、成功率泛化、
依赖兼容性、检查点实体、实时进程状态或真机效果。
历史证据包缺完整源码/依赖快照，部分引用被 Git 忽略；下一步建议与验收在 STATUS。
本地检查过程中用 `/tmp/mulyi-doc-audit-before.json` 保存 336 个相关源码/文档文件哈希及初始 status，
只用于本次保护核对，不把临时文件当作长期复现证据；资产/权重未做全量哈希。

### 交付检查结果

- 本次写入范围只有 `AGENTS.md`、`README.md`、`EXPERIMENTS.md`、`docs/PROJECT_MAP.md`、
  `docs/RUNBOOK.md`、`docs/STATUS.md`。AGENTS 为 111 行，原 GitNexus 块逐字保留；README 原文也逐字保留。
- 六份文件内 77 个本地 Markdown 链接/锚点均能对应；任务名与 4 个注册文件中 9 个 ID 对应。
  运行手册 10 条项目脚本调用的参数与本地 argparse / 共用参数模块对应；
  `--headless/--device` 仅确认交由 AppLauncher 提供，未验证本机框架版本。
- `git diff --check` 通过；新增 Markdown 另检查空白、代码围栏、链接与常见凭据模式，无发现。
  代码路径/关键类函数已静态核对并修正文档中的误写；不把静态核对标成运行通过。
- 336 个初始相关文件中，仅 README 因获准追加导航而变化；其余哈希一致，HEAD 未变，
  初始 Git status 项没有消失。未执行 reset/clean/checkout、安装、进程操作、提交或推送。
- 会话期间 status 另出现 `scripts/diagnostics/trace_rotate_bulb.py` 未跟踪项；不是本次创建，
  未读取/修改或纳入命令验收，保留给该工作的执行者。不能把结束时整份 git diff 都归因于本次。

### 开始时已有未提交改动

以下为本次开始时的 `git status --short`，目录项不等于文件总数；不代表这些改动已获审查或由本次产生。

```text
 M README.md
 M assets/d_peg_insertion/tools/validate_flexiv_pregrasp.py
 M docs/d_peg_insertion.md
 M docs/tactile_representation/repose_policy_integration.md
 M scripts/diagnostics/validate_repose_training.py
 M scripts/rsl_rl/bulb_diagnostics.py
 M scripts/rsl_rl/bulb_training.py
 M scripts/rsl_rl/play.py
 M scripts/rsl_rl/repose_training.py
 M scripts/rsl_rl/validate_bulb_revision.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/brainco/agents/visuotactile_rsl_rl_ppo_cfg.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/brainco/brainco_hand_visuotactile_env_cfg.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/visuotactile_inhand_manipulation_env.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/InsertDPegCustom/README.md
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/InsertDPegCustom/agents/rsl_rl_ppo_cfg.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/InsertDPegCustom/env_cfg.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/Revo3/agents/rsl_rl_ppo_cfg_insert_d_peg.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/Revo3/dexsuite_revo3_env_cfg_insert_d_peg.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/README.md
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/actions.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/assembly_env_cfg.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/env_cfg.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/d_peg_insertion.py
 M source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/d_peg_tactile.py
 M tests/test_bulb_bounded_actions.py
 M tests/test_bulb_bounded_policy.py
 M tests/test_d_peg_insertion_rewards.py
 M tests/test_d_peg_task_integration.py
 M tests/test_d_peg_v3_rewards.py
 M tests/test_d_peg_v4_rewards.py
 M tests/test_direct_repose_visuotactile_task.py
 M tests/test_repose_training.py
 M tests/test_rotate_bulb_custom_any_contact.py
 M tests/test_rotate_bulb_custom_failure.py
 M tests/test_rotate_bulb_custom_turns.py
 M tests/test_rotate_bulb_grasp_reset.py
 M tests/test_rotate_bulb_pure_collision.py
 M tests/test_rotate_bulb_reward_balance.py
 M tests/test_rotate_bulb_rewards.py
?? assets/d_peg_insertion/validation/depth_success_model_800_20260919/
?? assets/d_peg_insertion/validation/depth_success_resume_20260919/
?? docs/d_peg_depth_success_evaluation_2026-09-19.md
?? docs/d_peg_depth_success_resume_2026-09-19.md
?? docs/play_video.md
?? docs/tactile_representation/bulb_assembly_guidance_2026-09-19.md
?? docs/tactile_representation/rotate_bulb_custom_rewards_2026-09-19.md
?? recordings/
?? scripts/rsl_rl/evaluate_d_peg.py
?? scripts/rsl_rl/play_camera.py
?? source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbCustom/assembly_guidance.py
?? tests/test_bulb_assembly_guidance.py
?? tests/test_d_peg_depth_completion.py
?? tests/test_d_peg_unrestricted_control.py
?? tests/test_direct_repose_tactile_switch.py
?? tests/test_play_camera.py
```

## BULB-PHASE-ACTION-20260920-01

用户在分析方案后授权最小改动：释放前限制机械臂的大幅增量，释放后鼓励手指平滑控制。
仅本地修改 Custom `env_cfg.py` 中两个动作代价函数及旋出奖励参数，未连接远端或启动仿真/训练。

- 未释放时臂动作幅度代价系数从 −0.025 提高到 −0.25，释放后回到 −0.025；
  手指幅度仍为 −0.025。释放后手指动作变化代价从 −0.01 提高到 −0.02；
  释放前及机械臂平滑代价仍为 −0.01。各组使用平方均值，再由 RewardManager 乘实际 dt。
- 只在 Custom 旋出训练/PLAY 配置启用倍率 10/2；共享函数默认 1，独立装配配置及标准旋出保持原值。
  阶段读取已有 `_bulb_released`，不新增状态。动作 v3、归一化、探索分布、30 Hz 默认控制频率、
  资产、初态、物理、接触门控、成功与失败判据保持；所有改动基于 HEAD `4707074` 加用户既有改动。
- 新增 `tests/test_rotate_bulb_phase_action_penalties.py`，14 项 CPU 测试验证按名称的动作分组、
  混合阶段、释放/局部 reset、持续大动作、手指稳定闭合、倍率回退、非法参数和装配声明兼容；
  30/60/120 Hz 检查相同动作数值的积分预算，不代表物理轨迹跨频率等价。
  修正旧接触辅助预算测试，使调用传入真实配置 params，满幅反向动作在未释放阶段合计 −0.355/s。
- 本地 `/home/liuxinyu/miniconda3/envs/brainco/bin/python` 离线回归 **321 passed, 3 skipped in 11.42s**；
  范围包括新增阶段测试、bounded actions/policy、标准/Custom 奖励、turns/failure/软抓持/接触/抓姿、
  资产、纯碰撞、装配 guidance/contact/state 和任务注册。两项缺原生 Isaac 导出器，一项缺 `pxr.Usd`。
  日志 [pytest.log](logs/diagnostics/bulb_phase_action_20260920/pytest.log)。
- GitNexus 仓库绑定为当前 `mulyi-model-tac`，索引 commit 匹配并在修改前刷新；两动作函数仍返回
  UNKNOWN/not found，奖励配置初查 CRITICAL、刷新后 HIGH，已告知用户并补充源码引用核查。
  图中测试/文档出现异常跨模块流程，未当作完整依赖证明。原始影响分析及修改前源码保存在
  `logs/diagnostics/bulb_phase_action_20260920/`。无 Git 提交；已有其他装配改动不属于本次修改。
  `detect-changes --scope all` 包含全部既有改动并报 critical；CLI 展示省略条目，提高 limit 重跑仍不能
  作为本次变更的独立无影响证明。本次以修改前快照做 AST 比较，生产定义仅变化两个动作代价函数及
  Custom 旋出奖励配置类，补以以上行为回归；没有宣称图检查干净。
- 当前倍率为待验证初值；未测训练后的接触保持或独立成功数/总数，不声称已解决脱离。
  数值预算、Hydra 参数和恢复旧代价的方法见
  [奖励说明](docs/tactile_representation/rotate_bulb_custom_rewards_2026-09-20.md)。

## BULB-VIDEO-20260920-01

用户指定 `Rotate-Bulb_Custom_Dexonomy_GPU4_20260920` 最新 checkpoint，要求服务器录制并下载。
2026-09-20 10:30:40 UTC 选择最高已保存迭代 `model_300.pt`：
`/home/xinyu/training_runs/20260920_092213_rotate_bulb_custom_dexonomy_gpu4/train/model_300.pt`，
SHA256 `15a7ba15e21eb0ac7cc9aaab83edbb9dc127b56d450a3c2a54cb380ce567e83a`。

- 服务器输出 `/home/xinyu/recordings/rotate_bulb_dexonomy_latest_20260920_103040`；
  原 run `source_snapshot.tar.gz` 中源码恢复到独立 workspace，checkpoint、params、契约复制后回放。
  26 个资产哈希、52 个 RSL-RL fork 源文件哈希与原记录一致；未修改运行代码。
- 使用原任务 `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Custom-v0`、guided 物理、Dexonomy 抓姿、
  60 Hz、669 维观测、28 维 v3 动作，编码器关闭；seed 42、GPU 3、单环境、auto 相机。
  保留原训练覆盖，仅为单环境录像缩小 PhysX 缓冲区；完整参数及源码哈希见
  [manifest.json](recordings/xinyu21_rotate_bulb_dexonomy_20260920_103040/manifest.json)
  和同目录 `source_hashes.json`。回放生成的动作/观测契约与原 run 相同。
- 10:34:35 UTC 回放正常退出（exit_code=0）。请求 1800 控制步，实际 MP4 为 **1799 帧、
  29.9833 秒、60 FPS、1280×960**；精确 1800 帧断言未通过，随后 OpenCV/FFmpeg 独立解码均确认
  1799 帧且 FFmpeg 无错误，保留实际结果，未补帧。已查看初始预览和第 600 帧，画面有效。
- 本地 [视频](recordings/xinyu21_rotate_bulb_dexonomy_20260920_103040/rotate_bulb_dexonomy_model_300.mp4)
  大小 1,958,211 字节，SHA256 `3f64e1ceb9f4015d8212091d28022c4ca520976b8b0cc5b6e6b0b1254f24d3c5`，
  与服务器相同；[验证结果](recordings/xinyu21_rotate_bulb_dexonomy_20260920_103040/validation.json)
  及 `server_verification.json`、`play.log`、相机参数和抽帧同目录保留。
- 原 checkpoint 哈希未变；录制后 GPU 4 原训练 PID 275257 仍在运行，状态记录迭代 368。
  没有执行独立成功率评测，录像不代表模型达标或最佳。
- 交接文档修改前 GitNexus impact：STATUS 消歧后 LOW、4 个直接依赖；EXPERIMENTS MEDIUM、
  5 个直接依赖（README/PROJECT_MAP/RUNBOOK/STATUS/奖励专题）。图中的执行流程计数不稳定，
  补充文本引用核查确认本次仅追加文档记录，未修改业务函数或提交代码。

## ASM-VIDEO-20260920-01

用户指定 TensorBoard run `Bulb-Assembly_v3_GPU5_20260920` 最新 checkpoint，要求服务器录制并拉回本地。
2026-09-20 10:05:46 UTC 选择时最新已保存模型为
`/home/xinyu/training_runs/20260920_041251_task_revision_restart/bulb_assembly_v3_gpu5/train/model_1100.pt`，
SHA256 `2f3e214ef9343b07381c7938f9cd9c68258e17b4138f4a2d379005f26e1c7275`。

- 服务器输出：`/home/xinyu/recordings/bulb_assembly_latest_20260920_100546`；
  独立 workspace 核验原 run 的 205 个源码快照哈希，恢复 184 个 scripts/source 文件。
  使用原 60 Hz、固定桌面初态、纯碰撞、v3 动作、669 维观测、关闭编码器；未使用后续 30 Hz/随机 reset 修改。
- 单环境、seed 43、GPU 1、1200 控制步、auto 相机、1280×960；沿用已验证的单环境 PhysX 缓冲区缩减，
  完整命令、源码快照路径、checkpoint 来源及退出码见本地 [manifest.json](recordings/xinyu21_bulb_assembly_20260920_100546/manifest.json)。
  checkpoint 与契约/params 复制到独立目录后回放；原 checkpoint 前后哈希一致，原装配训练继续运行。
- 10:09:23 UTC 正常退出（exit_code=0）；视频为 **20 秒、60 FPS、1200 帧**，
  本地完整解码 1200/1200，SHA256 `30501031bc49f29f06afbb8c0f35424ef2eb54b6cb8ed7b989d9a64174cc4575` 与服务器一致。
- 本地 [MP4](recordings/xinyu21_bulb_assembly_20260920_100546/bulb_assembly_v3_model_1100.mp4)、
  [校验](recordings/xinyu21_bulb_assembly_20260920_100546/validation.json)、同目录 `play.log`、初始预览和抽帧保留。
  已查看预览及第 600 帧，画面有效；未执行独立成功率评测，不据视频认定模型达标。

## ASM-ANALYSIS-20260920-01

用户询问 position_tracking 权重较大为何录像没有明显抬起。只读检查服务器保存配置、原 run 源码快照、
TensorBoard 标量和实际 RewardManager/PPO 实现，未启动新仿真、未改训练配置、未停止作业。

- 对象：`Bulb-Assembly_v3_GPU5_20260920`，训练 PID 242934，采集时约迭代 1169；
  原配置 6144 环境、60 Hz、20 s 回合、位置权重 5、position_std=0.2 m。
- 原始标量：[tensorboard_snapshot.json](logs/diagnostics/assembly_reward_analysis_20260920/tensorboard_snapshot.json)。
  各序列最后 100 个记录点均值：position_tracking=1.373584、good_finger_contact=1.572809、
  any_finger_contact=0.643541、hand_shell_potential=0.715978、thread_downward=0.542732；
  insertion_tracking=0、success=0；Train/mean_reward=100.7139、mean_episode_length=1125.827 步。
  time_out=0.895674、bulb_dropped=0.029189；终止统计可能重叠，不能当独立评测比例。
- RewardManager 实际记录 sum(weight*term*dt)/max_episode_length_s；曲线不是位置误差或策略梯度。
  PPO 对采样动作的优势/概率比更新，使用优势归一化，不对奖励经 PhysX 反传位置梯度。
- [静态数值比较](logs/diagnostics/assembly_reward_analysis_20260920/numerical_comparison.json)：
  从保存桌面高度 -0.00969 m 与初始 root=0.022331 m 推算，上方目标 root=0.105031 m，
  横向偏差 0.18 m、高度差 0.082700 m。保持门控 g=1 时加权奖励率原位 1.212240、
  抬高 1 cm 为 1.255085、横移靠近 1 cm 为 1.311895；抬高 1 cm 但 g=0.8 时为 1.004068。
  这是由公式计算的假设状态，不是测得的 rollout；未考虑手指协调/动力学。
- 首要假设：桌面接触即可获得双侧接触及位置持续奖励；位置门控只看接触力模长，不证明力闭合或脱离桌面支撑。
  初期竖直收益小，抓持丢失可能使收益转负；插入奖励在严格 hover 达标前为零。
  长回合接触收益与高超时日志相符，但尚无同步高度/桌面力/分指力/门控轨迹确诊。
- bulb_dropped 需要历史 lifted 标志，非零日志表明部分训练样本跨过过 root 高于桌面 6 cm 的标记；
  不能由 model_1100 单段确定性视频推断所有带探索噪声的训练样本都从未抬起或都无物理抬起能力。
- PPO 的 KL 最近均值约 0.01698、clip fraction 0.22138、latent noise std 0.39511，
  未见足以单独认定优化器停止/数值失效的证据；1 rad 动作 scale/60 Hz 的协调难度仍是待测因素。
- 建议先补高度/支撑力/接触门控轨迹，再验证分阶段高度进展与桌面持续奖励的竞争；建议尚未实施，
  未改变碰撞、成功判据或困难样本。

## ASM-LIFT-20260920-01

用户确认实施首次抬起的小改动，本次只改本地桌面装配奖励，未连接服务器、同步或重启训练。
在已有 30 Hz、桌面 XY/yaw 随机 reset 的工作区上增加 `assembly_lift.py` 和装配配置接线。
源码 AST 对比确认原有全部奖励定义未变：位置/插入各 5，成功 20；动作/观测/物理/成功规则不变。

- 新项：抓持软门控乘离桌高度进度的历史最佳增量，每回合最多 8；有效抓持抬高 3 cm、
  桌面接触力 < 0.1 N 持续 0.3 s 后一次 +3。返回值除以控制周期，配置 weight=1。
  高度由实时姿态下启用碰撞网格最低点相对桌面求出，减去每回合实际 reset 基准。
  求线性最小值只需原网格凸包顶点，不改变真实碰撞几何；新奖励状态随 RewardManager 局部 reset。
- GitNexus：装配环境配置 LOW（PLAY 直接依赖）；奖励配置 UNKNOWN，补充引用确认由装配配置实例化。
  reset impact 报 CRITICAL 并含异常跨任务边，已提示且本次未修改 reset。
  文档 STATUS LOW、EXPERIMENTS MEDIUM、地图/专题消歧后 LOW；索引仍有异常关系，不能宣称排除所有影响。
  结果保存在 `logs/diagnostics/assembly_lift_20260920/*impact*.txt`；本机修改前无活动训练/回放进程。
- 离线 **55 passed in 2.82s**，brainco Python 3.11；真实 USD 检查通过（未安装新依赖）：
  `test_bulb_assembly_lift.py`、`test_bulb_assembly_guidance.py`、`test_bulb_assembly_state.py`、
  `test_bulb_assembly_contact_rewards.py`、`test_bulb_assembly_reset.py`。
  覆盖 30/60/120 Hz 奖励预算、25 Hz 非整保持步数、局部 reset、弱接触/抛起无抓持、保持中断、
  传感器缺失/非有限值、倾倒伪高度、坐标变换和真实 USD 碰撞网格最低点。
- Isaac 本机 GPU 0、2 环境、40 步、seed 42、纯碰撞、30 Hz：退出码 0，**20/20 探针检查通过**，
  RewardManager 加载 14 项，含新两项 weight=1。零动作 rollout 有限、桌面稳定、原有部分 reset 检查通过。
  [报告](logs/diagnostics/assembly_lift_20260920/probe/report.json)、
  [日志](logs/diagnostics/assembly_lift_20260920/smoke.log) 与逐步记录保留。
  探针不检查新项的实际抓持抬起轨迹；新项预算与局部 reset 的覆盖来自上述离线测试。
  日志仍有原有触觉网格正则路径警告，本次没有把这些警告消除。
- 未验证新策略学习效果；0.1 N 是可配置初值，未做动态标定。后续应同时观察离桌高度、
  分指力/门控、桌面支撑力、达到 3 cm 的比例与独立完整装配成功数/总数。
  此项记录修订上一条“建议尚未实施”的状态，上一条分析证据原样保留。

实际复现命令（仓库根目录；USD_RUNTIME 为本机现有 USD 扩展，异机需换成实际路径）：

```bash
USD_RUNTIME=/home/liuxinyu/miniconda3/envs/brainco/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
PYTHONPATH="$USD_RUNTIME" LD_LIBRARY_PATH="$USD_RUNTIME/bin:$USD_RUNTIME/bin/deps:/home/liuxinyu/miniconda3/envs/brainco/lib" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuxinyu/miniconda3/envs/brainco/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_bulb_assembly_lift.py tests/test_bulb_assembly_guidance.py \
  tests/test_bulb_assembly_state.py tests/test_bulb_assembly_contact_rewards.py tests/test_bulb_assembly_reset.py

PYTHONDONTWRITEBYTECODE=1 /home/liuxinyu/miniconda3/envs/brainco/bin/python \
  scripts/rsl_rl/probe_bulb_assembly.py --out logs/diagnostics/assembly_lift_20260920/probe \
  --num_envs 2 --steps 40 --seed 42 --headless --device cuda:0
```

再次执行仿真时必须另选输出目录以保留这次证据。


## ASM-RESTART-20260920-01

用户授权连接 xinyu21、同步本地项目改动并重启灯泡装配训练。本地/远端均为
main / `4707074194091cd723ebd18dea7824c184e99c6d` 加未提交改动；本次未改生产源码、未提交。

- 同步至 `/home/xinyu/mulyi-model-tac`：83 个源码/配置/文档/测试/小型资产证据文件，
  12 个新增或更新、71 个已一致，SHA256 83/83 一致；未传录像、未删除远端文件。
- 新 run：`/home/xinyu/training_runs/20260920_105032_bulb_assembly_lift_30hz_gpu5`。
  下文相对路径均属于此 run：`sync_backup/` 保留被替换文件，`sync_evidence/` 保存
  manifest、同步前哈希、远端到本地的源码差异及逐文件验证；`source_snapshot/` 和
  `source_manifest.json` 保存 206 个 scripts/source 文件；`runtime_versions.json`
  记录 Python、依赖版本及 52 个外部 RSL-RL fork Python 文件哈希。
- GitNexus 绑定当前本地工作区，索引 commit 与 HEAD 一致；装配配置及 AssemblyLiftReward LOW，
  奖励配置 UNKNOWN；共享动作惩罚及 reset CRITICAL，含异常跨任务边。已向用户提示，
  补查实际引用/继承与两端源码差异，确认装配惩罚仍使用默认倍率 1。未以 UNKNOWN 当作无影响。
- 本地/远端相同九个定向测试：各 106 passed、4 skipped；2 项缺 pxr.Usd、2 项缺原生 Isaac 导出器。
  远端使用已有 USD 扩展补查 `tests/test_bulb_assembly_lift.py` 为 24/24 passed（含两项真实 USD）。
  汇总在 `test_results.json`。未安装/升级依赖。
- 旧 run：`/home/xinyu/training_runs/20260920_041251_task_revision_restart/bulb_assembly_v3_gpu5`。
  核对监督/训练 PID 242931/242934 后向监督发送 SIGINT；正常停止于迭代 1262，exit_code=0。
  `train/model_1200.pt` 停止前后 SHA256 均为
  `677ce76546924fd4a8e36e88fb8dc63d0d6f4b45084f3405e177fc3b61cc35ef`。
  停止记录为 `stop_record.json`；旧日志、配置和检查点保留。
- 新训练启动于 2026-09-20 10:53:47 UTC（18:53:47 Asia/Shanghai），训练 PID 325605、监督 PID 325604。
  沿用 GPU 5、6144 环境、seed 43、15000 次迭代、原 CPU 绑定、纯碰撞、关闭编码器及原 PhysX 容量。
  旧控制契约为 60 Hz，新默认 30 Hz 不兼容，因此从头启动、`resume=false`；无父检查点加载。
  现有监督只管理一个训练子进程，不自动重启，不启动历史自主训练循环。
- 完整命令/环境在 `launch_spec.json`。采用当前默认桌面 XY ±0.05 m/yaw ±π/2、
  抬起最佳进展最多 8 分及稳定离桌一次 3 分；WarpSDF 更新周期由旧 1/60 改为 1/30 s，
  与控制步对齐。14 项奖励中位置/插入权重各 5、动作惩罚 -0.02/-0.01、成功奖励 20。
- 已观察 PPO 迭代 0–4、`train/model_0.pt` 保存，669 维观测、动作契约 v3、decimation=8。
  首模型 SHA256 `46d9d710383c4ae8aeb9a5e9cbd4d87ba47b2aa17f89766ce10c96b64fbc34da`。
  最终持续运行与配置核对保存在 `startup_verification.json`；这不是独立成功率评测。
- TensorBoard 新增 `Bulb-Assembly_Lift_30Hz_GPU5_20260920`，旧标签保持原路径。
  旋出 PID 275257、D-Peg PID 242935 继续运行；其源码虽已同步，既有进程没有热加载或重启。
- 本地证据副本：[assembly_restart_20260920_105032](logs/diagnostics/assembly_restart_20260920_105032/)。
  尚未验证新策略抬起改善、0.1 N 阈值的轨迹适用性或完整装配成功数/总数。


## ASM-VIDEO-20260920-02

用户要求用刚重启的装配训练最新 checkpoint 在服务器录制并拉回本地。
2026-09-20 12:20:20 UTC 选择时最新保存模型为
`/home/xinyu/training_runs/20260920_105032_bulb_assembly_lift_30hz_gpu5/train/model_200.pt`，
SHA256 `872d0f994efe1691f10942124d17b551a0a6845ce09c0912e10ddfb108071c82`。

- 服务器输出 `/home/xinyu/recordings/bulb_assembly_lift_latest_20260920_122020`；
  将训练 run 的 206 文件源码快照逐一校验并复制到独立 workspace，模型及 params/契约复制到
  独立 checkpoint 目录。回放导出写入该副本，原 checkpoint 前后哈希一致。
- GPU 2 单环境、seed 43、30 Hz、1200 控制步、1280×960、auto 相机；采用原训练命令覆盖，
  保留纯碰撞、随机桌面初态、抬起奖励及 v3/669 维契约，仅将 PhysX 缓冲容量缩至已验证单环境规模。
  录制未修改生产源码、未重启任何训练。完整命令和环境见本地 manifest。
- 12:30:13 UTC 正常退出，exit_code=0；视频 40 秒、30 FPS、1200 帧，1200/1200 完整解码。
  两端 MP4 SHA256 均为 `d26b8ea8e1ad9e11406442a0821fa1cf72d7564f9af969205eb09485c16d78dd`。
  完成核验时 GPU 5 原装配训练 PID 325605 仍运行，迭代 247。
- 本地 [MP4](recordings/xinyu21_assembly_lift_20260920_122020/bulb_assembly_model_200.mp4)、
  [manifest](recordings/xinyu21_assembly_lift_20260920_122020/manifest.json)、
  [解码验证](recordings/xinyu21_assembly_lift_20260920_122020/validation.json)、
  [服务器校验](recordings/xinyu21_assembly_lift_20260920_122020/server_verification.json) 及 play.log 保留。
  已查看初始预览与第 300/900 帧，画面有效；不据抽帧或本视频认定独立装配成功率。


## ASM-ZERO-HAND-20260920-01

用户要求把装配手部初始关节角全部设为 0 并重启训练，同时确认动作/探索没有过度约束。
按“手部”处理为 21 个 right_* 关节；7 个机械臂关节和基座位置/朝向保留。

- 生产改动仅 `RotateBulbCustom/assembly_env_cfg.py`：父类初始化后新建 joint_pos 字典，
  将 right_* 值置零；现有 reset_robot_to_grasp 从 default_joint_pos 恢复角度及驱动目标。
  没有修改共享参考资产或旋出/D-Peg 配置。GitNexus 装配类及 __post_init__ 为 LOW，
  直接依赖 PLAY；方法查询注明两处动态调用无法解析，已补查继承及 reset 引用。
- 本地九个相关 CPU 测试：100 passed、4 skipped（2 项 pxr.Usd、2 项原生 Isaac 导出器）。
  服务器 GPU 2 独立 2 环境/40 步实测：15/15 检查通过，覆盖 startup/full/partial reset 的
  手部零位、机械臂原值、驱动目标，非 reset 槽位的角度和目标保持；PLAY 零位及旋出参考未变。
  此为初始化/有限性验证，不是抓取或装配成功率评测。
- 新 run `/home/xinyu/training_runs/20260920_124630_bulb_assembly_zero_hand_gpu5`，
  20:49:20 Asia/Shanghai 启动（12:49:20 UTC）。训练 PID 360229、监督 PID 360228；
  GPU 5、6144 环境、seed 43、15000 次迭代、30 Hz、纯碰撞、关闭编码器，从头训练。
  继续使用随机桌面 XY ±5 cm/yaw ±90°、抬起奖励和原 PhysX 容量。
- 旧 run `20260920_105032_bulb_assembly_lift_30hz_gpu5` 在迭代 300 正常停止，exit_code=0；
  `train/model_300.pt` 停止前后 SHA256 均为
  `9b3db7dd9e15c933cf597e984c80fc19469cc1d3739cbbe089d7a1c0886f3511`。
  保留原模型/日志；旋出 PID 352175 和 D-Peg PID 242935 继续运行。
- 保存配置中 21 手关节全零、7 臂关节逐项等于原 run。14 项奖励及 policy/algorithm 配置
  与原装配一致：action_l2=-0.02、action_rate_l2=-0.01，均无阶段倍率；没有 hand_far 项。
  按归一化动作 [-1,1] 的理论上界，幅度惩罚 ≤0.02/s、变化惩罚 ≤0.04/s，20s 总上界 1.2。
  这是最坏动作序列预算，不是实际策略惩罚，也不保证其学习影响可忽略。
- 动作使用当前测量角加一次增量，每步各关节 scale=1 rad，再受关节位置限位；子步保持目标。
  policy=BulbActorCritic，std=0.05+softplus(parameter)，初值0.35、无上限；entropy_coef=0.001，
  clip_actions=None。保留 tanh [-1,1]、潜变量均值变换、物理限位、PPO clip_param=0.2、
  max_grad_norm=1.0/desired_kl=0.01，不声称完全无约束。本次未收紧这些设置。
- 已确认 PPO 迭代 0–4 及 `model_0.pt` 保存，首模型 SHA256
  `8b9b202d38c6638548866851f778dc8a920d138414d9ed8c83587a8905cbb20d`。
  TensorBoard 新标签 `Bulb-Assembly_ZeroHand_30Hz_GPU5_20260920`，旧标签保持原 run。
- run 下 `sync_backup/` 保存被替换配置；`sync_evidence/` 保存六行差异及两端哈希；
  `source_snapshot/`、`source_manifest.json` 为 206 文件运行快照，`launch_spec.json` 为完整命令/环境。
  `zero_hand_probe.json`、`verify_zero_hand.py`、`startup_verification.json` 和 `stop_record.json`
  保存实测/配置/停机证据。本地副本：
  [assembly_zero_hand_20260920_124630](logs/diagnostics/assembly_zero_hand_20260920_124630/)。
- 先前逐步动作分析因用户转向本次任务而停止：两个隔离探针缺少 headless viewport 模块，
  未形成有效轨迹；相关进程组 357692/358188 已定向终止，原始日志保留于
  `/home/xinyu/recordings/assembly_action_analysis_20260920_124108`。仅有 TensorBoard 读数，
  不据此认定“手初态是根因”；当前零位更改的学习效果尚待验证。


## BULB-SPEED-20260920-01

- 目标：保留三圈 guided 全任务动力学，提高采样吞吐；每个候选单因素实验。
- 基线：GPU 4 无其他 compute PID，生产源码 445/445 文件哈希一致；1024/3072/6144 各完成 8 轮 PPO，前三个正式同步 profiling 轮与后三个自然计时轮分开。
- 结果：2776.4 / 5631.7 / 8421.2 transitions/s。不能与旧 GPU 2 共享条件直接计算加速。
- 证据：[基线报告](docs/rotate_bulb_speed_baseline.md)、[原始统计](benchmarks/rotate_bulb/baseline_results.json)、[结构分析](docs/rotate_bulb_physics_analysis.md)。
- 当前限制：短吞吐测试不是任务成功评估；后续完整 episode 和候选结果另补，不宣称最终 sim2real 已验证。
- 生产曾暂停：原 run `/home/xinyu/training_runs/20260920_134756_rotate_bulb_segments90_armsteady_6144_gpu4` 的 model_100 保留；已按下方记录重启新v2 run。

- 阶段记录（2026-09-21）：[求解器单因素](docs/rotate_bulb_solver_experiments.md)、
  [碰撞/接触](docs/rotate_bulb_contact_experiments.md)、[NUMA](docs/rotate_bulb_numa_experiment.md) 已完成；
  接触扩大、rest=0、NUMA 调整均放弃，位置 8/4/2 的旋出进度明显下降。
- [30 Hz 触觉](docs/rotate_bulb_tactile_experiments.md)：修正缓存后 pad 位姿误差为 0；pose_only
  与完整参考 pressure 输出及 64 policy/64 zero 指标相同，吞吐差仅 +0.26%，不声称显著加速。
  引入 v2 检查点契约，候选只改 5 个接入文件；原生产尚未同步。
- 15 Hz 降频完整结束：5418.2 transitions/s，对30Hz −0.19%，严格成功0/64，pad滞后22.03mm。
  无速度收益且增加延迟，因此放弃；[原始结果](benchmarks/rotate_bulb/tactile15_results.json)。16/4组合因attachment误差2.34/2.36倍超预定2倍线被拒绝，独立3072吞吐虽达7954.7（+41.2%）仍不采用。
- 24/4组合通过固定的短测筛查和mechanical probe；64policy+64zero完整，policy最大圈数均值1.7704、
  释放3/64、严格成功0/64，attachment峰值6.080mm/0.2035rad；仍不能证明最终sim2real不变。
  [筛查结果](benchmarks/rotate_bulb/combined24_quality_acceptance.json)、[完整数据](benchmarks/rotate_bulb/combined24_quality_results.json)。
  三规模最终速度3005.7/6751.1/9820.0 transitions/s，提升8.3%/19.9%/16.6%；
  原基线复测5613.1（与初次差0.33%），优化3072相对复测仍提升20.3%。
  [最终结果](benchmarks/rotate_bulb/final24_results.json)、[推荐配置](benchmarks/rotate_bulb/recommended_profile.json)。
  已同步5个运行文件并从冻结快照启动新run `20260920_183647_rotate_bulb_physx24_4_pressure30_6144_gpu4`；
  GPU4/6144env/seed42/15000轮，训练459464/监督459463；02:44:29确认迭代0–4和model_0。
  77个保存张量有限、446个冻结文件哈希通过，原model_100未变，其他训练未重启。
  [实际重启验证](benchmarks/rotate_bulb/restart_verification.json)、[总报告](docs/rotate_bulb_speed_optimization.md)。
  此为启动与有限性验证，最终训练成功率及sim2real仍待新策略完成训练后评测。

## ASM-CURRICULUM-IMPL-20260921-03

- 目的：把已设计课程接入实际训练，保留原20秒任务、动作/观测/最终成功。
- 版本：开始于 `4707074194091cd723ebd18dea7824c184e99c6d` 的已有dirty工作区；本次仅修改3个既有Python接线文件，
  新增课程模块/模板/启动评测导出脚本/测试。原装配源文件与开始时哈希一致；非从干净HEAD做基线实验。
- 环境：本机RTX5070、brainco Python3.11、torch2.7cu128、IsaacSim5.1、IsaacLab已安装源码、RSL-RL3.1.1本地fork。
- 证据：`logs/diagnostics/bulb_curriculum_implementation_20260921/`，固定`stage0_probe_v3`、自由`stage0_free_probe`、
  `free_roundtrip`、`ppo_stage0_final`、`ppo_stage0_resume`、`frozen_stage0`；具体输出目录和命令见实现记录。
- 结果：固定/自由各2env160步；恢复诊断2env40步，无邻居污染；PPO2次128transition、续训1次累计192transition；
  冻结seed401共4完整回合，0成功，策略/归一化不变。124测试通过4跳过。
- 保留失败：初次train导入其他checkout未找到任务，已固定本地导入；初版事件参数/探针分组断言失败已修复。
  另有现存`test_custom_tactile_contract_identifies_new_task`的mock缺`env.cfg`失败，原测试与tactile源码未改。
- 限制：无真实成功抓持/hover库，未验证后续学习或三训练seed增益；不能将短PPO或静态状态恢复当任务成功。
  未做远端训练/同步/真机操作；Stage2/3拒绝无合格前驱checkpoint启动。
- 方案差异：当前策略负责接触重建且每步计PPO；80/20为episode概率；FIFO每episode一个快照；单rank；
  Stage1内部难度用训练窗口，跨阶段用3冻结seed评测。见[实现与运行](docs/rotate_bulb_curriculum_implementation.md)。

## ASM-CURRICULUM-SYNC-20260921-04

用户授权服务器文件同步；目标 `/home/xinyu/mulyi-model-tac`。
24个文件同步，63个相关文件哈希一致，服务器CPU测试124通过/4跳过；4个既有训练PID及start_ticks不变。
main/暂存区保留，四个课程提交导入独立分支；未启动新训练或仿真。
覆盖前备份 `/home/xinyu/mulyi-model-tac-sync-backups/bulb_curriculum_sync_20260921_105657/before/`。
详见[同步记录](docs/rotate_bulb_curriculum_sync_2026-09-21.md)，非学习效果实验。

## PEG-VIDEO-20260921-01

用户指定 `Insert-D-Peg_XY10cm_Yaw90_GPU6_20260920` 的3000号检查点，要求服务器录像并传回本地。
实际来源 `/home/xinyu/training_runs/20260920_041251_task_revision_restart/insert_d_peg_xyyaw_gpu6/train/model_3000.pt`，
SHA256 `8585395ea640265259a79df7b7603ae1f3ae8e1233b089653d229c954074aae8`。

- 独立目录复制检查点、params、契约及205项原训练源码，源码前后哈希通过；669维观测契约完全一致。
  使用原任务、seed44、孔XY各±10cm/yaw±90°、240Hz物理/30Hz控制、15秒回合。
  仅缩小单环境PhysX预分配容量，保留原动作、奖励、成功条件与碰撞。
- 原快照未归档完整资产及tacmap依赖，回放引用服务器既有目录；因此不声称所有依赖都已冻结至训练时版本。
  前两次初始化未产出视频，第二次诊断捕获缺失 `tacmap_sensor`；补充依赖目录后第三次成功。
  失败manifest/play.log保留在远端 `attempt1/`、`attempt2/`。诊断包装仅打印退出时异常，项目源码未改。
- GPU7最初核查空闲；录制期间另一会话启动课程PID529622，共享期间未干预它。
  19:53核验回放退出码0，原D-Peg PID242935继续至4662轮，GPU4/5既有训练启动时间未变。
- [MP4](recordings/xinyu21_insert_d_peg_model3000_20260921_114155/insert_d_peg_model_3000.mp4)
  为40秒、30FPS、1280×960、1200帧，OpenCV与FFmpeg完整解码通过。
  1,449,817字节，SHA256 `770fcba85102c813f7611a26969345ed3450ead9157c3788cf69967bcafd57a0`；本地与远端一致。
  桌面副本 `/home/liuxinyu/桌面/D型插销_model_3000_20260921.mp4` 同哈希。
- [manifest](recordings/xinyu21_insert_d_peg_model3000_20260921_114155/manifest.json)、
  [验证](recordings/xinyu21_insert_d_peg_model3000_20260921_114155/validation.json)、
  [服务器核验](recordings/xinyu21_insert_d_peg_model3000_20260921_114155/server_verification.json)及日志/抽帧保留。
  已查看初始及第150/900帧，画面有效，含插销脱手样本；本次未统计独立成功率。
  远端根目录 `/home/xinyu/recordings/insert_d_peg_model3000_20260921_114155/`。
- 文档影响分析：STATUS为LOW、4个直接引用；EXPERIMENTS为MEDIUM、8个直接文档引用。
  图的实验索引执行流程映射含杂项，另核对实际文档引用；本次未修改业务函数或提交Git。


## ASM-CURRICULUM-VIDEO-20260922-01

用户明确选择“各阶段初始场景合集，并标注阶段和目标”。本机创建单环境装配场景，
使用现有资产与命名预抓模板制作8段相机巡览；Stage1细分近/中/原臂零手3个起点。
交付[视频及章节说明](recordings/bulb_curriculum_stages_20260922_171826/README.md)：
48秒、1600×900、24FPS、1152帧、H.264/yuv420p，SHA256
`3382cfaf4ad065fd89eb10060e7b5e9d59875d0bf8156502f1bfbf43f88c6c3e`。

- 验证：17项通过，包含完整解码、所有段落关节/灯泡状态冻结、0控制步、预抓模板匹配、
  原任务零位手、悬停尖端距入口10cm及相关源码哈希不变。
- 边界：本地成功库为空，Stage2/3由现有运动学工具摆出几何示意，全程显著标注；
  未验证接触/动力学/碰撞可行性，未写成功库，未作为策略回放或任务成功证据。
- 版本：当前dirty工作区HEAD `4707074194091cd723ebd18dea7824c184e99c6d`；具体输入哈希、
  摄像机矩阵、展示关节/物体状态记录在同目录scene_manifest.json。
- 操作范围：只新增独立录像脚本和产物、更新交接文档；未改运行代码/资产，未训练、未连接远端。
- 失败保留：首帧RGB未就绪的预览日志及traceback，增加独立脚本输出预热重试后完成。
  Kit异常退出码不作为验收依据，最终以verification.json中的完整解码和状态检查为准。


## ASM-CURRICULUM-REWARD-AUDIT-20260922-01

用户要求判断服务器课程Stage0为何没有接触奖励及是否缺少接近引导。本次只读连接xinyu21，
核对run快照、实际env.yaml、TensorBoard文件/6006标签及完整438574条回合日志。
原接近/接触项权重为0，统一计入curriculum_progress；后者2000轮均非零。
全程原双侧>0.5N接触事件434308/438574，严格稳定抓持与Stage0成功0/438574；
最后10000回合接触10000/10000、成功0/10000，课程累计奖励均值5.900383。
按接触两项合计最多5分推得该窗口接近项平均至少约0.900383分，分项精确值未记录。
训练已完成0–1999轮、65536000转移，于9月22日约03:01（Asia/Shanghai）预算退出code2，
保留model_1999，未晋级，原3个PID已不存在。以上均为训练统计，未做冻结独立评测。

[报告](docs/rotate_bulb_curriculum_reward_audit_2026-09-22.md)区分已证实的日志合并与奖励/严格验收差异，
以及尚需逐步轨迹定位的接触力/滑移/转角中断原因。
证据目录`logs/diagnostics/bulb_curriculum_reward_audit_20260922_1735/`含服务器配置、源码哈希/内容、
标量序列、完整日志汇总及8项一致性检查。五个相关源码文件与本地/远端工作区均一致。
未改远端文件、未训练/仿真/重启；仅在本地保存分析证据和更新文档。

## ASM-CURRICULUM-REWARD-INVENTORY-20260922-02

用户要求提取当前各阶段奖励，以评审设置合理性；本次仅分析当前工作区和已下载的服务器证据。
分支main、HEAD 4707074194091cd723ebd18dea7824c184e99c6d，有既存未提交改动，非干净HEAD快照。
[完整清单](docs/rotate_bulb_curriculum_reward_inventory_2026-09-22.md)含五阶段权重矩阵（另覆盖固定/自由Stage0及Stage1九档）、
每步实际奖励公式、每回合预算/每秒奖励区别、成功条件、恢复验证/初态混合、Full十四项函数与源码入口。

静态提取24项检查全部通过：AST提取原奖励并用轻量对象执行实际configure_curriculum，核对五阶段权重/时限、
Stage3 wrapper、五份已下载服务器源码SHA256、控制dt、公式数值例及文档链接。没有启动Isaac或策略评测。
关键结果：Stage0辅助预算至多7，Stage1/2至多15，无严格阶段完成一次性奖金；
Stage3仅插入项最大5分/秒、动作成本、最终成功20；Full恢复原14项，best_progress权重200在30Hz下预算至多6.6667。
记录接触累计与连续保持不一致、Stage3抬起混合初态缺少搬运引导、持续插入/位置项的停留风险。
其中Stage0已有训练停滞统计佐证；具体计时中断原因及后续阶段实际学习影响仍未知。

仅新增清单和更新STATUS/本索引；未修改运行源码、奖励或判据，未连接远端、训练、评测或重启。

## BULB-RELEASE-DIAGNOSIS-20260925-03

- 用户请求分析是否故意控制不释放；本次仅诊断，不改奖励、物理判据或原训练。
- 当前训练4047–4146轮日志最高进度均值2.969814、最终2.578632、released指标1.9314%、成功0%；日志均值非独立概率。
- 用户最近视频对应model_3700（不是此时最新checkpoint），32环境/seed42/1200控制步；64完整20秒回合释放1/64、成功0/64、全部超时。
- 未释放63回合平均5.394秒到2.959370圈、最后2.737224圈；≥2.9圈仅0.410秒。一次满足双门槛后下一控制步正确解锁。
- 平均总回报12.723735；旋转5.766269/手心3.962177/接触1.653543/抓持0.999658，后段奖励与failure均0。
- 解释：历史最高进度信用不随回退收回、超时无失败扣分、释放后抓持/搬运衔接不足；支持阶段局部最优但未证实因果，不能单独归咎手心项。
- 385源码/52运行库/原模型哈希一致，训练PID/start_ticks保持，GPU1已恢复空闲；主动Finished结束由1200条记录及完成标记验证，未保留进程退出码。
- [完整诊断、命令、限制与改进建议](recordings/bulb_release_diagnosis_20260925/README.md)，[统计](recordings/bulb_release_diagnosis_20260925/summary.json)，[曲线](recordings/bulb_release_diagnosis_20260925/progress_reward.png)。

### BULB-RELEASE-DIAGNOSIS-20260925-03：22:58全精度训练奖励补查

正式run0–4194轮4195日志点：释放后稳定/搬运分别1670/1619点正值；释放抓稳与成功始终0。
最近100轮前两者均值5.456741e−8/7.400595e−7，非零26/18点；前段6项引导均100点非零。
终端0.0000不可视为精确零；日志点数不可称回合数。此前64回合冻结回放后段全0仍仅描述那个样本。
[详细口径与分析](recordings/bulb_release_diagnosis_20260925/guidance_analysis.md)，
[原始统计](recordings/bulb_release_diagnosis_20260925/guidance_audit.json)。仅只读事件，无训练干预。

## BULB-PROGRESS-PENALTY-20260925-04

- 用户授权在本地和服务器项目添加回退与不旋出惩罚；本次未重启或修改冻结训练快照。
- Randomized/PLAY新增两项，其他任务保持：新增最差回退超0.01圈部分−2/圈封顶−2；有效进展阈值0.02圈，
  接近3秒/停滞宽限2秒/渐增1秒，最高−0.3/秒封顶−4。释放/物理失败后不叠加。版本写env.unscrew_progress_penalty。
- 本地brainco及服务器相关CPU套件各134通过；本地原生2环境200控制步、212检查通过，687输入/双项共享状态/reset核验。
  首次原生脚本误把policy61维当完整687维而断言失败，修正为policy61+proprio434+perception192后通过，失败证据保留。
- 冻结model3700旧64回合离线重算：原12.723735、回退0.453556、停滞3.622809、新8.647370；释放后新增扣分0。
  无新策略训练，不能声称成功率改善或已消除局部最优。
- 定向同步6文件至/home/xinyu/mulyi-model-tac；覆盖前env_cfg SHA57a5213f…一致，旧文件备份于
  /home/xinyu/project_updates/bulb_progress_penalty_20260925_1515/backup。同步后全部SHA一致。
- 当前GPU4/6144训练PID852306及装配PID871712/start_ticks保持；旋出运行快照env_cfg哈希仍57a5213f…，
  新项目env_cfg为10dd71b7…；**运行中训练未使用新奖励**。插销仍暂停。
- 本次新增模块与env_cfg局部接入；GitNexus影响LOW/PLAY子类，动态注册经文本补核。服务器无node，
  采用本地绑定图谱和服务器匹配原始SHA核验。全工作区detect_changes含既有84文件/68符号，CRITICAL，非本次独立改动结论；未提交。
- [参数、公式和边界](docs/rotate_bulb_progress_penalty_2026-09-25.md)；
  [同步报告](logs/diagnostics/bulb_progress_penalty_20260925/sync_report.json)、
  [离线重算](logs/diagnostics/bulb_progress_penalty_20260925/trajectory_rescore.json)、
  [原生报告](logs/diagnostics/bulb_progress_penalty_20260925/native_checked/report.json)。

## BULB-PROGRESS-PENALTY-TRAIN-20260925-05

- 用户授权停GPU6装配、以原6144环境开启更新旋出并测耗时；随后9月27日询问成功率。
- 装配7380轮SIGINT停止（exit−2），model7300保留且哈希一致，USER_PAUSED；GPU6空闲后新run从头启动。
- 新run 20260925_152318_rotate_bulb_progress_penalty_6144_gpu6，seed42/max15000/PID1002974；原GPU4 PID852306保持。
- 9月27日10:03当前6553，checkpoint6500；近100轮释放99.917%、最高/最终2.999127/2.999089圈，
  完整成功与释放后抓稳奖励始终0。稳定接触8/100点非零、搬运1/100点非零，均值仅2.59e−8/2.81e−8。
- 最近100轮20.184秒，19.57–20.63；采样20.015、更新0.169、9741 steps/s。早期1–5轮18.584秒，不能替代当前统计。
- 正式配置6144、19奖励项及新惩罚version1、模型/优化器有限并已更新、源码/运行库/原契约一致，TensorBoard已可见。
- [完整记录及口径](docs/rotate_bulb_progress_penalty_training_2026-09-27.md)；
  [训练全精度审计](logs/diagnostics/bulb_progress_penalty_gpu6_20260925/status_audit_20260927.json)、
  [启动核验](logs/diagnostics/bulb_progress_penalty_gpu6_20260925/formal_verification.json)、
  [装配停止审计](logs/diagnostics/bulb_progress_penalty_gpu6_20260925/assembly_stop_audit.json)。
- 非独立成功率评测，没有继续改奖励或重启任何作业。

## BULB-CONTACT-RELEASE-DIAGNOSIS-20260927-08

- 用户要求最新双卡checkpoint录像，并排查released是否被bug卡住。选择时最新model1300，GPU6隔离运行，原GPU4＋7持续训练。
- 从正式run复制并验证390源文件、52运行库、checkpoint及两份契约；奖励1/40/−6、观测、事件、终止及三圈物理一致。
- 录像1环境/seed42/1200步，两完整20秒回合，40秒/1200帧/30FPS/1280×960；服务器和本地完整解码/视频SHA256/桌面副本通过。
- 独立128环境/seed42/1200步确定性回放：释放0/256、成功0/256、超时256/256；平均最高2.87930、最终2.87542圈，拇指接触力全0。
- 没有双门槛已满足而未解锁的样本；角度/轴向归一化差最大3.56e−5。另4环境人工注入机械测试通过门槛、关节关闭、自由落体、碰桌及局部reset，非策略成功样本。
- 每回合接触+15.718，停滞−0.365，回退−0.032，总回报24.983；平均10/20秒2.687/2.875圈。缓慢进展重置停滞计时，单侧接触收益高且后段奖励接不上；支持局部高收益未完成行为，未做因果消融。
- 21:37:37正式训练1351轮，1252–1351原始100点released均值3.2642%、成功/释放抓稳0，23.7418秒/轮。带噪声训练与确定性play、rank0标量与完整回合比例不可当作同口径比较。
- 模型/优化器77张量有限，当前训练日志无匹配致命错误；保留的默认NCCL故障日志来自正式启动前历史实验。原四个PID/start_ticks保持，GPU6结束2MiB。
- [录像与完整诊断](recordings/xinyu21_bulb_contact_transport_20260927_132600/README.md)、[逐回合审计](recordings/xinyu21_bulb_contact_transport_20260927_132600/policy_probe/audit.json)、[机械验证](recordings/xinyu21_bulb_contact_transport_20260927_132600/mechanical/evaluation.json)。本次未改生产代码、奖励或重启训练。


## BULB-PHYSICS-PERF-20260927-10

- 用户授权xinyu21空卡实际物理性能优化；仅使用GPU6。冻结旧接触1/跟踪40的Randomized/Custom guided物理、model1300，保留24/4、奖励、成功判据和20秒回合。
- 版本main/4707074，dirty源码390文件+运行库52文件经manifest核验；Sim5.1.0.0、Lab2.3.2.post1、PhysX扩展107.3.26+107.3.3及二进制SHA留存，SDK子版本未知。
- 6144×32基准12更新：预热0–1、同步诊断2–4、自然5–9、trace在10；自然n=5为22.008040秒/轮、8933.46 transitions/s，采样21.833838/PPO0.174202。非与旧25秒同口径的优化对照。
- CUPTI两控制步定位GPU累计1041.506ms：articulation/约束55.51%、碰撞37.55%、触觉4.29%、报告/力读取0.062%；最大kernel convexTrimeshNarrowphase 290.995ms。外层physics为simulate+fetch_results，非纯solver。CPU cProfile保存，perf原生采样受权限限制。
- 实际每环境45刚体、42 authored joints、64shape（58convex+4mesh+box+sphere），全局地面仅1个；环境过滤与Scene迭代钳制已核实。8环境零动作首子步无报告接触，不外推全部分布。
- 1024固定初态/动作短测：baseline两次9.7784/9.9381秒每32步，物理指标相同；7个候选全部淘汰。关闭自碰撞表面+35.09%却末态接触1024→7；partition4表面+12.12%却约束/穿透劣化；跳过pressure改变观测。所有诊断关闭保留为失败证据。
- baseline/partition4/16固定policy各64完整回合均0成功/0释放；partition4连接峰值1.881→4.134mm、0.03726→0.09668rad；policy/normalizer SHA前后不变。无释放后抓持覆盖，不能以零对零判不劣。
- 新增benchmark_bulb_physics.py、replay_bulb_physics_factors.py、run_bulb_physics_case.py；py_compile通过，新launcher实际运行三项完整评测和初态审计通过。无正式物理优化可采纳，未修改生产物理/资产、未提交。
- GPU6资源1299采样无其他计算进程，结束2MiB/0%；原冻结源码/运行库/model1300保持。期间另一项已记录工作停止原GPU4/7并以接触0.1/model100续训，非本实验操作，结果不代表新run。
- [完整报告/前后表/复现命令](docs/rotate_bulb_physics_performance_2026-09-27.md)、[原始证据与哈希](benchmarks/rotate_bulb/physics_perf_20260927/README.md)。下一最小实验为外螺纹凹形网格的误差受限简化，再做固定fixture质量检查；未执行。

## BULB-EPISODE-TIMING-20260928-02

- 比较对象更正：用户确认是桌面抓取Lift，下面9月13日旧灯泡不回答其问题。Lift源码默认物理120Hz/decimation2/4秒回合，对当前实际240Hz/8/20秒，单轮物理更新4倍、跑满回合物理步10倍；碰撞对象、自碰撞、螺旋约束也不同。两项GPU配对容量为库默认16倍；3GiB碰撞栈按文档属于pinned host，不能直接计入显存。历史灯泡PyTorch reserved2.434GiB对整卡25.965GiB，原生逐项分配未知。旧抓取保存配置/性能日志未取得，未新开GPU实验。[正确对照](docs/lift_vs_bulb_cost_2026-09-28.md)。
- 后续明确对比“更早每轮几秒”：历史final_status/validation/部署payload确认9月13日4.399秒、双卡各512、dt1/240/decimation4。当前各6144/decimation8，每轮环境数12倍、环境×物理子步计数24倍；这是工作量计数，不是同条件GPU成本比。旧版已含臂手/guided灯泡，当前约16589与旧约7300条/秒不可当作同条件提速证明。见报告比较节及historical_comparison.json。
- 用户要求查当前旋出episode及各部分耗时。本次读取xinyu21现有作业/配置/事件，不新开仿真、不改训练。
- 20:38:55采集，GPU4/7各6144；3397–3496连续100点采样23.38040秒、PPO0.32294秒、合计23.70334秒。前100点24.05795秒，未见最新窗口变慢。
- 20秒/600步回合、32步rollout、240Hz/30Hz、有效24/4核实；平均episode586.7484步，超时标量93.6167%、完整成功0。当前并行负载下跑满回合估算444.93秒，非单环境回放实测。
- 重算旧接触1/model1300/6144的同步profiling：物理20.13974秒/84.35%、压力1.18867秒/4.98%、奖励0.95801秒/4.01%；历史GPU约束55.51%、碰撞37.55%。明确不同于当前模型细粒度实测，不混加不同计时口径。
- 当前生产仍24/4，16/4仅独立256环境测试；reset和任务各阶段耗时未知。无新GPU实验、配置改动或重启；保留原四PID/start_ticks。
- [完整报告](docs/rotate_bulb_episode_timing_2026-09-28.md)、[本次原始快照](logs/diagnostics/bulb_episode_timing_20260928_2040/live_snapshot.json)、[重算摘要](logs/diagnostics/bulb_episode_timing_20260928_2040/summary.json)。


## ASM-FLOOR-20261003-01

- 当前授权：重构桌面装配课程；在 dsw-wpmt7kzoh4b53jpmov 的指定仓库同步、测试、训练、调整。非旋出任务。
- 基线：本地 main / ecdb0d1292604ec1651c210551c8fd21b371cfde 加已有用户改动；本任务保护这些改动。
  服务器原路径无Git，旧源码未覆盖；基线与候选位于 `.workspaces/assembly_floor_{baseline,v2}_20261003`。
- 代码：横躺自由物体课程；新接近奖励与关节1软成本；state-only 369维；120Hz/4步候选；批量reset/回合统计。
  原接触、抬起、螺纹啮合、18mm安装／1秒保持、失败口径保留。
- 公开参考：Isaac Lab 官方 Lift 配置／奖励及 NVIDIA 性能手册，链接与采用边界见课程README。
- CPU：`scripts/diagnostics/check_bulb_floor_contract.py` 6项通过；覆盖奖励方向、往返进度、双向关节成本、
  几何净空、晋级窗口和新旧策略输入契约。不是Isaac替代测试。
- 原生：首轮因隔离快照缺ground资产失败，保留 `probe8_120/report.json`，补齐后 `probe8_state120`通过。
  两个256环境候选均通过14项检查（实际步数、无触觉、369维、自由横躺、自碰撞、部分reset等）。
- 同服务器GPU0／256环境／seed42／30Hz控制／30预热+180测量步，小随机动作：

| 冻结场景 | 物理Hz／每动作步数 | 转移/秒 | 设备使用峰值GiB | PyTorch峰值MiB |
| --- | --- | --- | --- | --- |
| 旧v1自由竖直（legacy240_256） | 240／8 | 1117.65 | 4.354 | 293.74 |
| 早期v2横躺无触觉（state240_256） | 240／8 | 1290.74 | 3.143 | 15.06 |
| 同一早期v2（state120_256） | 120／4 | 2174.47 | 3.143 | 15.06 |

- 120/240同候选吞吐比1.685；旧→新组合比1.946／设备内存−27.8%。前者为物理步单因素，
  后者同时改变初态、奖励、传感器、日志和缓冲，不能归因于单项。物理碰撞与求解器保持。
  设备内存来自CUDA free/total采样，包含PhysX/上下文；不是仅PyTorch显存，也不是全程NVML峰值。
- 64环境10次PPO更新完成：`ppo_smoke64/model_9.pt`，130个完整回合成功0。没有跨阶段晋级。
- 后续调整：关闭不用于策略的点云可视化，接近模板指尖距6–7cm（首版约12cm），
  42样本路径通过，joint1 −8.35°至10.32°。最终候选与早期性能候选分开冻结；原生/1024容量/300轮试训/100回合评测尚在推进。
- 重启后`nearer_native2_headless`原生14项通过。`state120_1024_final`的状态检查曾报passed，
  但日志有碰撞栈溢出／丢接触，故该次结果判无效（原始报告保留，不删除失败）。
  已停止自动跟进的`ppo_trial1024`；没有用其恢复或晋级。课程栈从512KiB/环境提高至1MiB/环境，
  1024为1GiB（主机锁页内存），自动上限3GiB，保留更大显式配置。
  新探针会检查原生Kit日志，有限试训有溢出停机守卫；`assembly_floor_v2_capacity_20261003`为扩容冻结副本。
  重查旧v1／早期v2的三份256环境原生日志无此类错误。
- 本地证据：`logs/diagnostics/assembly_floor_v2_20261003/`（before、影响分析、原始摘要、报告、启动脚本）。
  服务器：`/home/admin/workspace/xinyu/mulyi-model-tac/experiments/assembly_floor_v2_20261003/`。
- 未验证：稳定抓起学习效果、第一关节策略绕转分布、120Hz完整旋入接触、完整任务成功率和跨seed收敛。

### ASM-FLOOR-20261003-01 后续：扩容与双卡共同训练

用户追加指定每张卡4096环境、两卡共同训练一个策略。
- 1GiB栈的1024复测`state120_1024_capacity`通过15项检查（含原生日志无错误）；旧512MiB失败保留。
  单卡300轮计划停止，`ppo_trial1024`与`ppo_capacity1024`未完成任何PPO更新，不计作训练结果。
- 扩展`distributed.py`与课程runner：PPO边界合并完整回合和物理成功状态库，统一晋级；
  reset无collective；主卡保存全局状态和每rank计数/RNG，恢复拒绝不同world-size/seed映射。
  共享train.py只为课程分开rank日志，保留现有NCCL梯度/归一化同步和参数审计。
- Gloo两进程检查：晋级窗口、跨rank前序状态及来源、计数/RNG恢复、错误物理契约拒绝、
  空边界不重复入库、两rank同步晋级停机和仅主卡保存均通过。
- GPU `ppo_ddp_smoke64`：两卡各64环境4更新，NCCL cuda:0/cuda:1；首梯度全均值核验通过，
  前3更新各20次梯度同步，参数最大差0。model3含全局16384转移/20回合。
- `ppo_ddp_resume64`从model3恢复2更新：model4全局24576转移/42回合，各rank12288转移，seed42/43。
  两个检查点模型/优化器各77张量全有限，rank计数之和与全局一致。
- model4冻结seed401、16环境、100完整回合：稳定抓持成功0/100、曾抓持17/100、lift及后续0；
  policy_unchanged=true，reset失败0。关节1行程均值13.20°/最大31.51°；相对默认姿态最大偏离40.86°，
  超60°为0/100。只证明短训模型的该分布，不能宣称学会抓持或彻底消除绕行。
- 4096容量换算：1024失败日志最大要求619736352字节，线性外推2.309GiB；候选每卡3.75GiB，
  约62%余量且小于uint32上限。三种配对容量各2²⁴，接触点2²⁴，接触片2²⁰；不是总8192再分配一次。
- 正式作业：`.workspaces/assembly_floor_v2_ddp4096_20261003`冻结源码；
  `experiments/assembly_floor_v2_20261003/ddp4096_curriculum`，两卡各4096/合计8192，120Hz物理/30Hz控制，
  每阶段2000轮上限，三seed各100回合晋级评测。启动于北京时间约17:13；17:19已完成更新0–4、保存model0。正式前三次更新各20次梯度同步，
  参数差0，原生日志无PhysX/CUDA错误。首五轮20.39/16.49/14.87/16.62/20.28秒；
  GPU0/1即时显存24849/24810MiB，仍是Stage0训练，未认定抓持达标。
- `guard_training.py`监测本次日志，发现丢接触/溢出/CUDA错误即停止本次进程树；
  本地真实子/孙进程注入PhysX错误测试返回42并全部退出。它不是无限重启监督循环。
- 原生核查／源码哈希／检查点审计／冻结评测报告在同一实验目录，未删除失败记录，未提交Git。

- 本次汇总：[可核查证据](docs/evidence/assembly_floor_v2_20261003.json)；启动验收`ddp4096_startup_validation.json`。


## ASM-PERF-20261004-05

- 用户请求：排查`dsw-wpmt7kzoh4b53jpmov`训练各部分用时，区分物理步、GPU/CPU同步等原因。
- 对象：`assembly_reward_ddp4096_20261004/train/stage0`，活动rank2417/2418，每卡4096环境。
- 2026-10-04 14:06–14:13 Asia/Shanghai：读取日志、配置、远端实际Isaac/PPO源码；两rank各50秒100Hz非阻塞py-spy（各4994主线程样本、5错误），rank0另做30秒20Hz原生采样（599样本、0错误）。工具独立解包到/tmp，未修改训练环境依赖。
- 40–59轮日志均值采样25.33425秒、学习0.14415秒；TensorBoard墙钟间隔25.95671秒。打印iteration time不含log/save。
- 非阻塞占比：物理90.05/89.95%，reset含回合日志2.88/2.52%，终止1.32/1.48%，奖励1.16/1.10%，观测1.02/0.82%；rank0日志2.10%，rank1归一化同步2.74%。父子分类互斥；秒数为采样折算，不是CUDA函数插桩计时。
- 核心证据：`physics_context.py:565 fetch_results()`；原生502/599样本在PhysX→carb tasking等待。配置32×4=128批物理步/轮/rank；TGS场景下限32/8。未发现课程reset额外物理步。
- 确有回合日志cpu()及归一化item同步，未见D2H主导；CPU限流计数不增，显存稳定。触觉编码器/压力传感器在该课程关闭，接触力传感器保留。
- 局限：原生采样会短暂停顿，迭代69受扰动，不能当自然基准；没有GPU kernel trace/单因素碰撞或solver对照，物理内部耗时未拆分。降频/降迭代的任务质量未验证。
- 训练代码、配置、检查点未改，PID启动身份未变，守卫仍running；未起额外GPU作业。仅补充本次诊断文档，未提交。
- [报告及复现边界](logs/diagnostics/assembly_perf_20261004/report.txt)、[统计](logs/diagnostics/assembly_perf_20261004/summary.json)、[资源监测](logs/diagnostics/assembly_perf_20261004/resources.txt)、[最终健康](logs/diagnostics/assembly_perf_20261004/final_health.txt)。

## ASM-PERF-P3-20261004-06

- 当前授权：同步P3 RTX5090主机并快速验证Stage0性能；用户随后确认第一关为stage0抓持，要求优化后4096环境开始训练。
- 同步目标`/home/brainco1/workspace/xinyu/mulyi-model-tac`，无删除；旧文件保存在相邻`assembly_perf_support_20261004/sync_backup`。
- 环境：RTX5090 32GiB、Core Ultra9 285K、驱动595.91.07、Sim5.1.0、Lab2.3.2.post1、torch2.7+cu128、定制rsl-rl3.1.1；复用隔离环境。
- 256环境固定小随机动作，60预热+300测量，独立action seed20261004；原32/8、16/2、8/1、4/1分别123.24/87.05/68.76/61.92ms，4/1为1.99倍。
- 4/1同时改scene位置和速度min/max；原生USD读回一致。保留120Hz物理/30Hz控制，原自碰撞、螺纹、桌面、观测369、奖励及成功判据。
- 256环境PPO各24轮、seed42，每组196608转移；8–23轮平均采样+学习6.2758→3.2906秒，1.907倍。完整回合1807/1770，严格成功均0，阶段未晋级。
- 探针4组各15项通过，无原生日志物理错误。原32/8与4/1采样最小间隙−0.225/−0.586mm，抓取事件36/536与19/522；不能据吞吐认定任务质量等效。
- 额外对照：分区4与8基本相同、16略慢；Kit线程4/8/16基本相同。原始每环境64collider含58凸包、3 SDF。
- 壳体凸包保持原螺纹，仅降到56.05ms（在4/1上再约1.10倍），采样穿入约1.14mm，暂不采用。原资产不变。
- GitNexus configure_curriculum风险LOW（调用make_curriculum_env），CurriculumSettings为LOW；未修改它们。probe和文档未索引为UNKNOWN，补充引用核查：probe仅直接CLI调用，文档为README/AGENTS导航；代码只改独立probe。
- 4096同机固定动作：120Hz/32-8为327.230ms/控制步（32步10.471s），120Hz/4-1为199.903ms（6.397s，仅1.637倍），90Hz/4-1为163.414ms（5.229s，2.002倍）。30预热+180测量，均完成所有原生检查；不能等同于PPO或跨机DSW25秒。
- 原/120快速/90快速采样最小桌面间隙−0.260/−0.418/−1.107mm，90Hz峰值设备使用约22.47GiB。90Hz最大采样灯泡速度2.631m/s；这些是离散样本，不能代表全部最大穿透或学习质量。
- 90Hz显式开关`fast_stage0_physics`仅允许Stage0，未开启仍拒绝90/60，Stage1–4拒绝标志；控制30Hz与原成功判据不变。原先核心类impact LOW；validate方法未解析为UNKNOWN，补查configure/runtime/StageScheduler调用；未修改既有任务判据或绕过恢复校验。
- 12项CPU检查（含90Hz显式标志、默认拒绝及后续阶段拒绝）通过；分布式晋级/状态库合并/恢复边界检查通过。共14个原生probe报告passed，无物理错误。
- 120Hz临时训练在14:54:46启动初始化，4096提速不足后于约14:56:56终止其自有进程组602228；无PPO迭代或模型保存。原目录与停止原因完整保留，守卫返回0不能视为训练完成。
- 最终90Hz训练于15:03:27启动，冻结`.workspaces/assembly_stage0_fast90_4096_20261004`，输出`experiments/assembly_stage0_fast90_4096_20261004`，守卫603746、训练603756；从零4096、seed42、2000轮预算，独立Stage0不自动进入stage1。
- 冻结1650文件哈希自校验；其中1644份源码/资产与本地相同，6份为远端安装egg-info元数据。首批0–7轮PPO已完成，model0模型有限、优化器step20，实际90Hz/30Hz、4/1、369维任务配置、保留奖励参数核验通过，GPU快照25318MiB，无物理错误，守卫running。
- 首次验收6762完整回合，严格成功0；启动成功与采样提速不等于任务已学会。当前继续运行至Stage0自身提名停止或2000轮上限，不自动启动Stage1。
- 最终证据：[4096对照](logs/diagnostics/assembly_5090_perf_20261004/large_scene_comparison.json)、[全部14探针](logs/diagnostics/assembly_5090_perf_20261004/all_probes_summary.json)、[启动验收](logs/diagnostics/assembly_5090_perf_20261004/startup_validation.json)、[冻结同步](logs/diagnostics/assembly_5090_perf_20261004/frozen90_sync_verification.json)。完整启动脚本/守卫/快照清单在同目录launch90；原120Hz临时启动记录在远端原experiments目录。
- 证据：[探针汇总](logs/diagnostics/assembly_5090_perf_20261004/probe_summary.json)、[PPO汇总](logs/diagnostics/assembly_5090_perf_20261004/ppo_summary.json)、同目录完整console/report/episodes/params与同步清单。

- ASM-PERF-P3-20261004-06最终健康补记：15:11:25（北京时间）最终健康快照：已完成26轮（0–25），最近5轮采样+学习均值13.924秒；显存25848MiB，守卫running、PID身份匹配、无物理错误。训练累计完整回合27654，严格成功0；为训练计数，不是独立评测。[快照](logs/diagnostics/assembly_5090_perf_20261004/launch90/final_health.json)。

## ASM-CONTACT-P3-20261004-07

- 用户询问当前接触情况；2026-10-04 16:12只读P3 fast90/4096日志与最近4000完整回合，训练running、至296轮。
- 277–296轮均值：no_strict_contact79.015%，strict且速度不合格20.894%，hold_incomplete0.0905%；严格双侧约20.985%，其中约99.6%速度不合格。阈值：两侧>1N、灯泡线速度<0.05m/s且角速度<0.5rad/s；稳定抓持另需5mm/5°锚点误差内连续1s。
- 最近4000回合均曾达到弱双侧>0.5N的grasp事件，stable_grasp及严格成功0/4000；3951超时、47掉落（终止标签可重叠）。
- 严格双侧接触100–119/200–219/277–296轮为16.892%/20.262%/20.985%，改善在放缓。单指奖励0.535495/满额0.7，不能把奖励值直接当成功率。
- slip55.896%、pose_error34.512%是可重叠诊断，不能相加；slip是相对速度标签，不等于位移超过5mm。
- 四份判据/聚合源码本地与远端冻结副本SHA256匹配。未改变训练或判据；尚不能认定90Hz/求解器或奖励为稳定性根因。
- [统计和源码哈希](logs/diagnostics/assembly_5090_contact_20261004/summary.json)。

### ASM-CONTACT-P3-20261004-07：19:05追加只读快照

- 至1120轮、PID603756/guard running、无PhysX/CUDA错误，model1100保存。
- 1101–1120轮严格双侧接触26.3715%（上次20.985%），严格且速度不合格26.236%，条件比值约99.5%；hold_incomplete0.135%，slip62.0905%，pose_error35.259%，后三种原因可重叠。
- 最近4000完整回合弱双侧>0.5N事件4000、stable_grasp/严格成功0；3968超时、31掉落、2越界，终止标签允许重叠。
- 接触覆盖有所改善，但没有稳定保持突破；当前只读检查，不把这一趋势归因为90Hz或某奖励项，不改训练。
- [19:05统计](logs/diagnostics/assembly_5090_contact_20261004/summary_1905.json)。

## BULB-P3-STAGE0-GUI-20261004-01

- 用户要求远端GUI play看当前训练；2026-10-04 19:23确认运行。
- 检查点：fast90_4096本轮model1100，独立副本SHA256与训练源一致，契约严格加载；1环境、90Hz、solver4/1。
- Sim5.1/Storm普通窗口不更新GPU Fabric姿态，单独调用PhysX update_transformations亦无效；两次尝试保留日志、进程均停止。
- 最终独立`play_storm_mirror.py`创建38个机器人视觉体及灯泡视觉副本，移除显示副本物理API，从真实GPU状态刷新USD；只隐藏原物体显示，不改原物理碰撞。
- 原入口GitNexus精确影响LOW、1文件调用者、0流程；修改仅在远端独立play脚本，训练冻结源未改。
- 验证：连续截图手／灯泡姿态变化；前4个30步间隔灯泡位置与无镜像版本逐字一致；步数210、PID613765及训练PID603756均存活。非独立成功率评测；共享GPU回放偏慢。
- 证据：`logs/diagnostics/assembly_5090_play_20261004/`；远端完整目录`/home/brainco1/workspace/xinyu/assembly_play_support/fast90_model1100_gui_20261004_1916`。

## BULB-P3-CONDITIONS-20261004-1931

只读训练1211–1230轮日志均值：严格双侧接触26.832%，接触且速度不合格26.693%，条件占比99.48%；
瞬时grasp_ok且hold未完成0.1395%。pose_error35.5655%、slip62.823%为重叠掩码，不能相加或解释为独立失败回合率。
最近4000完整回合稳定抓持/成功0；3962超时、36掉落、1越界、1丢失抓持。
日志未拆线/角速度、平移/转角；未声称哪个单项主导。当前checkpoint1200。
本地core/runtime/runner与冻结训练源码SHA256一致；未改训练或GUI。
证据`logs/diagnostics/assembly_5090_contact_20261004/summary_1931.json`。

## BULB-P3-REWARD-AUDIT-20261004-1243

核查冻结运行源码：Stage0成功条件均有奖励对应，非未注册；双侧力/相对位姿/抓稳质量各0.5历史最佳增量预算，严格双侧接触超过3步才开放后两项。
抓稳质量乘积含高斯速度因子，速度越大信号迅速趋零；超速惩罚在各阈值2倍处封顶。
1224–1243轮每回合靠近3.494、历史靠近10.946、任意指接触2.733，双侧接触0.494、位姿0.408，抓稳/保持/完成显示为0；速度惩罚-0.248、滑动-0.933。
Episode_Reward已依据远端RewardManager乘回5秒统一口径。支持引导量级/门控不平衡假设，未排除物理及探索原因；未修改运行。
证据`logs/diagnostics/assembly_5090_contact_20261004/reward_audit_1243.md`。
