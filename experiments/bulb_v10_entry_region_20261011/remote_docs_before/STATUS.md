# 2026-10-07 22:26：历史5帧MLP与单帧LSTM同轮数比较

只读远端原始rank0事件，对齐从零v5 MLP和v7 LSTM至1015轮，均2×8192环境/rollout32。
第一阶段100轮均值：700–799为0.4767%/4.5404%，800–899为8.4342%/11.2141%，900–999为49.9439%/19.2279%（MLP/LSTM）。
LSTM较早起量，MLP在900轮后反超；共同末100轮916–1015为56.4421%/21.3613%。
900–999轮平均16.1869/17.3905秒，LSTM约慢7.44%；全0–1015累计采样+学习4.365/4.832小时。
两者所有已读阶段二/三指标最大值仍0；混合阶段倾角不作为二阶段独立对齐效果结论。
两run奖励版本v5/v7不同（朝向门控、二阶段+10），只有一对种子，不能归因架构优劣；未执行独立评测、修改或重启训练。
[完整对齐统计与保存配置](../experiments/bulb_mlp_lstm_comparison_20261007/comparison.json)。

# 2026-10-07：本地v8第二阶段改为螺纹端点浅插

按用户要求，二阶段两项位置奖励改为实际姿态下的螺纹端点到浅插目标距离；目标深度仍5.5 mm，目标轴向相位跟随当前值，不要求先旋转。
晋级改为端点深度5–6 mm、端点径向≤0.5 mm、倾角严格<30°，单控制步达标即进入三阶段；删除二阶段速度和0.2 s保持门槛。
口沿轴线检查保留给三阶段；二阶段一次+10及原奖励权重保持。三阶段三圈/螺距/≤5°/低速/0.2 s/一次+20保持。
课程CPU52/52通过（含USD资产、阶段隔离和奖金回归），未运行Isaac物理、未同步或重启远端v7 LSTM。
这里只验证端点浅插的任务逻辑，不证明斜插物理可达，也未新增“螺纹尚未啮合”的检测。
MDP版本8；旧checkpoint同维度不等于旧评测协议。GitNexus过期/错配，HIGH/CRITICAL与UNKNOWN已补充引用核查，未称图检查干净。
[本次记录](../experiments/bulb_stage2_tip_entry_20261007/README.md)。

# 2026-10-07 20:30：旧v6 model10700远端回放已传回本地

旧v6最后实际保存checkpoint为10700（最终日志10750）。已录32秒/960帧/1280×960/30FPS，完整解码与10文件传输哈希通过，模型哈希未变。
使用原v6代码/运行库，ADR0、checkpoint重力−1、完整三阶段；连续未筛选，原训练源码未改。
阶段二412帧，倾角52.0270°–79.0081°、均值64.7067°，没有≤5°或第三阶段；7个完整回合均失败，第8截断，非独立成功率评测。
首轮wpmt录像显存不足退出后，改在共享同一源码/模型的c3g GPU0完成；失败日志保留。两台原训练均活动，录像进程已退出。
[视频及验证](../recordings/bulb_v6_model10700_20261007_2023_c3g/README.md)。

# 2026-10-07：TactSpace 对比设计审查

结论：适合作为 TactSpace-inspired RGB/Depth/Marker 表示基线起点；尚非完整复现或已完成的公平 sim-to-real 对比。
核心自/交叉重建与 InfoNCE 已保留，但现训练仅 sim 同域，缺可信跨域联合训练和正式下游评测。
现有三模态模型重建使用空间旁路，且受损训练/评价目标与 TactSpace 不同；不能直接比较 total loss 推断 latent 优劣。
建议先统一数据、输入、预算与冻结表示下游评测；迁移主张需另补可信物理配对及真实测试。
本次仅源码/论文/历史记录审查，未运行实验或改运行代码；GitNexus UNKNOWN/错配已保留限制。
[审查证据与建议](../experiments/tactspace_comparison_review_20261007/README.md)。

# 2026-10-07 20:05：当前LSTM倾角未下降，旧v6停止前也有回升

本机6006当前为v7 LSTM，至578轮；近三个100轮279–378/379–478/479–578混合阶段末态倾角为80.2159°/80.4629°/81.0076°。
最近≤5°比例0.124022%（此前100轮0.099460%），进入二阶段仅0.130727%；不能据全阶段均值判断二阶段专属姿态。
另只读核查旧v6原始事件至10750、最后17:20:48；10451–10550/10551–10650/10651–10750倾角59.7696°/61.2759°/61.8443°，停止前已有回升。
v6/v7阶段二完成指标均全0。v7从零LSTM与旧v6续训不是同一条连续学习曲线；没有运行修改或重启。
[角度趋势证据](../experiments/bulb_v7_lstm_dsw_2x8192_20261007/angle_trend_20261007_2005.json)。

# 2026-10-07 20:01：LSTM变慢主要落在采样侧，近期伴随关节超速终止增多

只读核对v5/v6/v7保存配置：环境数/物理频率/求解器/碰撞/动作/PPO更新数保持；v6→v7差异为奖励+10、历史5→1及LSTM，另训练从续训变从零。
旧v6末100轮12.7048秒（采样12.08547/学习0.61952），v7 459–558为17.0629（16.0439/1.01905）。
较接近训练阶段的v5/v7 400–499为14.963/16.2816秒，差1.32秒；成熟v6与从零LSTM不能视为公平消融。
v7 480–499→540–559：采样15.34455→17.96135、学习1.0249→1.0104；rank0异常终止26.378%→35.1715%，
双rank关节超速事件729.75→1072.85次/轮，接触/重置负担变化可疑，尚非因果证明。
collection还含LSTM推理、隐藏态复制、奖励与重置，不能等同纯PhysX耗时；目前无法精确分摊。
50项runtime、8项灯泡资产、55项机器人资产SHA相同。GPU采样P0/2670–2700MHz、RAM充足，无其他GPU训练PID。
未干预训练或新开GPU实验；下一定位应单独计时推理/物理/奖励/reset，不先改奖励判据。
[证据和局限](../experiments/bulb_v7_lstm_dsw_2x8192_20261007/slowdown_audit_20261007.json)。

# 2026-10-07 19:57：旧MLP上升区间及当前LSTM耗时核查

读取指定wpmt目录中的v5完整0–11496、v6续训9500–10750、v7 LSTM0–553原始事件。
v5 stage1_success按100轮均值：500–599为0.1542%，700–799为0.4767%，800–899为8.4342%，
900–999为49.9439%，1000–1099为71.9462%；明显提升在约800轮，900–1000快速上升。
100轮滚动均值首次连续30个窗口保持≥1%/10%/50%的结束轮为824/906/1000。
上述为抓取搬运达标进入二阶段指标；v5/v6浅插、旋入和完整成功全部记录仍为0。
当前LSTM至553，近100轮stage1_success0.10294%，尚未明显上升；网络/奖励不同，旧拐点不是保证。
当前近100轮16.908秒/轮（采样15.88933、学习1.01865），近20轮18.4905，最新19.47秒。
均为rank0日志/重置批次统计，非独立评测；未修改训练或TensorBoard。
[完整统计](../experiments/bulb_v7_lstm_dsw_2x8192_20261007/mlp_lstm_onset_comparison_20261007.json)。

# 2026-10-07 19:53：v7 LSTM至540轮，TensorBoard已接通

指定wpmt DSW双卡各8192作业持续活动，最新checkpoint model500；近100轮平均16.5866秒/轮。
rank0完整0–540共541点：stage2_success、stage3_success、screw_turns、position_success全0。
近100轮进入第二阶段指标0.106419%（最初100轮0.104730%），未见稳定完成第一阶段的证据。
平均reward由最初100轮6.56278到近期8.16275，不能等同学会任务；异常机器人终止约27.4303%，non_finite全0。
两卡显存39939/43571MiB、利用率97/97%；外层日志未见Error/Traceback/OOM，本次未重扫原生日志。
本机 [TensorBoard6006](http://127.0.0.1:6006/#scalars) 已打开并HTTP核验139个标签，run `Bulb_v7_LSTM_2x8192/.`。
远端新TB20216监听loopback6008，本机隧道381517映射6006→6008；旧远端v6 TB6006未改。
指标为rank0训练重置批次按轮均值，非独立评测，rank1无事件标量。训练配置/进程未修改。
[本次统计](../experiments/bulb_v7_lstm_dsw_2x8192_20261007/progress_20261007_1952.json)。

# 2026-10-07 17:29：v7 LSTM双卡各8192已启动并验收

用户最终要求双卡每卡8192、总16384；指定wpmt DSW的独立副本为
`/home/admin/workspace/xinyu/mulyi-model-tac-v7-lstm-20261007`。launcher14243、rank14259/14260。
从零、seed42/43、30000更新，v7完整抓取抬起→浅插→旋入三圈；单帧380维/actor与critic独立LSTM256。
1733项同步校验及DSW CPU LSTM7/7、课程48/48通过；原生0–10共11次PPO更新、17项启动验收通过。
两rank真实all-reduce/首次参数SHA一致，model0与optimizer有限且step20，配置/奖励/物理/容量符合计划。
显存39705/43499MiB（总46068/卡）；外层及两份Kit日志无Error/容量溢出。后期峰值与独立成功率仍未知。
旧v6获授权停止，13份检查点保留，最新model10700停止前后SHA一致；单卡尝试已依用户改双卡要求取消。
旧TensorBoard6006仍指向v6；新训练日志/操作见[双卡记录](../experiments/bulb_v7_lstm_dsw_2x8192_20261007/README.md)。

# 2026-10-07：TactSpace RGB/Depth/Marker 适配模型已接通（CPU 验证）

原表示训练入口新增 `--model-type tactspace`，默认 tri_modal 及原 denoising 流程保持。
独立来源 latent、纯 latent 共享解码、联合重建/双向多正样本 InfoNCE、保存恢复及冻结导出已实现；
固定各分支 L2→有效均值→L2，Marker 保持 `[x0,y0,dx,dy,valid]` 点序列。
12 项 CPU 检查通过，包含原入口/新入口短训、恢复、单模态、梯度、无泄漏、有效性和导出数值一致。
现有 robust/tri_modal best.pt 均只读严格加载通过，输出有限 `[1,256]`。
本地数据目录仅说明文件；短训为合成数据（原模型2步，新模型2+恢复1步，Marker单模态2步），不证明学习效果。
跨域真实配对证据缺失，旧对齐入口对 TactSpace 明确报错；未训练 real 分支禁止导出。
未运行 CUDA/AMP、长期训练、Isaac/PPO 或真机；未改任务环境和已有用户资产整理。
GitNexus 公共路径报 CRITICAL，部分符号 UNKNOWN/异常；已补查引用并进行旧格式回归，未声称图检查干净。
[命令/限制](RUNBOOK.md#81-tactspace-rgbdepthmarker-适配版2026-10-07)、
[验证记录](../EXPERIMENTS.md#tactspace-wiring-20261007)。

# 2026-10-07：5090 TacSL 4096环境训练已启动

用户新授权至少4096环境训练；有效作业为 `reports/train4096_20261007_v3/train`，
主机 `brainco1-ThinkStation-P3-Tower-Gen-2`，systemd用户服务 `tacsl-train4096-20261007-v3`。
4096环境、horizon128、计划1500更新，从零训练原默认14维状态MLP；关闭触觉采集计算，GelSight接触物理保留。
前5次更新/2621440样本通过，8192完整训练回合0成功；模型/optimizer/日志有限，显存约28274MiB。尚未证明学会任务。
v1/v2因碰撞栈溢出失败保留；v3扩容至4293918720字节并加错误guard，未降低碰撞质量或改变判据。
训练完成后自动独立评测1024回合；服务最多24小时。初期吞吐粗估约23小时完成1500更新，非保证。
本机 [TensorBoard 6007](http://127.0.0.1:6007/#scalars)，与其他任务6006分开。
[启动验收、失败记录和操作入口](../experiments/tacsl_train4096_20261007/README.md)。

# 2026-10-07：灯泡教师可选单帧 LSTM（本地）

专用train/play新增 `--lstm`，双卡启动器透传；启用后观测39/149/192共380维、历史1帧，
Actor/Critic各自单层LSTM256→原MLP512/256/128。默认仍为5帧1900维MLP，任务/奖励/动作/物理保持。
任务局部policy适配原探索噪声边界和PPO末步bootstrap；回放每回合重置对应隐藏状态，共享runtime不改。
CPU7/7通过，包含真实循环PPO两轮/40次优化及严格权重往返；旧MLP不能直接加载，新模式需从零训练。
未启动Isaac/GPU、未同步或重启远端，原生/DDP和学习效果未验证。
[使用说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md#2026-10-07可选单帧-lstm)、
[检查日志](../experiments/bulb_teacher_lstm_20261007/cpu_checks.log)。

# 2026-10-07：5090 独立 TacSL 插入移植验证完成

目标主机 `brainco1-ThinkStation-P3-Tower-Gen-2`，工程 `/home/brainco1/workspace/xinyu/tacsl_insertion_lab`，
任务 `TacSL-Task-Insertion-Original-v0`；保留原默认 YAML 的 Franka/GelSight、动作、14 维实际观测、奖励、重置范围和成功阈值。
已完成单环境 260 步、双侧 RGB/深度/法向/剪切接触与释放验证、8 环境 4 次 PPO 更新（8192 样本）。
新种子独立评测 **0/16**，未证明学会插入；代码/数值检查不能证明与旧 Gym 动力学等价。
关键差异：RTX 触觉渲染隐藏两侧壳体可视 Mesh（碰撞完整）、PPO 改用 RSL MLP、原默认之外的 AAC/力观测未移植。
官方资产下载需登录，已下载公开 ManiFeel 镜像并记录哈希/同源 URDF；未取得官方 zip 哈希。
测试作业已退出，现有 Lab 检出及其他任务未修改。[证据、差异与复现入口](../experiments/tacsl_insertion_20261007/README.md)。

# 2026-10-07 16:21：v6已出现位置单项达标，角度单项比例上升

用户确认关注`orientation_success`和`position_success`；本机6006当前run为v6，非c3g实例v5。
完整9500–10480共981个记录点：position_success首次10388非零，10388/10427/10452/10453共4个非零点；
近100轮位置单项按轮均值0.003223%，角度单项0.303642%（此前100轮0.195833%）。stage2_success仍全0。
核对远端v6课程源码：位置和角度分别统计；阶段二位置看插深/径向，对齐还需同时满足倾角、低速及持续保持。
因此是单项条件开始达标/改善，不能把两条曲线非零等同于同一回合同时完成浅插。
统计为rank0混合阶段重置批次，非独立评测。本次未改训练。
[证据](../experiments/bulb_v6_dsw_resume_20261007/component_success_20261007_1621.json)。

# 2026-10-07 15:37：指定c3g实例v5仍运行，第二阶段完成指标为0

本次明确连接`dsw-c3g3e0ieowszisvolv`，不是运行v6的`dsw-wpmt7kzoh4b53jpmov`。
PID948/1009/1010活动，两rank工作目录确认为v5副本；日志推进至10338，最新model10300。
完整原始事件0–10338共10339点，rank0阶段二/三成功、浅插候选/保持、第三阶段时长及旋入圈数全0。
近100轮10239–10338：进入二阶段87.5420%，混合阶段重置末态倾角74.8781°（前100轮76.9542°），≤5°比例0.0760832%。
这说明尚未可靠对齐，不等于任何样本都从未角度达标，也不能仅归因为倾角；浅插还要求深度/对中/低速保持。
rank1无阶段标量，不能将rank0零记录外推为双卡从未浅插；没有独立评测分母。
本次只读远端，未改变或重启训练。[本次统计](../experiments/bulb_v5_dsw_2x8192_20261006/progress_20261007_1537.json)。

# 2026-10-07：本地v7新增二阶段一次+10，远端仍运行v6

已按用户要求在原浅插达标晋级事件新增`stage2_success_bonus`，每回合一次+10；三阶段最终一次+20保持。
原成功判据、v6连续奖励与其他20项原奖励/物理/动作/观测不变。48项CPU检查通过，含dt、失败门控、重复领取、reset与晋级日志锁存。
未重启或修改指定DSW正在运行的v6；本地代码更新不会使活动进程生效，尚未原生验证v7。
原始事件审计到10264，rank0连续765点阶段二/三成功均严格0，且二阶段成功锁存到回合结束、不会因后来掉落丢失。
修正此前表述：rank1无独立阶段日志，不能断言两卡历史上从未浅插成功；两卡汇总的完整任务终态成功率也全0。
[实现和边界](../experiments/bulb_stage2_bonus_20261007/README.md)、[完整原始标量审计](../experiments/bulb_stage2_bonus_20261007/success_audit.json)。

# 2026-10-07 15:28：v6至10238，尚无达标浅插记录；二阶段无独立完成奖金

TensorBoard完整9500–10238共739个记录点：stage2_success、stage3_success、stage3_elapsed_s、旋入圈数及最终奖励全0。
近100轮倾角63.1831°、≤5°比例0.117725%、进入二阶段87.2918%。这是rank0混合阶段重置批次统计，不覆盖每个物理帧或独立成功率。
核对已保存配置与实现：二阶段没有单独的阶段完成奖金，`Episode_Reward/success`是位置/朝向连续奖励；
三阶段首次满足净3圈、深度/对中/倾角/低速连续保持成功时一次+20，同帧物理失败不给。
另有新最佳旋入进度最高9分/圈、季度圈里程碑最高3分/圈（受接触/速率门控），回退增量−14分/圈。
本次只读核查，训练进程及配置保持。[证据](../experiments/bulb_v6_dsw_resume_20261007/progress_20261007_1528.json)。

# 2026-10-07 14:47：v6倾角改善，尚未进入第三阶段

指定DSW双卡作业仍活动，TensorBoard记录至10050轮，本次续训551次更新，最新检查点model10000。
rank0近100轮（9951–10050）混合阶段重置末态指标：进入第二阶段87.8432%，
倾角66.7958°（v6最初100轮74.5026°，前一个100轮68.2360°），中心目标误差7.0383cm。
≤5°比例仅0.08832%（前100轮0.12623%），没有精确对齐比例持续改善证据。
v6全551轮的stage2_success、浅插candidate/保持时长、stage3_success、旋入圈数及最终奖励原始标量全0。
判断：出现粗略朝向改善趋势，但尚未学会可靠重定向/浅插；指标非二阶段专属或独立评测。
显存34405/35689MiB，ADR0、重力−1；外层完整日志无Error/致命/容量溢出匹配，本次未重扫原生日志。
本次仅只读核查，未修改或重启训练；建议先观察后续1000–2000轮的倾角和小角度占比是否继续改善。
[统计证据](../experiments/bulb_v6_dsw_resume_20261007/progress_20261007_1447.json)。

# 2026-10-07 12:48：v6已在指定DSW双卡各8192续训

`dsw-wpmt7kzoh4b53jpmov`独立副本`/home/admin/workspace/xinyu/mulyi-model-tac-v6-20261007`，
1735/1735文件/链接验证通过；旧v1/v5项目及检查点保留。GPU0/1各8192、总16384，seed42/43。
选择同SDF/三阶段物理契约的v5 `model_9500.pt`，保留模型、归一化、优化器及重力课程；
追加20500轮，runner索引9500–29999，每回合仍从完整第一阶段开始。
专用train入口新增可选`--initial_adr_difficulty`，本次显式0匹配父run，避免未保存的ADR回到默认0.7317。
重力恢复−1m/s²，未跳过阶段。PID1446，rank1468/1469；run为副本下
`logs/rsl_rl/bulb_grasp_teacher_v6/2026-10-07_resume9500_2x8192`。

每rank沿用3.75GiB碰撞栈及接触/配对/heap/temp原默认×2，未再翻倍。
CPU44项及ADR参数9个分支情况通过；原生观察9500–9511共12次PPO更新，两rank真实all-reduce且参数SHA一致。
新model9500优化器190020→190040、模型更新且有限、归一化保留、父checkpointSHA未改；
实际配置v6/14s/120Hz物理/30Hz控制、原动作观测及其余19项奖励保持。
显存34403/35687MiB，外层及两份Kit日志无Error/容量溢出匹配；原visual引用警告保留。
只验证启动、续训和初期容量，重定向改善及完整成功率未评测；没有创建长期监督/自动重启。
[运行与证据](../experiments/bulb_v6_dsw_resume_20261007/README.md)、
[启动验收](../experiments/bulb_v6_dsw_resume_20261007/startup_verified.json)。下方“未部署”属于此前本地实施阶段。

# 2026-10-07：本地v6仅调整阶段二success位置奖励

复核保存的v5配置与model8800轨迹：阶段二728帧中642帧中心误差≤1 cm，最小倾角39.11°，无≤30°帧；
中心误差最小2.72 mm时倾角77.66°，支持“位置到位仍可获得较多奖励”的停滞假设，非因果消融。
本地只将阶段二`success`乘`0.2+0.8*(1−θ/π)^2`，θ为螺纹朝下倾角；原权重10/位置尺度0.1不变。
阶段一/三、其余19项奖励、接触门控、物理/动作/观测、时间和成功判据保持。版本6；floor=1可恢复v5奖励数值。
CPU44项通过、原课程函数AST及源snapshot/奖励/旋入/几何模块SHA保持。
未运行原生物理或新训练，未连接/修改远端，学习效果和独立成功率未知；远端最新保存证据仍对应v5。
[设置与轨迹分析](../experiments/bulb_stage2_reward_20261007/analysis.json)、
[20项奖励提取](../experiments/bulb_stage2_reward_20261007/reward_settings.json)、
[实施记录](../EXPERIMENTS.md#bulb-stage2-reward-20261007)。

# 2026-10-07 10:21：v5 model8800录像已下载本地

[32秒视频](../recordings/bulb_v5_20261007_1013/bulb_v5_model_8800.mp4)，1280×960/30FPS，960帧完整解码与7文件SHA通过。
独立副本ADR初值0匹配当前训练，checkpoint重力−1；完整v5/物理/成功口径保留。2.9秒进入阶段二，232/728帧分别为阶段一/二，无第三阶段。
5个完整回合最终成功0，第6回合截断，不作为独立成功率。checkpoint未变，录像进程已退出，原训练继续。
[记录与局限](../recordings/bulb_v5_20261007_1013/README.md)。大倾角HUD Radial为外推轴线误差，不是真实位移。

# 2026-10-07 10:10：v5至8858轮，第二阶段仍未突破

只读核查：PID948/1009/1010活动，最新model8800，近期约12.5352秒/轮。
近100轮rank0训练指标：进入第二阶段87.4108%（昨日83.1108%），混合末态倾角74.3373°（昨日78.6031°），
≤5°比例0.0971001%（昨日0.0777800%），中心目标误差6.9028cm（昨日8.9316cm）。
0–8858全历史原始标量：stage2_success/第三阶段时长/旋入圈数/最佳圈数/完整成功及最终奖励仍全0。
短期倾角此前100轮72.5855°→最近74.3373°，不声称持续收敛；ADR0、重力−1m/s²保持。
GPU显存35921/35653MiB；近期外层日志无致命错误，未全量重扫原生日志。
[统计证据](../experiments/bulb_v5_dsw_2x8192_20261006/progress_20261007_101023.json)。
判断：搬运略有改善，精确朝向/浅插仍未学会；指标为rank0重置批次按轮均值，非阶段二专属或独立评测。未修改训练。

# 2026-10-06 12:39：v5 TensorBoard网页已接通

本机入口 http://127.0.0.1:6006/#scalars ，run为`Bulb_v5_2x8192/.`；HTTP核验已加载138个标量标签。
远端TensorBoard PID26967仅监听127.0.0.1:6006，本机SSH转发PID14143；训练进程未改。
[启动与验证](../experiments/bulb_v5_dsw_2x8192_20261006/tensorboard_verified.json)。连接依赖本机SSH进程和目标服务器存活。

# 2026-10-06 12:37：v5第二阶段奖励与灯泡朝向核查

核对远端保存配置与env_cfg/curriculum源码SHA一致。第二阶段中心位置奖励权重5（双侧指尖接触门控）及
success权重10（纯位置平方tanh，rot_std=null，无朝向/接触门控）；thread_down权重5，为
(1−θ/π)×阶段二标志×拇指和至少一其他指尖均>1N；另有−0.2×(1−cosθ)/2全阶段朝向惩罚。
θ为灯泡本地−X相对世界−Z倾角，绕螺纹轴自转自由。浅插深度/径向/稳定保持是晋级条件，未设独立逐项连续浅插奖励。
近100轮2867–2966：进入阶段二83.1108%，混合阶段末态平均倾角78.603°，≤5°比例0.07778%，阶段二完成0。
不能称已学会可靠朝向；早期0–99倾角80.527°，2400–2499为76.746°，近期没有持续收敛证据。
thread_down奖励上涨同时受接触/阶段占比影响，不能独立证明角度改善。以上为rank0重置批次按轮均值，非阶段二专属或独立评测。
[配置与趋势](../experiments/bulb_v5_dsw_2x8192_20261006/orientation_analysis_20261006.json)、
[原始精度近期指标](../experiments/bulb_v5_dsw_2x8192_20261006/orientation_exact_20261006.json)。未改训练。

# 2026-10-06 12:34：v5已大量进入第二阶段，第三阶段暂无记录

只读核查指定DSW：PID948/1009/1010仍活动，TensorBoard到2952轮，最新检查点model2900。
近100轮rank0重置批次指标按轮平均：stage1_success=83.1967%（进入第二阶段），最新84.1175%；
0–2952全历史未四舍五入标量中stage2_success（进入第三阶段）、stage3_success、screw_turns、
screw_best_turns、stage3_elapsed_s及最终奖励均严格为0。此结论限于已记录的rank0训练指标，非双rank独立评测。
最近100轮日志耗时12.6689s/轮；混合阶段末态倾角78.543°，ADR0，重力仍−1m/s²。
训练继续；没有修改/重启作业，也未放宽浅插判据。完整成功数/总数仍未知。
[本次进度](../experiments/bulb_v5_dsw_2x8192_20261006/progress_20261006_123400.json)、
[TensorBoard全历史阶段核验](../experiments/bulb_v5_dsw_2x8192_20261006/stage_history_20261006_1234.json)。

# 2026-10-06 00:41：灯泡v5已在新DSW双卡每卡8192启动

用户指定`dsw-c3g3e0ieowszisvolv`，并将实例从四卡重启为两卡后授权继续。
独立副本`/home/admin/workspace/xinyu/mulyi-model-tac-v5-20261006`；1729项文件/链接核验一致，旧项目和检查点保留。
GPU0/1各8192环境，总16384；v5完整抓取抬起→浅插→三圈旋入；从零、seed42/43、30000轮。
torchrun PID948，rank PID1009/1010；run为副本下`logs/rsl_rl/bulb_grasp_teacher_v5/2026-10-06_scratch_2x8192`。
专用训练入口新增可选PhysX覆盖，默认不变；本次每rank碰撞栈64MiB→3.75GiB，接触/配对/heap/temp七项翻倍。
未改变SDF、求解器、完整阶段或成功判据；重力/ADR按原从零课程初始化，非旧模型续训。

已核验0–9共10次PPO更新，两rank真实梯度all-reduce且更新后策略SHA一致；model0有限、优化器step20。
保存配置确认v5/8192/14s/120Hz物理/30Hz控制/+20；外层及两份Kit原生日志未发现容量溢出、丢接触、CUDA OOM或PhysX Error。
GPU显存约34.6/34.3GiB（每卡约45GiB总显存）；CPU40项通过。原有visual引用警告保留。
[运行与证据](../experiments/bulb_v5_dsw_2x8192_20261006/README.md)。只验证启动和初期容量，后期密集接触峰值及独立三阶段成功率未知。
本次未创建长期监督循环；后续检查先核实PID/作业身份，勿重复启动。下方“未同步/未训练v5”为此前历史状态。

# 2026-10-06：灯泡课程v5增加三圈旋入和一次性+20成功奖励（本地）

按用户要求增加第三阶段：阶段二浅插稳定后，以实际深度/轴向角为基线，顺时针旋入3圈。
第三阶段独立6 s，前两阶段仍共8 s，最长14 s；只在第三阶段关闭两项carry_relative速度惩罚。
真实四元数解包计圈并校验轴向螺距，目标只超前1/8圈；保留SDF碰撞/自由刚体/完整阶段。
最终验收为净3圈、深度接近起点+3螺距（±1 mm×倍率）、对中≤0.5 mm、倾角≤5°、低速连续0.2 s。
新增带力新进度9分/圈、首次季度圈里程碑合计3分/圈、最大回退增量惩罚−14分/圈（0.01圈死区）。
两项旋入三圈最多36分，高于6 s原四项接触收益约28.2分；原接触项保持。
新增final_screw_success，首次完整稳定成功+20/回合，已正确处理dt，物理失败同帧不给，不重复领取。

CPU40项通过；最终原生2环境×245步通过，28动作/1900观测、终止更新顺序及+20项注册已核验。
三倍率×13个螺旋位姿原生零重力释放39/39通过；只是制造位姿的几何检查，不是学会自主旋入。
[证据包](../experiments/bulb_grasp_screw_stage3_20261005/README.md)。未训练/独立评测v5、未同步或重启远端。
旧模型仅能作为初始化候选；任意初始相位下的学习效果和8192环境SDF资源开销未知。

# 2026-10-05：支座方槽Tag打印文件

新增[打印说明](../calibration/assets/calibration_bench_visuotactile_urdf/tag_print/README.md)及`output/pdf/revo3_support_tag_55mm.pdf`：DICT_4X4_50 ID0，42mm黑码、55mm虚线裁剪框。按+Y朝孔1/2侧、+X同孔1→孔2安装，中心物点PnP轴向对齐tag_center；槽底到纸面的纸胶厚度h须实测补偿。PDF目视、600 DPI唯一ID和角点/PnP轴向检查通过，未实物验证，URDF未改。GitNexus tag_center未索引UNKNOWN，补核URDF及坐标文档，仅新增输出与说明。

# 2026-10-05：纠正长方体安装方向和每面Tag数量

用户澄清：法兰接头装在正方形端面，四个长方形面各2枚tag。当前 [资产](../assets/head/tag_press_head/README.md) 已改为X/Y/Z=20×20×40 mm，正方形顶端面中心Ø6×8尾柄，总高48 mm；Ø3.2横销孔沿X，Z=44 mm。四张20×40 mm贴纸各两枚12 mm tag，ID40–47，PDF有虚线裁剪框。CAD/STL/3MF通过；PDF离线检出8/8，上下顺序与20 mm间距通过。下方4枚tag和28 mm总高记录为已纠正的误解版本；实物试装未验证。

# 2026-10-05：长方体压头改为单色实体与四面贴纸（当前）

按最新要求更新 [assets/head/tag_press_head](../assets/head/tag_press_head/README.md)：40×20×20 mm 主体无嵌件/凹槽，顶面 Ø6×8 尾柄，Ø3.2 横销孔沿X、距肩部4 mm。四个竖直面各一枚12 mm ArUco 40–43，A4 PDF包含等尺寸裁剪虚线与50 mm标尺。CAD单实体/STL闭合/3MF回读通过，PDF 600 DPI离线识别4/4。旧文件归档；实物打印、安装和承载未验证。此记录优先于下方旧方案描述。

# 2026-10-05：灯泡课程v4改为手持浅插5–6 mm（本地）

用户明确选择“进入螺纹入口5–6 mm，先不旋紧”。阶段一仍是随机灯座口上方15 cm、中心误差<5 cm晋级；
阶段二目标改为螺纹尖端深入5.5 mm，按实际倍率/偏移换算中心（名义高口沿48.26 mm）。
验收：插深5–6 mm，尖端及口沿轴线偏差≤0.5 mm，倾角≤5°，线速度≤0.02 m/s、角速度≤0.5 rad/s，连续0.2 s。
终态成功/ADR/重力课程同口径，不锁存离开后的成功、不要求松手/旋紧。原有奖励权重、阶段一、8 s和动作/观测保持。

恢复源外螺纹SDF及壳体SDF到独立bulb_insert.usd，灯座三角碰撞保留；原0.8–1.2倍灯泡随机化保留，
灯座配对同倍率，几何度量同步缩放，初始灯泡Z增加1 cm保证净空。原资产/点云/snapshot不改。
修复v2重置事件可选env_ids导致原生EventManager拒绝的问题；首次失败报告保留。
CPU30项、原生完整任务2环境×245步通过；三倍率自由刚体碰撞探针完成。
零重力目标释放1 s未挤出，正常重力自由落下会过深或歪斜，未将其记作成功。
[完整证据](../experiments/bulb_grasp_insertion_20261005/README.md)。
未训练/独立评测v4、未验证8192环境SDF资源开销；未同步或重启远端v1，旧成功率不代表插入成功。

# 2026-10-05：灯泡课程v3以灯泡朝向惩罚替代手掌朝向惩罚

按用户要求禁用palm_down_orientation，新增bulb_down_orientation，权重仍为−0.2。
以灯泡本地−X螺纹轴相对世界−Z的倾角θ计算(1−cosθ)/2，两阶段均生效，无接触门控，允许轴向自转。
阶段二thread_down权重5及接触门控保持；v2的随机灯座/15cm→3cm/5cm和30°成功判据/8s保持。
版本升至3，原snapshot与原手掌函数保留，不影响其他任务；CPU24项通过，未运行原生仿真/训练。
[检查日志](../experiments/bulb_grasp_bulb_orientation_20261005/cpu_checks.log)。未同步或重启远端，旧日志/成功率不代表v3。

# 2026-10-05：灯泡课程v2本地改为随机灯座上方15cm→3cm

本地BulbGraspTeacher训练/Play现为课程version2：每回合把灯座XY按原目标的机器人基座系范围
X∈[−0.85,−0.55]m、Y∈[−0.25,0.25]m均匀采样，Z贴当前桌面、开口朝上，回合内固定。
阶段一跟踪灯泡刚体中心到灯座口竖直上方15cm，误差<5cm后，同回合目标下降12cm到口上方3cm；
二阶段继续要求倾角≤30°，重定向权重5及原奖励/8s/28动作/1900观测保持。
3cm为向孔口送入的中心目标，标称螺纹朝下时尖端低于口沿2.376cm；继承缩放/凸包与5cm容差，
不宣称严格插入或旋紧成功。资产/原snapshot未改，旧v1模型只能作新任务初始化候选。
CPU23项及真实USD几何检查通过：随机XY/局部reset/同座两目标/定时器固定/成功边界；
随机范围的桌面保守边缘余量≥5.73cm，最大缩放灯泡初态任意姿态高于口沿≥4.84mm。
[验证](../experiments/bulb_grasp_curriculum_v2_20261005/validation.json)、
[任务说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md)。
未运行v2原生PhysX/训练或独立评测，未同步/重启DSW；22:38核查PID62901/62917/62918仍在旧v1作业。
下方旧版成功率/视频均不能当作v2效果。后续如部署需明确切换作业并另存日志，保留旧检查点。

# 2026-10-05：标定URDF新增四个安装孔坐标系

坐标换算后续核查：`calibration/坐标转换.yaml`明确安装点为`mount_hole_1`，补齐Tag系Z=-35.5mm。
用户确认-308.2mm为Tag原点设备高度，据此孔底面设备Z推算为-343.7mm；原XY/Z换算数值无须改变。
标注180°安装旋转已计入轴向；YAML解析、URDF孔坐标、孔距推算及XYZ双向换算检查通过，未做实测验证。

后续按用户反馈，为四个孔link分别添加三色visual坐标轴（X红/Y绿/Z蓝），方便通过显示几何选择独立名称。
Z轴长度50mm，从底面穿过孔口；X/Y轴长12mm，半径均0.6mm，无collision。
XML及增量比较通过，所有关节和原有几何不变；尚未在用户查看器中验证点击选取。

按用户要求，在 `calibration/assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf`
新增 `mount_hole_1` 至 `mount_hole_4`，通过固定关节直连 `tag_center`，rpy全零。
Tag系坐标（mm）依次为(-152,35,-35.5)、(-77,35,-35.5)、(-27,-190,-35.5)、(23,-190,-35.5)。
Z取支架最底面（源网格Z=0），孔位依据M6×16几何报告，当前STL SHA256与该报告一致。
本次CPU检查通过：XML、完整连通树、四孔变换、全部原有元素不变及mesh路径存在；
现为65 link/64 joint、24活动关节。未运行Isaac或真机。
本次仅修改目标URDF和此状态记录；既有manifest/坐标登记的URDF哈希及tag_only验证脚本属于新增孔frame前的版本，不能作为当前文件验证。
GitNexus tag_center影响分析为UNKNOWN（未索引URDF符号），已补查导出器与历史验证脚本引用。

# 2026-10-05 22:27：灯泡重定向仍在优化（只读核查）

双卡训练PID62901/62917/62918活动，日志14872/30000，已保存14800；thread_down仍为权重5，
二阶段接触门控下按1−θ/π连续奖励，30°仅为成功阈值，低于30°仍鼓励减小倾角。
视频模型13100附近100轮→最新100轮14773–14872：混合阶段末态倾角55.12°→50.06°，
朝向≤30°比例23.91%→32.13%，完整两阶段成功20.50%→28.42%。
最近两个相邻100轮：倾角50.37°→50.06°、完整成功26.78%→28.42%；短期改善较小且有波动，
最近500轮倾角48.02°，不声称单调改善或收敛。均为rank0重置批次日志按轮均值，ADR分布变化，
角度/朝向比例混合阶段，不能当作二阶段专属指标或独立评测。
[统计证据](../experiments/bulb_grasp_teacher_curriculum_resume_20261005/progress_20261005_2227.json)。本次未改训练。

# 2026-10-05 22:23：灯泡两阶段教师服务器视频已下载

已完成此前选定model_13100的headless连续回放：1环境、seed42、960帧/32秒、1280×960/30FPS，
位置/倾角/阶段HUD及逐帧trace齐备；本地完整解码、传输SHA256和原checkpoint不变检查通过。
2.57秒首次进入阶段二；前四个完成回合末态均未成功，第五回合28.83–31.97秒共95帧同时满足位置<5cm、倾角≤30°，
视频结束时该回合尚未结束。不能将片段或0/4终态样本当作独立成功率。
[视频](../recordings/bulb_grasp_curriculum_20261005_1942/bulb_curriculum_model_13100_complete.mp4)、
[验证与报告](../recordings/bulb_grasp_curriculum_20261005_1942/validation.json)。
22:20核查双卡训练PID62901/62917/62918仍活动；本轮仅回放，未改训练、物理或成功判据。
首轮曾因渲染长等待误判而提前停止，保留原产物及修订记录；第二轮复用缓存完成。
本视频固定19:41选择的13100，不是下方22:19下载的14700。

# 2026-10-05 22:19：DSW教师最新模型下载与训练进度

只读核查dsw-wpmt7kzoh4b53jpmov：两阶段灯泡抓取教师双卡8192环境仍活动，日志至14777轮。
近100轮14678–14777阶段一完成率81.07%、完整两阶段成功率26.90%；最新单轮完整成功25.79%。
均为rank0重置批次训练标量按轮平均，非双卡按回合加权或独立评测。
选择时最新model_14700.pt及配置已下载，五文件远端传输前后与本地SHA256一致；未修改远端作业。
[下载与统计证据](../checkpoints/bulb_grasp_teacher_curriculum_20261005/model_14700/download_manifest.json)，同目录保存原始日志尾部。未回放验证。

# 2026-10-05 19:14：灯泡二阶段触发与训练进度（只读核查）

DSW双卡续训仍活动，最新日志12889轮（从12000起约890轮），最新保存model12800。
近100轮12790–12889日志均值：进入阶段二79.92%、完整成功16.57%；启动首100轮分别65.98%/2.27%。
上一百轮完整成功16.02%，最新单轮17.78%；全阶段末态倾角均值72.79°→57.30°，关节异常终止均值35.94%→23.28%。
说明第二阶段已大量触发且完整成功有改善，姿态/异常终止仍需关注。均为rank0重置批次训练标量的轮均值，
不是双卡按回合加权成功数/总数或独立评测；混合阶段角度不能直接当作阶段二专属误差。
当前未记录切换时刻，尚不能判断阶段二剩余时间是否构成瓶颈。未改变训练/代码/模型。
[进度证据](../experiments/bulb_grasp_teacher_curriculum_resume_20261005/progress_20261005_191426.json)。

# 2026-10-05 17:48：灯泡两阶段教师已同步DSW并从12000续训

已部署到`dsw-wpmt7kzoh4b53jpmov`的`/home/admin/workspace/xinyu/mulyi-model-tac`，原单阶段作业停止，旧模型/日志保留。
从停机时最新`model_12000.pt`恢复模型、优化器和重力trigger5730；双卡每卡4096环境，总8192，继续至原30000预算。
两rank梯度同步及首更新参数哈希一致，新checkpoint优化器step240020→240040；有效重力−9.8，课程version1/8s/灯座/权重5已核实。
核验至12012轮：阶段一60.92%、新完整成功0.53%（训练日志，非独立评测）；新旧成功判据不可直接比较。
ADR逐环境状态未存于checkpoint，按当前配置0.7317重新初始化；不声称原MDP完全恢复。
本地CPU16项、原生4env×300步通过；93文件SHA一致。运行与证据见[续训记录](../experiments/bulb_grasp_teacher_curriculum_resume_20261005/README.md)。
后文“未同步/未原生验证”是本次部署之前的历史记录。

# 2026-10-05：标签压头侧面检查与Isaac识别

[证据](../calibration/assets/head/tag_cube_head/surface_inspection/README.md)：完整双色STL五面5万点共面，STEP实体黑白交集及目标体双向差体积均0。
Isaac原生RTX七视角40/40可见tag检出（覆盖20个ID），最大位姿误差0.493 mm／0.274°。
平滑法线可复现视觉凹凸；只导入白色主体则有0.6 mm真实槽。未改打印几何；用户拓竹所见问题未确诊，未验收切片／实物／动力学。

# 2026-10-05：标签压头拓竹导入恢复包

用户报告3MF导入失败；当前同名文件为Bambu Studio 2.7保存的空项目（0模型），源黑白STL完整。
新增[恢复包](../calibration/assets/head/tag_cube_head/tag_cube_head_Bambu_import.zip)，含两份共同坐标STL、一个对象两个体积的AMF及导入说明；原3MF保留。
网格／AMF回读通过，尚未在拓竹GUI导入或切片实测。使用STL同时导入→单对象多零件→指定黑白耗材→另存拓竹项目。

# 2026-10-05：压头与法兰的打印销

新增[横向销](../calibration/assets/head/press_head_pin/README.md)：默认Ø2.9×12 mm，附Ø2.8／3.0试配版本、STEP、装配截面和几何报告。三种压头与法兰名义横孔Ø3.2；法兰STL实际最小内接径约3.091 mm，默认最小单侧模型间隙0.0955 mm。三种销CAD单实体有效，STL闭合／法向／尺寸检查通过；实物打印、承载和防退出未验证，带头滑配销无自锁。原资产／URDF未修改。

# 2026-10-05：标签压头改为五面四标签正方体（当前方案）

按用户后续要求改为顶面圆柱法兰接口、不放tag，其余五面各4枚，共20枚。
[新资产](../calibration/assets/head/tag_cube_head/README.md)按已告知的默认30 mm边长制作，顶面沿用Ø6×8 mm尾柄、Ø3.2 mm横销孔，整体高38 mm。
STL/双色3MF/STEP/米制角点齐备；网格闭合、3MF回读和20/20正视识别通过，两个合成视角PnP通过。
尚未验证实物配合／载荷／相机精度；原URDF未改。下方40×20×20方案已被替代，保留历史证据。

# 2026-10-05：40×20×20 mm 标签压头独立资产

新增 [tag_press_head](../calibration/assets/head/tag_press_head/README.md)：白色长方体，侧面 ArUco 40/41、底面 42，DICT_4X4_50、12 mm 黑边外沿、0.6 mm 齐平黑色嵌件。
含双色 STL/3MF、单件凹槽 STL、STEP、米制角点和预览。CAD／网格闭合／3MF回读与合成图识别、PnP验证通过；
未验证实物精度、承载或法兰配合，未修改现有URDF。证据见资产目录两份 validation.json。

# 2026-10-05：灯泡抓取教师新增回合内两阶段课程

当前`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`及Play版本：阶段一原位置误差<5cm后，
同回合切换到灯座口上方15cm，保留原奖励并新增权重5的螺纹朝下连续奖励。
完整成功要求阶段二位置误差<5cm且本地−X螺纹轴对世界−Z倾角≤30°；不新增接触/速度/保持硬门槛。
ADR和重力课程使用完整成功；阶段一完成另记。阶段不回退，reset回阶段一，最后物理步的结果在reset前记录。
新增原灯座派生资产并保留碰撞；28动作、1900观测、30Hz控制不变，回合由4s延长至8s。
CPU16项回归及真实USD几何检查通过；未启动Isaac/GPU训练、未同步远端、未重启作业，学习效果和独立成功率未知。
旧模型仅为初始化候选，不能据相同维度声称MDP恢复等价；详情见
[任务说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md)、
[BULB-GRASP-CURRICULUM-20261005](../EXPERIMENTS.md#bulb-grasp-curriculum-20261005)。

# 2026-10-05：默认标定URDF支架frame收敛为Tag唯一世界根

按用户要求删除五个支架/重复frame，支架网格直接挂tag_center，手安装改为tag_to_right_hand。
默认URDF为v5、61 link/60 joint，24活动关节不变。手/支架/工具的142组姿态全部位姿一致，
8662 link和14768几何变换最大误差0；104个网格/轨迹/初态哈希不变。
YAML改为Tag直连机械台换算；manifest、坐标登记及采样点源哈希同步，旧文件独立备份。
未运行Isaac/真机；旧导出器未更改，v4验证不适用于当前包。
证据：[TAG-ONLY-20261005-01](../EXPERIMENTS.md#tag-only-20261005-01)。

# 2026-10-05：真机三模态空载会话与持久参考采集

新增`calibration/hardware/collect_real_trimodal.py`，调用现有只读相机worker与SDK推理；
原推理入口支持外部raw参考，默认旧行为保留。后续可按SN选采、提供确认指名、复用基准并核对哈希。
本次用户确认空载，采集五路各16帧共80帧，5张参考＋75组三模态，全部有限且源帧/Marker差分/哈希验证通过。
[数据](../calibration/captures/revo3_trimodal_unloaded_20261005_01/session.json)、
[核验](../calibration/captures/revo3_trimodal_unloaded_20261005_01/validation.json)。
SDK原尺度、400光流点和原ROI保留，training_ready=false；尚缺深度m标定、SN指名/几何映射、对应仿真状态。
本次未采加载数据、不读手部串口、不运行仿真/训练。见实验CALIB-TRIMODAL-BASELINE-20261005-02及hardware说明。

# 2026-10-05：用户确认空载的五路图像已保存

15:05（Asia/Shanghai）按用户授权采集，五路各8张640×480 RGB，共40张，NPZ与PNG逐像素核验通过。
目录：[revo3_unloaded_20261005_01](../calibration/captures/revo3_unloaded_20261005_01/)。
空载状态按用户确认登记，SN→具体手指仍未知；逐路采集非同步，未运行SDK推理或物理标定。
手部串口未发现端点，五路图像全部成功，但原报告整体失败；本次无配套关节／压力读数。
后续加载应显式复用本批空载参考，当前离线脚本尚不支持跨批次参考参数。
详见[CALIB-UNLOADED-20261005-01](../EXPERIMENTS.md#calib-unloaded-20261005-01)。

# 2026-10-05：抓取教师与装配课程对比（静态审计）

用户报告新增抓取任务接近训练成功；本次核对本地教师快照、MDP、装配v2及10月4日既有诊断，未读取远端最新指标。
教师以目标位置误差<5cm记success（position_only），无接触/低速/保持硬门槛；搬运稳定另有奖励。
装配Stage0要求双侧>1N、世界线/角速度<0.05m/s和0.5rad/s、相对位姿5mm/5°内连续>1s。
教师采用−1→−9.8重力课程、持续双侧接触/目标跟踪奖励、1900维五帧含接触观测；
装配为全重力、分阶段预算奖励、369维单帧无接触观测。动作更新及PPO契约也不同，不能直接替换checkpoint。
既有装配诊断支持“接触后稳定学习信号弱”的假设，尚未证明某一差异导致学习差距。
下一步建议同口径独立评测教师的稳定抓持/离桌保持，再决定移植或蒸馏；未实施。
详见[对比记录](../EXPERIMENTS.md#bulb-grasp-vs-assembly-20261005)。运行代码、判据、检查点和作业未改。

# 2026-10-04：dsw-wpmt7kzoh4b53jpmov 抓取教师从零双卡训练

按用户要求停止旧assembly_reward_ddp4096作业（5个进程），保留15个已保存checkpoint。
最新抓取教师部署至 `/home/admin/workspace/xinyu/mulyi-model-tac`；87文件逐一SHA256校验，覆盖文件有.sync-backups备份。
新任务`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`，无checkpoint，种子42/43，2×4096环境，30000迭代。
从零训练时ADR初始难度归零，重力触发状态重新开始。每rank单物理GPU/逻辑cuda:0和独立Kit目录。
使用任务独立runtime下的教师RSL-RL fork，首次优化后验证梯度归约执行、两rank模型参数SHA一致。
启动信息：[launch.json](../experiments/bulb_grasp_teacher_scratch_2x4096_20261004/launch.json)。
首次PPO更新已验证梯度all-reduce和两rank参数SHA一致；已推进至iteration6，单次全局采样262144，两卡显存约8.4GiB。
初始物理重力沿用教师快照−1m/s²（从零指模型/优化器/课程状态，不是0重力）。
当前启动日志在远端`experiments/bulb_grasp_teacher_scratch_2x4096_20261004/train.log`。

# 2026-10-04：独立灯泡抓取教师已移植并通过回放/续训接线

新任务`BrainCo-Dexsuite-Flexiv-Bulb-Grasp-Teacher-v0`（另有`-Play-v0`）位于
[BulbGraspTeacher](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/BulbGraspTeacher/README.md)。
使用本仓库机器人和灯泡派生资产，保留1900维/5帧历史、28维相对关节动作及教师MDP。
原始model_22200已复制到checkpoints/bulb_grasp_teacher_20261004且SHA256验证一致。
4环境180步回放通过；4环境1次PPO更新/优化器恢复/−9.8重力恢复通过。不是成功率评测。
查看：`bash scripts/rsl_rl/play_bulb_grasp_teacher.sh --num_envs 16 --real-time`。
通用train/play不支持该专用runner，应使用新增脚本；未启动长期训练或更改服务器作业。
GitNexus绑定本工作区main/167fc3c，注册文件impact LOW；新增符号初次UNKNOWN经显式导入/引用核查。
证据：[BULB-GRASP-PORT-20261004](../EXPERIMENTS.md#bulb-grasp-port-20261004)。

# 2026-10-04：机械标定主说明只保留calibration入口

按用户要求删除根目录 `机械标定台仿真搭建_README.md` 兼容符号链接，
唯一正文保留在 [calibration/机械标定台仿真搭建_README.md](../calibration/机械标定台仿真搭建_README.md)，正文SHA256不变。
EXPERIMENTS及项目地图/运行手册/状态中的旧Markdown链接已改为实体路径，迁移清单同步登记旧入口移除。
GitNexus impact为UNKNOWN，补查引用后执行；仅文档入口清理，未改运行代码。

# 2026-10-04：五指首轮采样预定位点已生成

[calibration/fingertip_sampling_points.yaml](../calibration/fingertip_sampling_points.yaml)保存每指DIP触觉面中央附近1点，
当前Tag坐标系、mm、手21关节全零；压头尖端统一Z=40mm，再沿-Z下压。
当前手/支架最高29.85384mm，预定位高出约10.14616mm；每点另列模型表面中心射线高度。
已验证5点向下射线先命中对应触觉面、XYZ关节名义行程及YAML解析；URDF未修改，未操作真机。
有限压头/法兰碰撞、真机外参/TCP和实际接触零点未验证；几何表面Z不能直接作为球压头下压终点。
证据：[CALIB-POINTS-20261004-06](../EXPERIMENTS.md#calib-points-20261004-06)。

# 2026-10-04：默认标定URDF改为Tag中心世界原点

按用户最新更正，Tag中心只作为全局原点，方向沿用原世界坐标系，不对齐base_link。
默认旧路径仍指向calibration/assets下同一URDF；新增tag_center固定frame，与calibration_world重合。
Tag名义位置采用支架系(170,200,35.5)mm贴码槽底中心，实际码纸厚度尚未提供。
两条世界根固定关节平移同步重表达；手/支架/压头相对关系、网格、关节轴/限位和轨迹保持。

平台世界Z现在0.5mm（支架局部36mm）；机械台原读数-308.5605mm保留。
coordinate_reference.json更新世界变换及未启用的条件公式；台架外参/TCP仍待实测。
当前包manifest=v4；旧v3导出器和历史verify_touch_asset不自动包含本次及M6×16派生修订。
142组状态/9088个link变换通过，最大矩阵误差5.56e-17；104处mesh引用存在，
104个网格/CSV/初态文件哈希一致，66 link/65 joint通过urdfdom解析。未运行Isaac或真机。
矩阵、原文件备份、可重复只读验证见[当前包说明](../calibration/assets/calibration_bench_visuotactile_urdf/README.md)，
证据索引[CALIB-TAG-20261004-05](../EXPERIMENTS.md#calib-tag-20261004-05)。

# 2026-10-04：资产与任务数据分离

按用户要求将抓姿、初始化模板、工作空间库及随附验证归档集中到 `task_data/`；
USD/URDF/CAD/网格、触觉表面和几何元数据保留 `assets/`，标定仍在 `calibration/assets/`。
标准旋出奖励说明移到 `docs/tasks/rotate_bulb_rewards.md`；任务配置 Python 保留原位置和字节内容。
总入口：[资产目录](../assets/README.md)、[任务数据](../task_data/README.md)。

13项迁移、68文件、26,814,304字节；旧位置全部为相对符号链接。
67文件字节不变，1文档仅修正相对链接；迁移前544个资产文件均仍可从旧路径读取，
其中543个哈希相同。215个source/scripts Python源码及4个既有标定链接不变。
两种D-Peg真实加载函数、Custom抓姿contract及装配模板加载通过（无需导入Isaac），
4份模板的41项来源哈希匹配；装配CPU检查12/12通过。

路径impact为UNKNOWN，补查默认加载、生成器、来源哈希和历史引用后保留兼容入口；
全工作区detect-changes为CRITICAL（169文件、761符号、215流程，包含原有改动），
不能当成本次迁移或整个工作区的干净图验证。未提交、未改物理/奖励/成功判据，
未启动Isaac/训练、未同步远端。复制时必须携带task_data实体与旧入口链接；原资产本地忽略策略延续。
迁移清单及验证见 [ASSET-REORG-20261004-01](../EXPERIMENTS.md#asset-reorg-20261004-01)。

# 2026-10-04：按用户指定清理四组旧资产

已删除 `calibration/assets/revo3_breadboard_mount/generated_calibration_v5_m6x20/`、
`assets/rotate_bulb_collision_fast/`、`assets/d_peg_insertion/collision_variants/`、
`assets/rotate_bulb_randomized/workspace_bank.json`，共157个文件、84,242,385字节（80.34 MiB逻辑大小）。
删除前GitNexus impact均为UNKNOWN；补查当前源码只见生成输出路径，未见任务／标定默认加载引用。
已确认四项目标不存在，旧手XYZ包、当前视触觉XYZ包、V4／V4-touch、主socket及high_z_v2／joint1_zero工作空间库仍在。
生成器与历史记录保留；历史实验若指定已删副本需重新生成。仅本地清理，未运行仿真、未核查或修改远端。
下文及其他文档中关于这些已删资产的交付记录属于历史状态。

# 2026-10-04：Stage0奖励核查到1243轮

成功条件都有对应奖励，但双侧接触奖励近0.5单回合预算上限，抓稳质量／保持／完成仍显示0。
每回合历史靠近约10.95、靠近3.49、任意指接触2.73，稳定相关信号相对弱。
历史最佳增量发奖、多条件乘积／连续接触门控及超速惩罚封顶，构成可检验的奖励设计瓶颈；尚非因果验证。
已核对冻结源码哈希及日志单位，仅审计，无奖励／判据／作业改动。
[公式和数值](../logs/diagnostics/assembly_5090_contact_20261004/reward_audit_1243.md)。

# 2026-10-04 19:31：Stage0单回合门槛拆分核查

1211–1230轮：严格双侧>1N接触统计步占26.832%；26.693%同时接触且速度不合格，
因此接触成立时约99.48%仍超线速度或角速度门槛。grasp_ok但保持尚未完成仅0.1395%。
相对位姿超差诊断35.5655%、相对滑动诊断62.823%，掩码重叠且后者不是独立成功门槛。
日志合并线／角速度及平移／转角，尚不能分别定量或判定哪个单项占主导。
最近4000回合stable_grasp/success=0，3962超时、36掉落、1越界、1丢失抓持。
最新保存model1200；只读检查，未改作业。证据：
[summary_1931.json](../logs/diagnostics/assembly_5090_contact_20261004/summary_1931.json)。

# 2026-10-04 19:23：P3远端Stage0可视化回放已开启

按用户要求，远端桌面`:1`开启Sim5.1 Storm窗口，独立play PID613765；
加载本轮训练model_1100.pt的哈希一致副本，1环境、90Hz、solver4/1、evaluation=true。
训练PID603756保留运行。普通Storm窗口的GPU Fabric姿态不刷新；独立回放脚本
创建38个无物理API的机器人显示体及灯泡显示副本，每步从真实GPU状态同步USD。
原碰撞／奖励／动作／归一化／检查点契约保留；前30/60/90/120步灯泡位置与未加显示副本时一致。
连续截图已确认手与灯泡运动，19:22:52步数210。共享GPU导致回放偏慢，不是实时速度评测。
远端目录`/home/brainco1/workspace/xinyu/assembly_play_support/fast90_model1100_gui_20261004_1916`；
入口`launch_storm_mirror.sh`，日志`play_storm_mirror.log`。首次普通play及失败显示同步尝试已停止。
[验证及回放脚本副本](../logs/diagnostics/assembly_5090_play_20261004/verified.json)。

# 2026-10-04 19:05：5090 Stage0到1120轮，接触增加但稳定抓持仍未突破

训练PID603756/守卫running，无物理错误，最新保存model1100。1101–1120轮严格双侧>1N接触
统计步占26.372%，较277–296轮20.985%增加5.386个百分点；其中约99.5%仍速度不合格。
最近4000完整回合均有弱双侧>0.5N事件，稳定抓持/成功0/4000；3968超时、31掉落、2越界，标签可重叠。
单指接触奖励0.5355→0.5477，slip62.09%、pose_error35.26%为重叠诊断，不能相加。
只读状态核查，未改奖励、判据或训练；仍需解决接触后的低速稳定保持。
[统计快照](../logs/diagnostics/assembly_5090_contact_20261004/summary_1905.json)。

# 2026-10-04：默认完整URDF已应用M6×16局部沉孔

用户要求适配最长M6×16螺丝。当前指定完整URDF路径保持，lower_mount视觉／碰撞改引
lower_calibration_v4_m6x16_print.stl；四处Ø12沉孔承压Z由24降到10 mm，Ø6.8通孔及孔中心保留。
总高45 mm、平台Z=36 mm、全部关节／坐标不变，坐标登记哈希已更新；原V4/V5与源包保留。
按未确认的圆柱头Ø10×6 mm包络，无垫片名义突出6 mm、1 mm垫片时5 mm。
CAD单实体、STL封闭单体、768孔位探针、四孔螺丝／插入包络、104个mesh引用通过；
URDF除两处网格文件名外逐元素一致。未做实物装配／承载或Isaac复测。
GitNexus资产及新生成器查询UNKNOWN，补查引用后限定修改；未提交。
[详细说明](../calibration/机械标定台仿真搭建_README.md)与
[验证](../calibration/assets/calibration_bench_visuotactile_urdf/modifications/m6x16/validation.json)。
旧采集器／导出器源包仍为V4-touch，重建须显式应用此派生网格。下一步核对实物头型、平台有效孔深。

# 2026-10-04：默认建系URDF与平台Z基准已确认

用户指定后续统一以 `assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf`
建立坐标关系（实际文件位于calibration/assets，同一兼容链接）。
用户进一步确认测量部位为右侧平台平面基面（避开圆形槽和贴码槽），机械台显示Z为 **−308.5605 mm**，不是手指触觉面接触零点。
URDF的calibration_plane世界Z为0.036 m；默认零关节尖端Z为0.0480797829510025 m。
在Z同向、读数对应工作尖端触平台的条件下：z_world=0.036+(z_display_mm+308.5605)*0.001；
对应平台的press_z_joint为−0.0120797829510025 m。轴方向、TCP与XY外参尚未实测核对。
[coordinate_reference.json](../calibration/coordinate_reference.json)保存原始读数、单位、指定URDF哈希及条件公式；
该文件是后续建系的基准登记，不是已接入采集器的控制配置。主说明与总入口同步更新。
本次仅XML／哈希／算术检查，未改URDF、未连接硬件或执行运动；测量证据来自用户报告。

# 2026-10-04：标定与采集集中到calibration，迁移验证完成

总入口：[calibration/README.md](../calibration/README.md)，映射见
[organization_manifest.json](../calibration/organization_manifest.json)。
20项文件／目录迁移：simulation、hardware、collection、pressure、tools、image_generator、
assets、reference及outputs/simulation；主机械README沿用用户已移入calibration的位置。
原脚本、资产、参考及输出路径保留相对符号链接；共享训练／传感器包保持原导入结构，在shared中索引。
默认新采集／压力输出归入calibration，datasets旧入口兼容；SDK、captures和原始证据未改写。

迁移1349文件／444093930字节，无丢失；1272个非代码／非文档文件哈希一致，
其余改动限路径／文档及运行缓存。48个Python AST通过、函数／类签名保持、12个URDF的606个mesh引用存在。
原5项CPU检查通过；新路径一回合法向仿真115帧、15项检查通过，CPU URDF导出完成。
新旧计划／采集离线帮助及旧投影、图像网络、压力辅助导入通过；无硬件操作、Isaac/GPU启动或训练。
压力L0入口在当前解释器缺isaaclab.utils，迁移前源码同样失败；未安装或扩展环境，完整压力／Isaac流程未验证。
4份旧复制marker报告中的12个历史相对链接仍缺目标，原报告保持，当前导航与主说明链接已核对。
全工作区GitNexus变更分析CRITICAL（包含旧训练改动及路径迁移），未宣称图检查干净；
未提交。相关用户原改动哈希保持。证据见
[迁移核验](../calibration/outputs/organization_20261004/organization_verification.json)与实验CALIB-REORG-20261004-03。

# 2026-10-04：机械标定执行框架已补全文档，受控标定尚未开始

按用户要求完善[机械标定说明第8–14节](../calibration/机械标定台仿真搭建_README.md#8-标定目标与系统分工)：
真机几何／TCP／XY与接触Z零点、参考力与ADC、时钟、恒力采集状态机、实验矩阵、
实测轨迹对应仿真、分阶段参数拟合、数据契约／三模态训练转换和独立验收。
用户确认XYZ实际位置可读、独立测力可用；具体SDK、额外姿态轴定义和测力轴数／量程仍待登记。
工作区新增的`calibration/标定流程.md`原文保留，五轴压力机与写芯片流程已纳入方案；未实施写参。

源码核查：标定器默认仍为V4视触觉手，只支持两种V4资产；V5尚无采集配置入口。
完整XYZ包v3的法兰link已位于共同尖端，与原模型法兰frame比较须用manifest完整变换。
当前采集器不支持实测轨迹导入；SDK20×20光流、180×320／240×320深度与旧编码器不能直接拼接。
35组未知接触参考的历史数据仍仅用于诊断。下一步先登记设备／实装版本，完成食指球压头小闭环，
再拟合与独立验收。文档新增链接的本地目标、7个页内导航及差异空白检查通过；
离线`run.py --plan-only`确认默认270回合／55,890帧，未写采集数据。原7.5–7.6节实测记录及状态历史保留。
此次仅文档、静态核查及离线计划预览，无新硬件采集、仿真采集、参数拟合、训练或芯片写入。

# 2026-10-04 16:12：5090 Stage0已有双侧接触，稳定抓持仍为0

只读核查90Hz/4096 run，训练仍running，日志至296轮。277–296轮日志均值：
严格双侧接触（拇指和另一指均>1N）约20.985%；双侧接触但速度不合格20.894%，
据同口径比值约99.6%的双侧接触统计步不满足低速要求（<0.05m/s且<0.5rad/s）。
最近4000完整回合全部曾有弱双侧接触（均>0.5N），稳定抓持和成功均0/4000；3951超时、47掉落。
严格双侧接触占比从100–119轮16.892%增至200–219轮20.262%，最近约20.985%，近期改善放缓。
slip55.896%、pose_error34.512%为可重叠诊断，不能相加；日志均值非逐步池化总体成功率。
当前瓶颈为稳定维持接触及低速，不能从这些日志独立归因于90Hz或奖励；未改配置/重启训练。
证据：[统计快照](../logs/diagnostics/assembly_5090_contact_20261004/summary.json)，相关源码哈希与冻结工作区一致。

# 2026-10-04：5090 Stage0 4096快速训练已启动，固定动作采样达到2.00倍

已同步至`brainco1-ThinkStation-P3-Tower-Gen-2:/home/brainco1/workspace/xinyu/mulyi-model-tac`，
复用隔离Sim5.1/Lab2.3.2、torch2.7+cu128环境；被替换内容在相邻`assembly_perf_support_20261004/sync_backup*`保留。
同一5090、4096环境固定动作对照：原120Hz/32-8为10.471秒/32控制步；
120Hz/4-1为6.397秒（1.64倍）；最终90Hz/4-1为5.229秒（2.002倍）。
这是固定动作采样基准，不能当PPO每轮时间或直接与DSW25秒跨机比较。
另有256环境、各24轮PPO的120Hz求解器对照，末16轮6.276→3.291秒（1.907倍）。
分区/线程调节无明显额外收益；外壳凸包仅再快约10%，不采用，保留原碰撞资产。

用户确认“stage1”指第一关Stage0抓持。最终显式`physics_hz=90, fast_stage0_physics=true`、
TGS位置/速度上下限4/1、30Hz控制/decimation3，原奖励、成功判据和自碰撞保持。
默认仍120Hz；90Hz开关仅Stage0可用，后续阶段拒绝，不能绕过物理契约直接晋级。
14个原生探针全部通过，无丢接触/容量错误；90Hz4096采样桌面穿入约1.11mm，最大采样灯泡速度2.631m/s，
不能视为精度或学习质量等效。12项CPU检查及分布式边界检查通过。

2026-10-04 15:03:27（北京时间）已启动独立Stage0，从零4096、seed42、2000轮上限；
工作区`.workspaces/assembly_stage0_fast90_4096_20261004`，输出`experiments/assembly_stage0_fast90_4096_20261004`。
守卫PID603746、训练PID603756，启动身份存于launch_identity及startup_validation；物理错误守卫在运行。
首批0–7轮PPO已完成并保存model0，模型参数有限、checkpoint优化器步20，生效配置/奖励/1650文件哈希核对通过。
其中1644源码/资产与本地相同，6个为远端安装元数据；GPU快照25318MiB。
当次验收6762完整回合、严格成功0，未声称策略学会。训练仍在继续，不自动启动Stage1。
15:11:25（北京时间）最终健康快照：已完成26轮（0–25），最近5轮采样+学习均值13.924秒；显存25848MiB，守卫running、PID身份匹配、无物理错误。训练累计完整回合27654，严格成功0；为训练计数，不是独立评测。
120Hz临时作业602228在初始化期间停止，无PPO或模型；原目录与停止原因保留，不算完成训练。
证据：[4096对照](../logs/diagnostics/assembly_5090_perf_20261004/large_scene_comparison.json)、
[启动验收](../logs/diagnostics/assembly_5090_perf_20261004/startup_validation.json)，
复现见RUNBOOK与实验`ASM-PERF-P3-20261004-06`。

# 2026-10-04：DSW装配训练性能诊断，主瓶颈在PhysX完成等待

对活动rank2417/2418原位采样，未改代码/配置或重启训练。迭代40–59采样25.334秒、
学习0.144秒；TensorBoard相邻墙钟约25.957秒（日志打印不含后续log开销）。
两个rank各50秒非阻塞主线程采样约90%在PhysX simulate/fetch_results，粗略约23.4秒/轮；
reset及回合日志2.5–2.9%、观测0.8–1.0%、奖励1.1–1.2%，rank0日志2.1%、rank1归一化等待2.7%。
配置32控制步×decimation4=每rank128批物理步，未发现reset额外步；场景TGS下限32位置/8速度。
确有批量cpu()/item()，但无证据表明D2H为主因；CPU限流计数不增、显存窗口稳定。
原生采样进一步落在PhysX任务等待；无法拆分内部碰撞/solver/拷贝时间。尚未做降迭代或环境数对照，
不宣称降低物理精度安全或直接提速比例。原生采样有短暂停顿扰动，不能作为自然吞吐基准。
详见[诊断报告](../logs/diagnostics/assembly_perf_20261004/report.txt)及[实验记录](../EXPERIMENTS.md#asm-perf-20261004-05)。

# 2026-10-04：新增接近/单指奖励的装配课程双卡4096已启动

用户授权同步至`dsw-wpmt7kzoh4b53jpmov:/home/admin/workspace/xinyu/mulyi-model-tac`并双卡训练。
按本轮上下文启动横躺装配课程`BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0`，从Stage0重新训练。
旧课程已预算结束（exit2、model1999保留）；GPU0/1空闲。新完整冻结副本为
`.workspaces/assembly_reward_ddp4096_20261004`，输出`experiments/assembly_reward_ddp4096_20261004/train`。
主根目录是缺课程模块的旧副本，短暂新增的4个文件已核对撤回；部署在其下完整冻结目录，原主副本/旧快照保留。
28项源码/资产哈希核验、服务器11项CPU检查通过。2026-10-04 13:35:22（北京时间）启动，
守卫PID2408、课程启动器2410、训练rank0/1为2417/2418；PID身份在launch_identity中记录。

13:43:36启动验收：两卡每卡4096，共8192，NCCL共训，120Hz物理/30Hz控制，Stage0回合5秒，
resume=false，阶段预算2000轮，晋级三seed各100回合。实际配置确认历史接近weight40/std0.4、
any_contact weight0.7/threshold1N，动作/变化−0.02/−0.01、臂路程/偏离−0.08/−0.5。
迭代0–2已完成并保存model0；前三次PPO各20次梯度同步、参数最大差0，梯度均值检查通过。
前三轮23.08/16.62/14.98秒，GPU即时24811/24772MiB。日志已有两新增项非零奖励，无PhysX/CUDA错误；
原容量/丢接触停机守卫运行。仅证明已开始学习更新，尚无稳定抓持/独立成功率结论。
详见[ASM-REWARD-DDP-20261004-04](../EXPERIMENTS.md#asm-reward-ddp-20261004-04)及
[启动验收](../logs/diagnostics/assembly_reward_ddp4096_20261004/server/startup_validation.json)。

# 2026-10-04：Revo3五路相机恢复，三模态诊断数据已生成

用户重新连接后，相机5/5枚举与各8帧采集成功；21关节和244有效点再次读通。
新增`calibration/probe_revo3_trimodal.py`对本次40帧执行SDK推理，每路第0帧作参考、其余7帧输出，
共35组三模态；5/5路成功，独立数据一致性60/60通过。GFBCT深度180×320，四路GFBCI为240×320；
Marker为20×20光流采样网格，各origin/current/offset均为20×20×2，不能当成400颗物理marker。
用户确认有接触或状态不确定，全部标记`unverified-current`，无空载／绝对深度／物理标定结论。
GFBCT位移本次全0，四路GFBCI为−1～1；未做受控加载响应或五路并发实时验收。
未发运动、清零或使能指令；原相机未枚举记录保留在下方，当前阻塞已解除。
接口、命令和证据见[机械标定说明7.6节](../calibration/机械标定台仿真搭建_README.md#76-五路三模态通路验证2026-10-04相机重新接通后)
及[CALIB-SDK-20261004-02](../EXPERIMENTS.md#calib-sdk-20261004-02)。

# 2026-10-04：Revo3本机主链路读通，三模态受相机未枚举阻塞（历史，已恢复）

按用户授权复用现有SDK环境，新增`calibration/probe_local.sh`本机入口。实际只读两次探测：
右手BCUVR1205J2600002，21关节和244有效点均可读，拇指指腹57点仍Disabled。
最终数据维度／有限值检查18/18通过；ViTai相机0/5，系统仅枚举笔记本摄像头，
故整次探测退出码2。RGB、Depth、Marker三模态未取得；历史5路图像不能替代本次证据。
ViTai深度／marker接口可导入，两份GFBC深度ONNX模型CPU加载成功，未实际推理或建立空载基准。
未发送运动／归零／使能命令，未安装依赖或修改训练代码。
接口和连接阻塞详见[机械标定说明7.5节](../calibration/机械标定台仿真搭建_README.md#75-本机连接配置与三模态复测2026-10-04)，
证据见[CALIB-SDK-20261004-01](../EXPERIMENTS.md#calib-sdk-20261004-01)。
下一步为确认相机USB数据线／供电，枚举恢复后确认SN映射及空载，再验三模态。

# 2026-10-04：Stage0 启用任意单指接触引导，本地11项检查通过

按用户要求，仅Stage0在账本替换后重新注册`any_finger_contact`，复用dextac适配函数：
任意一指对灯泡过滤接触力>1N就给+0.7分/秒，30Hz每步约+0.02333，5秒最多+3.5；
不以双侧接触、低速、位姿稳定为门槛，多指接触不叠加倍数。原双侧弱接触质量/稳定保持仍保留。
历史最佳接近+1分/cm、动作/机械臂惩罚和严格+8成功条件/reset均保持；没有新增good_contact持续项。
11项CPU检查通过，覆盖五指分别单独接触、严格阈值、总力与灯泡过滤力区分、缺失/异常数据、
时间积分，以及Stage0限定/其他阶段代价保持。未运行Isaac、未部署或训练，学习效果未验证。
见[ASM-ANY-CONTACT-20261004-03](../EXPERIMENTS.md#asm-any-contact-20261004-03)。

# 2026-10-04：Stage0 新增历史最佳接近大奖励，本地离线通过

用户指定移植dextac的历史接近进度并保留机械臂动作惩罚。已仅在横躺课程Stage0添加
`FloorBestReachReward`／`reach_best_progress`：五指尖到壳体中心的最大距离每创新纪录缩短1cm给+1分；
weight40/std0.4并消除dt对积分的缩放，3–5cm改善给+3～5分，严格抓持完成仍+8。
原持续接近、动作/变化、臂路程/偏离成本均保留；成功/失败/reset/物理未变。
10项CPU检查通过：新纪录/往返、局部reset、无效数据、失败/抬起屏蔽、30/60/120Hz等积分、
实际配置函数Stage0限定及所有阶段原成本保留。未运行Isaac或训练、未同步远端，无学习收益结论。
课程源码哈希已变化，旧checkpoint不能绕过严格契约直接恢复；旧冻结工作区及模型未改。
见[ASM-BEST-REACH-20261004-02](../EXPERIMENTS.md#asm-best-reach-20261004-02)与[课程说明](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum/README.md)。

# 2026-10-04：横躺装配 Stage0 稳定抓持瓶颈只读分析

用户提供末轮速度54.92%、无双侧接触42.11%、slip37.76%、pose_error1.13%。
按当前标签定义，前两项分别为`strict & ~slow`和`~strict`；若来自同口径统计，
双侧接触步中94.87%速度不合格，同时满足接触和低速的步约2.97%。
`slip`为相对速度诊断（>0.01m/s或>0.2rad/s），并非累计位移>5mm；
`pose_error`才对应5mm/5°锚点偏差。速度/接触/位姿不合格清零保持计数，不直接reset环境。
优先怀疑接触后稳定控制及奖励门控/历史最佳保持信用，不能由这些占比认定动作惩罚过大或reset过早。
本地model1999契约的17份任务/课程源码哈希匹配；末窗口成功0，未晋级。
已有本地回放3回合全部超时、无抓持事件，不代表训练总体；model400单轨迹只作早期补充证据。
未启动仿真/训练、未连接远端、未改任务判据或运行代码；用户百分比原始日志待核验。
详见[ASM-GRASP-DIAG-20261004-01](../EXPERIMENTS.md#asm-grasp-diag-20261004-01)。

# 2026-10-03：Isaac Sim 6.0.1 独立环境的训练与 RTX 回放链路已通过

在 `brainco1-ThinkStation-P3-Tower-Gen-2` 新建独立前缀
`~/workspace/xinyu/assembly_play_sim6`，安装 Isaac Sim 6.0.1.0、Isaac Lab
`v3.0.0-beta2.patch1`、RSL-RL 5.0.1；原 5.1 环境和系统驱动未改。RTX5090／595.91.07
下官方 Cartpole 完成 64 环境、8 次 PPO 更新、8192 步并保存 `model_7.pt`；随后加载该模型，
RTX 回放 120 帧并生成 1280×720、60fps 视频，正常退出。首次相机回放因 shader 缓存构建约需
80 秒，期间高 CPU、低 GPU 不是已证实的死锁。证据位于远端 `assembly_play_support/`。

指定 `assembly_floor_v2_20261003/stage0/model_1000.pt` 已在原 Isaac Sim 5.1／Lab 2.3 环境
完成 3 回合无界面评测，证明 checkpoint、任务和物理可加载；0/3 成功，仅为链路验证。
该模型尚不能在 6.0.1 原样可视化：项目任务可注册，但 Lab 3 已将
`AdditiveUniformNoiseCfg` 改名，且官方新入口会在 Kit 启动前遇到项目的 `isaacsim.core` 运行时导入；
项目旧 `play.py` 也不符合 Lab 3 的 `launch_simulation` 生命周期。已加入噪声配置的双版本导入回退，
但未声称完成整个 Lab 3 迁移或权重兼容。6.0.1 环境还存在上游 beta 元数据冲突：Sim 固定
`packaging==26.0`／`coverage==7.4.4`，Lab 声明旧约束；实际 CUDA、训练和回放均通过。

# 2026-10-03：5090 主机独立 Conda 环境已安装，GUI 回放受阻

用户授权同步及可视化回放 stage0/model_1000，并要求 conda-forge、隔离其他环境。
目标 `brainco1-ThinkStation-P3-Tower-Gen-2`（192.168.10.252），项目在
`/home/brainco1/workspace/xinyu/mulyi-model-tac`；9330 个文件同步后 rsync 校验和无差异。
独立前缀 `~/workspace/xinyu/assembly_play_conda`：Python3.11、IsaacSim5.1.0、
IsaacLab2.3.2.post1、torch2.7.0+cu128、定制RSL-RL3.1.1。pip check、5090 CUDA矩阵运算通过。
已有环境及系统595.91.07驱动未改。安装/冻结清单/启动脚本/日志在相邻 `assembly_play_support/`。
GUI 曾创建窗口并到达 app ready，随后在 `librtx.scenedb.plugin.so` 崩溃，模型尚未加载；
与NVIDIA已确认的Blackwell/595渲染问题相符，根因未用驱动对照独立证明。
未启动训练，未验证策略效果；不能将安装成功视为回放成功。见实验 `P3-ENV-20261003-01`。

# 2026-10-03：V5 支架底面已截短 10 mm，适配 M6×20

按用户更正后的 M6×20 规格，从 V4 底面整体裁去 10 mm，四个原 Ø6.8 通孔、Ø12 藏头孔及 XY
孔位不变。螺丝承靠面到新底面由 24 mm 减为 14 mm：无垫片名义伸出 6 mm，加 1 mm 垫片时
名义伸出 5 mm。新包为 `assets/revo3_breadboard_mount/generated_calibration_v5_m6x20/`，旧 V4 未覆盖。
CAD/STEP 单实体有效，STL/3MF 为毫米制封闭单连通网格；四颗 M6×20 包络与支架无干涉。
当前视触觉手组合 URDF 的支架、手和标定 frame 均随新底面下移 10 mm，X/Y 与手部定义保持。
尚未实物打印或拧入光学平台，平台入口倒角、孔深和实际有效牙深仍需装配确认。

# 2026-10-03：标定默认已换为任务视触觉手并重新装配

后续按用户要求调整当前压头运动坐标：X新=−Y原、Y新=+X原、Z新=+Z原；
`world_to_press_stage` yaw=−π/2。随后修复link／joint不一致：移除Z关节早期+π/2补偿，
用户随后明确原点也统一到工作尖端：法兰／压头／tip三frame完全重合，工具两个固定关节xyz/rpy全0。
stage和carriage零位同样落在尖端，stage初始xyz=(0.103214595,0.210899079,0.048079783)m。
XYZ关节origin全0；工具网格局部origin重表达保持实体几何。
XYZ限位／CSV／初态保持；三轴方向、20组非零位姿、121帧FK与视觉／碰撞位姿检查通过。
manifest升级v3，历史NPZ原法兰位姿与新工具frame间使用`T_source_flange_shared_tip`完整刚体变换，原记录未改。
坐标证据在新资产包 `coordinate_frame_validation.json`；未重新采集或改变支架配合。

新入口 `assets/calibration_bench_visuotactile_urdf/urdf/revo3_calibration_scene_xyz.urdf`；
使用 `assets/revo21_right_touch` DV2手、原V4支架及球压头XYZ。默认config/smoke配置切换；
无版本字段的旧数据继续按旧手读取。源资产／已有训练改动保持，未运行训练或连接硬件。
手部21关节、五路相机frame、碰撞／惯量／压力描述均保留；任务rubber→DIP对齐使用原任务函数，
特别保留中指约29.5mm固定偏移。当前采集仍为虚拟UV模型，未声称任务USD几何或encoder完全兼容。

本次65link/64joint/104mesh引用、urdfdom4及零／非零位FK检查通过；121帧XYZ最大误差1.67e-16。
食指8/8短回合1574帧、99项数据检查、五指有限探针通过；Isaac52网格、30次查看与截图通过。
**剩余机械问题**：零位500点/link抽样，拇指与支架最大穿入2.383mm，旧手相同抽样最大0.599mm；
机械配合仍REVIEW_REQUIRED。未切削支架、改安装位姿或关闭碰撞，未验证动力学与实物装配。
见[新资产说明](../assets/calibration_bench_visuotactile_urdf/README.md)及[实验记录](../EXPERIMENTS.md#calib-touch-20261003-04)。

# 2026-10-03：桌面横躺灯泡装配课程 v2，双卡每卡4096正式课程已启动

用户确认重构的是桌面抓取→抬起→对准→旋入课程，并授权 dsw-wpmt7kzoh4b53jpmov。
v2 取消固定竖直灯泡阶段，首阶段自由横躺；近距→远距／XY±5cm／yaw±90°，再经真实成功状态库
学习搬运、下降旋入，最后20秒完整桌面任务。原严格成功／失败与自碰撞／螺纹保留。
接近项改为持续距离反馈＋正负进度，新增第一关节路程和超过60°初态偏离的软成本；未硬限位。
确认旧“关闭触觉”仍运行285-taxel压力、15维接触观测与31个触觉传感器；v2移除它们，输入369维。
物理候选240→120Hz，每30Hz动作8→4步，求解器迭代保持；reset/记录改批量读取GPU。

服务器为无Git旧副本，GPU0/1初始空闲。本次在指定仓库 `.workspaces/` 下建立冻结基线／候选；
不覆盖原主副本、不触碰其他训练。8环境和256环境原生检查通过：自由横躺、2mm净空、369维、
无压力传感器、部分reset独立且没有隐藏物理步。CPU6项行为检查通过。
同一早期v2候选256环境：240Hz 1290.74、120Hz 2174.47转移/秒（+68.5%），设备使用峰值均3.143GiB。
旧v1自由竖直对照1117.65转移/秒／4.354GiB；组合改动快1.95倍、内存少27.8%，不能分摊各项因果。
均为30步预热＋180步小随机动作，不是成功策略评测。原生接触稳定不能证明120Hz完整旋入精度。

64环境10轮PPO完成，保存model9，130完整回合0成功；仅链路验证。随后进一步把近距指尖距壳体中心
从约12cm缩至6–7cm，42个离线碰撞路径样本通过，joint1约−8.35°至10.32°；该修订待原生与试训。
服务器重启后最终近距初态2环境14项原生检查通过。1024环境虽然状态检查通过，但原生日志
发现512MiB碰撞栈溢出／丢接触，故`state120_1024_final`判无效，其吞吐不纳入性能结论。
已停止随后启动的`ppo_trial1024`，未用其继续训练或晋级。课程碰撞栈改为1MiB/环境（1024为1GiB，
初次上限3GiB，后按用户4096要求提高至3.75GiB，保留更大显式预留）；这是主机锁页内存，不应直接计为显存。
新增原生日志判错与试训溢出停机守卫；扩容1024复测15项通过，日志无丢接触。

用户追加要求双卡共同训练、每卡4096环境。已新增PPO边界同步课程／真实成功状态库，
主卡检查点保存两rank计数和采样RNG；rank1日志独立。Gloo双进程晋级／状态库合并／恢复与同步停机检查通过。
GPU双卡64环境4轮及恢复2轮通过：NCCL两设备、梯度均值核验、前三次更新参数差0。
短训model4独立seed401评测0/100稳定抓持成功，17/100出现过抓持事件；模型／归一化未变、reset失败0。
关节1最大偏离40.86°，100回合均未超过60°；这是短训分布，不能证明学成后也无绕转。
最终冻结`.workspaces/assembly_floor_v2_ddp4096_20261003`，已启动`experiments/assembly_floor_v2_20261003/ddp4096_curriculum`：
每卡4096、合计8192环境，120Hz/30Hz，各阶段上限2000轮，三seed独立晋级评测。17:19核验已完成更新0–4并保存model0；
正式8192环境前三次PPO各20次梯度同步、参数差0，无PhysX/CUDA错误。
首五轮20.39/16.49/14.87/16.62/20.28秒，每卡即时显存约24.2GiB；训练仍在Stage0，未认定抓持达标。
每卡碰撞栈3.75GiB（uint32不能设4GiB），三配对容量各2²⁴、接触点2²⁴、接触片2²⁰。
容量不足／丢接触守卫会停止本次进程树；旧300轮单卡计划由此替代，不并行保留。
未证明学会抓起、消除策略绕圈或全流程成功；旧检查点与v2不兼容，不自动导出为旧Custom任务。
说明见[课程v2](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/RotateBulbAssemblyCurriculum/README.md)，
记录见[ASM-FLOOR-20261003-01](../EXPERIMENTS.md#asm-floor-20261003-01)。

# 2026-10-03：Revo3 SDK副本与真机只读采集已完成

用户授权复制SDK并读取已连接手。`calibration/brainco-revo3-sdk/`保存源码及原环境中的
bc-revo3-sdk2.1.0／pyvitaisdk1.0.17原生包和模型；676文件的SHA见`calibration/sdk_copy_manifest.json`。
新入口`calibration/probe_revo3_readonly.py`优先用副本运行包，其余依赖复用源SDK的`.venv`。
本次识别UltraVisionTouch右手BCUVR1205J2600002，ttyUSB0／5M／从站127，读取3次21关节反馈与触觉。
阵列声明301点，拇指指腹Disabled，其余4指腹＋掌部244点有效、Force模式mN；未修改使能／清零。
5路ViTai相机各8帧640×480读取成功，实际图像及独立检查见`calibration/captures/revo3_readonly_20261003_final/`。
请求30fps、驱动回报120fps、短窗口接收约58–62fps；逐路读取不证明并发带宽／同步。
健康读取error_code=0，但SafetyState.Unknown及240.84V回报待核实，未认定为健康验收。
接口、单位、数据要求、命令与边界已写入[搭建说明第7节](../calibration/机械标定台仿真搭建_README.md#7-revo3-sdk与真机只读采集2026-10-03)。
下一步需确认相机SN到手指、具体ViTai型号／模型、空载基准，再验深度／marker；台架SDK、真机运动及坐标对齐仍未处理。

# 2026-10-03：精简标定场景已导出含XYZ压头运动的URDF

交付 `assets/calibration_bench_urdf/urdf/revo3_calibration_scene_xyz.urdf`，自带109份mesh、
初始关节位置和121帧 `press_trajectory.csv`；共68link/67joint，保留手部21活动关节，新增3个prismatic。
无龙门架外观。`world_to_press_stage` 的origin集中定义待用户调整的压头坐标系，当前零位为原场景首帧。
原手／支架／压头URDF元素除mesh路径外保持，XML、urdfdom4、145处mesh引用与初态几何核查通过；
独立FK复算121帧法兰和tip最大误差4.17e-16，XYZ分别位移方向检查通过。
只验证运动学；新运动链／原压头质量惯量仍未知，未做Isaac动力学导入或重新采集。
入口与坐标调整见[资产包说明](../assets/calibration_bench_urdf/README.md)，生成器为 `scripts/calibration_bench/export_urdf.py`。

# 2026-10-03：标定场景已移除虚拟龙门架等机构

按用户要求，`scene.py:export_scene` 不再生成台面、两侧导轨／立柱、横梁、Z轴柱和滑块；
保留原压头／法兰、Revo3右手／V4支架及查看用相机／灯光。
当前场景与截图位于 `outputs/calibration_bench/head_fixture_only/`，README入口已更新。
复用 `validated_v1` 首回合数据重新导出，未重新采集；手／支架／工具的USD几何与动画段逐字节相同。
Isaac原生54网格拓扑、121帧变换核对及30次查看更新／截图通过，位姿最大元素误差3.71e-10。
旧场景作为历史证据保留；原始资产、模型参数与采集流程未改。

# 2026-10-03：机械标定仿真 v1 已搭建并完成有限验证

新增独立 `scripts/calibration_bench/`：读取 V4 手／支架和真实压头网格，虚拟 XYZ 受控运动，
默认球压头；CPU 采集法向几何深度、临时压力模型、既有曲面 HydroShear marker 和诊断 RGB，
USD 在本机 Isaac Sim 5.1 查看／回放。未改共享训练任务、原始资产、支架螺钉结构或 SDK。
源码反查确认原 encoder 三模态为 RGB＋Depth＋Marker；压力是额外通道。
HydroShear 为核心参考，未发现足以把全部任务归为单篇论文复现的完整稿件。

本次：球压头8/8回合（含6种XYZ和2种虚拟旋转）、1,574帧，99项数据检查通过；
三角和锥头各1/1回合、115帧、15项检查通过；5项解析／协议回归通过。
240×320额外7个位姿探针有限值、空载归零与递增载荷检查通过，未完成该分辨率全量采集或收敛验收。
原生54网格拓扑检查通过，首回合121帧动画变换与NPZ最大误差3.71e-10，30次查看更新和RTX截图通过。
两次早期USD导出语法错误保留在 `outputs/calibration_bench/first_smoke/` 与 `sphere_normal_v2/`，
已修复，最终证据在 `outputs/calibration_bench/validated_v1/`。

边界：规定运动的软接触近似，不是 PhysX/FEM 动力学；RGB光学和压力材料参数未经独立标定。
现有食指内部视觉mesh开口，抽样几何守卫不能认证全行程净空。当前虚拟UV和稠密网格
不等于旧相机／285-taxel硬件布局，不直接兼容旧encoder。无真机对齐、训练或完整270回合采集。
下一步先做分辨率／材料／光学与独立参考验证，再决定论文正式采集；真机联调按用户要求后置。
使用、字段和注意事项见[系统说明](../scripts/calibration_bench/README.md)，
记录见[CALIB-SIM-20261003-01](../EXPERIMENTS.md#calib-sim-20261003-01)。

# 2026-10-03：三种压头与连接法兰装配 URDF 已生成

基于 `assets/head/` 中三角、球形、锥形压头及连接法兰的原始 STL，在同目录新增三份固定装配
URDF。装配按 mesh 中直径6mm、长度8mm的压头尾柄插入法兰直径约6.4mm、深8mm的接口孔，
并将双方位于4mm插入深度的横向销孔同轴对齐。根 frame 定义在法兰盘安装面中心，压入方向为
`-Z`；每份资产提供工作端 `press_head_tip` frame。三份 URDF 通过 XML、拓扑、相对 mesh 引用和
离线变换核查；4份 STL 均闭合且法向一致。未填写未知的质量/惯量，未做 Isaac Sim 导入、动态
碰撞、实物装配或真机标定。路径与坐标契约见[机械标定台仿真搭建说明](../calibration/机械标定台仿真搭建_README.md)。

# 2026-10-03：本地输出、测试与旧支架资产已清理

按用户要求，已从当前工作区移除 `recordings/`、`outputs/`、`logs/`、`benchmarks/`、
检查点包内两个纯日志子目录、`tests/` 及 `.pytest_cache/`。其中 68 个 Git 已跟踪测试文件
现在表现为删除；历史文档中的测试命令和本地证据路径仅保留为历史记录，不再可在本工作区复现。
`runs/`、模型检查点、运行源码、其他任务资产及清理前已有用户修改均未触碰。

`assets/revo3_breadboard_mount/` 现只保留 `generated_calibration_v4/`；V1/V2/V3、构建/验证脚本、
说明文件和缓存均已移除，因此当前包可直接使用但不能在本仓库重建。保留包共 150 个文件、
61,332,716 字节，清理前后文件清单聚合 SHA256 均为
`27e562abb04a0df3cb038b76c27b015fd42f3161d4736d4e07d697b306e7e7c8`；两份 URDF XML
可解析，143 个 mesh 引用全部存在。ROS `check_urdf` 本次因本机缺少
`liburdfdom_model.so.3.0` 未执行；原 `validation.json` 保留。清理项已移入系统回收站，可恢复。

同日继续清理 `docs/`：删除 52 份带日期后缀或 experiment(s) 命名的历史日志型 Markdown，
保留 STATUS、RUNBOOK、PROJECT_MAP 及未按该规则命中的设计、实现、集成和使用说明。
旧索引中指向已删除专题文件的链接仅作历史路径记录；删除项已移入系统回收站，可恢复。

# 2026-10-01：Revo3标定槽V4已交付，根部/挡板不改

用户最终取消拆挡板/低位凸块，只扩大两个槽。独立`generated_calibration_v4/`交付
40mm口径/3mm深球面槽、净55×55mm/0.5mm深贴码槽及右手装配URDF。
原V3全部包内文件哈希保持；整段腕部区域、四孔/两定位区域CAD对称差0，
除两个槽外移除材料为0，未改固定结构或标定frame，外形仍220×252×45mm。
CAD单实体有效、STEP回读、62354面封闭单连通STL及两份URDF通过；
98条槽/平台射线最大标高误差0.01061mm。手部61link/60joint/21活动关节组合参数保持。
相同抽样继承局部干涉不增加，仍REVIEW_REQUIRED；无打印/真机标定，仍需绑带。
见[V4交付](../assets/revo3_breadboard_mount/CALIBRATION_V4.md)及
[验证记录](../EXPERIMENTS.md#revo3-calibration-mount-20261001-05)。

# 2026-10-01：Revo3矩形标定下支架V3已生成

基于V2补齐L形空缺区域，底面外轮廓220×252mm、总高45mm；新增平台Z=36mm。
按用户确认增加口径20mm/深3mm的球面槽，以及40.6mm方形/深0.5mm的贴码槽（40mm含白边贴片）。
球槽底点(170,125,33)mm和贴码槽底(170,200,35.5)mm均有URDF参考frame；
局部球槽底点不等于整机最低点/行程限位，探头工具补偿及实际码面厚度待实测。
有效单实体CAD、STEP回读、61634面封闭单连通STL、URDF及原手参数/网格保持通过；
STL两槽58条射线最大高度误差0.00866mm，V2材料未切除，安装孔/定位区域保持。
零位同批400点/visual抽样未增加>0.1mm穿入；仍继承拇指等局部干涉，装配REVIEW_REQUIRED。
向上压紧仍需实装绑带；未打印/真机探底或做刚度与标定精度验收。
见[V3交付](../assets/revo3_breadboard_mount/CALIBRATION_V3.md)及
[检查记录](../EXPERIMENTS.md#revo3-calibration-mount-20261001-04)。

# 2026-10-01：Revo3固定腕部下支架V2已生成并重新装配

用户确认要消除根部晃动。本次保留V1与源资产，在
`assets/revo3_breadboard_mount/generated_stable_v2/`新增STL/STEP/BREP、单支架URDF及组合URDF。
加高45mm腕部鞍座、按实际手截面留0.25mm余量重做槽、前端双角挡、两处掌背支撑，
并预留8mm软绑带通道；向上保持及沿手长方向的压紧仍需实际安装绑带。
CAD有效单实体/STL闭合单连通，尺寸220×252×45mm；原安装孔与定位柱保持。
腕部同一5000表面点旧版201点穿入、新版0点；±X/−Y移动0.5mm有几何阻挡。
手部姿态/21关节/全部手网格保持，URDF解析通过；拇指继承干涉最大抽样约0.689mm仍在，
无打印/预紧/抗振或动力学验收，完整装配REVIEW_REQUIRED。
见[V2交付](../assets/revo3_breadboard_mount/STABLE_V2.md)及
[记录](../EXPERIMENTS.md#revo3-stable-mount-20261001-03)。

# 2026-10-01：指定 brainco-description Revo3 右手＋下支架 URDF 已生成

按用户指定使用外部 `brainco-description/revo3_system/urdf/revo3_right.urdf`，
与 `1_stl/breadboard_mount_256` 上下支架恢复闭合装配，再删除上支架得到最终版。
交付 [assets/revo3_breadboard_mount](../assets/revo3_breadboard_mount/README.md)，
自带网格及预览；最终58 link / 57 joint，保留原21活动关节和手部全部参数。
两版urdfdom解析、相对路径、坐标、源网格哈希及移除上支架前后其余内容一致检查通过。
零位表面抽样发现下支架与新版手局部干涉，腕部最大抽样深度2.793mm；未改原几何，
标记REVIEW_REQUIRED，未做仿真/真实装配或全关节运动碰撞验收。
见 [实验记录](../EXPERIMENTS.md#revo3-breadboard-urdf-20261001-02)。

# 2026-10-01：双 Y 标定台已导出四 link 可动 URDF

从`calibration_bench_sim_package/assets/blender/dual_y_press_checked.blend`按原父子分组导出
固定底座→双Y同步龙门→X滑块→Z压头，四份米制STL共481268三角面；原Blend/STEP未修改。
交付`generated_urdf/`含概念CAD名义X/Y/Z ±250/±150/±75mm版，以及机械配置软件限位
X[-340,0]/Y[0,170]/Z[-310,0]mm版；后者Z范围超过候选CAD名义行程，只作控制契约参考。
两份URDF通过xmllint、ROS Humble urdfdom、相对mesh路径、四link三关节链及限位检查；独立临时
重建与交付逐字节一致。未做Isaac/Gazebo导入或全行程碰撞扫描；碰撞盒、质量/惯量、effort/
velocity仍为明确占位，装配`REVIEW_REQUIRED`和X/Z连接面约0.5mm间隙未消除。
见[交付说明](../calibration_bench_sim_package/generated_urdf/README.md)和
[实验记录](../EXPERIMENTS.md#calibration-bench-urdf-20261001-01)。

# 2026-10-01：本地与服务器旋出任务已恢复GPU4高成功率profile3

用户明确要求两端恢复旧版本。以`20260928_140146_rotate_bulb_phase_phys60_30hz_12s_scratch_gpu4/workspace`冻结源码为唯一基准，定向恢复Randomized的env_cfg.py、phase_rewards.py和README，两端三文件SHA均等于GPU4清单。
默认恢复profile3、旧机器人USD/原joint1=-8.35275°及约±160°限位、workspace_bank_high_z_v2，移除initial_arm_pose奖励；保持物理60Hz/控制30Hz/12秒及GPU4接触/旋出/搬运奖励。
本地与服务器各114项相关CPU测试通过，分别21份相关源码和10项资产来源SHA匹配。检查点契约原workspace SHA对应，model8400保留；未进行新的原生构造/回放，不将源码恢复当作重新评测成功。
本地新零位资产/模块保留但不再引用；撤销功能的初态惩罚测试已备份并移到诊断目录的retired_tests（.disabled）。装配课程及其他任务变化保留，不做全仓库回滚。
本地备份`logs/diagnostics/bulb_restore_gpu4_20261001/local_backup/`，服务器备份`/home/xinyu/project_updates/bulb_restore_gpu4_20261001/`；恢复前两端版本不同，均单独保留。GPU4 stopped/GPU6 completed，未新启动或停止训练。
见[本地恢复](../logs/diagnostics/bulb_restore_gpu4_20261001/local_restore.json)、[服务器恢复](../logs/diagnostics/bulb_restore_gpu4_20261001/remote_restore.json)、[验证范围](../logs/diagnostics/bulb_restore_gpu4_20261001/validation.json)。

# 2026-10-01：最新GPU6训练完成，完整成功仍0；旧GPU4末期约94.2%

北京时间14:56重新连到xinyu21实时读取两run完整TensorBoard标量与status。恢复既有本机TensorBoard SSH转发服务，未启动/停止训练。
GPU6 `20260929_125130_rotate_bulb_joint1zero_pose001_phys60_scratch_gpu6`于10月1日01:14正常完成15000轮（iteration14999、exit0），最新`train/model_14999.pt`。
完整15000点Train/success_rate、成功奖励及成功终止均0，release_grasp全0。末100轮旋出峰值2.9600圈、末态2.9463圈、释放指标5.20%、超时97.63%、回报8.17；position_tracking仅1.16e−5。已有旋转进展，但未学会完整释放抓取搬运。
旧GPU4 `20260928_140146_rotate_bulb_phase_phys60_30hz_12s_scratch_gpu4`仍停在8467轮，最新model8400；末100轮Train/success_rate94.23%，末点95.91%，释放指标99.996%。不同版本/预算，不能据此归因某一改动。
以上是训练日志统计，未新做独立回放评测，不能用旧model1000动作推断最终模型动作。见[实时快照及统计窗口](../logs/diagnostics/bulb_training_completion_20261001/training_snapshot.json)。

# 2026-09-30：可达性复算不支持必须160–180度后转

CPU按新USD复算4752个旧坐标见证（joint1已换算新零位）：全部末端误差<0.5毫米/0.5°。432灯座接近/抓取/抬起见证joint1最大67.14°；3456搬运解3448个在±90°，最大111.42°，并非最小必要角度。
model1000录像1080步FK重建：三回合末joint1约158/160/162°，掌心距整个灯座区域仍至少39.7–42.8厘米，均未抓持/释放，不像为了够到而完成的绕转。目标远近可能影响学习难度，不能据此证明完整动态成功或因果。
仅冻结资料CPU诊断，未改训练/分布、未重跑碰撞/仿真；见[可达性与轨迹分析](../logs/diagnostics/bulb_reachability_audit_20260930/README.md)。

# 2026-09-30：根坐标核查未发现翻转，任务在底座−X侧

组合USD顶层/base_link/root_joint与冻结配置、保存的原生底座均单位旋转。joint1父帧约171.65°是原运动链定义；新零位未旋转根。
原始抓取参考对原生reset的9个机械臂刚体最大位置误差3.93e−7米；手掌/灯座/全部3456搬运目标同在−X侧。目标世界→根、奖励根→世界及实际几何观测变换对应，4份相关源码与冻结录像SHA相同。
本次旧SSH隧道不可用，结论基于冻结资料与本地只读复算，非当前活动进程新快照；未改训练，未证明后转完整因果。见[根坐标核查](../logs/diagnostics/bulb_root_frame_audit_20260930/README.md)。

# 2026-09-29：关节1无单侧硬限制，探索噪声已接近下限

检查当前GPU6冻结8层USD/hash：joint1轴Z/±180、默认drive位置0/速度0，未见附加ForceAPI；实际训练ImplicitActuator为K300/D45/力矩上限300，动作每步覆盖目标。没有查到固定后转力矩或180度驱动目标。
训练策略Normal采样+tanh，范围双向；录像为无采样的均值回放。实际joint1 std初始0.35→model1000 0.0600→model1300 0.05455，下限0.05。model1000第一回合5秒状态的负动作条件概率约0.68%，但前2秒后平均约22–23%，不是全程单侧探索。
此前近端点原生测试双侧可达，尚不能排除从初态运动的碰撞/动力学不对称。未新开仿真/未改训练。见[资产与探索核查](../logs/diagnostics/bulb_joint1_asset_exploration_20260929/README.md)。

# 2026-09-29：动作惩罚参考系核验，未发现全0关节目标混用

当前profile6控制目标为当前实测角+动作增量，动作0是当前角目标；action_l2罚动作增量，action_rate_l2罚与上一动作之差，非关节绝对角。1077个非重置步复算误差均小于2.1e-9。
initial_arm_pose参考固定reset七臂角[0,52.63,-8.59,-86.65,-24.37,36.28,-158.42]度，不是全0；joint1因零位重定义才为0。arm_motion姿态只在grasped后用抓持参考。
action_rate存在延续已有动作的平滑偏好，但示例从+0.026反向-0.026仅增加joint1平滑扣分1.29e-7/步，不能据此认定主因。未发现错误符号/参考，无训练改动。见[补充核验](../logs/diagnostics/bulb_joint1_reward_audit_20260929/README.md)。

# 2026-09-29：GPU6后转奖励审计，姿态惩罚生效但较弱，存在少扣分诱因

model1000三回合joint1到约158/160/162度，前2秒后正奖励近0；搬运/成功项始终0。姿态公式逐步复算误差2.82e-10，单joint1累计约0.038–0.058分（不含重置末步），不是符号或零位失效。
最可疑组合：接近期arm_motion已罚速度、未grasped时不罚抓持参考姿态；hand_far依赖ever_grasped因而全程0，接近进度只奖新纪录不罚后退；90度/-0.01初态项很弱。末段速度成本明显下降，尚未证明后方方位本身被偏好。
训练901–1000与1122–1221窗口，回报改善中各奖励分量差分约70%来自arm_motion少扣，成功/搬运仍0；非因果消融。停滞项三回合均-1.955，未被后转规避。
仅只读/离线分析，未改奖励或重启。见[详细数值、图和建议](../logs/diagnostics/bulb_joint1_reward_audit_20260929/README.md)。

# 2026-09-29 22:56：GPU6新零位model1000视频已下载

按用户要求选择当前GPU6 profile6最新保存点model1000，原冻结零位/±180/姿态-0.01/90度免罚配置，在当时空闲GPU4录制36秒seed42单环境原始回放。
3完整回合全超时，未释放/抓持；joint1范围-19.42至162.10度（新坐标）。视频1080帧完整解码，服务器/本地/桌面hash一致；非独立成功率评测。
GPU6训练保持原PID/start_ticks，录像正常退出。GPU4旧训练本次只读发现此前已停在8467轮，非本次操作停止。
见[视频与核验](../recordings/xinyu21_bulb_gpu6_joint1zero_20260929_2253/README.md)。

# 2026-09-29：GPU6零位±180度、姿态权重-0.01，已从零重启

用户授权停GPU6、降低初始姿态偏离惩罚并重启。profile6将weight -0.025→-0.01（扣分降低60%），90度免罚及释放后0.2保持，同时部署前轮已验证的joint1初态零位/有限±180资产和派生workspace。
旧GPU6停在1267轮，model1200/日志保留；新run `20260929_125130_rotate_bulb_joint1zero_pose001_phys60_scratch_gpu6`，6144/32steps/seed42/15000轮，60Hz物理/30Hz控制/12秒，与原GPU6除资产坐标/姿态weight/profile外保持。
CPU57、原生12项和32环境13轮通过；正式至2轮，iteration0/禁止load/空优化器、新限位与零位、更新保存、配置/契约和源码运行库hash通过。
GPU4保持原PID；TensorBoard筛选`Joint1Zero_Pose001`。学习效果尚未评估。见[执行证据](../logs/diagnostics/bulb_joint1_zero_weakpose_gpu6_20260929/README.md)。

# 2026-09-29：本地关节1改为初始姿态零位、有限±180度，已原生验证

用户最终将连续旋转方案改为“原初始位置为资产0位，范围±180度”。Randomized独立USD层调整joint1父关节帧与两限位，init_state joint1=0；实际初始姿态保持。
原joint1初态-8.35275度成为0，新范围对应旧坐标[-188.35275,+171.64725]。原控制器按新软限位裁剪，90度姿态项及其余任务参数保持。
派生workspace保留3456样本，仅更新坐标参考/新层哈希；旧checkpoint契约拒绝。CPU32、USD仅3属性差异、原生9项通过；实际端点+179.99966/-179.70734度，reset姿态误差1.19e-7米。
GPU4/GPU6活动训练保持原冻结资产；本次仅在GPU7独立诊断副本验证，已退出。尚未部署新训练或评估学习效果。
见[最终实现和证据](../logs/diagnostics/bulb_joint1_zero_20260929/README.md)。此前连续方案已放弃，保留尝试证据。

# 2026-09-29：GPU6免罚范围改为90度，已从零重新启动

用户要求：15度→90度，停原GPU6从0训练。新profile5仅改初态姿态项deadband及版本标记，系数-0.025/90度尺度/释放后0.2和其他参数保持。
旧profile4停在1754轮，model1700保留；新run `20260929_104034_rotate_bulb_pose90_phys60_scratch_gpu6`，GPU6/6144/32steps/seed42/15000轮/初始LR0.001，物理60Hz/控制30Hz/12秒。
CPU53、原生12项及32环境13轮通过；正式核验至8轮，禁止load/iteration0/空优化器/更新保存/实际配置及哈希通过，未加载pilot或旧模型。
GPU4原PID保持；TensorBoard筛选Pose90。效果尚未评测，详见[执行记录](../logs/diagnostics/bulb_pose90_gpu6_20260929/README.md)。

# 2026-09-29 18:20：GPU6 model1500短录像已下载

按用户要求用当前GPU6 profile4最新保存点model1500录制36秒，单环境seed42、原冻结源码/物理/奖励/契约，未筛选。
3个完整回合均超时，0次成功，1080条步前记录均无释放/抓持；joint1范围-42.34至-7.05度，未复现之前的大幅反转。
视频1080帧全解码，本地/桌面哈希、配置/契约/源码验证通过。回放正常退出，GPU4/GPU6训练保持。
见[视频与验证](../recordings/xinyu21_bulb_gpu6_latest_20260929_1815/README.md)。短样本非独立成功率评测。

# 2026-09-29 18:08：GPU6未见中途报错，旋转仍进步但接近/释放滞后

GPU6运行至1452轮；完整终端匹配告警均在PPO开始前，未发现训练期新报错，两run标量无NaN/Inf。1351–1450轮GPU6 KL0.0141、噪声0.220，未支持数值坍缩。
同轮GPU4/GPU6回报7.458/4.310，旋转峰值2.832/2.925圈，成功均0；GPU6近几百轮回报持续上升。
关键差异是旧palm_above_support引导0.3096/0.00081，新增姿态项当前仅-0.00116；GPU6约99.19%回合超时，释放后抓取搬运尚未建立。
前次“尚未到900轮”的判断已过时；当前是行为路线分化的疑点，不能因果归罪姿态项或宣称长期卡死。两run实际配置差异仍仅姿态项/profile（PPO仅run_name），未改训练。
见[终端、同轮曲线及判断限制](../logs/diagnostics/bulb_gpu6_plateau_audit_20260929/README.md)。

# 2026-09-29：GPU6早期学习同轮核查，未改训练

GPU6 profile4实际只比GPU4增加initial_arm_pose和版本号；PPO配置相同、源码运行库哈希通过。按629-728轮比较，mean_reward为-1.831/-2.112，旋出峰值日志均值0.747/0.439，成功率均0；GPU6释放日志指标更高。
GPU4奖励明显转正在900–999轮（均值+2.653），GPU6尚未到此区间。新增姿态成本在100–299轮是同期旋转进度正奖励的30–48倍，成熟策略上4%的尺度标定不能证明从零阶段足够弱。
支持早期探索更保守的可能性，不能据单seed确定因果；GPU6最近旋出进展回升，未判定卡死。此次只读，未修改或重启任何作业。
见[同轮曲线、尺度核查及限制](../logs/diagnostics/bulb_initial_pose_learning_audit_20260929/README.md)。

# 2026-09-29 16:05：GPU4成功率约82%，model6600录像已下载

GPU4最近100个训练成功率记录均值81.79%，末点82.86%；GPU6新姿态引导版本同期仍在早期、成功率0。不是等阶段奖励对比或独立评测。
用GPU4原冻结profile3的model6600在临时GPU7录制约48秒，未筛选：5个完整回合中4次成功、1次超时，末尾另有未完成回合；成功终止与一次性成功奖励吻合。
视频1439帧完整解码、本地/桌面哈希、实际配置/契约及源码运行库核验通过；GPU4/GPU6训练进程保持，回放已退出。
见[录像及原始成功记录](../recordings/xinyu21_bulb_gpu4_success_20260929_080026/README.md)。

# 2026-09-29：GPU6新姿态引导从零训练已启动并验证

用户明确选择从0训练；原GPU6停在6430，model6400保留；临时model6300续训作业按新指令停在6300，禁止将其当作当前训练恢复。
最终run `20260929_071719_rotate_bulb_initial_pose_phys60_scratch_gpu6`，GPU6/6144/32steps/seed42/15000轮，初始LR0.001；60Hz物理/30Hz控制/12秒回合和PPO配置与GPU4一致。
仅新增profile4初始姿态弱引导，原主奖励/契约/物理/成功条件保持。原生10项、从零32环境13轮PPO通过。
正式从iteration0、空优化器开始，禁止load；已到6轮，更新/保存/新负奖励、两份契约、实际配置及源码运行库哈希通过。未加载pilot或历史权重。
GPU4原进程保持，学习收益尚未评测。TensorBoard筛选`InitialPose_Phys60_30Hz_12s_Scratch`。
见[同步、停止、从零启动与验证证据](../logs/diagnostics/bulb_initial_pose_gpu6_launch_20260929/README.md)。

# 2026-09-29：初始姿态弱引导已本地实现，尚未部署训练

Randomized奖励profile4新增`initial_arm_pose`，weight=-0.025；七臂关节相对固定reset角度，15度免罚/90度尺度，权重[2,1,1,1,.5,.5,.5]，二次增长不封顶，释放后乘0.2。
抓持状态不影响参考或激活；手指不受罚，现有奖励、动作/观测契约、物理和成功判据保持。当前固定reset明确写default_joint_pos，未来若随机化初态需改为记录实际reset参考。
用旧model5100四条轨迹离线估算新增扣分1.86–2.02/回合，占原回报3.68–4.15%；只是尺度验证，非训练改善结论。
CPU68项通过，尚未进行profile4原生仿真/训练，未同步服务器或重启GPU4/GPU6；活动作业仍profile3。
见[实现、参数、验证及复算](../logs/diagnostics/bulb_initial_pose_guidance_20260929/README.md)。

# 2026-09-29：第一关节反转已用同模型轨迹复现

model5100/seed42/单环境原配置重放1440步，四回合均从joint1=-8.35度转到约-160度限位，累计偏移约151.65度，发生在释放之前。
第一回合2秒已到-159.65度、5.43秒才释放，grasped始终false，前2秒59/60步策略输出负向指令，故大幅绕转时姿态成本未启用；非回合重置跳变。
第三回合抓持在2.2秒建立，q1已约-158.94度，此姿态被记为新参考。第一回合arm_motion累计-0.62、搬运引导+37.27，当前回报允许低成本绕转路线。
四回合均超时，未据此估计总体成功率；GPU4/GPU6保持原进程，临时GPU7已释放，未修改训练。
见[轨迹与复现证据](../logs/diagnostics/bulb_joint1_trace_20260929/README.md)。

# 2026-09-29：model5100机械臂大幅偏转的奖励实现核查

用户观察到视频机械臂第一关节反向大幅转动。本次核验服务器录像冻结源码与本地phase_rewards/mdp/actions/Custom env_cfg四文件SHA256一致，以及录像实际奖励配置。
`arm_motion=-0.3`：释放前始终罚速度，仅grasped时罚相对抓持建立时姿态的偏移；重抓会更新参考，释放后整项为0，非持续相对回合初态的位置约束。
位置死区0.05rad、尺度0.15rad、归一化后截断到1再平方，七关节取均值；单关节偏移达到0.20rad后封顶。action_l2=-0.05罚增量，action_rate_l2=-0.02罚相邻增量变化，均不直接约束累计偏移。
当前动作目标为当前实测角度加动作增量（scale1rad），并裁剪到关节限位；实际运动由驱动/物理决定。实现允许大幅偏转，但视频第一关节精确角度及发生阶段未提取轨迹确认。
建议另行实现释放前固定初始参考的关节偏移成本、合理接近死区、超过容差继续增长，并为释放后保留较弱且不阻碍搬运的姿态约束；本次未修改奖励或训练。

# 2026-09-29 11:22：GPU4 model5100 回放已录制并下载

选择时最新保存点model5100，沿用GPU4冻结源码、运行库、契约及60Hz物理/30Hz控制/12秒配置，在空闲GPU7完成单环境seed42原始回放。
视频约48秒（1439帧/30FPS），服务器全部帧解码通过，本地及桌面副本SHA256一致；未筛选成功片段或开展独立成功率评测。
抽帧可见灯泡离开底座并被握住抬起，尚不能据此确认最终目标成功。GPU4/GPU6训练进程保持，临时GPU7回放已退出。
见[录像与验证记录](../recordings/xinyu21_bulb_phys60_latest_20260929_031720/README.md)。

# 2026-09-29 10:27：GPU4已出现少量完整成功，尚未稳定收敛

GPU4物理60Hz/控制30Hz/12秒从头run约4904轮，最新model4900；本次只读训练事件，未干预训练。
4805–4904原始100点：释放状态日志99.9099%、最高旋出2.99927圈，释放后接触质量0.79941（非成功率）。
release_grasp和released_stability奖励持续非零；搬运奖励由相邻前100点2.85096→2.89734，目标位置误差27.876→27.396cm。
首次正成功日志在3197轮，但最近Train/success_rate均值0.113137%，前100点0.130320%；完整成功仍少，未随奖励同步改善。
这些是训练日志窗口均值，不是独立冻结模型评测的成功数/总数；未作本次视频或独立成功率验证。
见[原始事件窗口及解释](../logs/diagnostics/bulb_gpu4_progress_20260929/README.md)。

# 2026-09-28 22:10：GPU4物理60Hz/30Hz控制/12秒从头训练，正式更新/保存通过

用户要求停原GPU4＋7并在GPU4训练最新本地旋出；最终明确选择物理60Hz、decimation2、控制30Hz、12秒，从头训练。
原双卡run停在3680轮，四个已核验进程均退出，model3600和日志保留；GPU7释放。
GPU6分阶段奖励续训保持原PID/start_ticks及冻结240Hz/decimation8/20秒配置。
新run `20260928_140146_rotate_bulb_phase_phys60_30hz_12s_scratch_gpu4`，22:03:56启动GPU4/6144，最多15000轮。
分阶段奖励全量配置与GPU6相同，初始LR0.001/seed42，不加载历史或pilot权重。
本地相关CPU76项通过；32环境、13次PPO/416控制步短测通过，10项原生奖励检查、超时重置、有限值和模型更新通过。
物理dt/decimation/render_interval及回合时间是实际配置的预期差异，动作契约仅decimation变化；其余判据/碰撞/solver保留。
22:10:39正式到3轮，首个model0的optimizer step20/非零动量/77张量有限；配置、两契约及391源码/52运行库哈希核验通过。
GPU4约26636MiB，GPU6保持原进程到3709，GPU7为3MiB/0%；不声称物理精度等价、学习收益或成功率改善。
TensorBoard筛选`Phys60_30Hz_12s_Scratch`；证据见[GPU4启动记录](../logs/diagnostics/bulb_gpu4_timing_launch_20260928/README.md)。

# 2026-09-28历史：Randomized旋出修正为控制30Hz、decimation2、12秒，本地验证阶段

按用户要求，每动作物理更新8→2（与桌面Lift一致），回合仿真时间20→12秒。
用户随后明确控制30Hz；修正此前只改decimation却保留240Hz、导致控制120Hz的理解错误。
仅修改`RotateBulbRandomized/env_cfg.py`，训练及PLAY共用；物理dt改为1/60，
因此控制30Hz、回合上限360控制步/720物理步，render_interval同步为2。
Custom/Assembly默认、solver24/4、压力30Hz、奖励profile3、碰撞及成功判据未改。
GitNexus绑定本工作区main/4707074，impact LOW，直接继承PLAY；动态Gym注册另行核对。
真实配置函数/子类完整初始化的CPU替身检查通过，相关CPU39项通过；未启动Isaac，物理稳定性/速度/显存未测。
GPU4＋7原run及GPU6分阶段奖励run仍使用冻结副本的decimation8/20秒，未重启、未同步覆盖。
动作契约包含decimation；虽控制周期恢复30Hz，旧checkpoint仍不能直接按新物理子步配置无审查续训，保护保留。
见[修正后的解析结果及验证范围](../logs/diagnostics/bulb_timing_config_20260928/resolved_config_cpu_control30.json)。
旧`resolved_config_cpu.json`仅保留误改120Hz时的历史验证，不是当前配置。

# 2026-09-28：更正比较对象，用户指桌面抓取Lift而非旧灯泡

此前用9月13日旧灯泡4.399秒/512环境解释用户所说“之前抓取”的比较对象选错。
新核查：仓库Lift默认120Hz、decimation2、4秒回合；当前灯泡实际240Hz、decimation8、20秒。
同32控制步rollout物理更新64→256次；跑满回合每环境480→4800物理步，非实测耗时倍率。
抓取对象为方块/机器人自碰撞false；灯泡含SDF螺纹/耦合/释放约束、自碰撞true。
灯泡两项GPU配对容量从库默认2^21设到2^25；第三项本就2^25。3GiB collision stack按库文档为
pinned host，不能直接计入显存。历史6144灯泡PyTorch reserved峰值2.434GiB、整卡峰值25.965GiB，
显存大头在PyTorch之外，但没有逐项native分配清单，不能把全部差额归给某组件。
旧抓取当次保存配置和性能日志尚缺，仅对比源码默认与当前实际；未改训练、未新开GPU实验。
见[正确的抓取与灯泡对照](lift_vs_bulb_cost_2026-09-28.md)。下方旧灯泡比较只作历史补充。

# 2026-09-28 20:38：旋出采样占98.64%，完整回合约7.4分钟墙钟

此前误选的旧灯泡参照：9月13日原始记录为4.399秒采样、双卡各512环境、60Hz/decimation4；
当前各6144、30Hz/decimation8，每轮环境数12倍、每卡环境×物理子步计数24倍。
旧版已含机械臂/手和guided灯泡；不是新加机械臂所致。两版硬件/轨迹/任务版本不同，
这是工作量变化解释，未做同规模对照以排除额外退化；完整20秒回合两版均4800物理子步/环境。

只读当前接触0.1/跟踪40双卡run，3397–3496原始100点：采样23.3804＋PPO0.32294＝23.70334秒/轮；
实际日志间隔23.72965秒，前100点24.05795秒，最新窗口没有变慢。GPU4/7原worker/start_ticks保持。
实际仍每卡6144、rollout32、240Hz/30Hz、20秒/600步、TGS24/4；16/4仅独立小规模短测使用。
平均episode586.7484步、超时标量93.6167%、released99.9846%、成功0；
按当前并行速度折算跑满回合444.93秒，同时推进12288环境，非单环境play实测。
重算9月27日旧model1300分项：物理20.13974秒/84.35%、压力1.18867秒/4.98%、奖励0.95801秒/4.01%；
历史GPU kernel关节约束55.51%、碰撞37.55%。这是旧profiling定位，非当前checkpoint逐组件实测。
reset及任务各阶段耗时未单独测量；本次未新开GPU实验、改代码或重启训练。
见[详细分项与原始证据](rotate_bulb_episode_timing_2026-09-28.md)。

# 2026-09-28 20:53：分阶段接触奖励已在GPU6正式续训，首批PPO/保存通过

用户要求释放前减少接触奖励与机械臂动作、释放后引导抓取，并确认从当前双卡最新checkpoint续训。
Randomized旋出新增profile3：辅助接触释放前0，释放后单侧最高0.25/s、对向1/s、每回合最多4分；
关闭独立randomized_grasp支付；臂速成本释放前无需先抓稳、权重−0.3，释放后该项豁免。
保留真实旋转接触门控、手心引导、搬运40/失败−6、动作/观测、物理、随机化和最终成功判据。
代码仅Randomized/env_cfg.py及新增phase_rewards.py，标准Custom/Assembly保持。

新run `/home/xinyu/training_runs/20260928_123831_rotate_bulb_phase_contact_resume_gpu6`，
20:45:18启动GPU6/6144环境，从原GPU4＋7 run的model3500完整恢复，SHA以`2f084913c739842d`开头；
恢复学习率0.00013647718878928572，从3500追加11500，总编号上限15000。
CPU60通过、32环境原生10项及三次PPO通过；正式20:53:41核验到3504，
首个新model3500优化器step70040→70060、8个actor参数改变、77张量有限，两个契约匹配。
正式env.yaml除奖励版本/函数/权重/预算和路径外无差异，391源码/52运行库哈希通过。
GPU6约26296MiB，监督1174265/worker1174266；原GPU4＋7四PID/start_ticks保持，当时到3533。
本机TensorBoard HTTP200及新`PhaseContact_Resume3500_6144_GPU6_20260928`标量已验证。

首个pilot因验证脚本误认为零权重项会从active_terms移除而在加载前失败，证据保留；
修正检查后独立pilot2通过，正式未用pilot更新后的模型。当前仅启动/更新验证，未证明抓取改善；
最初几轮尚未完成20秒回合，初始零episode指标不能解释为学习退化。
见[完整修改、恢复及证据](../logs/diagnostics/bulb_phase_rewards_20260928/README.md)。

# 2026-09-28 20:07：model3300旋出回放已下载到本地及桌面

按用户要求在xinyu21空闲GPU6使用当前接触0.1/跟踪40双卡run回放；
19:59:48选取时最新保存点为model3300，当时训练3397轮。复制模型及契约至独立录制目录，
390源码/52运行库文件哈希与原冻结run一致；原生play、1环境、seed42、完整1200控制步。
视频40秒、1200帧、30FPS、1280×960，服务器完整解码通过；本地和桌面副本SHA256一致。
桌面文件`灯泡旋出_接触0.1_model_3300_20260928_1959.mp4`；原始过程保留，未作成功片段筛选。
原训练四个PID/start_ticks保持，录制后到3410轮；GPU6释放为3MiB。本次不是独立成功率评测。
见[视频与核验记录](../recordings/xinyu21_bulb_contact01_latest_20260928_115948/README.md)。

# 2026-09-28 19:54：双卡旋出到3383轮，释放指标99.976%，完整成功仍0

本次按用户要求只读核查xinyu21并接通TensorBoard。当前仍为接触0.1/跟踪40、
GPU4＋7共12288环境的`20260927_141100_rotate_bulb_contact01_tracking40_resume100_ddp`，
实际注册任务为继承Custom的Randomized旋出；19:54:06到3383/15000，最新保存model3300。
原worker1100555/1100556存在，start_ticks均78691089。72MB训练日志定向错误模式扫描为0。
主rank原始标量3283–3382共100点：released均值99.9761%，最高圈数2.999079，
reward10.95985；release_grasp和success_rate仍0，position_tracking均值2.4153e−6，
目标误差36.319cm，超时94.651%、掉落5.354%。释放后的抓稳/搬运尚无成功证据。
平均采样23.8755＋PPO0.2223＝24.0978秒/轮；这是训练日志窗口，非跨卡独立回合评测。
复用服务器TensorBoard66933，恢复本机http://127.0.0.1:6006/#scalars转发，HTTP200及
当前Contact0.1_Tracking40_Resume100标签/成功率接口已验证；未重启或修改训练。
见[原始标量、快照及连接说明](../logs/diagnostics/bulb_training_tensorboard_20260928_1954/README.md)。

# 2026-09-28：关节开销定位到Robot/手指链，16/4已短测及续训

按用户最新要求使用128环境分组、256环境短训练；不沿用旧严格物理误差淘汰门槛。
统一无接触且动态对象均唤醒，内部约束kernel（16子步）Robot＋Lamp36.22ms、Robot35.67ms、
Lamp3.37ms；进一步手部21DOF26.31ms、臂7DOF9.16ms。未发现重复Lamp attachment或驱动对抗mimic。
正式任务保留碰撞、240Hz/30Hz、全部DOF、三圈/释放/成功判据，只在独立case采用有效TGS16/4。
同model200、同初态，各256环境/12更新：整轮8.3005→6.9404s，吞吐986.93→1180.34/s（+19.6%）；
同分类关节/约束kernel228.630→173.237ms。实际env.yaml仅position min/max与log_dir不同。
候选再续24更新至`/home/xinyu/training_runs/20260928_joint_solver_gpu6/adapt256_16_4/train/model_234.pt`；
267结束训练回合227曾释放，非有限/大范围异常0。固定模型32完整回合释放13/32、成功0/32，
释放后抓持/搬运仍未解决；不把零成功视为等效，不将小规模收益外推6144。
五个固定tip省略诊断38→33links/DOF28仅缩短4.3%轮时，未作为正式合并；质量及传感映射保持原资产。
旧GPU6碰撞队列按用户切换要求暂停、证据保留；本轮短续训/评测结束GPU6已释放，GPU4/7原worker保持。
见[实际拓扑、分组、修改及结果](rotate_bulb_joint_solver_2026-09-28.md)。下方碰撞队列“进行中”为此前状态。

# 2026-09-27 23:52：collision_fast 碰撞副本与初筛完成，GPU6对照/适应训练进行中

本轮按用户新授权放宽接触精度门槛，冻结当前接触0.1/跟踪40的生产配置与model200；
240Hz、decimation8、TGS24/4、partitions8以及控制/观测/奖励/成功判据保持。
独立目录 `/home/xinyu/training_runs/20260927_1530_collision_fast_gpu6`；GPU4/7原训练保持。
1024固定初态/动作：基准10.300s、螺纹减面/SDF128 11.430s、Shell凸包11.006s、
Robot自碰撞关闭8.163s/32步；后者名义吞吐+26.18%，伴随接触/轨迹变化，尚非最终训练收益。
掌部28凸包合一的独立Robot资产已生成并通过离线属性检查，native形状/性能正测试。
首个6144对照因记录脚本未创建输出目录而在PPO前失败，日志保留；修复后32环境三次PPO通过，
完整模型/归一化/优化器恢复通过，实际配置除路径和测试环境数外与现行生产相同。
目前队列为掌部1024、重复基准1024、6144对照64更新；候选适应训练与最终选择未完成。
见[本轮记录](rotate_bulb_collision_fast_2026-09-27.md)。下方旧精度门槛淘汰结论不适用于本轮。

# 2026-09-27 22:45：物理性能实测完成，无可采用候选，GPU6已释放

本批固定旧接触1/跟踪40的Randomized（继承Custom物理）model1300，不代表下述新接触0.1续训。
GPU6/5090、6144环境、32步、dt1/240、decimation8、有效TGS24/4，排除启动/预热/trace后
5轮平均22.0080秒（采样21.8338、PPO0.1742），8933.46 transitions/s；历史25秒不可算作优化前后收益。
CUPTI定位GPU kernel时间：articulation/约束流水线55.51%、碰撞37.55%、触觉4.29%、报告/力读取0.062%。
原physics.engine是simulate+fetch_results外层，不是纯solver。
1024环境固定初态/动作完成两次基准及7项单因素：粗壳体凸化、关自碰撞/螺纹/attachment、
跳过压力、partition4/16均因物理/观测损失或无收益淘汰；未重试历史16/4，未采用诊断关闭功能。
另以固定model1300做baseline/partition4/16各64完整回合，均成功0/64、释放0/64；
partition4连接误差1.88→4.13mm、0.0373→0.0967rad；零对零不能证明完整任务质量合格。
新增3个隔离诊断/复现脚本，未修改生产物理、奖励、判据。原始日志、动作、初态、trace、配置与哈希已保存。
结束GPU6为2MiB/0%、无计算进程；原冻结390源码/52运行库及model1300保持。
GPU4/7由另一项工作切换新run，已单独记录，性能实验没有发出停启操作。下一最小实验为保留凹形的
外螺纹网格误差受限简化，再用同一fixture筛选；未实施。
见[完整报告与复现命令](rotate_bulb_physics_performance_2026-09-27.md)、
[原始证据](../benchmarks/rotate_bulb/physics_perf_20260927/README.md)、
[实验索引](../EXPERIMENTS.md#bulb-physics-perf-20260927-10)。下方保留生产训练状态与历史记录。

# 2026-09-27 22:25：接触0.1/跟踪40，已从刚停双卡run的model100正式续训

用户明确续训来源必须是`Rotate-Bulb_Contact1_Tracking40_12288_DDP_GPU4_7_20260927`自己的高释放保存点。
原run `20260927_041844_rotate_bulb_contact_transport_12288_ddp`按要求停在1425，model1400/日志保留。
比较其全部保存点之前完整50个released原始标量，选择train/model_100.pt：51–100均值39.2601%，
model200为30.0495%、model1400为3.5927%；这是历史训练窗口，非固定模型独立释放率。
旧GPU6 model7300的准备仅执行过三次隔离短测，已标记SUPERSEDED，未用于本次正式续训。

本地/服务器主项目与新冻结配置已将any_finger_contact从1改到0.1，position_tracking仍40、failure仍−6；
版本元数据1→2。实际env.yaml除该权重及版本外无差异，390项目文件仅一配置变化、52运行库文件相同。
新run `20260927_141100_rotate_bulb_contact01_tracking40_resume100_ddp`于22:16:07启动，
GPU4＋7各6144、共12288环境，32steps/env，总上限仍15000（从100追加14900）。
两rank完整恢复模型/归一化/优化器及原有效学习率0.00037290056934580207，父模型SHA以`1b4c9521`开头。
CPU64通过、32×2短测三次更新通过；正式前三次PPO同步通过，跨卡参数差0。
首个新model100优化器step2020→2040、actor权重改变、77张量有限、契约与原run一致；非仅成功启动。
22:25:42到110轮；106–110平均17.0748秒/轮，仅启动初期短窗，不代表后续稳定速度。
监督1100543、torchrun1100544、workers1100555/1100556；未操作其他GPU上的任务。
本机 http://127.0.0.1:6006/#scalars 已验证HTTP200，新标签
`Rotate-Bulb_Contact0.1_Tracking40_Resume100_12288_DDP_GPU4_7_20260927`可见。
新奖励的释放、抬升与完整成功改善尚未验证；仿真状态重新reset，不是采样轨迹逐步续接。
见[来源、修改及本地证据](../logs/diagnostics/bulb_contact01_resume_ddp_20260927/README.md)、
[实验记录](../EXPERIMENTS.md#bulb-contact01-resume-ddp-20260927-09)。下方保留历史状态。

# 2026-09-27 21:37：model1300录像已下载；256回合未释放，机械解锁测试通过

新双卡run到1351轮；1252–1351原始100点released均值3.2642%，完整成功/释放抓稳0，每轮23.7418秒。
GPU6隔离回放model1300：128环境、1200步、256完整20秒回合，释放0/256、成功0/256、全部超时；
平均最高2.87930圈、末态2.87542圈，拇指接触力全0；无已达双门槛但未解锁样本，角度/轴向基本同步。
每回合接触+15.718、停滞−0.365，总回报24.983；慢速新进展重置停滞计时，支持未完成任务仍高收益的解释。
独立4环境注入状态测试确认双门槛、连接关闭、自由落体、碰桌及局部reset均通过；未发现释放开关硬性卡死。
训练有探索噪声，确定性play没有；日志窗口与独立回合口径不同，未做权重因果消融，也未改训练。
40秒/1200帧model1300录像及桌面副本完整解码/哈希通过，原训练PID/start_ticks保持，GPU6已释放为2MiB。
见[本次录像、实测及局限](../recordings/xinyu21_bulb_contact_transport_20260927_132600/README.md)、
[实验索引](../EXPERIMENTS.md#bulb-contact-release-diagnosis-20260927-08)。下方保留历史状态。

# 2026-09-27 18:49：释放下降与静态接触收益冲突，尚未做因果消融

新双卡run日志截至917轮。500–599→800–899：释放3.297%→1.542%，达到三圈2.122%→0.855%，
最高圈数2.9405→2.9496、reward23.5648→24.2749，超时98.334%→99.389%，释放抓稳/成功仍0。
最符合证据的解释是接近三圈后保留接触收益、避免释放风险的局部高收益行为；接触最高1/s对比停滞最高-0.3/s，
而实际位置跟踪信号约1e-4、接触约0.758，单侧接触高分不等价于释放后的对向抓持。
近期std下降可能辅助固化，实际gamma0.999；未证明是短折扣、随机分布改变或数值崩溃所致。
仅主rank日志及奖励机制分析，没有最新物理轨迹/条件掉落率/单因素消融，未改训练。
详见[下降原因、趋势图与局限](../logs/diagnostics/bulb_release_decline_20260927_1845/README.md)。

# 2026-09-27 18:43：释放指标最新909轮，最近100轮均值1.61%

当前新双卡run的TensorBoard读取至909轮：`Metrics/object_pose/released`单点4.0104%，
最近890–909共20点均值1.8463%，810–909共100点均值1.6108%；完整任务`Train/success_rate`仍0。
释放状态指标与达到三圈、释放后抓稳、完整搬运成功不同；统计为主日志rank标量窗口平均，非跨卡独立评测。
原始曲线和摘要保存在`logs/diagnostics/bulb_release_rate_20260927/`。本次仅查询，未改训练。

# 2026-09-27 16:56：当前奖励及位置随机化已核实

核对GPU4＋7新run实际env.yaml、两rank初始化契约、5个源码文件与bank哈希；本次只读。
21个奖励配置条目中17个非零，接触权重1/最高1分每秒，位置跟踪权重40/每回合最多40分，失败一次-6。
位置随机化已启用：支架XY±10cm，目标相对支架XY±15cm、Z+28–42cm；支架Z/姿态及臂初态固定，手零位。
运行时从432个支架点×8个配对目标共3456行均匀抽样，每个环境startup/reset更新；两rank种子42/43。
不是五个坐标各自连续独立采样，目标相对高度不等于净抬升距离。详见
[全部奖励及随机化审计](../logs/diagnostics/bulb_rewards_randomization_20260927_1653/README.md)。下方保留训练状态。

# 2026-09-27 16:49：新训练已释放，目标接近奖励微增，未证实稳定抬升

GPU4＋7新奖励12288环境到604轮，最新model600。TB快照截至601，最近502–601释放指标均值3.3071%。
相邻50轮502–551/552–601：释放3.3101%→3.3040%，position_tracking均值0.00010694→0.00013523，
非零日志点42/50→47/50；release_grasp及成功率仍0，位置误差0.35008→0.35106m，稳定抓持奖励近零。
当前运行源码已核对：position_tracking奖励释放后带软接触的三维目标接近增量，没有独立正向Z位移条件；
因此只能确认极弱的后段目标接近信号，不能据此宣称已学会抬升或给出抬升高度。
主rank训练日志分析，非独立评测；本次未修改或重启训练。
见[释放/抬升信号记录](../logs/diagnostics/bulb_release_lift_20260927_1647/README.md)。下方保留历史状态。

# 2026-09-27 16:16：新双卡训练516轮，本机TensorBoard已接通

GPU4＋7新奖励12288环境正常运行，16:15:40到516轮，最新model500，原worker start_ticks保持。
本机 http://127.0.0.1:6006/#scalars 已转发至21服务器，HTTP200及当前run/23条标量读取通过。
选择`Rotate-Bulb_Contact1_Tracking40_12288_DDP_GPU4_7_20260927`；重连命令和曲线说明见
[本次查看记录](../logs/diagnostics/bulb_training_tensorboard_20260927_1614/README.md)。
TB截至512轮，最近413–512共100点：23.54579秒/轮，采样23.30351/PPO0.24228秒；100轮约39分钟。
总奖励23.0728、最高圈数2.93669、最终圈数2.89036，接触/前段旋转有进展；
释放指标均值3.847%、release_grasp为0，position_tracking均值5.8242e-5（80/100点非零），
全部513个训练成功率点为0，尚未学会完整释放后抓稳/搬运。仅主日志rank训练统计，非独立评测。
本次不修改或重启训练。下方保留历史状态。

# 2026-09-27 15:05：GPU6旧训练暂停，性能对照完成，GPU6已释放

旧GPU6回退/停滞惩罚旋出在7352轮正常停止，model7300/日志与哈希保留，已标记USER_PAUSED。
它仍是接触0.1/position_tracking4的6144环境旧奖励，不能自动恢复。GPU4＋7新奖励12288训练继续。
15:04:53新训练到339轮、最新model300，最近20轮25.2016秒，采样24.96435/PPO0.23725秒。
GPU6使用新训练model200私有副本完成6144环境8轮基准：自然采样25.256秒/PPO0.178秒；
同步分项物理84.49%、压力5.07%；其余为接触、奖励、观测等。模型/契约/优化器核验通过。
关点云显示25.5130秒，对照25.4339秒，无整体加速；主线程绑核ABA约1.33%短窗收益，已恢复原设置。
单卡4096实测20.0342秒，但样本数减少、单卡吞吐下降；三卡各4096可保持12288，约21%轮时缩短仅为外推，未实测/启动。
三个有效case各8次更新，诊断退出，GPU6为2MiB/0%；旧checkpoint哈希保持，生产390源码/52运行库及进程start_ticks保持。
本次只更新独立诊断入口与记录，没有采用未经验证的生产优化，没有降低物理精度或删触觉。
xinyu20实查GPU0约2.6GB可视化占用、0%利用率，有余量但非独占；其他卡有计算任务，本次未占用20。
见[性能测试记录](bulb_runtime_optimization_2026-09-27.md)。下方保留历史状态。

# 2026-09-27 13:01：新接触/搬运奖励双卡12288环境已正常训练

Randomized旋出any_finger_contact从0.1提至1、position_tracking从4提至40，failure仍−6；
本地、服务器主项目与新冻结快照哈希一致。CPU163通过，单卡及双卡3轮原生PPO通过。
用户确认后停止旧GPU4（9156轮，model9100保留）和临时GPU7单卡，改为4＋7共同训练。
GPU6原回退惩罚训练PID1002974继续至7086轮，start_ticks未变；插销和装配继续暂停。

新run `20260927_041844_rotate_bulb_contact_transport_12288_ddp`，12:51:53从头启动，
每卡6144、共12288环境，seed42/43、最多15000轮、每轮393216条采样。
13:01:00交接到日志编号7；监督1062658、torchrun1062659，worker1062666/1062667。
第1–5轮平均19.39秒，采样19.066秒/PPO0.323秒，约20281条/秒；GPU4/7显存25957/26311MiB。
正式首轮梯度均值及前三轮参数一致性检查通过，model0有限/优化器step20、契约、源码和TB入口均通过。
默认NCCL大数据通信独立复现非法内存访问；仅新run使用LL协议后解决，共享软件包未修改，失败证据保留。
新奖励尚无学习成功率或抬举改善结论。见[奖励及运行记录](rotate_bulb_contact_transport_2026-09-27.md)、
[核心核验快照](evidence/bulb_contact_transport_ddp_20260927.json)。下方保留历史状态。

# 2026-09-27 10:21：新奖励旋出 model6500 录像已下载

选取10:11:20时GPU6新run最新保存的model6500，在空闲GPU7用对应冻结源码/运行库单环境回放。
40秒/1200帧/30 FPS/1280×960；服务器、本地完整解码及桌面副本哈希一致，含回退/停滞奖励。
抽帧仍主要在支架附近，未见明显抬高搬运；灯泡受手遮挡，不能凭画面确定约束释放，非独立成功率评测。
10:19:46新训练到6603轮，原GPU6/GPU4 PID及start_ticks保持，GPU7已回到3MiB。
视频见[本次录像与核验](../recordings/xinyu21_bulb_progress_penalty_20260927_021120/README.md)，
桌面副本为“灯泡旋出_回退惩罚_model_6500_20260927.mp4”。下方训练指标仍为各自时间窗口。

# 2026-09-27 10:03：新惩罚旋出已正常训练，释放接近100%，完整成功仍0

GPU6新run `20260925_152318_rotate_bulb_progress_penalty_6144_gpu6` 已于9月25日从头启动，
6144环境/15000轮/seed42，PID1002974，当前6553轮，最新model6500。
装配当时按用户要求停止于7380、model7300保留；标记USER_PAUSED。原GPU4旋出继续，插销保持暂停。
新run最近100轮释放指标99.917%、最高/最终圈数2.999127/2.999089，但成功及释放后抓稳奖励始终0；
稳定接触/搬运均值仅2.59e−8/2.81e−8。释放前卡点减少，释放后的抓持/搬运仍是瓶颈。
最近100轮平均20.184秒（19.57–20.63），采样20.015秒/PPO0.169秒；未检出碰撞栈溢出或CUDA OOM。
新惩罚确实生效，源码/运行库/契约和模型有限性、优化器更新、TensorBoard入口均已核验。
这是训练日志分析，非最新模型独立评测。详见[当前训练记录](rotate_bulb_progress_penalty_training_2026-09-27.md)。
下方“尚未重启”仅为9月25日23:16历史状态；后续23:23启动已完成。

# 2026-09-25 23:16：Randomized旋出回退/停滞惩罚已加入两端项目，当前训练仍为旧奖励

用户授权本地与xinyu21项目添加惩罚。新增unscrew_regression：0.01圈容差，新增最差回退按−2/圈，
每回合封顶−2；unscrew_stall：接近3秒、有效新进展后宽限2秒、1秒渐增到−0.3/秒，封顶−4。
释放后免罚，物理失败当步不叠加，动作/观测/三圈/碰撞/成功判据及装配配置保持。
本地与服务器相关CPU测试各134通过；本地原生2环境/200步、212检查通过。
旧model3700的64回合离线重算平均12.724→8.647，回退/停滞分别扣0.454/3.623；非训练改善证据。
6文件定向同步到/home/xinyu/mulyi-model-tac、备份和SHA256核验通过。
**GPU4/PID852306仍使用原冻结workspace，本次未重启训练，新惩罚尚未在该进程生效。**
装配GPU6/PID871712保持，插销未恢复。未来按新项目启动/恢复时应标记奖励版本切换。
见[实现及参数](rotate_bulb_progress_penalty_2026-09-25.md)、
[同步核验](../logs/diagnostics/bulb_progress_penalty_20260925/sync_report.json)、
[实验记录](../EXPERIMENTS.md#bulb-progress-penalty-20260925-04)。下方保留历史记录。

# 2026-09-25 22:58：旋出后段引导曾触发，但当前信号极弱

读取正式训练TensorBoard全精度0–4194轮：release_grasp与success全部4195奖励点为0；
released_stability/position_tracking并非全程零，分别有1670/1619个正值日志点。
最近4095–4194轮，二者均值5.456741e−8/7.400595e−7，非零26/100和18/100；
前段接近、接触、旋转、抓持和手心奖励每轮非零。日志点非回合触发次数。
这补充终端0.0000精度不足，不改变此前64回合回放后段全0的样本结论。
见[完整精度核查](../recordings/bulb_release_diagnosis_20260925/guidance_analysis.md)及同目录guidance_audit.json。
仅只读核查，未改训练。下方保留历史记录。

# 2026-09-25 22:41：旋出释放停滞诊断，未改变训练

当前训练4047–4146轮：最高进度日志均值2.969814圈、最终2.578632圈，released指标均值1.9314%，成功0%。
对用户晚间视频对应model_3700做32环境/seed42/1200步冻结回放，64完整回合释放1/64、成功0/64、超时64/64。
未释放63回合平均约5.39秒到2.959圈，最后回退到2.737圈；≥2.9圈仅平均0.410秒，非持续精准停在门槛。
每回合仍平均获得12.724分（旋转5.766、手心3.962、接触1.654、抓持1.000），后段奖励全0；超时无failure扣分。
唯一达到双门槛的轨迹下一控制步自动解锁，机制能工作；支持释放前阶段局部最优假设，尚无奖励消融证明因果。
手心预算前几秒已基本领完，不是无限悬停刷分。建议验证释放后抓稳衔接与回退信用分配，未实施修改。
源码/运行库/模型哈希一致，原训练PID/start_ticks保持，GPU1诊断结束已释放。
见[诊断记录及曲线](../recordings/bulb_release_diagnosis_20260925/README.md)、
[实验索引](../EXPERIMENTS.md#bulb-release-diagnosis-20260925-03)。下方保留历史记录。

# 2026-09-25 20:07：晚间最新模型录像已下载，两项训练仍未达到任务目标

旋出GPU4/PID852306到3817/15000，装配GPU6/PID871712到6940，仍是Stage0-free自由抓取，
自3422起新增3519次更新，未晋级抬起。最近100轮日志成功率均为0%；旋出最大进度日志均值2.966圈、
释放比例约1.681%。装配20:00附近最后10000完整训练回合严格抓持0/10000。
完整训练日志定向扫描未检出原碰撞栈溢出、CUDA OOM、Python Traceback和PhysX error。

19:59:16选取时最新保存的旋出model_3700、装配model_6900已分别在空闲GPU1/7回放并下载：
各40秒/1200帧/30 FPS/1280×960；两端完整解码、本地FFmpeg与桌面哈希核对通过。
装配本次回放严格抓持0/8，非reset帧可见手臂移开、灯泡留桌；失败画面完整保留。
原训练PID/start_ticks、原checkpoint哈希及契约保持，训练未暂停，录制GPU已释放。
插销仍按用户要求暂停。本次未改奖励、任务代码、原训练配置或启动新训练。

视频见[晚间录像目录](../recordings/xinyu21_bulb_latest_20260925_120000/README.md)，桌面有中文副本。
详情见[实验记录](../EXPERIMENTS.md#bulb-latest-video-20260925-02)
和[核心核验快照](evidence/xinyu21_bulb_video_20260925_2000.json)。
下方保留各自时间点的历史记录。

# 2026-09-25 13:24：两项灯泡训练仍运行，最新保存模型录像已下载

GPU4旋出PID852306到2934/15000；GPU6装配PID871712到6026，仍在Stage0-free自由抓取，
自3422起新增约2605次更新，未晋级抬起。录制前后两个PID/start_ticks一致，插销保持用户暂停。
两项最近100轮日志成功率均值均为0%；旋出最大进度日志均值2.963圈、释放比例约1.861%，
装配最后10000个完整训练回合严格抓持0/10000。训练持续更新不等于已学会任务。
本次两份完整训练日志定向扫描未检出原碰撞栈溢出、CUDA OOM、Python Traceback或PhysX error。

13:10:31选取时最新保存的旋出model_2900与装配stage0_free/model_5900已在空闲GPU1依次回放。
各40秒/1200帧/30 FPS/1280×960，服务器与本地完整解码、桌面副本哈希核对通过；
装配回放严格抓持0/8，非完整装配评测，原始失败画面保留。
旋出可见手在灯泡上方，装配非reset帧可见手臂离开、灯泡留桌；尚未可靠完成当前目标。
对应快照、运行库、课程settings和契约核对通过；仅录制副本调整单环境/渲染/容量预分配。
原训练、checkpoint及源码未修改；GPU1录制后已释放。

视频和完整记录见[录像目录](../recordings/xinyu21_bulb_latest_20260925_051200/README.md)，
桌面有中文命名副本。见[实验记录](../EXPERIMENTS.md#bulb-latest-video-20260925-01)
与[核心核验快照](evidence/xinyu21_bulb_video_20260925.json)。
后续应区分训练窗口、8回合回放和独立评测；没有据此继续修改奖励或启动新训练。
下方保留各自时间点的历史记录。

# 2026-09-24 18:24：插销已暂停，装配扩容后进入自由抓取阶段并正常更新

已停止GPU6插销（末次日志13738、正常退出0），原model_13700及日志保留，哈希核对一致。
这是用户明确暂停，后续不要依据旧训练目标自动恢复插销。GPU4旋出PID852306保持运行。

装配碰撞栈从2GiB扩为3.75GiB（1.875倍，4GiB超过uint32上限）；接触、patch、三项配对、heap、temp共7项翻倍。
8192环境首试在首轮采样时CUDA OOM，失败日志保留；现保持所有扩容值，改4096环境以满足GPU6显存。
新run `/home/xinyu/training_runs/20260924_100337_bulb_curriculum_buffers_4096_gpu6`，
监督865565、课程启动器865566、当前自由阶段训练871712（start_ticks51341145）。
从固定Stage0 model_3419恢复策略/优化器/归一化/课程状态，新增4轮后保存model_3422，
三seed冻结评测91/100、97/100、83/100，合计271/300，通过原门槛后自动转入Stage0-free。
seed42/reward v2，每次阶段启动最多新增9000次更新；源码及训练库与原装配快照哈希一致，仅调整容量及环境数。

已验证自由阶段PPO3422–3424新增3轮、实际缓冲区参数、三份契约及127个有限标量，显存约29.0GiB。
未再触发原碰撞栈溢出或OOM；这只证明本次首批更新，长期稳定和完整装配成功仍未验证。
model_3422已保存且25个模型状态张量发生变化、优化器step68400→68480、所有张量有限；
自由阶段尚无新checkpoint，原100轮/晋级保存机制保持。
TensorBoard6006标签 `Bulb_Assembly_Buffers375G_4096_GPU6_20260924/stage0_free` 已实际可见。
见[实验记录](../EXPERIMENTS.md#asm-physx-resume-20260924-04)和[核验快照](evidence/bulb_curriculum_buffers_20260924.json)。
下方保留各自时间点的历史记录。

# 2026-09-24 16:35：手心上方奖励及高Z随机目标已同步，GPU4重新训练

用户根据录像要求旋出时鼓励手心在支架上方，并提高目标Z/加入高度随机化。
当前Randomized任务新增实际palm相对支架XY对齐及+10–18cm高度的软奖励；保持预算2分、
新增旋出进度预算2分，释放/失败后停止给新信用。原三圈、3cm/1秒成功判据及碰撞保持不变。
目标原有Z随机+18–30cm，现改为+28–42cm；3456对新库保留原432个支架位置，仍是有限筛选分布。
随机任务契约v2，输入仍687维；本次从头训练，不复用旧v1策略/优化器。

已同步xinyu21主检出及独立训练快照；新run
`/home/xinyu/training_runs/20260924_081649_rotate_bulb_palm_high_z_gpu4`，
GPU4/6144环境/15000轮/seed42；训练PID852306、监督852305。
CPU134通过/2跳过；原生8环境96项检查与32环境3轮PPO通过。
正式PPO0–3已完成、模型保存/优化器更新/张量有限/新契约/手心指标均核验通过，任务继续运行。
TensorBoard6006标签 `Rotate-Bulb_PalmAbove_HighZ_GPU4_20260924`。

旧随机旋出在11243轮正常退出0，原model_11200保留并核对停止前后哈希；插销PID242935未重启。
学习效果和独立成功率未验证，后续用新模型回放/完整回合评测检查姿态和搬运；不能拿旧日志成功率替代。
详情见[实验记录](../EXPERIMENTS.md#bulb-palm-high-z-20260924-03)和[核验快照](evidence/bulb_palm_high_z_20260924.json)。
下方保留各自时间点的历史记录。

# 2026-09-24 15:56：当前插销与随机旋出模型录像已下载

按用户要求在xinyu21 GPU6依次录制单环境回放，使用选取时最新保存的插销model_13400/seed44、
随机灯泡旋出model_11100/seed42，沿用各自训练快照、overrides及契约；未停止或重启原训练。
两段均40秒、1200帧、1280×960、30 FPS；两端OpenCV完整解码、本地FFmpeg解码及SHA256通过。
已下载至[录像目录](../recordings/xinyu21_peg_bulb_latest_20260924_074033/README.md)，桌面有中文命名副本。
插销抽帧可见脱手，原始连续视频保留；灯泡抽帧可见持物离开底座，不据此声明独立成功率。
原checkpoint哈希未变，原PID242935/489880启动时间未变，完成时仍分别训练至13498/11168轮。
仅录制进程使用单环境的小容量PhysX预分配；未更改原物理约束或训练配置。
详见[实验记录](../EXPERIMENTS.md#peg-bulb-video-20260924-02)。

# 2026-09-24 15:30：装配固定Stage0通过，切换自由阶段物理溢出停止；旋出与插销仍在训练

经跳板只读登录xinyu21，核对进程、GPU、监督状态及原始训练/评测记录。
8192环境新奖励课程已完成固定灯泡Stage0提名，保存`curriculum/stage0/model_3419.pt`；
seed401/402/403冻结评测分别92/100、99/100、97/100，逐回合复核一致，合计288/300（96%）。
这只证明固定灯泡稳定抓持阶段通过，不代表自由抓取、抬起或完整装配成功。
随后加载该模型切换`stage0_free`，于09:53:46（北京时间）出现PhysX
`collisionStackSize buffer overflow`并丢失接触；监督标记`fatal_simulation_error`/exit -9，
课程进程已不存在，自由阶段无checkpoint，未进入后续阶段。原GPU1现由其他用户占用。
固定Stage0状态记录6086023训练回合/896532480转移；仅重数末10000回合，成功8721/10000，
未全量扫描25GB左右逐回合日志，不能将此窗口外推为全程成功率。

GPU4随机旋出PID489880仍运行：11107/15000轮，最近100轮日志成功率算术均值98.7891%，
最新保存model_11100；GPU6插销PID242935仍运行：13450/14200轮，
最近100轮日志成功率均值97.8783%，最新保存model_13400。两者未做本次独立评测。
本次未同步代码、修改配置、重启或停止训练；装配恢复前需解决自由阶段碰撞缓冲区问题并重新核实GPU资源。
证据及限制见[核查记录](../EXPERIMENTS.md#training-status-20260924-01)与
[只读快照](evidence/xinyu21_status_20260924_153033.json)。

# 2026-09-23 10:41：8192环境装配课程重启，每阶段9000轮

用户要求扩为8192环境，随后接受每阶段9000轮上限；达到原提名及独立评测门槛可提前晋级。
新run：`/home/xinyu/training_runs/20260923_104131_bulb_reward_v2_8192_9000_gpu1`。
GPU5已被其他用户占用，改用空闲GPU1；seed42、OLD_REWARD=False、从零Stage0，
498份项目文件与45份RSL运行库文件哈希均与前次1024环境快照一致；未修改Python。
监督737978、课程启动器737979、训练737980，10:41:31启动，10:53核验已完成PPO 0–8轮。
model_0含78个有限张量，实际配置8192/9000、669输入/28动作及127个TensorBoard标签核验通过；
8470完整回合reward均有限，成功0/8470，8548回合success日志与原runtime一致；尚无学习效果结论。
显存20872/32607MiB（约20.4/31.8GiB），早期吞吐约1.04–1.25万转移/s；没有检出所查原生致命错误。
此前10:39启动的2000轮版本已在首次PPO更新前按用户新要求停止，原始日志保留。
TensorBoard 6006标签：`Bulb_Assembly_RewardV2_8192_9000_GPU1_20260923/stage0`。
详见[实验记录](../EXPERIMENTS.md#asm-reward-v2-8192-20260923-03)。

# 2026-09-23 10:28：新奖励Stage0预算结束，未学会严格抓持

实时只读核查：`20260923_010600_bulb_reward_v2_gpu5`已于07:56:06完成2000轮，
65536000转移、442422完整回合，严格Stage0成功0/442422；未晋级，保存model_1999。
监督exit code 2表示阶段预算用尽未获晋级提名，未检出所查PhysX/CUDA/溢出/Python致命错误。
三个原进程均不存在，GPU5当前空闲；没有自动续训或更改环境数。

训练显存峰值7790/32607MiB（约24%），1024是沿用上次课程的保守选择，未测容量上限。
最近100轮吞吐中位2656转移/s，采样12.2265s、学习0.1135s；训练期GPU利用率采样中位81%。
扩容可优先测试4096，但尚无4096/6144实测，不按空闲显存线性宣称速度提升。
最后10000回合仅201个曾满足双侧各>1N，最长严格稳定抓持11步/0.367s，要求为31步/1.033s；
原runtime独立日志最后10000回合也为0成功、203个宽口径grasp事件，未到stable_grasp。
正奖励仍主要来自接近：reach均值0.270124，contact_stability仅0.000350；新方案尚未证明学习有效。
证据及全回合统计见[ASM-REWARD-V2-STATUS-20260923-02](../EXPERIMENTS.md#asm-reward-v2-status-20260923-02)。

# 2026-09-23：新奖励装配课程GPU5启动记录（历史，现已预算结束）

按用户授权，同步8个奖励相关Python及2份文档，备份/哈希核对完成。
GPU5旧完整装配正常停于9328轮，model_9300哈希保持；GPU4旋出和GPU6插销未停止或重启。
新run `20260923_010600_bulb_reward_v2_gpu5`，01:11:13开始，1024环境/seed42/OLD_REWARD=False，
从零Stage0按原门槛逐阶段推进，每阶段最多2000轮，3冻结评测seed复核后才晋级。
监督724505、启动器724506、实际训练724507；01:14:14已核验PPO0–7轮、model_0保存、
78个张量有限、28动作/669输入、127个TensorBoard标签及奖励/失败诊断有效。
服务器CPU161通过/2跳过；短PPO6轮/32回合通过，成功0/32；正式早期诊断0/1094，不是冻结成功率。
6006标签 `Bulb_Assembly_RewardV2_Curriculum_GPU5_20260923/stage0`；新训练继续运行，监督器不自动重启失败。
见[同步、启动与停止说明](rotate_bulb_reward_gpu5_2026-09-23.md)、
[ASM-REWARD-V2-TRAIN-20260923-01](../EXPERIMENTS.md#asm-reward-v2-train-20260923-01)。

# 2026-09-22：Assembly reward v2 实施，1000环境短测

用户已确认下方设计；已实现默认 `OLD_REWARD=False` 的有限预算进展/稳定/完成奖励，
`OLD_REWARD=True` 保留旧奖励。加入全部分项TensorBoard及Stage0–3多标签失败诊断/逐回合JSONL。
观测、动作、模型、reset分布及原成功函数不变；新reward只读取成功标志，不重复推进保持计时。
定向CPU回归161通过、2跳过；Stage0原生1000环境×180步、Full 1000×620步均通过，进程均已停止。
首轮默认内存池PhysX OOM的结果已拒绝；通过的重测仅限制检查脚本的GPU预分配池，未改生产物理配置。
没有PPO更新、远程操作或完整学习评测；两条探针均成功0/1000，不代表策略学习结果。
原物理状态库契约逻辑保持，reward开关/版本在独立文件记录，开关不改变兼容初态库选择。
范围、命令、旧奖励计时差异与证据见[实施记录](rotate_bulb_reward_implementation.md)。

# 2026-09-22：Assembly Curriculum 奖励重设计方案（历史：现已确认并实施）

完成当前 reward、物理状态、连续保持、reset 混合与晋级流程的源码审计，交付
[重设计方案](rotate_bulb_reward_redesign_plan.md)：各阶段新增进展、有限稳定信用、一次完成奖金，
包含公式、权重、每秒/回合/折扣预算及消融设计。候选 Full 非最终成功正预算≤30.5、最终成功一次+80；
均为待验证建议，不是已实施配置或成功率结论。历史 Stage0 停滞仍需逐步轨迹定位计时中断原因。
本次仅文档分析及本地索引维护，未修改 Python、未连接远端、未启动训练/仿真；等待用户确认后再实施。

# 2026-09-22：Assembly Curriculum 全阶段奖励提取完成

按当前工作区源码提取固定/自由Stage0、Stage1九档、Stage2、Stage3与Full的权重、公式、门控和实际累计口径。
见[完整清单](rotate_bulb_curriculum_reward_inventory_2026-09-22.md)：Stage0辅助上限7，Stage1/2上限15，
三者均无独立阶段完成奖金；Stage3只保留插入率5、动作成本及最终成功20，Full恢复14项原奖励。
指出连续保持与奖励不对齐、Stage3混合抬起起点的搬运引导缺口、Full持续项尺度/hover锁存风险；后两者尚未训练验证。
24项静态提取核验通过，五份核心文件仍与此前服务器快照一致；本次仅新增本地文档，未修改运行代码或连接服务器。
见[ASM-CURRICULUM-REWARD-INVENTORY-20260922-02](../EXPERIMENTS.md#asm-curriculum-reward-inventory-20260922-02)。

# 2026-09-22：服务器课程Stage0奖励核查，预算结束未晋级

只读核查xinyu21的`Bulb_Assembly_Curriculum_GPU7_20260921/stage0`：
原接近/接触项权重为0，相关奖励合并到`curriculum_progress`，并非无接触或无接近引导。
全部438574训练回合中434308曾达到原双侧>0.5N接触事件，但严格稳定抓持/Stage0成功均为0；
最后10000回合全部有该接触事件，课程累计奖励均值5.900383，严格成功仍0/10000。
历史最佳接触与非连续接触累计可得较高奖励，未与连续>1秒/滑移门槛充分对齐；具体计时中断原因待逐步轨迹。
已于9月22日约03:01跑完2000轮预算，以启动器code2结束，保存model_1999；原3个PID均已不存在。
本次未改远端或启动新仿真/训练。见[核查报告](rotate_bulb_curriculum_reward_audit_2026-09-22.md)、
[ASM-CURRICULUM-REWARD-AUDIT-20260922-01](../EXPERIMENTS.md#asm-curriculum-reward-audit-20260922-01)。

# 2026-09-22：Assembly Curriculum 初始场景视频完成

按用户选择制作各阶段初态合集：48秒、1600×900、24 FPS，八段画面含六个课程阶段及Stage1三个起点。
真实场景资产、全景到近景、常驻全景参考、中文阶段/目标/时限；1152帧完整解码及17项检查通过。
本地无真实Stage2/3成功库，两段为明确标注的IK几何示意，不写入训练状态库、不证明动态抓持。
全片冻结场景，仅移动相机；未加载策略、未训练、未连接远端，未修改运行源码或原资产。
交付[视频与说明](../recordings/bulb_curriculum_stages_20260922_171826/README.md)，
证据见[ASM-CURRICULUM-VIDEO-20260922-01](../EXPERIMENTS.md#asm-curriculum-video-20260922-01)。

# 2026-09-22：Assembly Curriculum Stage0 最新模型录像完成

实时核查确认GPU7课程已于03:01（Asia/Shanghai）用完Stage0的2000轮预算，未达晋级条件，
保留 `model_1999.pt`；监督器标记failed/exit 2，日志明确为预算用尽，未进入后续阶段。
按用户要求使用该最新模型及原489份源码/资产、45份RSL源码快照，在GPU2单环境录制，
保留Stage0固定灯泡、seed42、30Hz、28动作/669输入；课程恢复兼容性检查通过。
40秒/1200帧/1280×960视频已下载，服务器与本地OpenCV完整解码、本地FFmpeg解码及SHA256核验通过。
本地[录像](../recordings/xinyu21_bulb_curriculum_stage0_model1999_20260922_092007/bulb_assembly_curriculum_stage0_model_1999.mp4)，
桌面副本为 `/home/liuxinyu/桌面/灯泡装配课程_Stage0_model_1999_20260922.mp4`。
录像记录8个完整Stage0回合、阶段成功0/8；单seed确定性回放，不是完整装配独立成功率评测。
原模型哈希、GPU4/5/6训练PID及启动时间保持，未重启训练或修改运行代码。
见[ASM-CURRICULUM-VIDEO-20260922-01](../EXPERIMENTS.md#asm-curriculum-video-20260922-01)。

# 2026-09-22：光学面包板 M6 固定孔与打印定位柱

按用户确认的内六角圆柱头 M6、只用打印柱开合，在手根开口版上新增四个下模安装孔及两组上下模定位配合。
安装孔为Ø6.8通孔/Ø12平底藏头孔，承靠面距底24mm；四孔属于同一25×25mm网格。
下模两柱Ø8×6mm，上模一个Ø8.6圆孔和一个8.6×10.6长圆孔、深7.2mm；占地仍220×249mm。
型腔和52.4×40.4mm手根通口保留；仅原贴合面和下模背面发生孔/柱变化。CAD、STL闭合及六个位姿开模检查通过。
交付[打印文件、孔位图和装配STEP](../1_stl/breadboard_mount_256/README.md)。
台面长宽及有效牙深未提供：示例摆放需至少250×275mm；不加垫片时M6×30伸出台面侧6mm，不等于已确认螺纹适配。
未实打、未验证配合松紧/承载；原版文件保持不变。网格局部严格检查失败及后续离散精度核验均有记录。
见[MOLD-BREADBOARD-MOUNT-20260922-05](../EXPERIMENTS.md#mold-breadboard-mount-20260922-05)。

# 2026-09-22：上下模手根端面完整开口

按用户插线需求，在上一版底边再去6mm，原CAD保留Y≥-19，越过手根原端面0.2mm。
合模后形成52.4×40.4mm完整矩形出口；上下模占地均为220×249mm，仍各为一整件、背面平放。
上模192/下模294个非根端型腔面逐面一致；每半模根端4面按本次要求移除或缩短。
343个通道内部点为空、4个外围材料点存在，CAD切面面积与完整出口吻合；两STL闭合、单连通且绕序一致。
交付[新版打印文件与装配STEP](../1_stl/root_open_256/README.md)，原文件和上一版保持不变；未实际插线或打印。
见[MOLD-ROOT-OPEN-20260922-04](../EXPERIMENTS.md#mold-root-open-20260922-04)。

# 2026-09-22：上下模裁边适配256mm打印平台

按用户确认的256×256mm平放范围，上下模各保持一件，统一保留原CAD的X≥-60、Y≥-25，
占地由230×270缩至220×255mm；三组外部定位柱/孔削平/填平。上模196、下模298个型腔面逐面几何核对一致。
新[打印STL与CAD](../1_stl/trimmed_256/README.md)已平放、型腔朝上，两STL回读均为单个闭合连通网格、绕序一致。
新下模采用双精度网格，并按原CAD曲面补回3个三角面；459个局部采样最大距离0.001455mm，非全局误差证明。
255mm长边不含Brim/裙边，居中时两端各余0.5mm；未做实际切片或打印验收。
源STEP、原五个STL及既有URDF未改；旧URDF下模网格的未闭合记录仍适用。
见[MOLD-PRINT-TRIM-20260922-03](../EXPERIMENTS.md#mold-print-trim-20260922-03)。

# 2026-09-22：1.stp 五实体来源核查

原文件已有5个实体：STL编号01/03为上下模，02/04/05为拇指近根部的三个小实体，距模具约2.9–4.7mm。
单PRODUCT、无装配实例链接；实体数不表示设计零件数。额外15个曲面壳位于隐藏图层，
其中三组指尖状参考曲面的同参数网格哈希相同。小实体可能为分模/布尔操作留下的孤立型芯或残片，
原始特征历史缺失，无法确认属于手还是模具；未删除或合并。原STEP、STL及URDF不变。
见[位置图与分析](../1_stl/analysis/README.md)、[MOLD-ENTITY-ANALYSIS-20260922-02](../EXPERIMENTS.md#mold-entity-analysis-20260922-02)。

# 2026-09-22：模具上下分件与法向开合 URDF（几何/运动版）

根目录 `1.stp` 已拆为上下模两个 link，通过 prismatic 关节沿实际贴合面法向开合；
保留原 CAD 闭合位姿，法向相对原 Z 倾斜约1.14155°。150mm 行程为可改建模参数。
交付见 [mold_split](../assets/mold_split/README.md)：原文件逐字节归档、STEP/BREP 分件、参考几何、网格、交互预览及验证。
原始3个微小实体与15个附加曲面完整留作参考，未随意分配运动归属。URDF解析、零位坐标抵消及CPU离散开模间隙检查完成。
下模网格仍有7条开边，碰撞闭合验收未通过；STEP重导出有ppm级体积差异，另保存原实体BREP。
质量/惯性未知，未运行Isaac/PhysX或接入训练；下一步是按需要定位网格缺口、配置凹形碰撞及实测物理参数。
来源和失败尝试见 [MOLD-SPLIT-20260922-01](../EXPERIMENTS.md#mold-split-20260922-01)。

# 2026-09-21 19:53：D-Peg XY10cm/Yaw90 的 model_3000 视频已下载

按用户指定的 `Insert-D-Peg_XY10cm_Yaw90_GPU6_20260920`，在xinyu21单环境回放 `model_3000.pt`。
205项训练源码快照哈希一致，补充引用服务器既有资产/tacmap依赖；669维观测契约与训练一致。
GPU7、seed44、30Hz、40秒/1200帧/1280×960，OpenCV与FFmpeg完整解码、本地/远端SHA256通过。
本地[视频](../recordings/xinyu21_insert_d_peg_model3000_20260921_114155/insert_d_peg_model_3000.mp4)，
桌面副本 `/home/liuxinyu/桌面/D型插销_model_3000_20260921.mp4`。
原checkpoint未变，本次未重启训练；录像未统计独立成功率。
来源、初始化失败尝试及复现边界见[PEG-VIDEO-20260921-01](../EXPERIMENTS.md#peg-video-20260921-01)。

# 2026-09-21 19:49：GPU7 已切换为 Assembly Curriculum

按用户要求停止 GPU7 原三圈旋出：19:37:22 正常退出于1871轮，旧 model_1800 哈希不变。
19:46:50 启动 `20260921_113720_bulb_curriculum_gpu7`，独立课程任务、1024环境、seed42、从零Stage0；
监督529620、启动器529621、实际训练529622。按窗口及3冻结评测seed门槛逐阶段晋级，恢复20秒full；
每阶段预算2000轮，用尽未达标则保存停止，不直接跨阶段。
已确认正式PPO 0–3轮、model_0保存、78个张量有限、28动作/669输入、534项快照哈希一致。
服务器原生169/169检查、短PPO6轮/33回合通过；短PPO阶段成功0/33，尚未证明抓取学习有效。
6006标签 `Bulb_Assembly_Curriculum_GPU7_20260921/stage0` 已读取53项标量。
GPU4/5/6原作业PID和启动时间不变；失败探针及旧日志/模型均保留。
详见[启动与验证记录](rotate_bulb_curriculum_gpu7_2026-09-21.md)、
[ASM-CURRICULUM-GPU7-20260921-04](../EXPERIMENTS.md#asm-curriculum-gpu7-20260921-04)。

# 2026-09-21：Assembly Curriculum已同步xinyu21

课程实现、所需10cm hover配置和测试已同步至 `/home/xinyu/mulyi-model-tac`。
63文件哈希一致，服务器CPU测试124通过/4跳过，4个既有训练进程未重启；没有启动新训练。
四阶段提交分支已导入，main与Git暂存区不变；覆盖前已备份。
范围和证据见[同步记录](rotate_bulb_curriculum_sync_2026-09-21.md)。

# 2026-09-21：装配curriculum已实现，本地运行验证完成

新增独立 `BrainCo-Dexsuite-Revo3-Right-Bulb-Assembly-Curriculum-v0`，实现Stage0固定抓取、撤夹具检查、
Stage1桌面抓取/稳定离桌、Stage2真实持物库到hover、Stage3真实hover库旋入及20秒full。
原Custom任务、28动作/669观测、原reward/success与sim2real接口保留；原有运行文件只增加课程识别/训练钩子。
课程支持成功前缀联合恢复、当前策略3步接触复建、按来源记账、窗口调度、3冻结seed晋级验证和元数据续训。

已通过124项相关测试（4跳过）、固定/自由2env原生检查、联合状态恢复、2次PPO更新及1次续训。
冻结Stage0评测0/4，参数和归一化不变；这只验证链路，尚未学会抓取。
没有运行Stage1–3长训练、建立真实成功库、执行三训练seed对照或同步远端。首版单进程、回合概率混合、FIFO状态库。
下一步是按独立课程入口运行Stage0，测量闭合/承载/接触复建，再按门槛晋级；不自动恢复历史作业。
命令和实现边界见[实现记录](rotate_bulb_curriculum_implementation.md#0-当前实现优先于后面的历史设计)，
证据见[ASM-CURRICULUM-IMPL-20260921-03](../EXPERIMENTS.md#asm-curriculum-impl-20260921-03)。

# 2026-09-21：装配四阶段curriculum设计补充（未实施）

当前请求仅做分析和方案，已补充[当前源码审计](rotate_bulb_curriculum_analysis.md#8-本次请求当前工作区源码审计2026-09-21只读运行代码)，
并新增[实施设计](rotate_bulb_curriculum_implementation.md)：Stage0固定抓取及撤夹具迁移、
Stage1自由桌面抓取/离桌、Stage2真实持物状态搬运、Stage3真实hover起点完成下降旋入。
优先解决双侧抓持→承载；当前已有soft抓持，但单侧零接触会使lift/transport归零，insertion仍受hover硬门控。
保留28维动作、669维观测及语义、原最终判据；来源编号仅进训练日志，不新增阶段观测。
设计明确联合reset/驱动目标/接触重建/历史初始化、向量环境warmup的PPO数据隔离，以及至少3个独立训练seed对照。

本次复核已有两种动作模式各96回合的计数，与旧汇总一致；没有重新运行评测或核实远端作业状态。
当前本地10cm与历史4cm hover基线分开，旧结论/数据保留。只修改文档并刷新本地GitNexus索引；
未修改运行代码/reward/reset、未启动训练、未同步、未提交。预抓动态可行性、状态库和课程增益均待验证。
下一实施增量为独立Stage0；当前没有因本方案自动获准训练或资源操作。
证据与设计边界：[ASM-CURRICULUM-DESIGN-20260921-02](../EXPERIMENTS.md#asm-curriculum-design-20260921-02)。

# 2026-09-21 17:35：装配20秒全流程与curriculum只读分析完成

对GPU5零手桌面装配run的冻结`model_3400`，以3个seed分别做96回合确定性和96回合随机动作评测。
抓取为0/96与1/96；两种模式稳定抬起、hover、alignment、insertion及最终成功均0/96。
随机模式仅1回合触发位置奖励，下降/旋入奖励0/96；历史最近100轮约99.924%正奖励来自接近/朝向。
建议Stage0固定抓取→Stage1桌面抓取→Stage2持物搬运→Stage3旋入，再恢复完整20秒协议。
阶段reset、随机范围、晋级建议及历史聚合统计局限见[分析报告](rotate_bulb_curriculum_analysis.md)。
本次未修改项目代码/奖励/reset、未启动课程；运行版本仍为4cm，本地10cm不混入统计。
192回合奖励累计核验、策略/归一化不变、206份源码哈希核验通过；原训练360229继续至3577轮。
证据见[ASM-CURRICULUM-20260921-01](../EXPERIMENTS.md#asm-curriculum-20260921-01)。

# 2026-09-21：装配悬停改为10cm，保留下降段的密集反馈（本地）

按用户建议，`AssemblyGuidanceCfg.hover_clearance_m` 从0.04改为0.10。
已有接近/抬起/搬运/下降旋入密集奖励；新增 `descent_reward_length_m=0.04`，
将较长自由下降段按完成比例换算，避免仅抬高目标使起点下降奖励缩小约19倍。
入口和螺纹内同轨迹奖励保持，最终安装目标、成功判据、碰撞、14项奖励权重、动作/观测和20秒回合不变。
本地62 passed、2 skipped（缺pxr.Usd）；未运行Isaac、同步远端或重启。
15:49核查远端装配PID360229继续至3273轮，运行配置仍为4cm；此前视频也展示4cm旧配置。
这属于目标/奖励数值修正，尚未验证10cm抓持可达性、动态净空或学习改善。
分析与证据见[ASM-HOVER-20260921-01](../EXPERIMENTS.md#asm-hover-20260921-01)。

# 2026-09-21 15:44：装配手臂初态与安装目标视频完成

按用户要求，用当前装配 `20260920_124630_bulb_assembly_zero_hand_gpu5` 的206文件源码快照，
在GPU7独立单环境录制18秒初态展示，包含全景、近景、俯视及中文位置标注。
21个手关节为零、7个臂关节与训练一致；灯泡XY各±5cm/yaw±90°，seed43单样本。
青色为最终安装参考点/灯泡示意，绿色为螺纹尖端离入口4cm时的悬停参考点；后者不是成功目标。
机器人和灯泡在录制前后状态逐元素不变，没有策略或物理推进。
1280×960、24FPS、432帧完整解码通过，配置及状态核验25项全部通过，本地与远端SHA256一致。
桌面视频：`/home/liuxinyu/桌面/灯泡装配_手臂初态与安装目标_20260921.mp4`。
原装配PID360229持续至3254轮；未修改/重启训练。本次仅可视化，不是成功率评测。
证据见[ASM-INITIAL-GOAL-20260921-01](../EXPERIMENTS.md#asm-initial-goal-20260921-01)。

# 2026-09-21 15:58：原三圈 Custom 从 model_1400 在 GPU7 续训

用户指定 `Rotate-Bulb_3Turns_PhysX24-4_GPU4_20260921` 最新 checkpoint；已核对其指向原
`20260920_183647_rotate_bulb_physx24_4_pressure30_6144_gpu4/train`，最新为 `model_1400.pt`。
在 xinyu21 GPU7 启动新 run `20260921_074610_rotate_bulb_3turns_resume1400_gpu7`，
训练 PID506697、监督506696；15:50:03启动，15:58核验已完成1400–1403轮更新。
继承6144环境/seed42/30Hz/PhysX24–4/三圈guided/673维/关闭编码器，恢复策略、归一化、优化器，
学习率取父checkpoint的9.203429591335062e-05；按原15000轮目标追加13600轮，计划末标号14999。

446项源文件与17项补充USD资产已冻结并复核，动作/触觉契约完全一致；env YAML只变化路径。
首个新 `model_1400.pt` 的优化器步数28020→28040，归一化计数275447808→275644416，
策略权重更新且77个张量均有限，父模型哈希不变。首轮缺公共地面USD在PPO前退出，
已补齐并保留 `attempt1_missing_ground/`；不能将该次退出码0视为成功。
新标签 `Rotate-Bulb_3Turns_PhysX24-4_Resume1400_GPU7_20260921` 已在6006读到46项标量。
GPU4 Randomized489880、GPU5装配360229、GPU6 D-Peg242935仍在运行，未重启。
这里只确认完整续训及更新，尚无本次独立成功率评测。
见[验证证据](../benchmarks/rotate_bulb_resume_gpu7_20260921/resume_verification.json)、
[实验记录](../EXPERIMENTS.md#bulb-resume-gpu7-20260921-01)。

# 2026-09-21 13:14：Randomized 新任务已同步并开始正式训练

已新增独立 `BrainCo-Dexsuite-Revo3-Right-Rotate-Bulb-Randomized-v0`，保留原 Custom/Assembly。
臂初态固定、21个手关节归零；支架与灯泡整体X/Y各±10cm，目标相对支架X/Y各±15cm、Z+18–30cm。
训练均匀抽取离线IK/碰撞路径检查后的3456个离散位置—目标对（432个支架点，每点8个目标），
不是连续方盒全域保证。阶段臂幅度倍率1/5/1，687维新契约；三圈guided和原成功/失败条件保留。

已按用户授权定向同步到xinyu21的 `/home/xinyu/mulyi-model-tac`，冻结382个源码/资产文件。
旧旋出在13:02正常停于1417轮、退出码0，`model_1400.pt`哈希不变；装配360229和D-Peg242935未重启。
新run `20260921_050053_rotate_bulb_randomized_6144_gpu4` 于13:05:49启动，
GPU4/6144环境/15000轮/seed42/from scratch，监督489879、训练489880。
13:14已确认正式PPO迭代0–5、`model_0.pt`保存及77个张量有限，显存25255/32607MiB；
687维、零手/固定臂、24/4求解器、30Hz控制/pressure及新采样库哈希全部通过，未见致命错误。
6006标签 `Rotate-Bulb_Randomized_ZeroHand_GPU4_20260921` 已实际读取57条标量标签。

本机/服务器原生检查各121/121；服务器32环境3轮PPO通过。扩大离线回归424通过、3项既有测试失败、2跳过，
具体原因已披露。尚无新任务独立成功率评测；下一步先观察接近/抓持学习和完整任务曲线，再选择检查点评测。
详见[实现及验收](rotate_bulb_randomized_training_2026-09-21.md)、
[证据索引](../EXPERIMENTS.md#bulb-randomized-20260921-01)。以下条目保留各自历史核查时间。

# 2026-09-21：随机支架/目标与零手初态的新旋出任务方案（历史，后续已实施）

用户明确：支架与灯泡 X/Y 各随机约 ±10 cm，扩大上方/左右目标范围；机械臂保持当前初态，
21 个手部关节归零，让策略自行接近抓持。建议新增独立 Randomized 旋出任务，按接近/旋出/搬运
调整臂代价，保护原抓持初态 Custom 任务；**本次只完成设计与 CPU 初筛，尚未实现或注册新任务**。
支架完整几何在 ±10 cm 方形内最小桌边余量 14.2 cm；候选目标相对灯座 X/Y ±15 cm、Z +18–30 cm，
固定抓持朝向的局部 IK 为 207/243 合格，不能据此保证整个范围可达；还需零手初态、碰撞与路径验收。
未启动 Isaac、GPU、远端或训练。详见[方案](rotate_bulb_randomization_design_2026-09-21.md)、
[证据索引](../EXPERIMENTS.md#bulb-randomization-design-20260921-01)。下方运行状态为各自历史核查时间。

# 2026-09-21 10:54：当前旋出日志入口修复，model_1100 录像已下载

当前三圈 guided run 仍为 `20260920_183647_rotate_bulb_physx24_4_pressure30_6144_gpu4`。
10:35核查至1098轮，1047–1096轮训练日志的 `Metrics/object_pose/success` 平均99.5044%，
平均最大有效旋出2.999738圈、每轮27.6716秒；这是训练日志点均值，不是独立回合计数评测。
三圈计数采用3−1e−5门槛，而机械释放保留角度/轴向进度0.1%数值容差，两指标不可混称。

6006端口TensorBoard正常，但历史 `tensorboard/active_runs` 展示目录漏收录当前run。
新增软链接 `Rotate-Bulb_3Turns_PhysX24-4_GPU4_20260921` 后，本机现有SSH隧道HTTP200，
已读取1126轮成功率曲线；用户可在 `http://localhost:6006` 选择此run并搜索
`Metrics/object_pose/success`。未重启TensorBoard或训练、未移除其他日志入口。

10:46选择当时最新 `model_1100.pt`，在GPU3独立单环境录制40秒/1200帧/30FPS/1280×960。
原run源码446项、资产110项及动作/触觉契约核验通过；保留24/4、三圈guided、pose_only/30Hz，
仅缩小单环境预分配缓冲。首轮资源引用不完整且无视频，失败证据保留；修正隔离目录引用后完成。
完整解码通过，下载SHA256与服务器一致；桌面文件为
`/home/liuxinyu/桌面/旋转灯泡_model_1100_20260921.mp4`。
[录像及证据](../recordings/xinyu21_rotate_bulb_latest_20260921_024605/)、
[实验记录](../EXPERIMENTS.md#bulb-video-20260921-01)。10:54原训练PID459464继续至1138轮。
本次未统计录像回合成功数；训练最终策略质量与sim2real仍需独立评测。

# 2026-09-21：旋灯泡速度优化完成，训练已重启

当前采用三圈guided Custom，TGS位置24/速度4、物理240Hz、控制/pressure30Hz、pose_only新鲜位姿。
1024/3072/6144环境吞吐为3005.7/6751.1/9820.0 transitions/s，比原基线提高8.3%/19.9%/16.6%。
原基线3072复测5613.1，与初次差0.33%；最终3072相对复测仍提升20.3%。
16/4虽更快，但attachment误差超预定筛查线而被淘汰；大contact offset、rest=0、15Hz与NUMA调整均未采用。

新run `20260920_183647_rotate_bulb_physx24_4_pressure30_6144_gpu4` 于02:36:48（Asia/Shanghai）启动，
GPU4/6144env/seed42/15000轮，从头训练v2触觉契约；训练PID459464、监督459463。
02:44:29确认迭代0–4和model_0，77个保存张量有限，446个冻结文件哈希验证通过，无PhysX错误。
原model_100保留且哈希不变；装配360229/D-Peg242935未重启。

奖励/碰撞/三圈门槛/终止与成功条件保持。固定早期model_100的最终配置评测仍为0/64严格成功，
不能声称最终训练成功率或sim2real不下降；24/4的attachment角误差仍高于正确参考，完整数值已披露。
后续应在新契约训练收敛后做独立完整任务评测；当前没有启动额外长期对比训练。

[总报告](rotate_bulb_speed_optimization.md)、[基线](rotate_bulb_speed_baseline.md)、
[结构分析](rotate_bulb_physics_analysis.md)、[重启验证](../benchmarks/rotate_bulb/restart_verification.json)。
独立分支`perf/rotate-bulb-speed-20260920`，优化审查基线`4d29801`；任务分阶段提交，原main既有工作保留。

# 当前状态与交接

核查日期：2026-09-20（Asia/Shanghai）。Git 根为当前 `mulyi-model-tac` 工作区，
分支 `main`，HEAD `4707074194091cd723ebd18dea7824c184e99c6d`，非 detached HEAD。
**适用范围是该 commit 加核查时的未提交改动，不是纯 commit。**
开始时已有 39 个已跟踪文件修改、16 条未跟踪文件/目录项；清单见
[DOC-AUDIT-20260920-01](../EXPERIMENTS.md#doc-audit-20260920-01)。
交付核对时另出现非本次创建的未跟踪 `scripts/diagnostics/trace_rotate_bulb.py`，未纳入本次调查。

## 当前目标与完成情况

2026-09-20 22:21 用户要求拆分旋出采样耗时，已完成两个隔离诊断（GPU2、1024/3072环境），
各6轮PPO、预热2轮后同步测量3轮。物理联合求解占63.5%/70.4%，压力阵列13.4%/7.3%，
接触力读取1.9%/1.6%；详细分项与可复算证据见
[采样性能报告](rotate_bulb_sampling_profile_2026-09-20.md)、
[BULB-PROFILE-20260920-01](../EXPERIMENTS.md#bulb-profile-20260920-01)。
这是GPU共享条件下的缩小规模测量，不能作为6144环境25.83秒的原位分解，也未分离臂/手/灯泡的耦合求解。
诊断结束，生产代码/配置未改；正式GPU4任务继续至59轮，装配与D-Peg进程保持。
附带发现：压力阵列内部位姿传感器在无reset的测量轮内没有刷新；其1/30秒周期配合零dt更新，
CPU调度复现支持位姿缓存停留在reset时刻的疑点。尚未比较原生逐帧位姿误差，未修复；
下一步应独立验证并修正该观测正确性问题，不能将少算位姿当作性能优化。

2026-09-20 用户明确：当前灯泡旋出任务只需要 **三圈引导版**。
后续旋出训练、回放、诊断和调参均以 Rotate-Bulb-Custom 的 guided 模式为准：
`bulb_pure_collision=false`、`initial_screw_turns_range=(3.0, 3.0)`，
保留约三圈旋转与对应 18 mm 轴向行程完成后释放的现有判据及数值容差。
除非用户另行要求，不切换或启动纯碰撞旋出版本，也不引入“一圈后 100 N 拔出”条件。
此要求针对旋出任务，不改变独立桌面装配任务。本次仅记录要求，未修改代码或重启训练。

2026-09-20 21:48 用户要求重新检查并实施臂手分工/长程旋转引导，已完成代码与调研：
连续新角度1.5分/圈+12个90°里程碑各0.125分，三圈总预算仍6分；
单指接触即可驱动附着灯泡、0.2秒换指记忆，新增释放前真实臂漂移/速度代价（最高−0.2/s）。
动作v3/臂倍率40/远离−0.3及三圈释放/最终成功条件保留；新小目标观测669→673维。
本地352通过/5跳过，远端2环境原生检查与3次PPO更新通过，装配保持原669维配置。
GPU4新run `20260920_134756_rotate_bulb_segments90_armsteady_6144_gpu4` 已于21:48:42启动，
6144环境/seed42/15000迭代、from scratch；PID374552，监督374551。
21:56已确认正式PPO迭代0–3与model_0保存，673维/引导契约v1及奖励参数生效；
逐项配置对比仅涉及两项旋转奖励、臂状态项、小目标观测和run路径/名称，物理/成功条件一致。
当前初期显存25723MiB，无接触容量/非有限值/内存致命错误；尚无完整任务成功率评测。
旧run停在170轮，检查点保留；装配与D-Peg进程未变。
详见 [三圈分段引导](rotate_bulb_segmented_guidance_2026-09-20.md)、
[BULB-SEGMENTS-20260920-01](../EXPERIMENTS.md#bulb-segments-20260920-01)。

2026-09-20 21:30（Asia/Shanghai）只读补查：该三圈 guided run 正常至136/15000，
GPU4/6144环境，显存27299MiB，最新保存model_100；完整成功日志从启动至今均为0。
最近20轮（117–136）日志点均值：回合最大旋出1.252圈、达到3圈指标4.61%、超时指标99.28%；
最新单轮最大旋出均值2.218圈，短期回升尚未证明持续改善。
对比50–69轮，接触奖励0.004661→0.012823、远离代价−0.032616→−0.002376，
两项旋出奖励合计0.010071→0.022996；mean_reward −3.183→−0.947。
说明接触/留在附近及获奖旋转改善，但尚未完成稳定三圈释放和目标搬运。
窗口均值按日志点计算，非合并回合成功率或独立评测；此次未录像、改配置或重启。
证据：`logs/diagnostics/bulb_status_20260920_2130/metrics.json`；实验索引沿用BULB-CONTACT-LOG-20260920-01。

2026-09-20 当前旋出训练的初始姿态/目标可视化已录制：单环境、seed42，固定重置状态，仅移动相机，
含机械臂全景、手部近景、俯视及青色目标/3cm容差标记；本次初始物体参考点距目标13.2cm。
[18秒视频](../recordings/xinyu21_bulb_initial_goal_20260920_130439/initial_pose_and_goal.mp4)。
初始关节配置和目标采样范围与当前训练一致；前后状态不变，未执行策略、未重启任何训练。
详见 [BULB-INITIAL-GOAL-20260920-01](../EXPERIMENTS.md#bulb-initial-goal-20260920-01)。

2026-09-20 20:58（Asia/Shanghai）旋出接触奖励/历史日志只读比较完成，未修改或重启训练。
当前GPU4/6144环境训练至78；有单指接触最高+0.02/s、合计+0.1/s，但独立双侧抓稳塑形未启用。
当前接触与减少远离正在改善，旋出最大圈数却下降，完整成功日志仍为0；
相比上一版臂10配置不是全面退化，但仍未形成持续抓持旋转。旧60Hz的高圈数/回报包含无抓持门控的收益。
显存已升至27299MiB，启动时的1.45GiB节省不是长期峰值结论。
详见 [接触与日志分析](rotate_bulb_contact_log_analysis_2026-09-20.md)、
[BULB-CONTACT-LOG-20260920-01](../EXPERIMENTS.md#bulb-contact-log-20260920-01)。

2026-09-20 20:49（Asia/Shanghai）已按用户要求将装配任务 **21 个手部关节初值全部设为 0**，
保留 7 个机械臂关节/基座初态；只改 `assembly_env_cfg.py` 的装配配置，启动、完整/局部 reset、PLAY 共用。
远端 2 环境/40 步检查的 15 项验证通过，含零位/驱动目标、机械臂保持、局部 reset 隔离及旋出初态保持；
本地相关 CPU 回归 100 passed、4 skipped。未修改奖励、动作/观测契约、碰撞或成功判据。
装配已在 GPU 5、6144 环境、30 Hz、seed 43、15000 次迭代从头重启：
`/home/xinyu/training_runs/20260920_124630_bulb_assembly_zero_hand_gpu5`，
训练 PID `360229`、监督 PID `360228`，TensorBoard `Bulb-Assembly_ZeroHand_30Hz_GPU5_20260920`。
已确认迭代 0–4 的 PPO 更新与 `model_0.pt`；旧 run 在迭代 300 正常停止，`model_300.pt` 哈希不变。
保存配置确认 14 项奖励及策略/PPO 参数与旧装配完全一致：动作惩罚 -0.02/-0.01，无旋出 40 倍/远离惩罚；
动作 ±1 rad/控制步，探索 std 初值 0.35、下限 0.05、无上限，entropy=0.001，无额外 wrapper clip。
仍保留 tanh 动作边界、物理关节限位、PPO clip=0.2/max_grad_norm=1.0，不能称作完全无约束。
旋出和 D-Peg 训练继续；先前动作诊断按本次用户转向停止，未取得可用动作轨迹，未据其声称根因。
零位是否改善抓取尚待训练验证。证据见 [ASM-ZERO-HAND-20260920-01](../EXPERIMENTS.md#asm-zero-hand-20260920-01)。

2026-09-20 新装配 run 最新已保存 `model_200.pt` 已在服务器录制并下载：
来源为 `20260920_105032_bulb_assembly_lift_30hz_gpu5`，使用该 run 的 206 文件源码快照、
30 Hz、纯碰撞、随机桌面初态、seed 43；GPU 2 单环境录制正常退出，GPU 5 原训练继续。
本地 [40 秒视频](../recordings/xinyu21_assembly_lift_20260920_122020/bulb_assembly_model_200.mp4)
为 1280×960、30 FPS，1200/1200 帧完整解码、两端 SHA256 一致，原 checkpoint 哈希未变。
未执行独立成功率评测。详见 [ASM-VIDEO-20260920-02](../EXPERIMENTS.md#asm-video-20260920-02)。

2026-09-20 20:13（Asia/Shanghai）按用户要求同步并重启 Custom 旋出，GPU 4 从 4096 提至 **6144 环境**。
新 run：`/home/xinyu/training_runs/20260920_121222_rotate_bulb_far_arm40_6144_gpu4`，
训练 PID `352175`、监督 PID `352174`；30 Hz、seed 42、15000 迭代，从零开始。
已核对 `hand_far.weight=-0.3`、释放前臂倍率 40；三项配对缓冲各从 2^27 降至 2^25，
碰撞栈从 2 GiB 提至 3 GiB，保留 guided、自碰撞、原动作/观测/成功协议。
9 个文件备份后同步；443 个相关源码/资产与本地一致，远端 CPU 回归 **252 passed**。
已观察迭代 0–8 的 PPO 更新、保存 `model_0.pt`；显存 26143 MiB（旧27627 MiB），初期吞吐中位8266 steps/s。
并行数增加50%时显存减少约1.45GiB；长期峰值仍需监督。
旧训练停在迭代 254，原模型哈希不变；装配和 D-Peg 进程启动时间不变。
TensorBoard：`Rotate-Bulb_FarArm40_6144_GPU4_20260920`。
详见 [BULB-MEM-RESTART-20260920-01](../EXPERIMENTS.md#bulb-mem-restart-20260920-01)。
下方“本地未同步/未重启”为历史状态，现已完成同步重启；学习效果尚无独立成功率验证。

2026-09-20 用户确认的远离/机械臂惩罚已本地实现：Custom 旋出启用 `hand_far.weight=-0.3`，
8–20 cm 平滑平方代价，保留前 0.5 秒、目标保持和终止帧豁免；释放前后均生效。
`arm_before_release_multiplier` 从 10 调至 40，释放后恢复基础代价；手指变化倍率仍为 2。
运行代码仅改两项奖励配置，成功/失败判据、动作/观测、物理及独立装配配置未变。
相关 CPU 回归 **252 passed**，覆盖不同频率的时间积分、局部 reset、阶段切换及标准/装配奖励兼容。
本次未同步服务器、未重启训练、未运行新物理回放；学习效果待训练验证。
详见 [BULB-FAR-ARM40-20260920-01](../EXPERIMENTS.md#bulb-far-arm40-20260920-01)。
下方 model_100 分析中的“候选未实施”为分析时历史状态，现已按本条在本地实施。

2026-09-20 当前 Custom `model_100.pt` 录制及惩罚预算分析完成：当前30 Hz与旧60 Hz各录21秒，
同seed42/各自源码契约；首个完整20秒回合均超时、未释放、未成功。当前最大0.527圈、距离代理最大1.398m，
0.725s后全机器人接触消失；旧最大2.506圈，但双侧>1N接触时长为0。当前虽放弃，回报仍+0.07354，失败罚分0。
当前臂20秒幅度代价仅−0.02547；未释放失握终止不生效，是优先关注的奖励路径。
候选方案（**未实施**）：距离≤8cm不罚、≥20cm最高−0.3/s、前0.5s宽限，未释放也生效；
释放前臂倍率10→40。固定原轨迹重算总回报约−5.718，属于数量级校准，不是学习改善证明。
[当前录像](../recordings/xinyu21_bulb_model100_compare_20260920_113845/current_model_100.mp4)、
[并排对照](../recordings/xinyu21_bulb_model100_compare_20260920_113845/model_100_side_by_side.mp4)、
[完整分析](rotate_bulb_model100_penalty_analysis_2026-09-20.md)，实验 [BULB-M100-20260920-01](../EXPERIMENTS.md#bulb-m100-20260920-01)。
未改运行代码/配置、未重启训练，三训练进程及原模型哈希保持；核验时旋出迭代188。

2026-09-20 本次按用户要求重启远端 Custom 旋出：已核对 `/home/xinyu/mulyi-model-tac`，
83/83 个本地改动/新增文件原已与远端一致，443/443 个相关源码/资产快照文件也一致；无业务代码覆盖。
GPU 4 新 run 为 `/home/xinyu/training_runs/20260920_105800_rotate_bulb_custom_phase_30hz_gpu4`，
训练 PID `337170`、监督 PID `337169`，北京时间 18:58:53 启动；4096 环境、seed 42、15000 迭代，从头训练。
实际配置已核对：30 Hz、guided、自碰撞开启、Dexonomy 28 关节初态、动作 v3、669 维观测、
关闭编码器，释放前臂动作倍率 10、释放后手指动作变化倍率 2。已观察迭代 0–4 的 PPO 更新并保存 `model_0.pt`。
旧旋出停在迭代 504，`model_500.pt` 哈希未变；装配 PID `325605`、D-Peg PID `242935` 未重启。
服务器相关回归 378 passed、11 skipped、1 failed；失败是原有 8192 环境预算测试自相矛盾，
4096 环境和分阶段惩罚测试通过。未执行独立成功率评测。详见
[BULB-RESTART-20260920-02](../EXPERIMENTS.md#bulb-restart-20260920-02)。下方未同步/未重启条目为历史状态。

2026-09-20 18:53（Asia/Shanghai）已按用户要求同步到 `xinyu21:/home/xinyu/mulyi-model-tac`
并重启桌面灯泡装配。83/83 文件 SHA256 一致，12 个新增/更新；旧文件有独立备份。
新 run：`/home/xinyu/training_runs/20260920_105032_bulb_assembly_lift_30hz_gpu5`，
TensorBoard：`Bulb-Assembly_Lift_30Hz_GPU5_20260920`；训练 PID `325605`、监督 PID `325604`。
沿用 GPU 5、6144 环境、seed 43、15000 次迭代、纯碰撞、关闭编码器；新默认为 30 Hz、
桌面 XY ±5 cm/yaw ±90°、两项抬起奖励，触觉采样周期同步为 1/30 s。
旧 run 为 60 Hz，控制契约不兼容，故本次从头训练（`resume=false`）；旧任务在迭代 1262
正常停止，保留 `model_1200.pt` 且哈希未变。旋出 PID 275257 与 D-Peg PID 242935 继续运行。
本地/远端定向测试各 106 passed、4 skipped；服务器补载 USD 后抬起模块 24/24 通过。
已验证 PPO 迭代 0–4、`model_0.pt`、14 项奖励与新配置；206 个源码快照及运行依赖版本已归档。
此项更新下文“未同步/未重启”的历史状态；不代表已验证学习改善或独立装配成功率。
证据与完整路径见 [ASM-RESTART-20260920-01](../EXPERIMENTS.md#asm-restart-20260920-01)。

2026-09-20 Custom 旋出分阶段动作惩罚已本地实现：未释放时机械臂增量幅度惩罚提高到原来的
10 倍，释放后恢复；释放后手指动作变化惩罚提高到原来的 2 倍，释放前保持原值。
参数为 `rewards.action_l2.params.arm_before_release_multiplier=10.0`、
`rewards.action_rate_l2.params.hand_after_release_multiplier=2.0`，均设为 1 可恢复旧代价。
使用已有 `_bulb_released`；共享函数默认倍率为 1，独立装配沿用原代价，标准旋出不变。
动作缩放/契约、探索、控制频率、奖励成功判据及物理未改；本机未发现活动训练/回放，未同步远端或重启训练。
本地 CPU 回归 **321 passed, 3 skipped**（Isaac 原生导出器两项、`pxr.Usd` 一项）；
含 14 项新增阶段、动作分组、持续大动作、稳定闭合、局部 reset、时间积分和装配兼容验证。
尚未验证训练后接触时长或成功率改善。详见
[BULB-PHASE-ACTION-20260920-01](../EXPERIMENTS.md#bulb-phase-action-20260920-01)、
[参数与代价公式](tactile_representation/rotate_bulb_custom_rewards_2026-09-20.md)。

2026-09-20 装配抬起小改动已在本地实现：新增 `lift_best_progress`（每回合最多 8 分）及
`lift_stable_once`（抬高 3 cm、双侧抓持、桌面过滤接触力低于 0.1 N，连续 0.3 s 后一次 +3）。
高度用当前姿态下灯泡碰撞网格最低点相对桌面的高度，减去每回合 reset 时的实际值；
最佳进展结合原有接触软门控，下降不扣分、重复抬到相同质量不重复给分。
新增模块 `assembly_lift.py`，参数位于 `env.assembly_lift`；原位置/插入权重仍为 5，
保持现有 30 Hz、随机桌面 reset、动作/观测/碰撞/成功判据。运行代码只新增模块及装配配置接线。
本地 **55 项离线测试通过**（包括真实 USD 几何），Isaac GPU **2 环境/40 步检查通过**，
实际加载 14 项奖励。新奖励预算/局部 reset 由数值测试验证，仿真检查仅为零动作接线与有限性检查。
未同步远端或重启训练；0.1 N 阈值尚未由策略轨迹标定，学习效果及独立成功率待验证。
证据：[ASM-LIFT-20260920-01](../EXPERIMENTS.md#asm-lift-20260920-01)，
参数与 TensorBoard 口径见 [装配引导](tactile_representation/bulb_assembly_guidance_2026-09-19.md#2026-09-20首次抬起引导)。

2026-09-20 Dexonomy 旋出录像完成：为 `Rotate-Bulb_Custom_Dexonomy_GPU4_20260920`
录制选择时最新保存的 `model_300.pt`，使用该 run 原源码快照、guided 物理、60 Hz、seed 42，
服务器 GPU 3 单环境回放退出码 0。本地
[视频](../recordings/xinyu21_rotate_bulb_dexonomy_20260920_103040/rotate_bulb_dexonomy_model_300.mp4)
为 1280×960、60 FPS、1799 帧（约 30 秒；请求 1800 控制步），完整解码通过、两端 SHA256 一致。
动作/观测契约与原 run 相同，原 checkpoint 哈希未变，GPU 4 原训练继续运行。
录像不是独立成功率评测；来源与校验见 [BULB-VIDEO-20260920-01](../EXPERIMENTS.md#bulb-video-20260920-01)。

2026-09-20 装配停滞分析（只读曲线/源码，未改奖励或重启）：`Bulb-Assembly_v3_GPU5_20260920`
约迭代 1169，最近 100 个 TensorBoard 记录点位置奖励均值 1.374、双侧接触 1.573、
插入/成功奖励均为 0。主要假设为桌面接触的持续回报与抬起时接触损失形成局部停滞；
初态满抓持的静态计算中上抬 1 cm 只增加约 0.043/s，抓持系数降至 0.8 则抵消该收益。
这不是物理轨迹测量；掉落终止非零，不能声称全部训练环境从未抬起。仍需高度、桌面支撑力、
分指接触与门控同步轨迹定位原因。证据见 [ASM-ANALYSIS-20260920-01](../EXPERIMENTS.md#asm-analysis-20260920-01)。

2026-09-20 装配录像已完成：为 TensorBoard run `Bulb-Assembly_v3_GPU5_20260920`
录制选择时最新已保存的 `model_1100.pt`，使用原 run 源码快照、60 Hz、固定桌面初态、seed 43。
服务器 GPU 1 单环境录制，退出码 0；本地视频
[bulb_assembly_v3_model_1100.mp4](../recordings/xinyu21_bulb_assembly_20260920_100546/bulb_assembly_v3_model_1100.mp4)
为 20 秒、1280×960、60 FPS，1200 帧完整解码、两端 SHA256 一致。
原训练持续运行、原 checkpoint 哈希未变；录像不是独立成功率评测。
详见 [ASM-VIDEO-20260920-01](../EXPERIMENTS.md#asm-video-20260920-01)。

2026-09-20 本次远端同步：按用户要求连接 `xinyu21`，将本地源码、配置、文档、测试及所需初态同步到
`/home/xinyu/mulyi-model-tac`；79/79 文件 SHA256 一致，15 个文件新增/更新，其余内容一致。
被替换版本保存在 `/home/xinyu/training_runs/20260920_095303_rotate_bulb_custom_softgrasp_30hz/sync_backup/`。
本地/远端相关 CPU 回归均为 **289 passed, 1 skipped**（未加载 `pxr.Usd`）。
服务器 8 张 GPU 当前均有进程，旧旋出在 GPU 4，装配 GPU 5，D-Peg GPU 6，均未停止或重启。
新 run 的候选方案为 GPU 1 共享、2048 环境、15000 迭代、seed 42、30 Hz、新奖励、从零开始；
用户的 GPU 共享选择尚待回复，**此时未启动新训练**。记录见
[BULB-SYNC-20260920-02](../EXPERIMENTS.md#bulb-sync-20260920-02)。

最新请求：桌面灯泡装配的位置及桌面内朝向随机化。已在装配独立 reset 中启用世界
X/Y 各自均匀 ±0.05 m、绕世界 Z 的 yaw 均匀 ±π/2；每次相对原始桌面位置重新采样，
不累积偏移，灯座保持原位，高度与原始倾斜保持。X/Y 是方形采样，角点距离约 7.1 cm。
参数为 `assembly_env_cfg.py:DexsuiteRevo3BulbAssemblyEnvCfg` 的
`bulb_reset_x_range`、`bulb_reset_y_range`、`bulb_reset_yaw_range`（单位 m/m/rad）；
startup/reset、训练/PLAY 共用。范围设为 `(0.0, 0.0)` 可恢复固定初态。
原 `assembly.py:reset_bulb_assembly_on_table` 接收但未使用 x/y 参数，本次使其作用于灯泡。
本地 brainco 解释器离线 **46 passed**：`test_bulb_assembly_reset.py`、
`test_bulb_assembly_state.py`、`test_bulb_assembly_guidance.py`、
`test_bulb_assembly_contact_rewards.py`、`test_rotate_bulb_grasp_reset.py`；
覆盖范围/端点、世界 Z 旋转、重复及局部 reset、配置接线和已有装配/抓姿回归。
未运行 Isaac、未同步远端或重启训练；这里保证的是 reset 位姿分布，动态落稳后的分布待验证。
本机进程核查仅发现编辑器 Python 服务。GitNexus 索引已刷新至当前 HEAD，
reset impact 报 CRITICAL 并返回异常跨任务边；配置类 impact 为 LOW（1 个依赖），
不能据图宣称完整排除影响。补充源码引用核查确认 reset 由装配配置注册，PLAY 继承装配配置。

此前请求：最小改动将 Custom 旋出与桌面装配的默认控制频率改为 30 Hz。
共享 `RotateBulbCustom/env_cfg.py:configure_rotate_bulb_scene` 的 decimation 从 4 改为 8，
物理步长仍为 1/240 s；装配继承生效，训练与 PLAY 使用相同频率。未同步远端或重启训练。
验证：配置/reset 与装配离线回归 18 passed；静态核对 Custom 30 Hz、标准旋出仍为 60 Hz，未运行仿真。
旧进程不会自动生效；旧 60 Hz 检查点的动作契约与新默认频率不同，恢复时仍须匹配原契约。

此前请求：本地修改 Custom 旋出奖励，搬运使用原连续抓取质量软门控，由用户另行同步到服务器训练。
已完成：两项旋出接入软质量/0.1 秒换抓记忆；搬运改为释放后新增最佳进度、最高 4 分；
启用释放后抓稳 +2 和稳定接触封顶 +2；释放后持续无接触且远离才判失握，与碰桌/越界共享一次 −6。
弱接触及单侧接触仍有部分搬运奖励。成功保持原释放后 3 cm/1 秒/+10 判据。
本地离线 **289 passed, 1 skipped**（独立 Python 缺 `pxr.Usd`）；覆盖标准/Custom 奖励、
弱接触、记忆衰减、局部 reset、30/60/120 Hz 预算、任务注册、资产及装配状态/奖励回归。
本次未运行 Isaac、未连接远端、未启动/重启训练，尚无新奖励的成功率结论。
生产改动集中在 `RotateBulbCustom/env_cfg.py`，动作/观测/初态/物理/成功规则保持；
同步后需新建环境，核对 `params/env.yaml` 中的奖励及终止参数，旧启动覆盖仍可改变默认值。
详情：[当前奖励说明](tactile_representation/rotate_bulb_custom_rewards_2026-09-20.md)、
[BULB-REWARD-20260920-01](../EXPERIMENTS.md#bulb-reward-20260920-01)。

此前请求（本次奖励修改前）：同步本地改动至 `xinyu21`，使用 Dexonomy 新抓姿从头重启 Custom 旋出训练。
已同步至 `/home/xinyu/mulyi-model-tac`，112/112 文件 SHA256 一致（54 更新/新增、58 原已一致）。
本地及服务器定向回归均为 41 通过、2 跳过。旧旋出 run 停在迭代 1704，保留 `model_1700.pt`；
新 run 为 `/home/xinyu/training_runs/20260920_092213_rotate_bulb_custom_dexonomy_gpu4`，
沿用 GPU 4、4096 环境、seed 42、15000 迭代、guided 物理、关闭编码器；不恢复旧模型。
训练 PID `275257`、监督 PID `275256`，启动于 2026-09-20 17:22:58（Asia/Shanghai）。
已确认迭代 0–4 的 PPO 更新及 `train/model_0.pt` 保存；生效配置的 28 个关节角与新模板完全一致，
`resume=false`、观测 669 维、动作契约 v3、初始 3 圈。装配与 D-Peg 原进程未重启。
这里只验证重启后能训练，未执行独立成功率评测。
证据与备份见 [BULB-RESTART-20260920-01](../EXPERIMENTS.md#bulb-restart-20260920-01)。

此前请求：将编辑器经 Dexonomy 优化的新灯泡模板用作 Rotate-Bulb-Custom 初态，已完成。
旋出训练及 PLAY 默认读取项目内 `grasp_reference/dexonomy_20260920/simulation_state.json`，
按名称设置 28 个关节，启动及完整/局部 reset 同步恢复位置和驱动目标。
机械臂、灯泡/灯座及默认 3 圈保持不变；装配、D-Peg 保留旧参考。
本机 Isaac Sim 2 环境 **440/440 状态检查通过**，最大位置误差约 0.707 微米；
离线 **46 通过、2 跳过**（独立 Python 未加载 `pxr.Usd`）。
验证为精简观测下的姿态/reset 检查，未验证 PhysX 动态抓持或旋出成功率，未启动训练。
证据：[BULB-GRASP-20260920-01](../EXPERIMENTS.md#bulb-grasp-20260920-01)、
[模板与复查说明](../assets/rotate_bulb_custom/grasp_reference/dexonomy_20260920/README.md)。

此前文档审计已建立规则、地图、运行手册、状态和实验索引；下文保留其历史核查范围。
当时未授权业务实现及仿真，该限制仅属于旧请求；此次按新请求完成局部接入和本机状态验证。

仓库呈现多个候选工作方向：Flexiv D-Peg 深度判据与续训、Custom 灯泡 v3 控制/旋出、
桌面灯泡装配引导、Repose revision 1/状态与视觉触觉对照。
依据为已有改动、专题记录和录像清单；**未将其中任一方向认定为用户当前训练目标**。
旧 `PEG_TRAINING_GOAL.md`、`bulb_assembly_autonomous.md` 和 `.autonomy/` 的任务/预算不是本次执行授权。

## 已确认的能力及证据

| 结论 | 证据与边界 |
| --- | --- |
| 训练/回放/三个任务评测入口及主要 Gym 注册存在 | 当前源码静态核查；路径与接口见 [PROJECT_MAP](PROJECT_MAP.md)，命令见 [RUNBOOK](RUNBOOK.md) |
| 存在不启动 Isaac 的轻量测试入口 | 本次 `tests/test_play_camera.py` 15/15 通过；12 个入口/注册文件 AST 解析、2 个启动脚本 shell 语法通过；没有验证整个任务运行 |
| D-Peg model_800 历史深度保持评测为 1/16 | [PEG-EVAL-20260919-01](../EXPERIMENTS.md#peg-eval-20260919-01)：本次重数 JSONL 一致，评测器/判据源码哈希与当前一致；完整代码环境仍缺快照 |
| D-Peg 历史恢复确实发生 PPO 更新 | [PEG-RESUME-20260919-01](../EXPERIMENTS.md#peg-resume-20260919-01)：归档记录优化器步数、归一化计数及模型变化；151/1299 是训练统计 |
| 旧版旋出开发评测失败样本仍可追溯 | [BULB-EVAL-20260919-01](../EXPERIMENTS.md#bulb-eval-20260919-01)：0/20，seed 4040；不可推到当前 v3 |
| 装配引导曾通过短仿真探针 | [ASM-PROBE-20260919-01](../EXPERIMENTS.md#asm-probe-20260919-01)：2 环境/40 步；当前 assembly 配置哈希已不同，不是完整装配成功证据 |

## 版本差异、已知问题与未验证项

1. **运行环境未确认**：本次系统 Python 3.12.3 未发现 torch/Isaac/RSL-RL 的包 metadata；
   不等于其他环境不存在。仓库缺完整依赖锁定，历史启动清单还依赖外部 RSL-RL fork。
   下次用获准环境记录解释器、包版本及实际 import 来源；本次没有尝试启动任务。
2. **检查点语义变化**：灯泡 `actions.py` 当前为 v3、臂/手 scale=1.0；旧 v2 权重不能直接恢复。
   Repose 新训练 revision 1、10 s、gamma 0.99；v2 记录中的 15 s、gamma 0.999 不适用于新默认。
   D-Peg Custom 当前无动作裁剪、单位残差、孔位置/姿态随机化也与旧版本不同。以契约和保存配置核对。
3. **D-Peg 成功协议差异**：当前只要求轴向 27 mm 连续 0.5 s，不要求 entry history/containment；
   旧目标文档要求真实几何插入。历史 1 个成功的入口标志为 false，不能因此推断穿模，也不能证明完整几何验收通过。
   需明确所用协议并检查原始轨迹；本次未改判据。`d_peg_tactile.py` 的 v4 注释仍写 contained，
   与当前判据不一致，且协议变化未仅靠 schema 数字完整区分；源码哈希必须一并记录。
4. **文档/默认值冲突保留待校对**：Repose 接入文档写 `tactile_encoder_chunk_size=32`，
   当前配置为 256；基础 Repose 的 `observation_space=147` 与共享 full 拼接的 152 维不一致，
   是否被运行时正确处理尚未验证。`integrate/README.md` 的 `~/BC_Tactile_Lab` 是历史目录示例，
   不是本工作区。Custom README 下方旧奖励/物理条目不能替代其顶部版本说明和当前源码。
5. **失败与阻塞线索**：Flexiv 预抓姿 README 记录旧候选零动作漂移失败，并明确未物理标定；
   当前初始化分布是否稳定需新版本有限探针。装配专题保留螺纹接触稳定性疑问，未证实为当前根因。
   `.autonomy/bulb_custom/state.json` 记录旧评测停在环境建立后，KVDB/生命周期归因已标未证实；
   不应重复当作确诊，也不应为此干预无关进程。

## 基线、运行记录与检查点

- 没有在本次认定“最佳模型”。D-Peg 的 `model_800.pt` 是明确历史评测/恢复父模型；
  旧灯泡 v018 `model_450.pt` 是开发失败基线。完整路径/哈希及实验 ID 在实验索引维护。
- `recordings/xinyu21_20260920_110956/manifest.json` 可将视频对应到 Repose 3750、
  D-Peg 3300、旋出 5700/6100、装配 6300 标号；只是录像来源，不是独立成功率。
- `recordings/xinyu21_v3_20260920_153650/recording_manifest.server.json` 与 `server_verification.json`
  是更新的 v3 录像/服务器检查副本，包含装配 model_600、旋出 model_1100 的线索；
  本次只读清单，没有加载模型或观看/重新解码视频。不可把旧版与 v3 录像混为同一基线。
- `.autonomy/bulb_custom/state.json` 的历史状态为 `PAUSED_BY_USER`，内部存在前后不一致的时间字段；
  `.autonomy/bulb/state.json` 的 contact_reward_v6 也早于新引导配置。它们是不同历史工作线，
  不能据其旧 PID/预算重启作业。**所有训练及监督的当前运行状态未核实**；未连接远端或检查进程。

## 指令与证据可携带性

适用祖先与本次代码/文档范围未发现 `AGENTS.override.md`，未验证新会话自动加载。
`.gitignore` 忽略根 `AGENTS.md`、`.autonomy/`、`logs/` 等；本地入口和许多证据不会随普通 Git 提交传播。
README 增加人工导航；本次不改变忽略规则、不提交。新建文档仍是未提交交付。
GitNexus 索引在 2026-09-20T04:07:17.072Z 建立、commit 匹配，但 status 报 2 个录像 JSON 未索引。
原 AGENTS 图统计也较旧，予以原样保留；`d_peg_success` impact 为 CRITICAL（136 直接依赖、18 流程），
部分关联需源码复核。Markdown impact 为 UNKNOWN，非低风险放行。没有重建图或执行提交前检查。

## 下一步建议（本次未获授权实施）

1. **优先补齐可携带的实验证据包**：针对一个确认要继续的 run，保存非敏感源码差异、
   未跟踪源码、完整生效配置、依赖/fork 版本、父模型/资产/契约哈希和评测协议。
   验收：另一会话能明确对应任务、代码、输入、命令与逐回合结果，不再只有 commit 或模型文件名。
2. 在预算与任务目标明确后，按当前版本冻结判据并做有限独立评测；插销区分深度保持与几何验收，
   灯泡区分旋出/桌面装配及 v2/v3。验收：每 seed 成功数/总数、所有失败/超时、权重不变与代表轨迹齐全。
3. 分别处理已定位的文档默认值冲突和 Repose 独立评测缺口。验收：默认值与源码一致；
   新评测能从原始状态逐回合判定，不能以目标切换次数或 reward 代替最终成功。
