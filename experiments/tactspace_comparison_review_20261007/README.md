# TACTSPACE-COMPARISON-REVIEW-20261007

本次为用户请求的对比设计审查，未运行训练、真机或远端操作，未修改运行代码。
检出为 main，HEAD `167fc3c79fbf0c882fe85e3e3d8bb3f7d180f77e`；TactSpace 与其他多项改动已存在于未提交工作区。

## 判断

选择 TactSpace 作为表示学习对照合理。当前实现适合作为 **TactSpace-inspired RGB/Depth/Marker 适配基线** 的起点，尚不能作为原论文完整复现或已完成的公平 sim-to-real 对比。
现有设计不用推倒重做；应先明确比较主张，再补齐统一评测与对应数据链路。

## 原论文与适配范围

[论文 III–IV 节](https://arxiv.org/html/2606.18959v1)以同一物理刺激为配对依据，联合优化独立模态编码器、自/交叉重建与 InfoNCE；原输入包括 RBS 穿透深度、FEM 应力和真实电容。真实电容编码使用历史帧。
[官方项目页](https://leggedrobotics.github.io/tactspace-web/)说明 ViT 编码器与共享解码结构；冻结表示后的下游网络在仿真数据训练、真实数据测试，覆盖分类、几何与力预测。查阅当日项目页代码链接仍标 Coming Soon，不能声称已与官方实现逐项核对。
“zero-shot”针对下游任务迁移，不等于对齐阶段没有真实数据。

把传感器换成本项目的 RGB/Depth/Marker 有合理性，可统一双方输入信息；无需为此强行改用电容传感器或 FEM。但这改变了实验问题，必须披露适配，不能直接搬用原论文数值作同条件排名。

## 本地代码证据

| 检查项 | 观察与判断 | 证据 |
| --- | --- | --- |
| 核心目标 | 每个来源独立编码；从单一 latent 重建所有有效目标；模态对间双向 InfoNCE，与方法思想一致 | `models/tactspace.py:71,114,197` |
| 编码/解码 | 图像为本地 CNN token 编码器，Marker 为点序列 Transformer；共享 latent MLP 接 FiLM 图像头和 Marker 头，非论文原结构 | `models/tactspace.py:78–110` |
| 跨域 | 默认仅 sim；loss 每次处理一个 domain 内的模态，没有跨域配对联合损失；CLI 明确拒绝真实数据，旧跨域入口也拒绝 TactSpace | `models/tactspace.py:197,279`；`train_latent_alignment.py:561` |
| 下游表示 | 各分支 L2→有效模态平均→L2；这是本项目多输入聚合规则。损失监督各单模态 latent，并没有单独监督平均后的表示 | `models/tactspace.py:148–164,197–276` |
| 时间 | 每帧独立编码、历史留给外部策略；不能当作已经复现编码器内历史建模 | `models/tactspace.py:57–67` |
| 数据隔离 | loader 按完整物理条件分组切分 episode，并非把相邻帧随机拆开，这一点合理；尚不自动等于完成 OOD 形状/尺寸及 sim–real 对齐集/下游集隔离 | `training/data.py:342` |
| 现有验证 | 历史日志 12 项 CPU 合成接线检查通过；本次未重跑。该证据不能证明表示质量或下游成功率 | [此前实验](../../EXPERIMENTS.md#tactspace-wiring-20261007)；`logs/diagnostics/tactspace_20261007/cpu_checks.log` |

表内 `models/`、`training/` 相对 `source/BrainCo_DexHand/BrainCo_DexHand/tactile_representation/`；训练脚本相对 `scripts/tactile_representation/`。

## 当前比较中的关键混杂因素

1. **缺少 sim–real 联合训练链路。** 定义 real 分支不等于已训练或已对齐。当前可讨论同域表示比较；要声称迁移方法优劣，必须有物理刺激可信配对、有效性标记，以及真实分支参与的联合训练。
2. **重建误差不等于 latent 质量。** 现有 `TriModalCrossAutoencoder.forward` 的解码器接收空间 token、Marker 位置及图像上下文（`models/tri_modal_cross_autoencoder.py:379`）；TactSpace 只接收 latent。这些旁路本身可以改善重建，不足以证明导出的 latent 更好。应在冻结表示上训练相同容量的下游头；系统重建指标可以另报。
3. **损失和受损数据评估不同。** 当前 `evaluate` 对 TactSpace 只算干净输入的联合损失并选 checkpoint（`train_tri_modal_cross_autoencoder.py:482`）；旧模型评估 clean 与三种受损模式，并以另一种组合选 checkpoint。旧目标还含前景/方向等加权项（`training/tri_modal_objectives.py`）。不能直接比较双方 total loss 或 selection loss。
4. **多模态平均属于适配，需要验证。** 建议同时报告各单模态及均值聚合表示。不能先验认定归一化平均必然保存互补信息，也不能在没有评测时判定其必然更差。
5. **容量、预算、输入可用性尚无完整对比证据。** 共用部分编码器有利于控制变量，但不证明总参数量、训练量或推理成本相等。也不能把提供正确坏模态 mask 的结果称作自动抗损伤能力。

## 建议的最小可发表对比（均未实施）

- **共同数据和输入：** 双方使用相同 RGB/Depth/Marker 可用集合、归一化、物理条件切分；独立保留位置、形状/尺寸泛化集。sim–real 比较还需整组物理刺激跨域隔离，防止同一配对组进入训练和测试。
- **表示评测：** 冻结编码器，训练相同结构与预算的下游头；优先用有可信标签的力/接触几何预测及任务相关状态预测。不要把本来已作为输入的 Depth 重建作为唯一结论。
- **机器人任务评测：** 若主张对控制有帮助，在同一任务/机器人/奖励/物理/动作与观测条件下比较；对齐策略容量、训练环境步数、评测分布，报告多个种子、成功数/总数及不确定性。不要直接复用 latent 语义不同的旧策略权重。
- **鲁棒性评测：** 统一干净、噪声、单模态受损/缺失的测试样本。报告双方训练时见到的增强；必要时另设 `TactSpace-adapted + same augmentation` 对照，区分结构收益与受损数据训练收益。各方法可保留各自目标，最终指标必须共用。
- **最少消融：** TactSpace-adapted 的 reconstruction-only、alignment-only、joint；单模态与均值聚合。涉及真实滞后/动态任务时，再做相同时间信息条件下的历史消融。原始方法式设置与控制变量适配设置应明确命名。
- **训练与选模：** 固定数据划分并保存清单，给双方合理且可记录的调参预算；披露参数量、训练样本/更新数、对比学习实际 batch、耗时及推理延迟。最终冻结选模后只在保留测试集报告统一指标。

优先顺序：先冻结比较协议与公共评估；若目标仅为仿真表示消融，可继续当前适配路线；若目标为 sim-to-real 主结果，再补物理配对、跨域联合训练和真实下游测试。

## 本次验证与局限

- 本次核对源码、入口、既有实验日志及论文/官方项目页；本地 `calibration/datasets/` 可见内容仅 README，未取得可用于正式比较的数据集。
- 未重新执行 CPU 检查，未测 CUDA、长期训练、控制任务或实际迁移；未声称任何方法胜出。
- GitNexus 显式绑定当前仓库，meta 的 HEAD 与当前相同，indexedAt 为 `2026-10-07T07:59:05.116Z`。但 query/context 未找到新目标，另一 query 出现符号/路径错配；impact 对 `compute_tactspace_objective` 返回 UNKNOWN，文档目标只命中历史 snapshot 候选。不能把该图视为当前工作区的可靠调用证据。
- 已直接核对训练/验证调用、导出包装器、跨域拒绝分支，以及 README/AGENTS/导航文档对 STATUS 和 EXPERIMENTS 的引用。本轮只增加审查记录及两处入口摘要，保留原内容，不改运行符号、不提交；未为此次分析重建全仓库索引。
