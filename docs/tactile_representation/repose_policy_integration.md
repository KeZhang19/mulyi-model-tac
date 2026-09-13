# Repose-Cube：恢复网络与 CLIP 策略观测

任务 `BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0` 默认使用冻结的多模态编码器与 CLIP projection，也支持下述尚未对齐的仿真基线。PPO 只训练原有 Actor/Critic MLP。

每帧观测为 `concat(state[152], z_little, z_ring, z_middle, z_index, z_thumb)`，默认历史长度为 1，与原版 Repose-Cube 一样只使用当前帧。每指 z 都经过 L2 归一化。projection 为 64 维时，观测为 `(152 + 5 × 64) × 1 = 472` 维；任务在创建 Gym space 前从编码包读取实际维度。原 ResNet 观测或四帧堆叠观测训练的 PPO 权重不能直接用于当前单帧配置。

## 尚未对齐时：先验证仿真 PPO

两份原始预训练权重不包含 CLIP 对齐头。尚无配对数据或 alignment checkpoint 时，可以直接使用仿真 encoder 的 L2 归一化特征 `h`，先验证传感器、观测与 PPO；不需要随机初始化并冻结一个 projection。该模式不提供 sim–real 对齐，也不能直接用真机塔替换仿真塔。

```bash
git lfs pull --include='runs/revo3_cross_modal_restoration_v1/best.pt'
python scripts/tactile_representation/export_tactile_encoders.py \
  --sim-checkpoint runs/revo3_cross_modal_restoration_v1/best.pt \
  --unaligned-sim --output-dir runs/revo3_encoder_exports

python scripts/rsl_rl/train.py \
  --task BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0 \
  --num_envs 4 --max_iterations 3 --headless --logger tensorboard \
  --run_name unaligned_sim_smoke \
  env.tactile_policy_checkpoint=runs/revo3_encoder_exports/sim_policy_encoder_unaligned.pt
```

真实仿真 checkpoint 的 `d_model=256`，因此当前帧观测是 `152 + 5 × 256 = 1432` 维。导出的独立文件明确标记 `feature_mode=unaligned_sim`、`alignment_id=null`，并记录原始权重 SHA-256。运行契约区分 `normalized_h` 与已对齐的 `normalized_z`，阻止混用编码器或恢复不兼容的 PPO。完成对齐后，应使用新的运行目录训练对应 64 维 projection 的策略。

上述短运行用于验证训练流程。正式仿真训练可使用 `--num_envs 32` 并去掉 `--max_iterations 3`，恢复任务默认的 10000 次迭代；短运行成功不代表策略已经收敛或真机迁移有效。

## 对齐权重与训练

在带有 PyTorch、Isaac Lab 的环境中，从仓库根目录运行。以下 `python` 应指向该环境。本地 Git LFS 指针不是模型权重，首先获取两份原始模型：

```bash
git lfs pull --include='runs/revo3_cross_modal_restoration_v1/best.pt,runs/revo3_tri_modal_cross_v1/best.pt'
```

离线对齐使用真实配对的仿真和真机数据目录；正样本以 `(episode_id, episode_step)` 匹配。两个目录必须包含采集格式的 RGB、Depth 和二维 marker 数据。同一个仿真数据集分别送入两塔只构成接线实验，不能证明真机对齐。

```bash
python scripts/tactile_representation/train_latent_alignment.py \
  --sim-checkpoint runs/revo3_cross_modal_restoration_v1/best.pt \
  --real-checkpoint runs/revo3_tri_modal_cross_v1/best.pt \
  --sim-model-type robust --real-model-type tri_modal \
  --sim-dataset /path/to/paired_sim --real-dataset /path/to/paired_real \
  --output runs/revo3_latent_alignment_cttp_v1 \
  --device cuda:0 --projection-dim 64 --epochs 100 --batch-size 64

python scripts/tactile_representation/export_tactile_encoders.py \
  --sim-checkpoint runs/revo3_cross_modal_restoration_v1/best.pt \
  --real-checkpoint runs/revo3_tri_modal_cross_v1/best.pt \
  --alignment-checkpoint runs/revo3_latent_alignment_cttp_v1/best.pt \
  --output-dir runs/revo3_encoder_exports
```

输出 `sim_policy_encoder.pt` 和 `real_policy_encoder.pt`，各自包含一塔最终权重、projection、模型配置、归一化参数和共同的 alignment ID。即使对齐阶段微调了 encoder，导出也采用对齐 checkpoint 中的最终权重。原始两个 checkpoint 提供架构及归一化元数据，必须与对齐训练所用模型对应。省略 `--alignment-checkpoint` 时仍执行原有的单 encoder 导出。

```bash
python scripts/rsl_rl/train.py \
  --task BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0 \
  --num_envs 32 --headless \
  env.tactile_policy_checkpoint=runs/revo3_encoder_exports/sim_policy_encoder.pt

python scripts/rsl_rl/play.py \
  --task BrainCo-Direct-Revo3-Repose-Cube-Visuotactile-v0 \
  --checkpoint /path/to/run/model_1000.pt --num_envs 2 --headless \
  env.tactile_policy_checkpoint=runs/revo3_encoder_exports/sim_policy_encoder.pt
```

恢复训练使用训练脚本现有的 `--resume --load_run ... --checkpoint ...` 参数。运行目录中的 `tactile_policy_contract.json` 记录编码包、标定资产和观测布局；恢复与播放先校验该文件再加载 PPO。复制或部署策略时须携带它。RL-Games 与 SKRL 的配置继续使用同样的扁平输入；使用外部训练入口时，应在保存新 run、恢复或播放加载模型之前调用 `prepare_policy_run(env, log_dir, resume_path=checkpoint_or_none)`，以执行同样的校验。

可用任务参数：`tactile_encoder_chunk_size`（默认 32）、`tactile_reconstruction_diagnostics`（默认 False）、`tactile_taxim_chunk_size`（默认 32）。分块大小控制每次编码的指尖图像数。重建诊断输出在 `env.latest_tactile_reconstruction`，其值为训练归一化单位；不会作为 z 的输入。RL 使用与对齐训练相同的 `encode → projection → normalize` 路径。

## 传感器适配

Direct 场景复用采集链路的标定相机射线、局部 TacMap 加密、240×320 dense Depth、100 点 HydroShear 及真实参考背景上的 Taxim 残差。内部计算顺序为 middle/index/ring/pinky/thumb，返回策略前显式转换成 little/ring/middle/index/thumb。

标定在 rubber 子链坐标系，任务在 DIP 父链坐标系。适配器读取 URDF 中固定安装变换，在临时目录生成对应射线布局，不修改源资产；中指约 29.5 mm 的平移也由该变换处理。marker 在目标 Direct 网格上重新投影，避免将参考手指网格的三维点直接当作目标表面。表面射线完全无命中时会报错。

图像/marker 的像素坐标系保持采集约定：Depth 使用米；marker 使用 `[x0_px,y0_px,dx_px,dy_px,valid]`；RGB 使用 0–255。RGB 量化与采集器一致，Depth/marker 尺度从 checkpoint 读取。当前标定要求 H=240、W=320、K=100；不通过简单拉伸旧的 30×30 深度图来伪造匹配。

每个控制步只推进一次观测历史和 HydroShear。局部 reset 清空对应环境的历史与剪切状态，并使共享传感器缓存失效。

## 真机输入示例

推理模块不依赖 Isaac 环境。Tri-Modal 真机塔仍要求三模态输入；RGB-to-depth、marker tracking、同步与状态估计由硬件系统提供。state 保留的 152 维含物体位姿和速度，必须遵循 Direct 任务的字段、顺序和尺度；只有关节编码器数据不足以直接部署该策略。

```python
import json
from pathlib import Path
import torch
from BrainCo_DexHand.tactile_representation.policy import (
    FrozenTactilePolicyEncoder, TactileObservationHistory,
)

encoder = FrozenTactilePolicyEncoder(
    "runs/revo3_encoder_exports/real_policy_encoder.pt", domain="real"
).to("cuda:0")
contract = json.loads(Path("/path/to/run/tactile_policy_contract.json").read_text())
encoder.validate_observation_contract(contract, history_length=contract["history_length"])
history = TactileObservationHistory(1, 152, encoder.projection_dim, contract["history_length"], "cuda:0")
policy = torch.jit.load("/path/to/run/exported/policy.pt", map_location="cuda:0").eval()

# 所有张量在 cuda:0，手指顺序固定为 little/ring/middle/index/thumb。
# rgb: uint8 [1,5,3,240,320]
# depth_m: float32 [1,5,1,240,320]
# marker: float32 [1,5,100,5]，像素坐标与位移
# marker_valid: bool [1,5,100]
# state: float32 [1,152]，字段与训练一致
with torch.no_grad():
    z = encoder(rgb=rgb, depth_m=depth_m, marker=marker, marker_valid=marker_valid)
    actions = policy(history.append(state, z))

# 新 episode：history.reset([0])，并清空硬件侧 marker 历史。
```

## 验证

```bash
OMP_NUM_THREADS=2 python -m pytest -q \
  tests/test_tactile_policy_encoder.py \
  tests/test_direct_repose_visuotactile_task.py \
  tests/test_tactile_contrastive.py \
  tests/test_tactile_representation_network.py \
  tests/test_tactile_representation_training.py \
  tests/test_tri_modal_cross_autoencoder.py \
  tests/test_tri_modal_cross_training.py
```

张量测试覆盖训练数据预处理对照、最终对齐权重导出、两塔 z 一致性、冻结与分块推理、无接触/无效 marker、历史隔离和 checkpoint 契约。完整效果验证还需使用真实预训练权重及配对真机数据；合成权重只能用于接口、传感器和 PPO 保存加载的冒烟检查。

### 2026-09-11 服务器流程验证

在 `dsw-8prvxfz53nbu67qzsu` 的 `/home/admin/workspace/xinyu/mulyi-model-tac` 验证了未对齐仿真基线，使用真实第 112 epoch 的仿真编码器；原始 checkpoint SHA-256 为 `c8e30bd4005aefba60d18aae46f992b722088d05a2cb9cd2ec66b316181c632a`。

| 检查 | 结果 |
| --- | --- |
| 编码器与任务相关测试 | 17 项通过 |
| 4 环境训练 | 3 次 PPO 迭代，退出码 0 |
| 32 环境恢复训练 | 从 `model_2.pt` 恢复，完成 2 次迭代，退出码 0 |
| 策略输入 | 1432 维，冻结真实 encoder，仅更新 Actor/Critic |
| 更新与保存 | Actor/Critic 参数均变化；保存模型的张量均为有限值 |
| 32 环境性能 | 单张 A100，约 78–92 steps/s，约 5.5–6.5 秒/迭代 |

日志及可重跑的脚本在服务器的 `remote_train_logs/repose_unaligned_smoke_20260911T094424Z.*` 和 `remote_train_logs/repose_unaligned_resume32_20260911T094928Z.*`；汇总记录为 `remote_train_logs/repose_unaligned_validation_summary.json`。最终短测 checkpoint 为 `logs/rsl_rl/brainco_repose_cube_visuotactile/2026-09-11_09-49-36_repose_unaligned_resume32_20260911T094928Z/model_3.pt`。

这些结果确认能够执行 PPO 更新、保存和恢复，不证明长期收敛或任务成功率。此次只做短测，未启动长时间训练。
