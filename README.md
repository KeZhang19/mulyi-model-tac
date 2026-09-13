## 触觉可视化观测

Repose-Cube 的多模态恢复与 CLIP latent 策略接入、权重导出及真机张量接口见 [接入说明](docs/tactile_representation/repose_policy_integration.md)。

Rotate-Bulb 默认直接使用已训练的多模态恢复 encoder，无需先完成 CLIP 对齐；训练命令、观测维度及恢复要求见 [Rotate-Bulb 接入说明](docs/tactile_representation/rotate_bulb_policy_integration.md)。

```bash
python scripts/rsl_rl/visualize_rl_tactile_obs.py \
  --task BrainCo-Dexsuite-Revo3-Right-Lift-v0 \
  --num_envs 1 \
  --focus-finger index \
  --presser square_4 \
  --target tacmap \
  --press-start-offset 0.035 \
  --press-motion-actor object \
  --presser-body-mode dynamic_axis \
  --presser-force-n 100 \
  --presser-axis-max-travel 0.5 \
  --enable-presser-collision \
  --robot-hold-mode hard \
  --tacmap-contact-shell 0 \
  --tacmap-resize-mode surface \
  --focus-visuotactile-only \
  --tacmap-display-max-mm 1 \
  --tacmap-local-roi-margin-mm 1
```

## 数据采集

从仓库根目录运行。该命令采集 index 指尖同步的 RGB、Depth 和 2D Marker Motion 数据，输出为压缩 NPZ 分片；数据集不会上传到 GitHub。

```bash
python scripts/tactile_representation/collect_index_cross_modal.py \
  --task BrainCo-Dexsuite-Revo3-Right-Lift-v0 \
  --output datasets/revo3_index_sweep_parallel_v1 \
  --sampling-mode sweep \
  --pressers square_4 cylinder_D4 \
  --num-envs 4 \
  --seed 7 \
  --headless
```

采集完成后，建立 mmap 缓存以加速训练：

```bash
python scripts/tactile_representation/prepare_mmap_cache.py \
  --dataset datasets/revo3_index_sweep_parallel_v1
```

## 网络一：Robust Cross-Modal Tactile Network

```bash
python scripts/tactile_representation/train_cross_modal.py \
  --dataset datasets/revo3_index_sweep_parallel_v1 \
  --output runs/revo3_cross_modal_new \
  --device cuda:0 \
  --epochs 1000 \
  --batch-size 32 \
  --grad-accum-steps 2 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --dataset-backend mmap \
  --num-workers 2
```

最佳权重默认保存在输出目录的 `best.pt`。

## 网络二：Tri-Modal Cross-Autoencoder

```bash
python scripts/tactile_representation/train_tri_modal_cross_autoencoder.py \
  --dataset datasets/revo3_index_sweep_parallel_v1 \
  --output runs/revo3_tri_modal_cross_new \
  --device cuda:0 \
  --epochs 1000 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --dataset-backend mmap \
  --num-workers 2
```

最佳权重默认保存在输出目录的 `best.pt`。

## CLIP/CTTP 风格 Latent 对齐

该阶段使用 CTTP 的 two-tower 结构：两个已经训练好的 encoder，各接一个两层 MLP projection head；相同 `(episode_id, episode_step)` 的 sim-real 观测是正样本，batch 内其他错配观测自动作为负样本，使用双向 InfoNCE 对齐投影后的 latent。

先把两个原始 encoder 导出成独立 checkpoint（数据集和 optimizer 状态不会被复制）：

```bash
python scripts/tactile_representation/export_tactile_encoders.py \
  --sim-checkpoint runs/revo3_cross_modal_restoration_v1/best.pt \
  --real-checkpoint runs/revo3_tri_modal_cross_v1/best.pt \
  --sim-model-type robust \
  --real-model-type tri_modal \
  --output-dir runs/revo3_encoder_exports
```

默认 alignment 阶段冻结 encoder，只训练 projection head。若要按照 CTTP 论文让两个 encoder 也参与端到端训练，使用 `--unfreeze-all-epoch 1`；该选项会在第 1 轮前解冻两路 encoder，并用更小的 encoder learning rate 微调：

当前仓库提供的两个 best checkpoint 可作为对齐实验的两个 tower：

```bash
python scripts/tactile_representation/train_latent_alignment.py \
  --sim-checkpoint runs/revo3_cross_modal_restoration_v1/best.pt \
  --real-checkpoint runs/revo3_tri_modal_cross_v1/best.pt \
  --sim-model-type robust \
  --real-model-type tri_modal \
  --dataset datasets/revo3_index_sweep_parallel_v1 \
  --output runs/revo3_latent_alignment_cttp_v1 \
  --device cuda:0 \
  --dataset-backend mmap \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 2 \
  --learning-rate 3e-4 \
  --encoder-learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --projection-dim 64 \
  --temperature 0.1 \
  --unfreeze-all-epoch 1 \
  --save-every 10
```

对齐模型的最佳权重保存在输出目录的 `best.pt`，同时会自动保存两个单独的最佳 encoder：`sim_encoder_best.pt` 和 `real_encoder_best.pt`。如果使用真正分开的仿真和真机数据集，额外指定：

```bash
  --sim-dataset /path/to/simulation_dataset \
  --real-dataset /path/to/real_dataset
```

两个数据集需要共享可匹配的 `episode_id` 和 `episode_step`。数据集、中间 checkpoint、alignment 输出和独立 encoder 导出文件均不上传；仓库仅保留上面两个原始 encoder 的 best 权重。

## Latent 匹配可视化

用全局 top-1 检索把 query、模型选中的候选和 ground-truth 配对帧画成 contact sheet：

```bash
python scripts/tactile_representation/visualize_latent_matches.py \
  --sim-checkpoint runs/revo3_cross_modal_restoration_v1/best.pt \
  --real-checkpoint runs/revo3_tri_modal_cross_v1/best.pt \
  --alignment-checkpoint runs/revo3_latent_alignment_cttp_v1/best.pt \
  --sim-model-type robust \
  --real-model-type tri_modal \
  --dataset datasets/revo3_index_sweep_parallel_v1 \
  --split test \
  --direction sim-to-real \
  --device cuda:0 \
  --dataset-backend mmap \
  --output runs/revo3_latent_alignment_cttp_v1/match_visualization_test/sim-to-real
```

输出目录中的 `sim_to_real_correct.png` 和 `sim_to_real_incorrect.png` 分别展示正确 top-1 和高置信度错误匹配；`matches.json` 保存对应的 episode/frame key 和 cosine similarity。

> 注意：可视化命令中的 `alignment-checkpoint` 需要先按上面的训练命令在本地生成；该 checkpoint 不随仓库上传。
