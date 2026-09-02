## 触觉可视化观测

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

该阶段冻结两个已经训练好的 encoder，只训练两个 projection head。相同 `(episode_id, episode_step)` 的 sim-real 观测是正样本，batch 内其他错配观测自动作为负样本，使用双向 InfoNCE 对齐投影后的 latent。

当前仓库提供的两个 best checkpoint 可作为对齐实验的两个 tower：

```bash
python scripts/tactile_representation/train_latent_alignment.py \
  --sim-checkpoint runs/revo3_cross_modal_restoration_v1/best.pt \
  --real-checkpoint runs/revo3_tri_modal_cross_v1/best.pt \
  --sim-model-type robust \
  --real-model-type tri_modal \
  --dataset datasets/revo3_index_sweep_parallel_v1 \
  --output runs/revo3_latent_alignment_pilot \
  --device cuda:0 \
  --dataset-backend mmap \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 2 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --projection-dim 64 \
  --temperature 0.1 \
  --save-every 10
```

对齐模型的最佳权重保存在 `runs/revo3_latent_alignment_pilot/best.pt`。如果使用真正分开的仿真和真机数据集，额外指定：

```bash
  --sim-dataset /path/to/simulation_dataset \
  --real-dataset /path/to/real_dataset
```

两个数据集需要共享可匹配的 `episode_id` 和 `episode_step`。数据集和训练过程中的其他 checkpoint 不上传；仓库只保留两个网络各自的 `best.pt`。
