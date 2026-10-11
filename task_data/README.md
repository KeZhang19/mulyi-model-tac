# 任务数据

保存任务使用的抓姿、初始化模板、采样分布，以及随模板归档的来源和验证记录。
模型/网格见 [assets](../assets/README.md)，标定见 [calibration](../calibration/README.md)，
奖励、动作、观测、终止与训练参数的 Python 定义仍在
[任务配置目录](../source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/)。

| 目录 | 数据及当前用途 |
| --- | --- |
| `d_peg_insertion/` | 标准 D-Peg 的 `pregrasp.json`、Flexiv Custom 的 `pregrasp_flexiv.json`；其余 `pregrasp_*` 为原候选，`flexiv_grasp_reference/` 保存来源，`validation/` 保存历史验证 |
| `rotate_bulb_custom/grasp_reference/` | 根 `simulation_state.json` 为旧共享参考；`dexonomy_20260920/` 为 Custom 旋出抓姿，不能用于替换装配/D-Peg 的默认参考 |
| `rotate_bulb_randomized/` | 当前默认 `workspace_bank_high_z_v2.json`；`workspace_bank_joint1_zero_v1.json` 为保留的零位变体，两者不能互换 |
| `rotate_bulb_assembly_curriculum/` | 当前运行读取 `floor_pregrasp_v2.json`；`pregrasp.json` 为保留的原模板，详见目录内说明 |

这里的“当前”来自 2026-10-04 本地源码读取路径，不表示远端运行配置或任务成功率。
候选、失败记录及历史验证状态原样保留；同名字段或相同维度不证明契约兼容。

## 路径与维护

- 2026-10-04 从 `assets/` 移入；[迁移清单](organization_manifest.json) 记录原路径、新路径和迁移前 SHA256。
- 旧位置为相对符号链接。运行代码仍使用旧字符串并读到这里的同一文件，源文件哈希和 JSON 内的来源路径保持不变。
- 新数据使用本目录；修改现有数据会同时影响新旧入口，应按原任务协议重新验证。
- 现有生成器仍在 `assets/*/tools/` 或 `scripts/diagnostics/`；需生成新候选时显式指定新的输出路径。
- 不单独删除旧链接：模板的 `source_sha256` / `asset_hashes`、保存的配置和历史工具仍引用它们。
- 同步/备份必须包括本目录和兼容链接；只复制 `assets/` 会漏掉实体数据。本次没有同步远端。
- 实体数据延续原 `/assets/` 的本地忽略策略；新检出不会仅靠 Git 获得这些文件。目录说明和迁移清单可随代码交接。

迁移验证结果见本地 `organization_verification.json`，仅证明文件、路径和来源哈希一致，
不替代 Isaac 动态仿真或独立任务评测。
