# 标定目录溯源

| 清单 | 含义 |
| --- | --- |
| [layout_20261011.json](layout_20261011.json) | 本次9项归类、旧路径兼容方式和737个迁移文件的原SHA256 |
| [organization_manifest.json](organization_manifest.json) | 2026-10-04首次集中归类记录；内容保持原样，canonical字段描述当时布局 |
| [sdk_copy_manifest.json](sdk_copy_manifest.json) | 初次复制厂商SDK的来源及SHA256；内容保持原样，不代表本次重新下载 |

旧 `calibration/{organization_manifest,sdk_copy_manifest}.json` 均为相对兼容链接。
本次验证报告保存于 `calibration/outputs/organization_20261011/verification.json`；
2026-10-04报告仍在 `outputs/organization_20261004/`。
两份报告只证明各自检查范围，不代表完成真机或Isaac动力学验收。
