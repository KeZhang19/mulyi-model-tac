# 厂商依赖

[brainco-revo3-sdk/](brainco-revo3-sdk/README.md)保存已有Revo3源码、原生运行包、模型和许可证副本。
内部目录和文件字节保持原样；旧 `calibration/brainco-revo3-sdk` 为相对兼容链接。
本项目硬件脚本从 `vendor/brainco-revo3-sdk/runtime/python` 加载副本。

依赖解释器仍使用原 `/home/liuxinyu/workspace/brainco-revo3-sdk/.venv/bin/python`，
启动器也支持 `REVO3_PYTHON`。本目录没有复制虚拟环境、安装软件或更新厂商版本。
原复制来源／SHA256见[SDK复制清单](../provenance/sdk_copy_manifest.json)；
该清单中的旧destination是历史记录，不随本次移动改写。
