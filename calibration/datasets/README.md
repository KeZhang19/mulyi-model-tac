# 派生训练数据

后续采集的NPZ分片、完整manifest与mmap缓存统一放在各自的新子目录。
本轮没有创建训练数据集；原始真机证据放在 `../captures/`，机械仿真回合放在 `../outputs/simulation/`。
由机械标定／SDK原始数据进入训练schema必须显式转换，见主框架第13节。
根目录 `datasets`是兼容链接，保证旧训练命令访问同一位置；新文档优先写 `calibration/datasets/...`。
