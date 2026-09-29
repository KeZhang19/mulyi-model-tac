# 装配课程预抓模板

`pregrasp.json` 是本工作区机器人、桌面灯泡和 Dexonomy 参考姿态的几何派生模板。
包含3个命名28关节起点：开手预抓、中间接近位、原臂与零手；不是成功策略或演示轨迹。

生成器：`scripts/diagnostics/build_bulb_curriculum_pregrasp.py`，依赖现有USD、NumPy、SciPy和
`build_bulb_randomized_workspace.py`的运动学/碰撞筛查工具。准备USD Python环境后：

```bash
python scripts/diagnostics/build_bulb_curriculum_pregrasp.py --out assets/rotate_bulb_assembly_curriculum/pregrasp.json
```

输出已存在时拒绝覆盖。重新生成应先选择新路径并重新验证，而非直接覆盖旧训练契约。
本模板筛查了42个离散路径点；离散无穿透不保证连续路径安全，也不证明动态抓持成功。
运行时核对所有来源资产哈希，变更后不能静默沿用旧状态库。
原生固定/自由和联合恢复验证见 `docs/rotate_bulb_curriculum_implementation.md`。
