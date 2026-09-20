# 本次仓库打包验证

本文件引用的是打包时重新执行的结果，而不是把上一轮报告直接当作新实验。代码核心评分方法未修改。

## 自动测试与环境

完整测试：**89 passed**；日志见 [pytest.txt](../reports/release_checks/pytest.txt)，JUnit 明细见 [pytest.xml](../reports/release_checks/pytest.xml)。

本地 Python 3.13.5、PyTorch 2.10.0+cpu；其余版本记录在 [environment.json](../reports/release_checks/environment.json)。测试包含未来数据不变性、标签隔离、时间顺序、事件匹配、有限链数值关系、因果注意力和实际梯度更新。远程 GitHub Actions 尚未执行。

## 完整合成比较

按源代码默认协议重新执行 24 条开发场景运行、48 条留出场景运行，每条 700 步。五种方法全部使用开发集选择的阈值。

主方法：24 个事件中提前命中 16 个；42 次总告警中 16 次为有效提前告警；提前召回率 66.7%，提前告警精确率 38.1%，每千条健康观测 0.5844 次误报，成功预警提前量中位数 31 步。该结果只对提供的合成生成机制成立。

依据：[汇总](../reports/benchmark/heldout_summary.csv)、[分场景结果](../reports/benchmark/heldout_by_scenario.csv)、[每条运行指标](../reports/benchmark/heldout_per_run.json)、[阈值选择](../reports/benchmark/development_threshold_search.csv)、[cluster bootstrap](../reports/benchmark/bootstrap_seed_cluster_intervals.json)。

## 实际小模型训练

正常 seeds 7/11/23、50× LR 干预 seeds 7/11/23、5000× 强干预 seeds 31/41/59，总计 9 条实际训练运行，默认 42,240 参数。强干预运行触发非有限数值保护后提前结束。

正常运行无标签事件且无告警。50× 干预没有产生 Loss excursion 标签，却各有一次告警，按此事件定义是误报。5000× 强干预出现 3 个事件，3 次成功提前告警和 3 次事后硬故障告警；严格提前告警精确率为 50%，不是 100%。其 21 步提前量来自人工极端干预，不是自然 LLM 故障预测结果。

强干预方案历史上是在温和干预未产生事件后增加的，此次按同一脚本重跑，不把它重新标成盲测。

依据：[六次训练摘要](../reports/torch_suite/runs_summary.json)、[强干预摘要](../reports/strong_stress/runs_summary.json)、[九条原始测量日志合表](../reports/all_tiny_training_telemetry.csv)。合表含多个运行，使用时先按 `evidence_suite` 与 `evidence_run` 拆分；不要直接送入单运行分析器。

示例 [normal_7 报告](../reports/tiny_normal/report.html) 配有该次运行的原始 CSV、scores 与观察标签。所有检查点和完整逐运行目录均可通过 `examples/` 重跑生成，未分发二进制模型文件。

## 数学示例

重新执行 `math-demo`，结果见 [math_results.json](../reports/math_demo/math_results.json)。它验证已知过程和拟合有限链计算，不把有限链尾概率称为 LLM 故障概率。适用范围见 [MATH_SCOPE](MATH_SCOPE.md)。

## 耗时

本次单条 `OnlineMonitor.update()` 中位数为 **0.200 ms**，P95 为 **0.380 ms**；仅含监控计算，不含 IO、作图、参数复制、模型训练或 GPU 同步。环境中的调度负载会改变数值，不能当作稳定的生产吞吐承诺。依据：[monitor_latency.json](../reports/latency/monitor_latency.json)。

## 复现范围与未执行项

重新执行了自动测试、默认合成 demo、完整合成 benchmark、九条小模型训练、数学 demo 和耗时测试。流式示例及解压后检查见 [打包检查记录](../reports/release_checks/package_checks.json)。本次没有重新分析旧项目的 20,000 行日志，没有训练真实预训练大模型，没有执行 GRPO/PPO、GPU 或多机实验。

相对路径文档和包完整性已检查。仓库中的测试日志是本地结果；CI YAML 的存在不能替代 GitHub 远程执行记录。
