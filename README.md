# LLM Training Dynamics SDE

**面向模型训练的可解释稳定性监控与随机动力系统研究原型**

Python · NumPy / SciPy / Pandas · PyTorch（可选） · Version 0.2.0

将训练日志转换为因果滑动窗口特征、异常分数与可解释告警，并用独立事件标签评估提前预警和事后检测。项目同时提供 OU、双阱与有限状态马尔可夫链数值实验，用于研究训练状态分析方法的适用边界。

> **当前交付：日志监控 + CPU 小型 Transformer 实际训练 + 随机过程数值实验。**
> 真实预训练大模型的 GRPO/PPO 后训练、TRL/verl 原生日志适配和 RL 专用指标监控尚未实现，见 [后训练扩展路线](docs/ROADMAP.md)。Risk Score 是启发式异常分数，不是故障概率。

[快速运行](#快速运行) · [输入接口](docs/INTEGRATION.md) · [实测验证](docs/VALIDATION.md) · [模型说明](docs/MODEL_CARD.md) · [上传 GitHub](docs/UPLOAD_GITHUB.md)

## 输入、处理与输出

```text
一次训练的 CSV / JSONL / 实时观测
        ↓
字段校验与 run / phase 隔离
        ↓
只使用当前及此前观测的因果特征
        ↓
Loss / Gradient / LR / 实际 Update 的启发式评分
        ↓
暖启动 → 持续超阈值 → 去重与冷却
        ↓
告警时间、触发指标、逐步分数、自包含 HTML 报告
```

监控器只提供诊断，不会自动停训、调学习率或回滚。数学研究模块独立运行，目前没有为在线告警提供实际大模型的遍历性或大偏差理论保证。

## 已实现与边界

| 模块 | 当前实现 | 边界 |
|---|---|---|
| 日志接入 | 单次训练的 CSV、JSONL、`OnlineMonitor.update()` | 原始日志需先符合本项目 schema |
| 在线监控 | 因果窗口、五种评分方法、告警状态机、解释 | 尚不消费 Reward / KL / Entropy 等 RL 指标 |
| 评估 | 开发集选阈值、留出集、事件匹配、误报与提前量 | 无独立标签时不计算准确率 |
| 真实反向传播 | 42,240 参数的两层因果 Transformer | CPU、算法生成 token 数据；不是预训练大模型后训练 |
| 数学实验 | OU、双阱、有限链稳态与倾斜谱计算 | 拟合链的结论不直接迁移到真实训练 |
| 工程测试 | 89 项本地测试通过；附 CI 配置 | 远程 CI、GPU、多机验证尚未执行 |

## 快速运行

在解压后的项目根目录执行，安装依赖时需要可访问相应软件包源；示例运行本身不下载模型或数据。

```bash
# 核心监控和数学实验，不需要 PyTorch
python -m pip install -e ".[dev]"
python -m pytest -q
python -m llm_sde demo --out runs/synthetic_demo
python -m llm_sde math-demo --out runs/math_demo
```

打开 `runs/synthetic_demo/report.html` 查看报告。未安装 PyTorch 时，训练相关测试会跳过；要运行完整的 89 项测试和小模型训练：

```bash
python -m pip install -e ".[dev,train]"
python -m pytest -q
python -m llm_sde train-demo --scenario normal --out runs/tiny_normal
```

默认训练运行 320 个 optimizer step。`train-demo` 不会覆盖已有的 `telemetry.jsonl`，重复运行请更换输出目录。使用本地 UTF-8 语料的字符级示例：

```bash
python -m llm_sde train-demo --corpus my_corpus.txt --out runs/local_corpus
```

**版本口径：**打包实测为 Linux / Python 3.13.5 / PyTorch 2.10.0+cpu；准确环境见 [environment.json](reports/release_checks/environment.json)。`pyproject.toml` 声明 Python ≥3.10，其他解释器和操作系统未在本次本地验证。CI 配置用于后续检查，不代表已通过跨平台测试。

## 接入自己的日志

```csv
step,loss,grad_norm,learning_rate
0,3.2,1.7,0.001
1,3.1,1.6,0.001
2,3.0,1.5,0.001
```

```bash
python -m llm_sde analyze my_training.csv --out runs/my_analysis
python -m llm_sde analyze my_training.jsonl --out runs/my_analysis_jsonl
```

`step` 为严格递增的 optimizer step；一个文件只对应一个运行。`grad_norm` 是本步观测的、约定采集位置下的梯度范数，示例记录裁剪前范数。当前接口要求该字段存在，`None` 会进入非有限数值告警路径，**不能用来表示“这一条没有采集”**。详见 [接入说明](docs/INTEGRATION.md)。

实时接口：

```python
from llm_sde import MonitorConfig, OnlineMonitor

monitor = OnlineMonitor(MonitorConfig(window=64, threshold=0.7))
result = monitor.update({
    "step": 0,
    "loss": 3.2,
    "grad_norm": 1.7,
    "learning_rate": 0.001,
    "run_id": "experiment_01",
    "phase": "training",
})
print(result["status"], result["risk_score"], result["factors"])
```

一个可以直接执行的流式示例：

```bash
python examples/streaming_monitor.py --out runs/streaming_example
```

它使用合成观测展示 API 和日志写盘，不冒充真实模型训练。

## 打包时重新执行的验证

以下是此次打包环境实际执行的结果，不是预计效果；详细协议与限制见 [VALIDATION](docs/VALIDATION.md)。

| 实验 | 实测结果 | 证据 |
|---|---|---|
| 自动测试 | 89 passed | [测试记录](reports/release_checks/pytest.txt) |
| 合成留出集 | 48 条运行，24 个事件；多指标提前命中 16 个 | [基线比较](reports/benchmark/heldout_summary.csv) |
| 提前告警质量 | 提前召回率 66.7%，提前告警精确率 38.1% | [分场景结果](reports/benchmark/heldout_by_scenario.csv) |
| 正常小模型训练 | 3 次正常运行，无告警 | [六次训练汇总](reports/torch_suite/runs_summary.json) |
| 50× 学习率干预 | 3 次均无所定义的损失事件，但各产生一次告警 | 相对于该事件定义，这些是误报 |
| 5000× 强干预 | 3 次均出现损失事件，各提前 21 步告警 | [探索性压力测试](reports/strong_stress/aggregate_metrics.json)，不是自然故障预测能力 |

已观察到无害扰动误报；无前兆突发异常没有被提前预测。不用“预测准确率”代替召回率，不把未产生故障的异常指标告警计为成功预警。

可直接查看 [合成日志示例报告](reports/synthetic_demo/report.html) 与 [真实小模型训练示例报告](reports/tiny_normal/report.html)。HTML 为本地自包含文件；仓库中的 PNG、CSV 和 Markdown 也可单独检查。

## 复现实验

```bash
# 24 条开发运行选阈值 + 48 条留出运行评估
python -m llm_sde benchmark --out runs/benchmark

# 固定配置下的六次 CPU 小模型训练
python examples/run_torch_suite.py --out runs/torch_suite

# 与温和干预分开的三次强扰动测试
python examples/run_stress_suite.py --out runs/strong_stress

# 仅监控器自身耗时，不是训练总开销
python -m llm_sde latency --out runs/latency

# 拟合有限状态代理链，不是实际 LLM 故障概率
python examples/analyze_markov_proxy.py runs/tiny_normal/telemetry.csv --out runs/markov_proxy
```

`configs/default.json` 的阈值为 0.7；`configs/calibrated_synthetic.json` 的 0.5 来自合成开发集，不是任意真实模型上的最优阈值。

## 仓库目录

```text
llm-training-dynamics-sde/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-core-tested.txt
├── .gitignore
├── .gitattributes
├── .github/workflows/tests.yml
├── configs/
├── src/llm_sde/
│   ├── io.py                  # 日志 schema 与 CSV/JSONL
│   ├── features.py            # 因果历史特征
│   ├── monitor.py             # 评分与告警状态机
│   ├── torch_demo.py          # TinyCausalLM 与真实训练循环
│   ├── evaluation.py          # 事件级评估
│   ├── synthetic.py           # 六类合成场景
│   ├── benchmark.py           # 阈值选择、留出比较、耗时
│   ├── sde.py                 # OU 与双阱
│   ├── markov.py              # 拟合有限链的数学计算
│   ├── report.py              # HTML / 图表
│   └── cli.py                 # 命令行入口
├── examples/                  # 流式、训练套件、数学研究示例
├── tests/                     # 89 项测试
├── docs/                      # 接入、评估、模型卡、数学边界、路线图
├── reports/                   # 实测摘要与可复核示例日志
├── REPOSITORY_TREE.txt
└── MANIFEST.sha256
```

完整目录见 [REPOSITORY_TREE.txt](REPOSITORY_TREE.txt)。`reports/` 保存此次实测摘要与精选日志；全量逐运行产物和检查点可用上述命令生成到被 Git 忽略的 `runs/`。仓库不包含模型权重、原始旧工程压缩包、私人简历或访问令牌。

## 后续：强化学习后训练

目标是先完成真实开源语言模型的 GRPO 训练，再扩展 Reward、KL、Entropy、组内奖励差异、回答长度与截断指标。优先验证一个具体诊断问题，而不是先声称平台化或普适故障预测。

当前仓库没有 `GRPOTrainer`、PPO 实现、奖励模型训练或 DDP/FSDP 原生适配。明确的验收项见 [ROADMAP](docs/ROADMAP.md)。

## 发布与联系

上传前先解压，使 `README.md`、`pyproject.toml` 和 `src/` 位于仓库根目录；不要仅提交压缩包本身。详见 [上传说明](docs/UPLOAD_GITHUB.md)。

许可证尚待项目负责人选择，此次打包没有代为添加 MIT / Apache 授权。所有新增实现和实验应由维护者复核、理解后对外使用。

**Name:** 刘珂  
**Email:** 15030368689@163.com
