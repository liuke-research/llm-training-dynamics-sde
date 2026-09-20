# 训练日志接入：当前契约与尚未完成的部分

实现依据：[io.py](../src/llm_sde/io.py)、[features.py](../src/llm_sde/features.py)、[monitor.py](../src/llm_sde/monitor.py)。本文描述当前行为，不把计划中的功能当作已实现。

## 当前字段

| 字段 | 必需 | 语义 |
|---|---|---|
| `step` | 是 | 非负整数 optimizer step，严格递增 |
| `loss` | 是 | 当前观测的目标函数数值；有限负值允许 |
| `grad_norm` | 是 | 本步观测的梯度范数，有限时应非负 |
| `learning_rate` | 是 | 实际使用的学习率，必须有限且非负 |
| `update_norm` | 否 | 相邻更新前后的实测参数差范数 |
| `param_norm` | 否 | 更新前的参数范数 |
| `clipped_grad_norm` | 否 | 裁剪后梯度范数；原始辅助观测，当前不直接参与评分 |
| `run_id` | 否 | 运行标识；开始后不得更改 |
| `phase` | 否 | 阶段标识；变化会重置统计窗口及告警状态 |

CSV 第一行为列名；JSONL 每行一个 JSON object。一个文件只放一个 `run_id`。恢复训练后 step 回退时，另建文件和监控器。`step` 不是行号，稀疏日志的提前量仍按实际 optimizer step 计算。

## 缺失值与非有限值：当前限制

当前 `grad_norm` 缺列会报错；`grad_norm=None` 或 `loss=None` 会转换为非有限数值，并触发 `NONFINITE`。此行为适用于真实数值故障，**不适合表示没有采集到某项指标**。不要补零、填未来数据，或随意前向填充冒充本步梯度。

真实框架接入前，应只向当前接口传入具有有效必需观测的记录；原始完整日志另存。采样频率变化会改变窗口所覆盖的训练跨度，需要重新做阶段内校准。后续计划增加 metric-availability 标记，尚未实现。

额外字段可以由 `JSONLLogger` 保留在原始文件中，但 `coerce_record()` 只向监控端传递支持字段。`reward`、`entropy`、`kl` 目前不会参与评分。不要把“日志已存下来”解释为“该指标已被模型使用”。

## 实时调用

```python
from llm_sde import MonitorConfig, OnlineMonitor
from llm_sde.io import JSONLLogger

monitor = OnlineMonitor(MonitorConfig())
with JSONLLogger("runs/my_run/telemetry.jsonl") as logger:
    # 在真实 optimizer step 后，以本步实际采集的值构造 record。
    record = dict(step=0, loss=3.2, grad_norm=1.7,
                  learning_rate=0.001, run_id="my_run", phase="train")
    logger.write(record)
    result = monitor.update(record)
    print(result)
```

上例数值仅用于展示接口，不代表真实训练数据。完整流式演示见 [streaming_monitor.py](../examples/streaming_monitor.py)。

## 采集位置

小模型示例按 `backward → 测量裁剪前梯度 → 梯度裁剪 → optimizer.step → 测量真实参数差` 的顺序执行。梯度、Loss 与实际更新都属于同一个 optimizer step，但 Loss 是更新前前向传播计算的。

真实工程中必须约定梯度累积、精度缩放、跳过更新和分布式聚合的语义，再做适配；本包没有自动处理这些场景。不要把某一 rank 的局部范数当全局范数；不要把学习率乘梯度当成 AdamW 的真实参数更新量。

`parameter_snapshot()` 会完整复制模型参数，只用于小模型验证。大模型接入不得不加评估地照搬这一探针。实际更新量为可选字段，可先不采集，但必须如实说明此信号未启用。

## 独立事件标签

```json
{"events": [{"onset": 150, "end": 170, "kind": "independently_verified_event"}]}
```

```bash
python -m llm_sde analyze telemetry.csv --events events.json --out runs/labeled
```

事件必须处于日志时间范围内且互不重叠。没有标签时不输出精确率/召回率。不要把 LR 干预开始时间直接当作故障发生时间。

## GRPO/TRL/verl

本次只完成通用接口打包；还没有这些训练框架的原生 adapter、分组奖励统计或后训练算法。后续字段设计、缺失语义修复和验证安排见 [ROADMAP](ROADMAP.md)。
