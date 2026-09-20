# 数学研究模块的对象与边界

模块：[sde.py](../src/llm_sde/sde.py)、[markov.py](../src/llm_sde/markov.py)。以下公式用于说明本代码计算的对象，而不是声称已证明实际大模型训练的性质。

## 已知随机过程

OU 模型为 `dX = theta * (mu - X) dt + sigma dW`。代码提供精确离散转移与 Euler–Maruyama；在 `theta > 0` 下，用均值 `mu`、方差 `sigma^2 / (2 theta)` 与 lag-1 相关 `exp(-theta dt)` 校验数值输出。

双阱模型的势函数为 `U(x) = x^4 / 4 - x^2 / 2`，漂移为 `x - x^3`。代码提供 Euler 和 tamed 数值步；比较的连续 SDE 稳态密度与 `exp(-2 U(x) / sigma^2)` 成比例。有限步长链的经验分布不等于连续 SDE 的精确稳态；短时间观察到的跃迁次数不能替代混合时间证明。

## 拟合有限状态链

`QuantileChain.fit()` 在校准窗确定分位数分箱，以相邻观测统计转移次数，加入正 pseudocount 并按行归一化。后续窗口使用冻结分箱，不用未来数据重新定义状态。

`stationary_distribution()` 计算拟合矩阵的唯一稳态向量；`absolute_spectral_gap()` 计算矩阵特征值模意义下的间隙。对非可逆链，不应把该数值不加常数与非正规性分析地直接写成有限时间混合率。

正 pseudocount 会让拟合链具有正转移概率，这是建模正则化，并不证明底层训练过程不可约、Harris 遍历或符合马尔可夫假设。

## 倾斜谱与占用尾概率

到达状态观测 `f` 的倾斜矩阵为 `P diag(exp(k f))`。代码计算 `lambda(k) = log rho(P diag(exp(k f)))`，再在有限 `k` 网格上取 `k a - lambda(k)` 的最大值。它是标量可加观测的有限网格数值近似，**不是通用 level-2 Donsker–Varadhan 求解器**。

有限时占用尾概率由动态规划在给定链、给定初始分布下精确计算，但仍然只有“假定这条有限链是模型”这一层的意义。它不是经过校准的 LLM 故障概率。

## 与在线报警的关系

上述模块目前不参与在线 Risk Score。状态投影是否足够、训练阶段是否近似平稳、不同窗口是否一致，均需要另行验证。经验 JS 差异是漂移描述量，不自动具有显著性检验解释。

实测例子见 [math_results.json](../reports/math_demo/math_results.json)；数学相关单元测试见 [test_math.py](../tests/test_math.py)。
