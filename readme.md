LLM-Training-Dynamics-SDE

https://img.shields.io/badge/python-3.9+-blue.svg

https://img.shields.io/badge/status-demo--in--progress-yellow.svg

🎯 项目定位：本项目旨在利用随机偏微分方程 (SDE)​ 与随机动力系统理论，为大模型（LLM）训练中的“Loss Spike”、“梯度爆炸”等稀有异常事件提供可解释的数学建模与早期诊断指标。

📖 项目背景 (Motivation)

在大模型分布式训练中，现有工程手段（如 GradNorm、LayerClip）多为“黑盒经验调参”，缺乏对底层动力学的深刻理解。本项目尝试填补这一空白：

视角转换：将 SGD/Adam 优化过程视为高维空间中的带噪声扩散过程。

理论武器：利用遍历理论（Ergodicity）估计训练收敛的混合时间，利用大偏差原理（Large Deviation）计算系统逃逸尖锐极小值（Sharp Minima）的壁垒。

目标：构建一套基于数学理论的“训练稳定性雷达”，提前预警潜在的训练崩溃风险。

🛠️ 技术栈 (Tech Stack)

核心语言：Python

数值计算：NumPy, SciPy, Pandas

可视化：Matplotlib, Plotly (可选)

深度学习框架：PyTorch (预留接口，用于接入真实 LLM 梯度流)

数学工具：欧拉-丸山法 (Euler-Maruyama), Lyapunov 函数分析

📂 目录结构 (Directory Structure)

目前规划如下，随着 Demo 迭代会逐步完善：

text

.

├── README.md          # 项目说明

├── requirements.txt   # 依赖包列表

├── LICENSE            # 开源协议

│

├── 01_baseline_sde/   # 🔹 第一阶段：基准 SDE 模拟

│   ├── ou_process.py  # 模拟 Ornstein-Uhlenbeck 过程（类比简单凸优化）

│   └── visualize.py   # 轨迹可视化

│

├── 02_loss_surface/  # 🔸 第二阶段：损失景观中的粒子漫步

│   ├── sgd_trajectory.py # 模拟 SGD 在 2D/3D 损失面上的随机游走

│   └── quasipotential.py # 尝试估计准势垒 (Quasipotential)

│

└── 03_llm_demo/       # 🚀 第三阶段：面向 LLM 的动力学分析 (8月下旬 Demo)

├── data_loader.py

└── stability_monitor.py

🚀 快速开始 (Quick Start)

确保你的环境安装了 Python 3.9+。

bash

1. 克隆仓库

git clone https://github.com/liuke-research/llm-training-dynamics-sde.git

cd llm-training-dynamics-sde

2. 创建虚拟环境 (推荐)

python -m venv venv

source venv/bin/activate  # Linux/Mac

venv\Scripts\activate # Windows
3. 安装依赖 (目前暂无，后续会添加 numpy 等)

pip install -r requirements.txt

🗓️ Roadmap & Milestones

[x] Step 1 (2026.08.05)：仓库初始化，撰写 README。

[ ] Step 2 (2026.08.10)：完成 01_baseline_sde，跑通第一个随机游走动画。

[ ] Step 3 (2026.08.20)：完成 02_loss_surface，展示数学理论如何解释 Loss Spike。

[ ] Step 4 (2026.08.30)：接入 PyTorch，产出第一版针对 LLM 的 Stability Monitor Demo。

📧 联系我 (Contact)

姓名：刘珂

邮箱：15030368689@163.com

求职意向：LLM 训练稳定性与强化学习动力学
