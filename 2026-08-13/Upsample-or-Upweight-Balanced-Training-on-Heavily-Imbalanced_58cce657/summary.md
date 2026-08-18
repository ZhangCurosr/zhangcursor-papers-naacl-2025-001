---
title: "Upsample-or-Upweight-Balanced-Training-on-Heavily-Imbalanced"
source: https://aclanthology.org/2025.naacl-long.171.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:59:31"
field: "多语言自然语言处理"
keywords: ["多语言语言建模", "数据不平衡", "温度采样", "Scalarization", "COOLDOWN", "梯度方差", "多任务学习"]
innovations: ["证明全梯度下降下 Scalarization 与 Temperature Sampling 等价，但在 SGD 下温度采样梯度方差更低收敛更快", "提出 COOLDOWN 温度调度策略（高温加速收敛→降温防过拟合），无需代理模型", "系统性验证稀疏温度调度优于细粒度在线重加权"]
benchmarks: ["OPUS-100 机器翻译子集", "mC4 多语言语言建模子集"]
---

# 论文速读：Upsample or Upweight? Balanced Training on Heavily Imbalanced Datasets

## 一句话总结
论文理论证明了全梯度下降下损失加权（Scalarization）与温度采样（Temperature Sampling）等价，但在SGD下温度采样梯度方差更低、收敛更快；据此提出 COOLDOWN 调度策略（先用高温快速收敛、再降温防过拟合），在多语言机器翻译与语言建模任务上有效平衡高/低资源语言性能。

## 研究问题与动机
- 多语言数据呈长尾分布：极少数高资源语言（如英语2733B tokens）占主导，大量低资源语言（如斯瓦希里语1B tokens）严重稀缺，均匀采样会导致模型偏向高资源语言。
- 业界普遍将 Scalarization（加权损失）与 Temperature Sampling（重采样）视为等价方法并互换使用，但缺乏严格理论依据。
- 实际训练中需在高资源语言快速收敛与低资源语言避免欠拟合/过拟合之间取得平衡。
- 现有动态调度方法（如 Unimax、Order Matters、DoReMi）实现复杂或依赖代理模型，缺乏简洁有效的温度调度方案。

## 核心贡献（创新点）
1. **首次严格证明 Scalarization 与 Temperature Sampling 的等价性与差异性**：在全梯度下降下两者等价（Theorem 1），但在 SGD 下温度采样梯度方差更低（Theorem 2）。
2. **揭示温度与方差的关系**：证明当近似温度升高或数据分布偏斜度增加时，Scalarization 的梯度方差单调增大（Theorem 3），为实践中选择采样策略提供理论依据。
3. **提出 COOLDOWN 温度调度策略**：训练初期使用高温（如 τ=5） aggressive upsample 低资源语言以加速收敛，后半程降至 τ=1 防止过拟合，无需代理模型即可实现高效平衡训练。
4. **系统性实证验证**：在多语言机器翻译（OPUS-100 子集）和多语言语言建模（mC4）两个任务上验证理论结论与 COOLDOWN 的有效性，对比静态/动态温度、DoReMi 等基线。

## 方法详解
- **Scalarization (S)**：在均匀采样基础上对各域损失加权，目标函数为 $\mathcal{L}_S(\mathbf{w}) = \mathbb{E}_{x \sim \mathcal{D}}[w_{f(x)} \mathcal{L}(x)]$，其中 $w_i$ 为域 $i$ 的权重。
- **Temperature Sampling (TS)**：按温度 $\tau$ 调整各域采样概率 $p(i;\tau) = \frac{|D_i|^{1/\tau}}{\sum_j |D_j|^{1/\tau}}$，目标函数为 $\mathcal{L}_{TS}(\tau) = \mathbb{E}_{k \sim p(\cdot;\tau)}[\mathcal{L}(x)]$。
- **Theorem 1 等价性证明**：当 $\tau$ 固定时，存在权重 $w_i = \frac{p(i;\tau)}{p(i;1)}$ 使得两种方法在全数据上损失相同。
- **Theorem 2 方差分析**：在 SGD 下，Scalarization 的梯度方差满足 $\text{Var}(\nabla \mathcal{L}_S) \geq \text{Var}(\nabla \mathcal{L}_{TS})$，即温度采样梯度估计更稳定。
- **Theorem 3 温度-方差单调性**：当 $\tau \geq 1$ 时，两者方差差 $\Delta = \text{Var}(\nabla \mathcal{L}_S) - \text{Var}(\nabla \mathcal{L}_{TS})$ 单调递增，表明高温度下 Scalarization 方差劣势更显著。
- **COOLDOWN 调度**：前半程（如前 30k/50k 迭代）使用 $\tau = 5$（或更高）aggressive upsample 低资源语言，后半程降至 $\tau = 1$（比例采样）防止过拟合；属于稀疏更新策略，避免细粒度动态加权带来的性能下降。

## 实验与结果
- **数据集**：
  - 机器翻译：OPUS-100 子集，8 种语言（英→es/fi/ga/gl/ha/hi/it/kk/ug），平行句数从 76K 到 1M 不等。
  - 语言建模：mC4 子集，EN(2733B)/IT(162B)/ZH(39B)/SW(1B)。
- **评估指标**：SacrenBLEU（翻译）、validation loss（语言建模）。
- **主要结果**：
  - **翻译任务（Table 3）**：相比比例采样（τ=1），COOLDOWN 使中低资源语言平均提升 +0.7 / +3.1 BLEU，高资源语言仅下降 -0.2 BLEU；相比静态 τ=5，COOLDOWN 在高资源语言上提升 1.5 BLEU，低资源语言持平；优于 Unimax、Order Matters，并与 DoReMi 相当且无需训练代理模型。
  - **语言建模任务（Table 4）**：COOLDOWN 在所有低资源语言（IT/ZH/SW）上均取得最优 dev loss，仅在 EN 上略低于比例采样。
  - **调度设计**：降温策略（5→1）优于升温策略（1→5）；稀疏更新优于细粒度在线重加权（Dense Update 1/2 表现更差）。
- **收敛验证**：图 4、图 8 显示 TS 收敛速度显著快于 Scalarization，且温度越高差距越大；充分训练后两者最终损失趋同，验证全梯度下的等价性。

## 相关工作脉络
- **梯度方法（MTL）**：PCGrad、RL-based 采样（Wang et al. 2020）、基于梯度相似性的温度调度（Fan et al. 2024）——本文认为这些方法相比简单加权损失并无显著提升，且 Scalarization 方差更大是根本原因之一。
- **损失方法（DRO）**：Oren et al. 2019、Xie et al. 2023 (DoReMi)、Zhou et al. 2021——COOLDOWN 可视为 DRO 的高效近似，但无需代理模型与密集更新。
- **Scaling Laws**：Chen et al. 2023、He et al. 2024、Jiang et al. 2024——探索最优数据混合比例，但小模型最优策略未必泛化到大模型；本文提供不依赖代理训练的启发式调度方案。
- **多语言干扰缓解**：Unimax（Chung et al. 2023）、Order Matters（Choi et al. 2023）——前者先均匀采样后移除低资源数据，后者先训高资源再加低资源，COOLDOWN 统一为单调降温策略且更简单。
- **类不平衡 vs 域不平衡**：本文关注域大小不均衡（多语言），与类不平衡方法（Buda et al. 2017）有概念关联但问题设定不同。

## 局限性与未来方向
- 仅评估预训练验证损失，未系统研究不同温度调度对下游任务性能的影响。
- 理论分析适用于重度数据不均衡场景，但多语言与多域建模在子域划分上存在差异，需进一步验证泛化性。
- 未穷举所有降温调度策略（如连续衰减、自适应降温），最优温度 schedule 仍依赖经验选择。
- 未考虑低资源语料本身可能存在的质量偏差， aggressively upsample 可能放大语料中的偏见。

## 研究启发与可借鉴点
- **方差视角解释收敛差异**：将"为何上采样比加权更好"归因于 SGD 下的梯度方差差异，这一理论框架可迁移到其他领域不平衡问题（如长尾分类、多任务学习）。
- **稀疏温度调度优于密集更新**：COOLDOWN 的"两段式"稀疏调度在实验上优于细粒度 online reweighting，提示在实际工程中应避免过于频繁的参数/权重更新。
- **降温策略的通用性**：初期 aggressive sampling + 后期回归比例采样的范式可推广至其他多域/多任务场景，作为零样本调度的 baseline。
- **无需代理模型的轻量方案**：COOLDOWN 不依赖 DoReMi 式的 proxy training，计算开销低，适合资源受限的科研/生产环境。
- **理论与实践结合的典范**：从 Theorem 1-3 的严格证明到 Figure 1-9 的系统实证，展示了理论指导实验设计的完整闭环。

## 关键术语表
**Scalarization**：在均匀采样基础上对_domain_损失乘以权重 $w_i$ 以平衡域间贡献的方法。
**Temperature Sampling (TS)**：通过温度参数 $\tau$ 调整各域采样概率，使低资源域被更频繁采样的方法。
**COOLDOWN**：本文提出的温度调度策略，训练初期使用高温（τ>1）加速收敛，后半程降温至 τ=1 防止过拟合。
**Gradient Variance**：SGD 下单样本梯度估计的方差，方差越低收敛越稳定快速。
**Distributionally Robust Optimization (DRO)**：最小化最坏子群损失的优化框架，与本文方法在目标上有交集但实现路径不同。
**Proportional Sampling**：温度 τ=1 时的采样策略，按原始数据量比例均匀采样。
**Unimax**：Chung et al. (2023) 提出的调度方法，先均匀采样直至低资源数据被访问固定次数后移除。
**Order Matters**：Choi et al. (2023) 提出的调度方法，先仅训练高资源语言，后期再加入低资源语言。

## 可复现要素
- **数据集**：OPUS-100（Zhang et al. 2020）、mC4（Xue et al. 2021），均为公开数据集。
- **代码/权重**：论文未提供官方开源代码或模型权重；实验基于 fairseq（Ott et al. 2019）与 Huggingface Transformer 实现。
- **关键超参**：机器翻译实验 τ=5 用于前 30k 迭代、τ=1 用于后 30k 迭代；语言建模实验 τ=5 用于前 50k、τ=1 用于后 50k；BPE vocab size=64k（翻译）/32k（对比实验）、学习率 5e-4、Adam β=(0.9, 0.98)、label smoothing=0.1、warmup=4000、max tokens=16384/32768（Table 5/6）。
