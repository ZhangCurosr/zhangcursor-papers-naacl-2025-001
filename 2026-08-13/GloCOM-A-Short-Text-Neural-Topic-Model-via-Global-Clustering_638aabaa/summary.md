---
title: "GloCOM-A-Short-Text-Neural-Topic-Model-via-Global-Clustering"
source: https://aclanthology.org/2025.naacl-long.51.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:03:32"
---

# 论文速读：GloCOM-A-Short-Text-Neural-Topic-Model-via-Global-Clustering

## 一句话总结
本文提出 **GloCOM**（Global Clustering COntexts for Topic Models），利用预训练语言模型（PLM）嵌入对短文档进行全局聚类，将聚类上下文拼接为虚拟长文档并注入 VAE 重建目标，同时通过可学习自适应变量解耦全局语义与局部个体分布，在低计算开销下同步缓解短文本主题建模的数据稀疏与标签稀疏问题。

## 研究问题与动机
1. **数据稀疏性**：短文本（如新闻标题、搜索片段）词量极少，词共现模式匮乏，传统与神经主题模型难以捕捉潜在的共现结构。
2. **标签稀疏性**：基于 VAE 的主题模型在计算 ELBO 重建项时仅关注输入中出现的词，忽略未观测但语义相关的高概率词，导致训练信号有偏。
3. **现有聚合方法的局限**：传统自聚合（伪文档、TF-IDF 聚类）在大数据下时间复杂度高、聚类质量差，且难以恢复单篇文档的主题分布；最新 SOTA 模型 kNNTM 虽采用 kNN+最优传输增强重建目标，但 pairwise 距离计算代价极高（4×A100 需约 50 小时）。
4. **PLM 表示带来的新契机**：现代预训练语言模型（如 all-MiniLM-L6-v2）能高质量编码短文本语义，为低开销、高语义保真度的文档聚类提供了可行路径。

## 核心贡献（创新点）
1. **PLM 驱动的全局聚类上下文构建**：使用 PLM 嵌入替代传统 TF-IDF 进行 K-Means 聚类，拼接簇内短文档形成全局文档；与依赖词频统计的传统聚合或 kNNTM 的高维最优传输相比，计算复杂度大幅降低且语义聚类质量显著提升。
2. **全局-局部双层级主题分布推断**：引入对数正态先验的全局分布 $\theta^g$ 与可学习自适应变量 $\rho_d$，通过 $\theta_d^g = \text{softmax}(\theta^g \odot \rho_d)$ 生成单文档分布；区别于直接对聚合文档建模或独立编码单文档的做法，在保留群体语义共性的同时精确刻画个体差异。
3. **上下文增强的重建损失机制**：构造增强文档 $\tilde{x}^d = x^d + \eta x^g$ 作为 VAE 重建目标；与 kNNTM 仅依靠局部 kNN 邻居增强相比，全局上下文能更全面地捕获跨文档的未观测相关词，从根本上平滑标签稀疏带来的损失偏差。
4. **统一且高效的端到端训练框架**：将聚类上下文生成、双分布推断与 ECR 正则化整合至单一变分下界中；相比 kNNTM 需离线预计算 $O(D^2)$ 距离矩阵，GloCOM 全程端到端可微，单卡 RTX 3090 训练耗时不足 10 分钟。

## 方法详解
- **文档聚类与全局上下文构造**：输入词袋表示 $\mathbf{X}=\{x^d\}_{d=1}^D$，经 PLM 编码得 $x_{PLM}^d$，使用 K-Means 划分为 $G$ 个簇；同一簇内所有短文档拼接为全局文档 $x^g$。
- **主题-词分布建模**：采用可微分解形式 $\beta_{ij} = \frac{\exp(-\|\mathbf{w}_i - \mathbf{t}_j\|^2/\tau)}{\sum_{j'}\exp(-\|\mathbf{w}_i - \mathbf{t}_{j'}\|^2/\tau)}$，词嵌入 $\mathbf{w}$ 以 GloVe 初始化，温度系数 $\tau=0.2$；配合最优传输正则化 $\mathcal{L}_{ECR}$（Sinkhorn 求解）防止主题坍塌与重复。
- **生成过程**：
  1. 对簇 $g$：$\theta^g \sim \mathcal{LN}(0, I)$。
  2. 对簇内文档 $d$：$\rho_d \sim \mathcal{N}(1, \epsilon I)$，计算 $\theta_d^g = \text{softmax}(\theta^g \odot \rho_d)$。
  3. 对第 $n$ 个词：$z_{dn} \sim \text{Multinomial}(\theta_d^g)$，$w_{dn} \sim \text{Multinomial}(\beta_{z_{dn}})$。
- **变分推断与优化目标**：使用完全因子化近似后验 $q_{\phi,\gamma}(\theta^g, \rho_d|x^g, x^d) = q_\phi(\theta^g|x^g)q_\gamma(\rho_d|x^d)$，两端均通过 MLP 网络参数化。单文档下界为：
  $\mathcal{L}^d = -(\tilde{x}^d)^\top \log(\text{softmax}(\beta\theta_d^g)) - D_{KL}(q_\phi(\theta^g|x^g)||p(\theta^g)) - D_{KL}(q_\gamma(\rho_d|x^d)||p(\rho_d|\epsilon))$
  整体损失 $\mathcal{L}_{GloCOM} = \sum_d\sum_g \mathbb{I}[x^d\in g]\mathcal{L}^d + \lambda_{ECR}\mathcal{L}_{ECR}$，采用重参数化技巧与蒙特卡洛采样进行端到端优化。

## 实验与结果
- **数据集**：GoogleNews（11,019 篇，152 标签）、SearchSnippets（12,294 篇，8 域）、StackOverflow（16,378 篇，20 标签）、Biomedical（19,433 篇，20 组），均经词频>3、长度≥2 预处理。
- **评估指标**：主题质量（TC $C_V$、TD）、文档主题分布质量（Purity、NMI）。
- **主要结果（K=50）**：
  - GoogleNews：GloCOM 获 **TC=0.475, NMI=0.817, Purity=0.586**，全面超越 kNNTM（TC 0.435, NMI 0.795）。
  - SearchSnippets：**TC=0.453, NMI=0.502, Purity=0.806**，领先 TSCTM 与 kNNTM。
