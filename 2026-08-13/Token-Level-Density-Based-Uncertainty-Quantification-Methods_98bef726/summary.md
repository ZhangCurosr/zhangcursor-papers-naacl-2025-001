---
title: "Token-Level-Density-Based-Uncertainty-Quantification-Methods"
source: https://aclanthology.org/2025.naacl-long.113.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:58:11"
field: "大语言模型可信性与不确定性估计"
keywords: ["不确定性量化", "大语言模型", "马氏距离", "词元级分析", "选择性生成", "幻觉检测"]
innovations: ["将马氏距离适配到LLM词元级别，首次在生成任务上实现有效的密度类UQ", "提出基于逐层密度特征的监督学习方法SATMD/SATRMD，通过PCA+线性回归实现高效不确定性估计"]
benchmarks: ["LM-Polygraph", "XSum", "SamSum", "PubMedQA", "MedQUAD", "TruthfulQA", "GSM8k", "CoQA", "SciQ", "TriviaQA", "MMLU"]
---

# 论文速读：Token-Level-Density-Based-Uncertainty-Quantification-Methods

## 一句话总结
本文首次将分类任务中成熟的马氏距离（Mahalanobis Distance, MD）密度类不确定性量化方法适配到 LLM 文本生成的**词元级别**，并在此基础上提出一种高效的监督式 UQ 方法（SATMD/SATRMD），在 11 个数据集的选择性生成任务和事实核查任务上大幅超越现有基线，且推理开销仅增加约 5–8%。

## 研究问题与动机
1. **LLM 幻觉问题与 UQ 的必要性**：大型语言模型在文本生成中不可避免地产生幻觉和非事实性陈述，在安全敏感场景中亟需可靠的不确定性量化手段以支持选择性生成/拒绝。
2. **密度类方法在生成任务中长期失效**：Lee 等（2018）的马氏距离等密度方法在分类任务中表现优异，但 Vashurin 等（2024）的基准测试表明，将其直接应用于 LLM 时（使用序列级 embedding），PRR 接近随机水平，几乎无效。
3. **层选择与 token 粒度的关键发现**：近期研究表明 LLM 内部状态包含不确定性信息，且中间层的表征比顶层更适合 decoder-only 模型；然而已有监督方法（如 SAPLMA、Factoscope）依赖简单特征，未有效引入成熟的密度类度量。
4. **计算效率与实用性的矛盾**：采样类方法（Semantic Entropy、SAR、Lexical Similarity）需要多次前向传播，带来 315%–700% 的计算开销，难以部署到大规模场景。

## 核心贡献（创新点）
1. **首次将密度类 UQ 方法适配到 LLM 词元级**：证明 token-level MD 在全数据集上显著优于 sequence-level MD/RMD，打破此前"密度方法对生成任务无效"的结论。
2. **提出基于层间密度特征监督学习的 SATMD/SATRMD**：逐层计算 token 级马氏距离，经 PCA 降维后训练线性回归预测不确定性，方法简洁且计算开销极低。
3. **设计嵌入选择策略提升密度估计质量**：从训练集中筛选高质量回答（Exact Match 或 AlignScore > 0.3）的 token embedding 构建协方差矩阵与质心，避免噪声拉偏分布估计。
4. **实验覆盖双任务与 11 数据集，全面验证有效性与泛化能力**：在 Llama 8b v3.1 和 Gemma 9b v2 上的选择性生成任务及 Mistral 7b v0.1 上的事实核查任务均取得 SOTA 结果；超参对阈值和 PCA 维数不敏感，少量训练数据（200–500 条）即可取得稳定性能。

## 方法详解

### 4.1 逐层不确定性分数（Layer-Wise Uncertainty Score）
**Embedding 提取与选择**：
- 从训练集生成响应中，依据正确性标准（Exact Match 或 AlignScore > 0.3）筛选出高质量 token 子集 $\mathcal{D}$，用其构建每层 $l$ 的协方差矩阵 $\Sigma_{\mathcal{E}_l}$ 和质心 $\mu_{\mathcal{E}_l}$。
- 该策略避免了使用预训练数据的困难，同时确保密度估计的准确性。

**Token 级马氏距离**（核心公式）：
$$
U^{\mathrm{MD}}(\mathbf{t}_i^k, l) = (h_l(\mathbf{t}_i^k) - \mu_{\mathcal{E}_l})^T \Sigma_{\mathcal{E}_l}^{-1} (h_l(\mathbf{t}_i^k) - \mu_{\mathcal{E}_l})
$$
- 对每层 $l=1,\dots,L$ 分别计算每个生成 token 的 MD 值。
- **ATRMD**（Relative 版本）：额外利用背景数据集（如 C4）计算背景协方差 $\Sigma_l^0$ 和质心 $\mu_l^0$，得分 $U^{\mathrm{RMD}} = U^{\mathrm{MD}} - U_0^{\mathrm{MD}}$。
- **序列级聚合**：对所有 token 的 MD 取平均得到 ATMD / ATRMD。

### 4.2 基于逐层分数的线性回归（Supervised Learning）
**特征构造**：
- 将各层 ATMD/ATRMD 拼接为向量 $f^*(\tilde{\mathbf{y}}^k) = [U^*(\tilde{\mathbf{y}}^k, 1), \dots, U^*(\tilde{\mathbf{y}}^k, L)]$。
- 对特征向量进行 PCA 降维，保留前 $N=10$ 个主成分以缓解多重共线性：$\tilde{f}^*(\tilde{\mathbf{y}}^k) = \mathrm{PCA}_N(f^*(\tilde{\mathbf{y}}^k))$。
- 可将序列概率 $P(\tilde{\mathbf{y}}^k|\mathbf{x}^k)$ 作为额外特征（MSp 补充）。

**训练流程**：
1. 将训练集 $\mathcal{T}$ 拆分为 $\mathcal{T}_1$（拟合协方差/质心）和 $\mathcal{T}_2$（训练回归器）。
2. 目标变量为质量度量的负值：$\mathbf{q}^k = -\mathcal{Q}(\tilde{\mathbf{y}}^k)$。
3. 在 $\mathcal{T}_2$ 上训练线性回归 $G^*(\cdot)$ 预测不确定性。
4. 最终用全量 $\mathcal{T}$ 重新估计协方差参数。
5. 最终监督分数：$U^{\mathrm{S}^*} = G^*(\tilde{f}^*)$，带 MSp 的版本记为 $U^{\mathrm{S}^*+\mathrm{MSP}}$。

### 4.3 混合不确定性量化（Hybrid UQ, HUQ）
- 将序列概率得分 $U_1 = 1 - P(\tilde{\mathbf{y}}^k|\mathbf{x}^k)$ 与 SATMD/SATRMD 得分 $U_2$ 通过 ranking 融合。
- 定义 ID 集合和模糊 ID 集合，引入混合超参 $\alpha$ 得到线性组合总不确定性 $U_T$，最终分段合成 HUQ 分数。

## 实验与结果
- **数据集（11 个）**：文本摘要（XSum, SamSum, CNN/DailyMail）、长答案 QA（PubMedQA, MedQUAD, TruthfulQA, GSM8k）、阅读理解（CoQA, SciQ）、短答案 QA（TriviaQA）、多选型 QA（MMLU）。
- **模型**：Llama 8b v3.1、Gemma 9b v2（选择性生成）；Mistral 7b v0.1 Instruct（事实核查）。
- **评测指标**：选择性生成用 PRR（Prediction Rejection Ratio），事实核查用 ROC-AUC 和 PR-AUC。
- **主要结果（Llama 8b v3.1）**：
  - HUQ-SATRMD 在均值排名上以 **2.94** 位列第一，显著优于所有基线。
  - 摘要任务：XSum（PRR-ROUGE-L 0.441）、SamSum（0.486）均取得最佳。
  - 长答案 QA（PubMedQA/MedQUAD/TruthfulQA/GSM8k）：**SATRMD+MSP** 以 **2.56** 均值排名领先，明显超越 SOTA。
  - 阅读理解 CoQA：HUQ-SATRMD 最优（PRR-AlignScore 0.515）。
  - MMLU：SATRMD+MSP 显著优于所有基线。
- **事实核查（Mistral 7b v0.1）**：
  - HUQ-SATRMD 取得 ROC-AUC **0.750**，优于 CCP 基线（0.716）；SATRMD+CCP 取得 PR-AUC **0.414**，优于 CCP（0.388）。
- **计算开销**：HUQ-SATRMD 和 SATRMD+MSP 仅增加 **5.3%–7.6%** 推理时间，远低于采样类方法的 315%–700%。
- **泛化性**：留一法 OOD 实验中，HUQ-SATRMD 以 **1.71** 均值排名仍为最优，对训练分布外的多选型 QA（MMLU）也有较强泛化。
- **鲁棒性**：对正确性阈值（0.1–0.8）和 PCA 组件数（5–20）变化不敏感；仅 **200–500 条** 训练数据即可稳定优于 MSP 基线。

## 相关工作脉络
1. **Lee 等（2018）马氏距离**：首次在 OOD 检测中提出 MD 作为不确定性度量，本文将其从分类迁移到 LLM token 级 UQ，是本工作的核心理论来源。
2. **Vashurin 等（2024）LM-Polygraph 基准**：系统性评测了 LLM UQ 方法，发现序列级密度方法（MD/RMD/RDE）在生成任务上接近随机，本文指出其根本原因在于使用了序列级平均 embedding 而非 token 级分析。
3. **Azaria & Mitchell（2023）SAPLMA**：证明 LLM 内部隐藏状态包含不确定性信号，使用 MLP 在单一层上训练；本文扩展为多层逐层特征融合 + 线性回归，并引入密度类特征。
4. **He 等（2024b）Factoscope**：使用深层神经网络（RNN+CNN）和多层次特征；本文对比发现，更简单的线性模型配合更精准的密度特征即能超越复杂架构。
5. **Kuhn 等（2023）Semantic Entropy / Duan 等（2024）SAR**：基于多次采样的自洽性/UQ 方法，计算开销大；本文方法单次前向即可计算，实现数量级的效率提升。
6. **Vazhentsev 等（2023a）HUQ**：混合不确定性量化框架，本文将其与提出的 token-level MD 特征结合，形成 HUQ-SATRMD，证明该融合框架的有效性。

## 局限性与未来方向
1. **依赖有标签训练数据**：方法为监督式，性能受限于训练数据的质量与规模；OOD 设置下性能有所下降，需谨慎应用于监督域外场景。
2. **未测试超大模型**：受算力限制，实验仅覆盖 7–9B 级别模型，LLaMA 3 70B 等大模型上的表现未知。
3. **误报潜在风险**：模型可能将安全、正确的生成文本错误标记为高不确定性，影响实际应用中的召回率。
4. **任务泛化边界**：尽管在 11 个数据集上验证了有效性，但在跨任务迁移时的鲁棒性仍需进一步探索。

## 研究启发与可借鉴点
1. **Token 级粒度是密度类 UQ 在 LLM 上成功的关键**：此前失败的主因是使用序列平均 embedding，改为 token 级后效果剧变——这一洞察可直接迁移到其他基于嵌入分布的分析任务。
2. **高质量 embedding 子集构建策略**：用 Correctness/AlignScore 阈值筛选 token 构建协方差矩阵，可复用于其他需要"干净参考分布"的模型诊断或校准任务。
3. **简单线性模型 + PCA 降维的稳定性**：在多层 MD 特征上以 10 个 PCA 分量训练线性回归，既有效又避免了过拟合，这种"深度特征+浅层模型"的思路值得在其他 UQ 变体中尝试。
4. **混合融合框架（HUQ）的可复用性**：将密度特征与序列概率通过 ranking 融合的策略，可推广到任意两个互补的不确定性信号的组合。
5. **与团队方向的结合机会**：可将 token 级 MD 特征作为幻觉检测、事实核查、选择性输出等下游任务的可迁移特征工程模块，或与 RAG 系统结合实现动态拒答机制。

## 关键术语表
**Mahalanobis Distance (MD)**：衡量某点到分布质心的距离，考虑特征间协方差结构，对 OOD 检测尤为有效。
**Token-level UQ**：在生成的每个词元上单独计算不确定性分数，而非对整个序列聚合后再评估。
**ATMD / ATRMD**：Average Token-level Mahalanobis Distance / Relative MD，逐层 MD 的平均聚合形式，其中 RMD 通过减去背景分布得分来增强区分力。
**HUQ (Hybrid Uncertainty Quantification)**：将多种不确定性信号（如序列概率与密度特征）通过排序融合策略合并的综合不确定性度量。
**PRR (Prediction Rejection Ratio)**：选择性生成的评估指标，衡量按不确定性排序后依次拒绝低置信生成时的平均质量提升。
**AlignScore**：基于对齐函数的事实一致性评分指标，用于评估生成文本与源文本的语义匹配程度。
**PCA（主成分分析）**：对高维特征向量进行线性降维，提取正交主成分以缓解多重共线性问题。
**SAPLMA / Factoscope**：分别基于 MLP 和深网络从 LLM 隐藏状态预测不确定性的监督方法。

## 可复现要素
- **数据集**：全部使用公开数据集（XSum, SamSum, CNN/DailyMail, PubMedQA, MedQUAD, TruthfulQA, GSM8k, CoQA, SciQ, TriviaQA, MMLU），均已公开。
- **代码开源**：论文未提及具体代码仓库，但使用了 LM-Polygraph 框架（Fadeeva 等，2023），可参考 https://github.com/ai-forever/lm-polygraph。
- **模型权重**：Llama 8b v3.1、Gemma 9b v2、Mistral 7b v0.1 Instruct 均为开源模型，公开可获取。
- **关键超参**：PCA 主成分数 $N=10$；token 筛选阈值（精确匹配或 AlignScore > 0.3）；采样次数 5；训练集拆分 $\mathcal{T}_1/\mathcal{T}_2$；HUQ 超参 $\alpha$ 在 $\mathcal{T}_2$ 上调优。
- **GPU 资源**：6 × NVIDIA H100，总实验时间约 400 GPU 小时。
