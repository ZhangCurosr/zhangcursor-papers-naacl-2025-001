---
title: "Conformalized-Answer-Set-Prediction-for-Knowledge-Graph-Embe"
source: https://aclanthology.org/2025.naacl-long.32.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:02:24"
---

# 论文速读：Conformalized-Answer-Set-Prediction-for-Knowledge-Graph-Embe

## 一句话总结
本文首次将共形预测（Conformal Prediction）理论引入知识图谱嵌入（KGE）的链接预测任务，通过设计适配KGE得分机制的非一致性度量与分裂共形框架，生成具有严格概率覆盖保证的答案集，解决了传统排序输出缺乏不确定性量化与形式化统计保障的问题。

## 研究问题与动机
- KGE模型通过标量得分对候选实体排序，但高排名实体并不必然对应高真实概率，导致预测结果缺乏可靠的可解释性与风险可控性。
- 现有不确定性量化方法（Platt scaling、Isotonic regression）属于经验校准，对校准集高度敏感且无法提供形式化的概率保证，难以支撑医疗、药物发现等高敏感决策场景。
- 直接截断Top-K或使用Naive概率累加的方法，要么无法保证覆盖率，要么生成过大/固定大小集合，缺乏对单条查询不确定性的自适应反馈。

## 核心贡献（创新点）
1. **首次将共形预测应用于KGE链接预测**：提出即插即用的答案集预测模块，从理论上保证输出集合以至少 $1-\epsilon$ 的概率包含真实答案，填补了KGE不确定性量化缺乏统计严格性的空白。
2. **设计三种KGE适配的非一致性度量**：针对距离型（NegScore）与语义匹配型（Softmax/Minmax）模型的得分分布差异，分别构造原始负得分、归一化相对位置与概率化残差三类度量，实现不同架构的最优适配。
3. **采用分裂共形预测提升计算效率**：避免为每个候选实体重训练模型，仅需一次训练+一次校准集评分，将渐近时间复杂度降至 $\mathcal{O}(|E|)$，在保持理论保证的同时具备工程可扩展性。
4. **系统验证答案集的三大 desirable 性质**：在四个基准数据集与六种主流KGE模型上，实证覆盖率达理论边界、集合紧度显著优于基线、且答案集大小能随查询难度自适应伸缩。

## 方法详解
- **问题重构**：将链接预测 $(h,r,?)$ / $(?,r,t)$ 转化为答案集预测任务，目标为输出 $\hat{C}(q_{n+1}) \subseteq E$，满足 $\mathbb{P}(e_{n+1} \in \hat{C}(q_{n+1})) \ge 1-\epsilon$，同时追求集合紧度与查询难度自适应性。
- **非一致性度量**：
  - **NegScore**：$S = -M_{\mathcal{T}}(t)$，直接取KGE模型得分负值，适用于TransE/RotatE等距离型模型。
  - **Minmax**：$S = -\overline{M}(t)$，对单查询所有候选实体得分做min-max归一化后取负，消除跨查询尺度差异，关注相对排名位置。
  - **Softmax**：$S = 1 - \hat{M}(t)$，将得分经softmax转为伪概率分布后取残差，突出真实答案与分布中心偏离程度，适用于RESCAL/DistMult/ComplEx/ConvE。
- **分裂共形预测流程**：
  1. 划分真训练集 $\mathcal{T}_{1:m}$ 与校准集 $\mathcal{T}_{m+1:n}$。
  2. **校准阶段**：在 $\mathcal{T}_{1:m}$ 训练KGE模型，计算校准集所有三元组的非一致性得分 $\alpha_i$，取 $(1-\epsilon)$ 分位数作为阈值 $\tau$。
  3. **预测阶段**：对测试查询 $q_{n+1}$ 遍历全量实体 $e \in E$，将 $S(\mathcal{T}_{1:m}, tr(q_{n+1}, e)) < \tau$ 的实体纳入 $\hat{C}(q_{n+1})$。
- **复杂度**：因无需逐候选重训，主要开销为一次前向打分与排序，渐近复杂度 $\mathcal{O}(|E|)$，与训练图谱规模解耦。

## 实验与结果
- **设置**：四个公开链接预测基准（WN18、WN18RR、FB15k、FB15k237）；六种KGE backbone（TransE、RotatE、RESCAL、DistMult、ComplEx、ConvE）；基线包括Naive、Platt、TopK；误差率统一设为 $\epsilon=0.1$，重复15次取均值。
- **覆盖率与紧度**：所有共形预测器严格满足 $\ge 0.90$ 覆盖率。在WN18上，TransE的NegScore平均集合大小为 $20.99$，显著优于TopK（$48.01$）与Platt（$4043.41$）；在FB15k上，RotatE的Softmax集合大小低至 $1.27$。距离型模型最优为NegScore，语义匹配型最优为Softmax/Minmax。
- **自适应性**：按真实答案排名分组（每100名为一个难度区间），
