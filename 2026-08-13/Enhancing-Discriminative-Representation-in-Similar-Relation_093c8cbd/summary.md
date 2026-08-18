---
title: "Enhancing-Discriminative-Representation-in-Similar-Relation"
source: https://aclanthology.org/2025.naacl-long.123.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:00:25"
field: "持续关系抽取"
keywords: ["Few-Shot Continual Relation Extraction", "Catastrophic Forgetting", "LLME", "Contrastive Learning", "Label Description", "Dynamic Clustering"]
innovations: ["利用关系描述嵌入进行动态层次聚类识别相似关系簇，替代不稳定的样本原型", "设计WSC/CMI/DT三类加权判别损失，从样本-描述对齐与簇级边界双重角度缓解相似关系混淆导致的灾难性遗忘"]
benchmarks: ["FewRel (10-way-5-shot)", "TACRED (5-way-5-shot)"]
---

# 论文速读：Enhancing-Discriminative-Representation-in-Similar-Relation Clusters for Few-Shot Continual Relation Extraction

## 一句话总结
本文提出SIRUS方法，通过关系描述动态聚类识别相似关系簇，并设计三类判别性损失函数强化相似关系间的样本区分度，有效缓解少样本持续关系抽取中的灾难性遗忘问题；同时首次系统探索大语言模型嵌入（LLMEs）在FCRE场景下的潜力，验证其优于传统BERT及因果语言建模LLM。

## 研究问题与动机
- **相似关系导致的灾难性遗忘**：现有FCRE方法多依赖记忆回放+对比学习范式，但未充分考虑语义相近关系（如"country of origin"与"country of citizenship"）之间的混淆问题，这是导致持续学习中灾难性遗忘的关键因素。
- **原型表示的局限性**：ConPL等方法基于实体样本平均构建关系原型来识别相似类，但少样本下原型代表性不足，且不同上下文会导致原型不稳定，难以可靠识别相似关系。
- **LLMEs在持续学习中的未知性**：LLMEs（通过移除因果掩码、改用对比学习训练的解码器型LLM）在表示学习方面表现优异，但其在FCRE场景下的灾难性遗忘特性及适应性尚未被系统研究。

## 核心贡献（创新点）
- **提出CRLD动态聚类框架**：利用关系描述嵌入进行自底向上聚类识别相似关系簇，替代基于样本原型的静态相似性判断，聚类阈值由距离自动确定无需人工调参，且每批次动态更新。
- **设计三类加权判别损失**：加权监督对比损失（WSC）、基于簇的互信息损失（CMI）、双三元组损失（DT），三者均通过描述相似度加权硬负样本，从样本-描述对齐、簇内凝聚、簇间分离三个角度增强判别性。
- **首次系统评估LLMEs在FCRE中的表现**：发现LLMEs（如LLM2Vec、BGE）仍存在灾难性遗忘，但结合SIRUS后可显著超越BERT基线及原始因果LLM，证明LLMEs在持续关系抽取中的巨大潜力。
- **端到端可复现的实验验证**：在FewRel（10-way-5-shot）和TACRED（5-way-5-shot）两大基准上，SIRUS+BERT在最终任务上分别达到69.16%和60.68%，较次优方法CPL_MI提升约3%和1.2%。

## 方法详解

**CRLD（Clustering Relations via Label Description）**
- 将每个关系的描述文本$d_i$直接输入编码器$f_\mathcal{M}$，经mean pooling得到嵌入$\mathbf{d}_i$；
- 对当前批次所有关系描述嵌入执行Agglomerative Clustering，以距离阈值$\theta$自动确定簇数$K$；
- 聚类在每批次训练后动态更新，反映编码器参数变化。

**三类判别损失函数**

1. **加权监督对比损失（WSC）**：
$$\mathcal{L}_{\text{WSC}} = -\sum_{p \in P(x)} \log \frac{f(z_x, z_p)}{\sum_{\bar{x} \in \mathcal{D}\setminus\{x\}} w(x,\bar{x}) \cdot f(z_x, z_{\bar{x}})}$$
其中权重$w(x,\bar{x}) = 1 + \alpha \cdot \gamma(d_x, d_{\bar{x}})$仅当两关系同簇时生效，$\gamma$为余弦相似度，$\alpha$控制加权强度。

2. **基于簇的互信息损失（CMI）**：
$$\mathcal{L}_{\text{CMI}}(x) = -\log \frac{h(z_x, d_x)}{h(z_x, d_x) + \sum_{\bar{x} \in \mathcal{N}(x)} w(x,\bar{x}) \cdot h(z_x, d_{\bar{x}})}$$
其中$h(z,d)=\exp(z^TWd/\tau)$，$\mathcal{N}(x)$为不同于$x$关系的样本集合，通过可学习矩阵$W$对齐样本嵌入与描述嵌入。

3. **双三元组损失（DT）**：
$$\mathcal{L}_{\text{DT}}(x)=\max(0, D(z_x,d_x)-D(z_x,c_x^+)+m_1)+\max(0, D(z_x,c_x^+)-D(z_x,c_x^-)+m_2)$$
其中$c_x^+$为$x$所属簇的 centroid，$c_x^-$为距$x$最近的**不包含**其关系的簇centroid，$D$为余弦距离。

**总损失**：$\mathcal{L}(x)=\lambda_1\mathcal{L}_{\text{WSC}}+\lambda_2\mathcal{L}_{\text{CMI}}+\lambda_3\mathcal{L}_{\text{DT}}$

**LLMEs输入模板**：采用指令格式"x. The relation between [e_h] and [e_t] is: "，mean pooling输出作为嵌入表示。

**推理**：Nearest-Class-Mean分类器，综合原型$p_r$与描述$d_r$相似度：$y^*=\arg\max_r(\gamma(z_x,p_r)+\gamma(z_x,d_r))$。

## 实验与结果

**数据集**：FewRel（100关系，80个用于8任务，10-way-5-shot）和TACRED（41关系，8任务，5-way-5-shot）。

**基线方法**：RP-CRE、CRL、CRECL、ERDA、SCKD、ConPL、CPL、CPL_MI（共8个SOTA基线）。

** backbone**：BERT-base-uncased、LLM2Vec（Llama2/Llama3/Mistral变体）、BGE。

**主要结果（BERT backbone）**：
- FewRel最终任务：SIRUS达**69.16%**，较CPL_MI（66.27%）提升**+2.89%**，较次优ConPL（62.46%）提升约6.7个百分点。
- TACRED最终任务：SIRUS达**60.68%**，较CPL_MI（59.48%）提升**+1.2%**。
- 各任务阶段SIRUS均领先，混淆矩阵显示相似关系对（如"country"/"country of origin"）误分类率显著降低。

**LLMEs结果**：
- SIRUS+LLM2Vec(Llama3)在FewRel上达**78.82%**，SIRUS+BGE(Mistral)在TACRED上达**73.97%**，均大幅超越BERT基线（约+10%）。
- 无记忆缓冲的LLMEs仍出现严重灾难性遗忘（如LLM2Vec纯对比学习在FewRel T8仅63.38%），验证SIRUS损失的必要性。
- LLMEs+原始因果LLM（CPL）性能远逊于LLMEs+Causal Mask移除版本，证实表征学习能力的关键作用。

## 相关工作脉络
- **RP-CRE / CRL / CRECL**：早期基于记忆回放+对比学习的FCRE方法，侧重原型稳定与知识蒸馏，未显式建模相似关系混淆问题。
- **ConPL（Chen et al., 2023）**：首个利用focal loss处理相似类的FCRE方法，但依赖样本均值原型识别相似类，少样本下原型不稳定；SIRUS用描述嵌入+动态聚类替代，更鲁棒。
- **CPL（Ma et al., 2024）**：引入prompt learning+margin contrastive loss，并结合ChatGPT数据增强缓解过拟合；SIRUS不依赖LLM数据增强，以判别损失直接刻画相似关系边界。
- **CPL_MI（Tran et al., 2024）**：将互信息最大化应用于LM head以保留预训练知识；SIRUS将MI应用于样本-描述对齐，并扩展至描述级聚类加权。
- **LLMEs（LLM2Vec/BGE）**：将解码器型LLM改造为双向编码器用于嵌入学习；本文首次将其系统引入FCRE，揭示其遗忘特性及与SIRUS的协同增益。

## 局限性与未来方向
- **仅限端到端关系抽取的上层任务**：假设实体已给定，未覆盖联合实体识别+关系抽取的端到端场景，后者需同时应对过拟合与跨任务遗忘双重挑战。
- **未显式处理少样本过拟合**：SIRUS聚焦相似关系混淆引起的遗忘，未引入数据增强或正则化手段应对样本稀缺导致的过拟合，作者计划后续结合数据增强进一步提升性能。
- **计算开销略增**：SIRUS每批次需前向编码描述嵌入并执行层次聚类，平均额外耗时约12秒/epoch（vs. CPL的10.34秒），虽可接受但可进一步优化聚类效率。

## 研究启发与可借鉴点
- **描述嵌入替代原型作为相似性度量**：在少样本场景下，标签描述比样本均值原型更稳定，可将此思路迁移至其他少样本持续学习任务（如持续分类、持续NER）。
- **动态聚类+硬负样本加权**：每批次重新聚类并依据簇内相似度调整对比损失权重，是一种轻量且即插即用的"相似关系感知"模块，可复用于其他持续表征学习任务。
- **LLMEs作为FCRE backbone的可行性**：本文证明了LLMEs移除因果掩码后的表示能力在持续关系抽取中远超原始decoder-only LLM，后续可探索LLMEs在其他持续NLP任务（如持续语义角色标注、持续事件抽取）中的应用。
- **三元组损失与互信息损失的协同设计**：CMI负责拉近样本-描述对，DT负责簇级边界约束，两者互补——此类多粒度判别目标的设计范式可推广至持续向量量化或持续检索任务。

## 关键术语表
- **Few-Shot Continual Relation Extraction（FCRE）**：在连续任务流中，仅用极少标注样本持续学习新关系并保留旧知识的抽取任务。
- **Catastrophic Forgetting（灾难性遗忘）**： continual learning中模型在新任务上训练后对旧任务性能急剧下降的现象。
- **CRLD（Clustering Relations via Label Description）**：本文提出的利用关系描述嵌入进行动态层次聚类以识别相似关系簇的方法。
- **LLME（Large Language Model Embedding）**：通过移除因果掩码、替换训练目标（如对比学习）将decoder-only LLM改造为双向文本编码器。
- **WSC（Weighted Supervised Contrastive）损失**：在监督对比学习基础上，对同簇关系的负样本按描述相似度加权，强化相似关系间的分离。
- **CMI（Cluster-based Mutual Information）损失**：最大化样本嵌入与其关系描述之间的互信息，并用簇内描述相似度加权硬负样本。
- **DT（Double Triplet）损失**：约束样本靠近自身描述、靠近所属簇centroid、远离最近异簇centroid的双三元组边界损失。
- **Nearest-Class-Mean Classifier**：推理阶段基于样本与各类原型/描述的余弦相似度之和进行最近质心分类。

## 可复现要素
- **数据集**：FewRel与TACRED均为公开数据集；实验划分方案参照Qin & Joty（2022）设定。
- **代码/权重**：论文未声明代码开源仓库链接；预训练backbone为Hugging Face公开模型（BERT-base-uncased、LLM2Vec系列、bge-en-icl）。
- **关键超参**：学习率{1e-5, 2e-5, 1e-4}；$\alpha\in\{0.1,0.15,0.2,0.25\}$；$\lambda_1\in\{0.5,1.0,1.5,2.0,2.5\}$，$\lambda_2$同域；$\lambda_3\in\{0.25,0.5,0.75,1.0\}$；$\tau_{\text{CMI}}\in\{0.01,0.02,0.03,0.04,0.05\}$；$m_1,m_2\in\{1.0,2.0\}$；聚类阈值$\theta\in\{0.1,\dots,0.8\}$；LoRA target modules为q/k/v/o/gate/up/down_proj。
