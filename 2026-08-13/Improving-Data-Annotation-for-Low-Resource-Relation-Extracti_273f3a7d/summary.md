---
title: "Improving-Data-Annotation-for-Low-Resource-Relation-Extracti"
source: https://aclanthology.org/2025.naacl-long.70.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:04:07"
field: "低资源关系抽取"
keywords: ["低资源关系抽取", "逻辑规则", "大语言模型", "知识蒸馏", "协作标注", "伪标签"]
innovations: ["引入逻辑规则作为可解释任务知识，通过演绎推理提升LLM关系抽取准确性", "提出CLMA迭代协作框架，LLM与SLM互补实现高效高质量伪标注", "设计基于预匹配度的NOTA识别与一致性过滤机制，减少错误累积"]
benchmarks: ["SemEval 2010", "TACRED", "TACREV", "Re-TACRED"]
---

# 论文速读：Improving-Data-Annotation-for-Low-Resource-Relation-Extracti

## 一句话总结
论文提出**CLMA（Collaborative Language Models-based data Annotation）**框架，通过引入可解释的逻辑规则辅助大语言模型（LLM）进行关系抽取，并将规则诱导能力蒸馏至小规模任务模型（SLM），迭代协作从大量无标签数据中获取高质量伪标注，在低资源关系抽取任务上取得新SOTA。

---

## 研究问题与动机

1. **低资源关系抽取（LRE）依赖大量人工标注数据，成本高昂**：现有神经网络方法需要充足标注样本，而真实场景中标注数据稀缺是主要瓶颈。
2. **LLM直接用于关系抽取存在两大缺陷**：① 传统 demonstrations 以 (样本, 标签) 对形式呈现，无关上下文中的噪声模式会误导LLM理解输入-输出映射；② LLM本身存在幻觉和不确定性，低资源场景下预测可靠性进一步下降。
3. **仅用LLM标注无标签数据效率低且成本高**：大模型推理速度慢、计算开销大，难以直接处理海量无标签样本。
4. **已有弱监督规则方法依赖人工或人工校验**：如PRBOOST、KICE等方法仍需人工参与规则评估，自动化程度不足。

---

## 核心贡献（创新点）

1. **引入逻辑规则作为可解释的任务知识表示**：将关系抽取的输入-输出映射抽象为由主体实体类型 $f_s$、客体实体类型 $f_o$ 和关系模式 $f_r$ 构成的前提，与结论（关系标签 $r$）对应，使LLM通过演绎推理而非表面模式匹配进行预测。
2. **提出CLMA迭代式协作标注框架**：LLM作为教师生成高质量规则并验证伪标注，SLM作为学生高效处理海量无标签样本，二者在尺度上互补，解决单一模型可靠性与效率的两难问题。
3. **设计数据选择与一致性过滤机制**：针对高频NOTA类别的不平衡问题，提出基于预匹配度的错配规则识别策略，并通过 Filter(·) 函数确保LLM标注结果与SLM高置信度预测一致，减少错误累积。

---

## 方法详解

### 3.1 问题定义
- 样本 $x = \{S, e_s, e_o\}$，标签集合 $\mathcal{Y} = \{r_1, ..., r_R, \text{NOTA}\}$
- 少量标注集 $\mathcal{D}_a$（K=8/16 per class），大量无标注集 $\mathcal{D}_u$

### 3.2 逻辑规则诱导（Logical Rules Induction）
**规则形式**：$\rho = [f_s; f_r; f_o] \rightarrow r$，其中前提 $p$ 为语义模式 conjunction。

**三阶段诱导流程**：
1. **零样本前提诱导**（公式1）：使用指令 $I_p$ 让LLM从标注样本中提取 $[\tilde{f}_s, \tilde{f}_r, \tilde{f}_o]$
2. **可能关系预测**（公式2）：让LLM预测前提 $\tilde{p}_a$ 可能对应的关系集合，得到错误关系集 $\mathcal{R}_{error}$
3. **前提精炼**（公式3）：以真实标签 $r_a$ 为参考，让LLM修正模糊模式，重复 $L=2$ 次得到最终规则集 $\mathcal{G}_0$

**知识蒸馏**：将规则诱导能力从LLM $M_L$ 蒸馏至小模型 $M_S$（Flan-T5-large, 780M），输入为自然语言描述的句子+实体，输出为 tokenized 规则+标签序列，使用因果语言建模损失（公式6）。

### 3.3 协作数据标注（Collaborative Data Annotation）
在 $T=10$ 轮迭代中，每轮执行：

**数据获取**（3.3.1）：
- SLM快速预测无标签样本（公式7），计算标签置信度 $s_r$（公式8）
- 取 top-$N_r$ 高置信度样本作为 $\mathcal{D}_L$
- 针对NOTA类别，计算样本与现有规则的语义匹配度 $s_p$（公式9-10），取 top-$N_p$ 最低匹配度样本作为 $\mathcal{D}_M$

**演绎推理**（3.3.2）：
- 对 $\mathcal{D}_L \cup \mathcal{D}_M$ 中的样本，检索最相似的 $Z=5$ 条规则（公式11-12）
- 构建展示列表 $\hat{\mathcal{C}}_d$，引导LLM诱导新前提（公式13）
- 将匹配规则以 if-then 形式嵌入提示，引导LLM进行演绎推理得出标签 $r_u$（公式14-16）

**一致性过滤**（3.3.3）：
- 使用 Filter(·) 函数验证LLM输出与SLM高置信度预测是否一致（公式17）
- 仅保留通过过滤的样本扩充标注集，更新规则集 $\mathcal{G}_t$ 并重新训练 $M_S^t$

### 3.4 推理模式
- **LLM直接推理**：直接用LLM在少样本数据上测试（无未标注数据场景）
- **LLM标注数据**：用增强后的SLM进行推理（实时/低延迟场景）

---

## 实验与结果

### 数据集
- **SemEval 2010**：19类关系，训练集144/662
- **TACRED**：42类关系，训练集334/662
- **TACREV**：TACRED纠错版本
- **Re-TACRED**：重新定义关系并修正误分类样本
- 采用 **True Few-shot** 设置（K=8/16 per class），五轮随机种子平均

### 评估指标
Micro F1 score

### 主要结果（Table 2）

| 设置 | TACREV (K=8) | SemEval (K=8) | Re-TACRED (K=8) | TACRED (K=8) |
|------|-------------|---------------|------------------|--------------|
| CLMA → $M_S$ | **51.16** | **67.12** | **63.65** | **43.86** |
| Base ($M_S$) | 30.98 | 61.18 | 55.59 | 28.90 |
| GPT-RE → $M_S$ | 35.94 | 38.01 | 38.65 | 33.00 |
| LLMaAA → $M_S$ | 38.95 | 57.39 | 35.94 | 24.87 |
| PGDG → $M_S$ | 30.19 | 66.86 | 53.80 | 28.31 |
| QA4RE → $M_S$ | 17.34 | 44.86 | 36.27 | 14.33 |

**最强结果**：CLMA → $M_S$ 在TACREV K=8上达到 **51.16 F1**，相比Base模型提升 **+20.18**，相比次优基线LLMaAA提升 **+12.21**。

### 消融实验
- **w/o Unl. Data**：平均F1下降 **10.9**，证明无标签数据的重要性
- **w/o Rule**：平均F1下降 **11.5**，证明逻辑规则的核心作用
- **w/o SLM**：性能下降，说明单一LLM难以处理困难样本
- **w/o LLM**：SLM会强化自身预测错误，说明LLM的纠错能力关键

### 伪标签质量
- TACREV上伪标签平均准确率 **85%**
- 规则前提一致性：来自 $\mathcal{D}_a$ 的规则Context Relation准确率达 **0.95/0.90**；来自 $\mathcal{D}_u$ 的规则达 **0.87/0.85**

---

## 相关工作脉络

1. **QA4RE（Zhang et al., 2023）**：将RE对齐为多选问答，通过关系模板 verbalize 标签。本文认为其样本-标签对形式易受无关上下文干扰，改用逻辑规则显式表达映射关系。

2. **GPT-RE（Wan et al., 2023）**：基于实体的 in-context learning，检索相似 demonstrations。本文指出 demonstration 中噪声模式会误导LLM，引入规则增强演绎推理。

3. **PGDG（Ding et al., 2023）**：让LLM合成新标注样本。本文认为合成数据与真实数据分布可能存在差异，CLMA直接从真实无标签数据中提取规则。

4. **LLMaAA（Zhang et al., 2023b）**：active learning循环中让LLM标注最低置信度样本。本文发现该方法效果不佳，推测是因为低置信度样本本身难标注；CLMA选择高置信度样本进行验证而非直接标注难样本。

5. **PRBOOST / KICE（弱监督规则学习）**：需要人工参与规则评估。本文完全自动化规则诱导，利用LLM知识替代人工校验。

6. **知识蒸馏（Sun et al., 2019; Jiang et al., 2023）**：传统方法需访问教师模型内部参数。本文采用黑盒方式，将LLM的规则诱导能力蒸馏至SLM，适用于闭源LLM。

---

## 局限性与未来方向

1. **LLM依赖性**：方法效果依赖于基础LLM的能力（如指令遵循能力），不同LLM可能产生差异结果。
2. **SLM架构局限**：使用Flan-T5-large作为小模型，参数量或架构变化可能影响蒸馏效果。
3. **实验设置局限**：仅评估 true few-shot 设置（K=8/16 per class），未涵盖 N-Way K-Shot、固定百分比低资源等其他设定。
4. **可扩展性未验证**：仅在句子级RE上验证，论文自述未来将扩展到文档级RE等更挑战性任务。
5. **伦理风险**：使用LLM生成数据训练小规模模型时，需注意敏感数据泄露风险和LLM许可证合规性。

---

## 研究启发与可借鉴点

1. **逻辑规则作为可解释知识载体**：将RE任务映射抽象为 $[f_s; f_r; f_o] \rightarrow r$ 的规则形式，既便于LLM演绎推理，也可作为模型间知识传递的统一介质。此思路可迁移至其他结构化预测任务（如事件抽取、知识图谱补全）。

2. **"高置信度验证"而非"低置信度标注"的策略**：不同于LLMaAA选择难样本，CLMA选择SLM高置信度样本让LLM验证，避免LLM在困难样本上犯错后污染数据集。这一"验证优于标注"的设计思路值得借鉴。

3. **错配规则处理稀有类别**：针对NOTA等长尾类别，通过语义匹配度识别与现有规则不匹配的样本，将其作为潜在NOTA候选。该策略可扩展至其他类别不平衡的场景。

4. **两阶段蒸馏+迭代协作范式**：先诱导规则→蒸馏至SLM→SLM筛选→LLM验证→规则扩展→重训练SLM的闭环，为其他"大模型+小模型"协作的数据增强任务提供了可复用框架。

---

## 关键术语表

**Low-Resource Relation Extraction (LRE)**：关系抽取的低资源变体，指仅有少量（K=8/16 per class）标注样本可用的关系抽取任务。

**Logical Labeling Rule**：由前提（主体实体类型、客体实体类型、关系模式）和结论（关系标签）构成的规则，形式为 $[f_s; f_r; f_o] \rightarrow r$。

**Collaborative Language Models-based data Annotation (CLMA)**：本文提出的协作标注框架，通过LLM（教师）和SLM（学生）的迭代协作从无标签数据中获取高质量伪标注。

**In-context Learning (ICL)**：利用大模型在提示中提供的 demonstrations 进行零样本或少样本推理的技术。

**NOTA (None-of-the-above)**：关系抽取中的负类标签，表示句子中实体对的关系不在预定义关系集合中。

**Teacher-Student Distillation**：将大模型（教师）的知识迁移至小模型（学生）的过程，本文特指将LLM的规则诱导能力蒸馏至SLM。

**Deductive Reasoning Prompt**：以 if-then 规则形式呈现前提，引导LLM通过演绎推理而非模式匹配进行预测的提示策略。

**True Few-shot Learning**：Perez et al. (2021) 提出的低资源学习设置，从每个类别的真实标注样本中采样K个，而非随机采样。

---

## 可复现要素

- **数据集**：SemEval 2010（CC BY 3.0）、TACRED（LDC license）、TACREV、Re-TACRED，均为公开数据集
- **代码**：论文附录提到 "All other hyperparameters are provided in our source code"，但正文未给出GitHub链接，需进一步确认
- **模型权重**：Flan-T5-large（Apache 2.0 license，HuggingFace开源）；GPT-3.5-turbo（API调用）
- **关键超参**：
  - LLM：gpt-3.5-turbo，temperature=0.0
  - SLM：Flan-T5-large（780M）
  - 演示数量 Z=5
  - 迭代轮数 T=10
  - 每轮选取样本数：TACRED系 $N_r=N_p=200$，SemEval $N_r=N_p=50$
  - 前提精炼次数 L=2
  - SLM训练：30 epochs，batch size=8

---
