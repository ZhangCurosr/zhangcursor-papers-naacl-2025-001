---
title: "Generating-Complex-Question-Decompositions-in-the-Face-of-Di"
source: https://aclanthology.org/2025.naacl-long.55.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:02:54"
field: "复杂问答与推理"
keywords: ["问题分解", "合成数据生成", "小语言模型", "LLM-as-judge", "分布偏移", "投票机制"]
innovations: ["仅需5个标注样本的CLLM面板合成数据生成方法", "扩展LLM-as-ranker至分解质量排序并结合STV投票", "证明小模型面板+微调可匹敌大模型分解性能"]
benchmarks: ["break", "musique", "2wikimultihop"]
---

# 论文速读：Generating-Complex-Question-Decompositions-in-the-Face-of-Di

## 一句话总结
论文提出了一种仅需5个标注样本即可生成合成问题分解数据的方法，通过使用小型LLM（CLLMs）面板进行多候选生成与投票排序，并将合成数据用于微调小型模型，在分布偏移场景下实现优于或少量损失地接近大模型的性能。

## 研究问题与动机
- **复杂问题分解的价值与瓶颈**：将复杂多跳问题分解为子问题序列可帮助LLM提升QA表现，但当前SOTA为监督方法，而监督模型在面对分布偏移时性能显著下降。
- **零/少样本LLM方法的不足**：纯few-shot提示的LLM在分解任务上表现不如监督方法，存在提升空间。
- **合成数据生成的资源瓶颈**：现有合成数据生成依赖大型模型（如GPT-4、Llama 70B），计算成本高且难以规模化应用于新领域。
- **分布偏移下的泛化需求**：需要一种仅需少量标注即可生成高质量合成数据的方法，以支持跨领域/新题型的数据自动构建。

## 核心贡献（创新点）
1. **提出仅需5个标注样本的合成分解数据生成流程**：通过CLLM面板生成+投票排序，大幅降低对人工标注数据的依赖；与现有方法依赖大量标注或超大模型的本质区别在于"近零样本+小型模型面板"的组合。
2. **创新性扩展LLM-as-judge/LLM-as-ranker至问题分解排序任务**：将RankLLM从零样本段落排序扩展至分解质量排序，通过构造带人工注入错误的示例实现two-shot提示；与Verga等人仅使用二元评分的judge机制不同，本文采用排序偏好+STV投票。
3. **使用小型LLM面板替代大模型进行数据生成与质量评估**：选用4个3-9B参数的CLLM组成生成与排序面板，证明在固定计算预算下小模型组合优于单一大模型；与Agrawal等人使用540B模型生成数据的做法形成资源效率对比。
4. **系统性验证分布偏移下的泛化能力**：在break/musique跨域训练-测试设置下评估FT-PANEL模型，揭示不同数据集结构对OOB性能的影响机制。

## 方法详解
**整体流程分为三步：**

1. **多候选分解生成（Candidate Generation）**
   - 使用4个CLLM（Llama 3.1 8B、Gemma 2 9B、Phi-3.5 3.8B、Qwen 2.5 7B）作为$\mathrm{CLLM}_{dqg}$，每个模型对复杂问题$q_c$生成一个分解候选$Q_s$。
   - 采用5-shot + Chain-of-Thought (CoT) 提示，仅需从训练集中随机选取5个示例，无需答案信息。

2. **分解质量排序（Ranking via Panel）**
   - 使用同一组4个CLLM作为$\mathrm{CLLM}_{rank}$组成评审面板，对每个问题的所有候选分解进行质量排序。
   - 构造two-shot排序提示：通过GPT-4o系统性地对参考分解注入多种类型错误，按预定义评分方案生成不同质量层级的示例作为prompt exemplars。
   - 为缓解位置偏差（position bias），在每次排序任务中随机打乱候选呈现顺序。

3. **投票选择（Selection via STV）**
   - 采用单一可转移投票（Single Transferable Vote, STV）机制从面板排序中选出每个问题的最优分解，记为PANEL数据。
   - 将STV类比选举过程：每个$\mathrm{CLLM}_{rank}$为"选民"，每个候选分解为"候选人"，排序即为"投票"。

**模型微调：**
- 使用PANEL数据微调Llama 3.1 8B和Qwen 2.5 7B，采用LoRA适配器（rank=32）+ RSLoRA控制α参数。
- 使用3-shot CoT提示，在A100 GPU上训练2个epoch（约8小时/模型）。

## 实验与结果
**数据集**：
- **break**：专为问题分解设计，使用来自HotpotQA的约45%子集，采用QDMR形式标注。
- **musique**：多跳QA数据集，子问题为自然语言形式，附带答案。
- **2wikimultihop**：仅用于监督模型的OOB性能评估。

**评估指标**：EM（精确匹配）、SARI（文本简化指标适配）、GED（图编辑距离）；下游QA评估使用LLM-CoQ和 supervised QA模型。

**关键结果**：
- **PANEL vs 单模型**：PANEL在break上SARI/GED优于多数单模型，在musique上全面超越所有单CLLM。
- **FT-PANEL vs Few-shot**：微调后Llama 3.1 8B在break上SARI提升4.33点（0.6728 vs 0.6295）、GED提升5.61点；Qwen 2.5 7B在musique上SARI提升4.33点、GED提升2.67点。
- **FT-PANEL vs 大模型**：Llama 3.1 8B (FT-PANEL) 在break上SARI 0.6728超越GPT-4o的0.6314；在musique上与GPT-4o相当（0.6211 vs 0.6050）。
- **OOB性能**：在musique上训练的FT-PANEL测试到break时，Llama 3.1 8B获得>2% SARI和>4% GED提升；反之则受限于break数据集的受限语言形式，提升有限。

## 相关工作脉络
1. **Press et al. (2023)** 提出CoT/CoQ分解式QA框架，本文聚焦于改进分解生成质量而非端到端QA。
2. **Verga et al. (2024)** 使用3个小模型面板进行二元judge，本文扩展至排序偏好+STV投票机制，适用于序列生成任务的质量评估。
3. **Pradeep et al. (2023a,b)** 的RankLLM用于零样本段落排序，本文将其思想迁移至分解质量排序并解决无标注数据问题。
4. **Agrawal et al. (2023)** 使用540B模型生成多语言QA数据，本文证明小模型面板在固定计算预算下可取得更优或相当效果。
5. **Wolfson et al. (2020)** 提出break数据集和QDMR形式化，本文在其benchmark上验证合成数据生成方法的有效性。
6. **Bansal et al. (2024)** 发现小模型生成合成数据微调后数学推理表现优于大模型，本文将其发现推广至问题分解任务。

## 局限性与未来方向
- **位置偏差未完全解决**：LLM排序结果受输入顺序影响，虽然随机打乱可部分缓解，但重复采样+顺序打乱求中心排序可能更有效但成本更高。
- ** weaker模型不适合做ranker**：实验表明使用Aya/Mistral/Olmo等较弱模型参与排序会降低整体质量，面板组成需谨慎选择。
- **仅聚焦多跳MRC问题**：单文档复杂问题的分解可能更容易，但本文未系统验证方法在更广泛场景的适用性。
- **合成数据依赖风险**：伦理声明指出过度依赖LLM生成数据可能导致"model collapse"，需要人类数据补充。

## 研究启发与可借鉴点
1. **小模型面板+投票机制可用于序列生成质量评估**：本文的STV投票框架可迁移至其他序列生成任务（如摘要、翻译、代码生成）的合成数据筛选。
2. **通过GPT-4o注入可控错误构造排序示例**：这一自动化构造few-shot exemplars的方法可推广至任何缺乏排序标注的新任务。
3. **跨域OOD评估揭示数据集结构性差异的影响**：break数据集的受限语言形式导致跨域迁移困难，提醒后续工作需关注训练数据的语言多样性。
4. **LoRA+RSLoRA微调小模型获得大模型级性能**：参数高效的微调策略结合合成数据，为资源受限场景下的高质量模型部署提供可行路径。

## 关键术语表
**Question Decomposition**：将复杂多跳问题拆解为有序子问题序列的任务，便于增量求解。
**CLLM (Compact LLM)**：指3-9B参数规模的小型语言模型，本文用于数据生成与排序以降低计算成本。
**LLM-as-Judge / LLM-as-Ranker**：利用LLM对生成结果进行质量评分或排序的技术范式。
**Single Transferable Vote (STV)**：一种偏好投票机制，通过迭代剔除得票最少候选并转移票数来选出最优者。
**Chain-of-Questions (CoQ)**：通过生成子问题序列并逐步作答来求解复杂问题的推理框架。
**SARI**：源自文本简化评估的指标，衡量生成文本相对参考文本的增删改token比例。
**GED (Graph Edit Distance)**：衡量两个分解结构之间最小编辑操作成本的图距离指标。
**Model Collapse**：模型循环训练于自身生成数据导致分布遗忘和性能退化现象。

## 可复现要素
- **数据集**：break、musique、2wikimultihop均为公开数据集；本文仅使用break和musique的主要实验。
- **代码**：已开源，链接为 https://github.com/hankelvin/complex_question_decomposition
- **模型**：使用Llama 3.1 8B、Gemma 2 9B、Phi-3.5 3.8B、Qwen 2.5 7B的公开checkpoint。
- **关键超参**：LoRA rank=32，RSLoRA控制α，训练2个epoch，约8小时/A100；提示采用5-shot（生成）/2-shot（排序）/3-shot（微调）。
- **GPT-4o版本**：gpt-4o-2024-05-13，用于构造排序示例。
