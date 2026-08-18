---
title: "LLMs-Are-Not-Intelligent-Thinkers-Introducing-Mathematical-T"
source: https://aclanthology.org/2025.naacl-long.161.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:27:24"
field: "大语言模型评测与推理能力诊断"
keywords: ["mathematical reasoning", "large language models", "benchmark", "chain-of-thought", "multiple-choice evaluation", "explainability"]
innovations: ["提出覆盖12个数学主题、层次化组织的MaTT基准，填补广泛数学领域评测空白", "通过有/无选项对照与人工解释标注，量化LLM真实推理比例并归纳选项利用策略", "揭示CoT在复杂数学问题上增益有限，挑战通用推理提示有效性假设"]
benchmarks: ["MaTT (Mathematical Topics Tree)"]
---

# 论文速读：LLMs-Are-Not-Intelligent-Thinkers-Introducing-Mathematical-T

## 一句话总结
本文提出了**数学主题树（Mathematical Topics Tree, MaTT）**基准，通过覆盖12个纯数学与 applied 数学主题、共1,958道多层次选择题，系统评估大语言模型（LLM）的数学推理能力；实验表明，即使GPT‑4在多项选择题上也仅达到54%准确率，移除选项后准确率大幅下降最高达24.2个百分点，且手工评估显示GPT‑4仅有53.3%的正确回答附带了完整、正确的解释，揭示当前LLM在复杂数学问题上仍大量依赖记忆、选项工程或循环推理，而非真正推理。

## 研究问题与动机
1. **现有基准覆盖过窄**：当前数学推理评测多聚焦特定题型（如应用题）或单一子领域，无法反映LLM在广泛数学主题上的综合能力。
2. **难以区分记忆与真实推理**：即便模型给出正确答案，也不清楚其是否经过完整推导，还是利用了训练数据中的记忆、选项特征或其他捷径。
3. **选择题格式的影响未被系统探究**：多项选择题可能为模型提供“猜测‑排除”的便利，但其对LLM推理行为的具体影响及在无选项情况下的性能衰减尚不明确。
4. **Chain‑of‑Thought (CoT) 在复杂数学问题上效果存疑**：尽管CoT在简单多步任务上有效，但对于需要创造性或长链条推理的数学问题，其增益有限，需要更细致的评测来验证。

## 核心贡献（创新点）
1. **提出首个覆盖广泛数学主题的层次化基准MaTT**：基于Wikipedia“数学主题列表”及经典教材目录构建包含12个主主题、772个节点、1,958道题目的树状结构，相比MATH、TheoremQA等基准具有更广的主题覆盖和更细的子主题粒度。
2. **系统量化选择题对LLM表现的助推效应**：通过对比有选项与无选项两种设置，揭示GPT‑4、ChatGPT、Mistral等模型在去除选项后准确率分别下降15.9、24.2、16.1个百分点，证明现有高分部分来源于选项利用而非纯粹推理。
3. **引入对模型解释的手动 Completeness 评估**：对GPT‑4正确回答的解释进行人工标注，发现仅53.3%的解释属于完整推理，其余归为选项/弱推理或无/错误推理，为“模型是否真正推理”提供了可解释的度量维度。
4. **归纳LLM在非推理情况下的多种策略**：明确区分并示例了choice engineering（choice use、deduce plausible answer、choice expert、middle ground rule）、theorem use、circular reasoning、blind memorization等模式，为后续诊断LLM推理质量提供分类框架。
5. **验证CoT提示在复杂数学问题上的局限性**： Across multiple topics, zero‑shot CoT (“let’s think step by step”) 未能带来稳定、显著的提升，挑战了CoT在各类推理任务中普遍有效的假设。

## 方法详解
- **主题树构建**：以Wikipedia“Lists of mathematics topics”为起点，提取12个主要数学领域（纯数学6个：Algebra、Calculus and Analysis、Number Theory、Combinatorics、Geometry and Topology、Logic；应用数学6个：Game Theory、Probability、Operations Research、Differential Equations、Statistics、Information Theory and Signal Processing），并为每个领域选取1–2本标准本科/研究生教材，利用教材目录建立层级子主题。
- **题目采集与去污**：从各教材对应章节提取题目，每题配套4个单选题选项（干扰项设计为相近数值、遗漏步骤答案或易混淆组合）；人工审核与独立核查保证质量，约95%的题目无需修订；剔除过于常见、答案紧接题干或易造成数据污染的题目。
- **评估设置**：测试5个模型（GPT‑4、ChatGPT‑turbo、Mistral‑7B‑Instruct‑v0.2、Llama3.1‑70B、OpenAI o1‑mini），采用三种提示形式：①有选项（要求先解释后选A/B/C/D）；②有选项+零样本CoT（末尾加“let’s think step by step”）；③无选项（仅要求解释并给出最终答案）。
- **解释质量标注**：由作者对GPT‑4在有选项时回答正确的样本进行人工分类：Complete（完整逻辑推导）、Choice/Weak（依赖选项或给出部分推理）、No/Wrong（解释缺失或错误）；同时统计在无选项时同样能给出完整解释的比例。
- **统计报告**：提供各主题准确率及95%置信区间（假设高斯分布，CI = 1.96√[p(1‑p)/n]）。

## 实验与结果
- **整体表现**：GPT‑4（有选项）54.0%，ChatGPT 42.9%，Mistral 23.1%，Llama3.1‑70B 53.5%，o1‑mini 79.2%。
- **CoT效果**：除个别主题（如Algebra GPT‑4从71.1%→73.6%）略有波动外，大多数主题CoT未带来稳定增益，部分主题甚至下降（如GPT‑4在Calculus从52.2%→50.9%）。
- **去选项冲击**：GPT‑4准确率从54.0%降至38.1%（‑15.9pp），ChatGPT从42.9%降至18.7%（‑24.2pp），Mistral从23.1%降至7.0%（‑16.1pp）。
- **主题差异**：各模型在不同主题间表现方差最高达41.2个百分点；o1‑mini在Algebra、Calculus、Number Theory上分别达89.7%、88.3%、84.9%，但在Combinatorics（73.1%）、Logic（79.4%）、Game Theory（48.5%）上明显落后。
- **解释完整性**：GPT‑4在所有正确回答中，仅53.3%的解释被判定为Complete；在Algebra、Calculus等主题中Complete比例较高（80.5%、79.6%），而在Geometry & Topology仅20.0%、Combinatorics仅33.3%。
- **结论**：现有LLM在广泛数学主题上的推理能力仍有限，且大量正确预测依赖选项特征或表层策略，CoT无法根本改善复杂数学问题的推理质量。

## 相关工作脉络
1. **MATH (Hendrycks et al., 2021)**：面向竞赛级数学问题，但主题集中于算术、代数、几何、概率等少数领域；MaTT扩展至12个大学级主题并提供细粒度子主题划分。
2. **TheoremQA (Chen et al., 2023)**：侧重定理驱动的问答，缺乏层次化主题结构；MaTT以教材目录为骨架，形成可直接定位知识点薄弱处的树形评测。
3. **LILA (Mishra et al., 2022)**：统一数学推理基准，涵盖23个子任务，但未按数学分支的系统主题树组织；MaTT强调主题覆盖率与子主题可比性。
4. **CoT / ToT / GoT 提示策略研究**：现有工作多在简单多步推理或代码生成上验证CoT有效性；本文指出在需要长链条、创造性推理的数学问题上CoT增益微弱，提示需针对问题复杂度差异化设计。
5. **幻觉与解释评估**：与Hallucination survey (Huang et al., 2023) 呼应，本文通过人工标注区分“正确‑完整解释”“正确‑弱解释”“错误解释”，为LLM可解释性评测提供细粒度分类。
6. **Program‑of‑Thought (PoT)**：附录C测试PoT在MaTT上的表现，发现将数学问题转化为代码本身仍需要同等复杂的推理，PoT未带来显著提升，说明自动化代码辅助并非解决根本推理瓶颈的银弹。

## 局限性与未来方向
- **模型覆盖有限**：仅评测5个商业/开源模型，未涵盖更多架构（如小参数模型、专门数学微调模型），结论的外推性受限。
- **解释评估依赖人工标注**：主观性强、规模有限；需开发自动化、可复现的解释质量度量方法。
- **题目难度校准依赖本科/研究生课程大纲**：不同高校课程标准存在差异，可能在某些子主题上引入难度偏差。
- **未深入分析训练数据污染**：虽已剔除常见题目，但仍可能存在部分教材内容进入训练集的情况。
- **未来方向**：扩展至更多模型与语言；构建动态更新、去污染机制更严的题库；结合程序执行、形式化验证等客观验证手段替代纯人工标注；探索针对性推理强化训练与提示策略。

## 研究启发与可借鉴点
1. **层次化主题树构建范式**：以权威教材目录为骨架、从具体章节抽取题目并人工审核的流程，可迁移至其他学科（如物理、计算机理论）的基准构建。
2. **选项‑无选项对照实验设计**：通过同一题目两种呈现形式量化“选项利用”程度，为诊断模型真实能力提供了一种低成本、高信息量的评测方案。
3. **解释完整性分类框架**：将模型输出分为Complete/Choice‑weak/No‑wrong三类，并统计“有选项‑无选项”双场景下的完整解释重叠率，可作为通用推理质量评估模板。
4. **策略归纳方法**：系统命名choice engineering的子类型（choice use、deduce plausible、choice expert、middle ground rule）及theorem use、circular reasoning、blind memorization，为后续错误分析提供可复用的标签体系。
5. **复杂推理任务上CoT的审慎应用**：本文表明在需要创造性或多步深度推导的任务中，单纯追加“let’s think step by step”收益有限，提示团队在构建复杂推理管线时应结合工具调用、分模块验证等更强机制。

## 关键术语表
- **MaTT (Mathematical Topics Tree)**：本文提出的数学主题树基准，以层级化主题结构组织1,958道数学选择题，覆盖纯数学与应用数学12大领域。
- **Chain‑of‑Thought (CoT) prompting**：通过在提示末尾追加“let’s think step by step”引导模型逐步推理的零样本提示 technique。
- **Choice engineering**：模型利用多选题选项特征（如排除明显错误、选择中间值、匹配常识数值）来得出答案，而非进行完整推导的策略集合。
- **Circular reasoning**：论证前提与结论相互依赖的逻辑谬误；本文观察到GPT‑4在几何拓扑问题中频繁使用该策略。
- **Blind memorization**：模型直接回忆答案而省略推导步骤的行为，常见于训练数据中高频出现的标准问题。
- **Completeness of explanation**：人工标注维度，指模型解释是否包含逻辑严密、步骤完整且与结论一致的推理过程。
- **Confidence interval (CI)**：假设准确率服从高斯分布，以1.96√[p(1‑p)/n]计算的95%置信区间，用于评估各主题准确率的统计可靠性。

## 可复现要素
- **数据集**：MaTT benchmark（1,958道题目，12主题，树状节点772），论文声明将开源。
- **代码/权重**：论文声明“will release code and data for MaTT”（具体发布链接待出版后确认）；所用模型为现有商业/开源模型（GPT‑4、ChatGPT‑turbo、Mistral‑7B‑Instruct‑v0.2、Llama3.1‑70B、o1‑mini），代码调用方式见附录A。
- **关键超参**：提示模板固定（见Appendix A）；零样本CoT仅追加“let’s think step by step”；无Temperature/Top‑p声明，通常默认值；评估指标为选择题准确率及人工解释分类比例。
- **外部依赖**：Wikipedia主题列表、指定教材目录；人工标注过程需至少两位标注者独立审核（文中提及95%一致性）。
