---
title: "PAPILLON-PrivAcy-Preservation-from-Internet-based-and-Local"
source: https://aclanthology.org/2025.naacl-long.173.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:32:02"
field: "LLM隐私与安全性"
keywords: ["推理时隐私保护", "PII泄露", "本地-云模型协作", "Prompt优化", "隐私基准数据集", "Large Language Models"]
innovations: ["提出Privacy-Conscious Delegation任务，以本地弱模型代理用户隐私访问云端强模型", "构建PUPA真实场景PII泄露基准数据集，含901条标注样本", "设计PAPILLON双阶段管道并通过DSPy MIPRO v2自动化优化，实现85.5%质量保持率与7.5%低泄露率"]
benchmarks: ["PUPA", "WildChat"]
---

# 论文速读：PAPILLON-PrivAcy-Preservation-from-Internet-based-and-Local

## 一句话总结
本文提出了**Privacy-Conscious Delegation**新任务，设计了一种名为**PAPILLON**的多阶段LLM管道，让本地弱模型作为用户的隐私代理，在不泄露个人身份信息（PII）的前提下调用云端强模型完成生成任务。通过构建**PUPA**基准数据集，验证了最优配置（Llama-3.1-8B-Instruct + GPT-4o-mini）可将响应质量保持在基线的85.5%，同时仅泄露7.5%的私有信息。

## 研究问题与动机
1. **推理时隐私泄露风险**：用户在使用API类大模型（如ChatGPT）时可能无意泄露PII，现有训练去噪无法解决推理阶段的隐私泄露问题。
2. **简单删改损害效用**：直接对查询做文本匿名化/删改会降低LLM生成质量，因为模型失去了利用被删实体的上下文知识。
3. **本地模型能力不足**：可本地部署的开源模型通常弱于顶级闭源API模型，用户面临"质量"与"隐私"的两难选择。
4. **缺乏系统化评测基准**：现有隐私评测多聚焦于训练数据记忆化或差分隐私，缺乏针对用户-LLM交互中PII泄露的真实场景评测。

## 核心贡献（创新点）
1. **提出Privacy-Conscious Delegation新任务**：与现有差分隐私或防御性攻击研究不同，本文关注在查询发起阶段就阻止PII被外部模型获取，而非事后防护。
2. **构建PUPA基准数据集**：从WildChat语料中筛选并重新标注含PII的真实用户交互数据，包含901条有效样本，填补了真实场景推理隐私评测的空白。
3. **设计PAPILLON多阶段管道**：通过本地模型作为"Prompt Creator"（生成隐私安全提示）和"Information Aggregator"（聚合云端输出），实现两阶段协作生成。
4. **自动化Prompt优化框架**：利用DSPy的MIPRO v2优化器联合优化两个模块的提示词，结合质量保持、泄露控制和提示规范性三项指标进行端到端搜索。

## 方法详解

### 任务设定
- 给定两个模型：可信但较弱的**本地模型** $M_{\mathrm{LOCAL}}$ 和不可信但较强的**远程模型** $M_{\mathrm{REMOTE}}$
- 输入：含PII的用户查询 $q$，含隐私信息 $p_1, p_2, \cdots, p_n$
- 目标：生成高质量响应 $r$，同时最小化向 $M_{\mathrm{REMOTE}}$ 泄露的PII比例

### PAPILLON管道结构
管道由两个核心组件构成，均在本地模型上运行：

1. **Prompt Creator**：接收用户原始查询 $q$，生成经隐私脱敏的中间提示 $q'$，发送给 $M_{\mathrm{REMOTE}}$ 获取中间响应 $C_R$
2. **Information Aggregator**：接收原始查询 $q$ 和中间响应 $C_R$，聚合生成最终输出 $C_L$

### Prompt优化目标函数
使用DSPy MIPRO v2进行200轮优化，最大化：

$$\frac{\mathrm{QUAL}(C_L, C_T) - \mathrm{LEAK}(q', p_{1..n}) + \mathrm{PWF}(q')}{2}$$

其中：
- $\mathrm{QUAL}$：LLM judge评估最终响应与目标响应的质量相似度
- $\mathrm{LEAK}$：计算中间提示 $q'$ 中包含的PII比例，公式为所有PII单位中存在比例的平均值
- $\mathrm{PWF}$：提示规范性指标，确保生成的 $q'$ 是结构良好的语言模型提示

### 评测指标
- **Quality Preservation**：采用LLM-as-judge，通过正反序对比避免位置偏差，输出0/1二分评分
- **Privacy Preservation (LEAK)**：迭代检查每个PII单位是否出现在中间提示中，计算泄露百分比

## 实验与结果

### 数据集
- **PUPA**：从WildChat提取的901条真实用户交互数据
  - PUPA-TNB：237条（用于评测，不参与优化）
  - PUPA-New：664条（用于Prompt优化）
  - 三大类别：Job/Visa/Application（16-40%）、Financial/Corporate（28-47%）、Quoted Emails/Messages（22-30%）

### 模型配置
- 远程模型：GPT-4o-mini（主要实验）、GPT-4-turbo、GPT-4o、GPT-3.5-turbo
- 本地模型：Llama-3.1-8B-Instruct、Llama-3-8B-Instruct、Llama-3.2-3B/1B-Instruct、Mistral-7B-Instruct、Mistral-Small
- 部署环境：单卡A100 80GB，使用SGLang/vLLM

### 核心结果
| 配置 | 优化前 QUAL | 优化前 LEAK | 优化后 QUAL | 优化后 LEAK | ΔQUAL | ΔLEAK |
|------|------------|-------------|------------|-------------|-------|-------|
| GPT-4o-mini [无删改] | 88.2 | 0.00 | - | - | - | - |
| GPT-4o-mini [删改] | 77.2 | 27.8 | - | - | - | - |
| Llama-3.1-8B-Instruct | 75.7 | 36.9 | **77.6** | **7.50** | +1.90 | **-25.0** |
| Llama-3-8B-Instruct | 71.8 | 17.4 | **85.5** | 21.2 | +7.50 | -9.70 |
| Llama-3.2-1B-Instruct | 41.2 | 23.8 | 58.0 | 39.3 | +16.8 | +11.5 |

**最强结果**：Llama-3-8B-Instruct + GPT-4o-mini 组合，优化后质量保持率85.5%，泄露率21.2%；若以Llama-3.1-8B-Instruct为本地模型，则在7.5%低泄露下达到77.6%质量保持率。

### 关键发现
1. 简单文本删改导致质量大幅下降（88.2→77.2），验证了方法的必要性
2. Prompt优化对所有本地模型均有正向效果，且无需额外训练
3. Mistral-Small虽质量高但泄露率偏高，反映不同模型的隐私规范意识差异
4. 成本分析：PAPILLON相较直接API调用仅增加约0.013美元（\$0.076 vs \$0.063），但减少了平均193.9个输入token

## 相关工作脉络

1. **训练时隐私研究**（Carlini et al., 2021; Kim et al., 2024）：聚焦训练数据记忆化与成员推断攻击，与本文关注的推理时隐私保护定位不同。
2. **差分隐私LLM**（Zhang et al., 2024; Hong et al., 2023）：通过噪声注入或差分隐私保证防护，属于事后防御思路，而非本文的"事前阻断PII访问"范式。
3. **文本匿名化方法**（Feyisetan et al., 2020; Staab et al., 2023）：依赖词嵌入等假设，不适用于黑盒API场景，且会严重损害生成质量。
4. **隐私评测基准**（PrivacyLens: Shao et al., 2024; WildChat: Zhao et al., 2024）：前者使用合成对话，后者包含真实但缺乏结构化PII标注；PUPA填补了真实场景+结构化PII的空白。
5. **PII披露研究**（Mireshghallah et al., 2024）：揭示了WildChat中意外的PII隐式泄露模式，本文在其基础上构建可量化的评测任务与防御方法。
6. **Prompt优化框架**（DSPy: Opsahl-Ong et al., 2024）：本文将其引入隐私保护场景，实现多阶段管道的联合优化，是方法层面的迁移创新。

## 局限性与未来方向

1. **仅处理显式PII**：当前方法依赖可检测的PII单位（姓名、公司、地址等），但性行为偏好、医疗条件等隐式隐私难以提取和防护。
2. **评测指标依赖LLM Judge**：质量和泄露评估均使用LLM作为judge，可能存在偏差；需更多人工标注验证。
3. **Prompt优化方案的内在缺陷**：Prompt Creator有时会过度泛化或添加与原始意图不一致的冗余信息，导致最终输出偏离用户请求。
4. **GPT-4o-mini标注质量参差**：用于数据收集和PII提取的模型存在过删改（over-redaction）问题，需人工校验。
5. **未来方向**：开发专门的隐私感知小型本地模型、探索差分隐私的理论保证、扩展至多轮交互场景。

## 研究启发与可借鉴点

1. **两阶段协作范式可迁移**：本地模型负责隐私敏感任务、云端模型负责复杂推理的"委托-聚合"架构，可推广至其他隐私敏感NLP任务（如医疗问答、法律咨询）。
2. **Prompt优化结合隐私约束**：将隐私泄露指标纳入DSPy优化目标，为其他约束条件下的prompt engineering提供了可复用的方法论。
3. **PII提取流程设计**：先删改再提取的三步流程（LLM删改→正则提取→去重）兼顾长上下文场景的召回率和准确率，可借鉴于其他PII研究。
4. **质量评估的位置偏差缓解**：正反序双调用的LLM-as-judge设计，可作为通用质量评测的参考方案。
5. **成本-隐私权衡分析**：文中对token消耗和API成本的分析框架，为实际部署中的性价比评估提供了量化参考。

## 关键术语表

**Privacy-Conscious Delegation**：一种新的推理时隐私保护任务范式，让可信的本地弱模型作为用户的隐私代理，通过生成脱敏提示调用不可信的云端强模型完成任务。

**PUPA (Private User Prompt Annotations)**：本文构建的基准数据集，包含901条来自WildChat的真实用户-LLM交互，标注有显式PII信息和对应的隐私泄露情况。

**PAPILLON**：Privacy Preservation from Internet-based and Local Language Model Ensembles的缩写，本文提出的两阶段LLM管道系统，包含Prompt Creator和Information Aggregator两个模块。

**LLM-as-judge**：使用大语言模型作为自动评估器，替代人工标注进行质量评判和隐私泄露检测的评测方法。

**MIPRO v2**：DSPy框架中的Prompt优化器，通过迭代采样训练数据、生成候选提示并基于多目标指标评估选择最优提示的自动化优化方法。

**PII (Personally Identifiable Information)**：个人身份信息，包括姓名、地址、公司名、社会安全号等可唯一标识个体的敏感信息。

## 可复现要素
- **数据集**：PUPA基于WildChat构建，论文声明将在发布时替换真实姓名为随机名称，并过滤不当内容；WildChat本身公开可用
- **代码**：论文未明确提供开源代码仓库链接
- **关键超参**：MIPRO v2优化200轮；训练集/验证集各150条样本；使用Llama-3.1-8B-Instruct/Llama-3-8B-Instruct + GPT-4o-mini组合
- **部署环境**：单卡A100 80GB GPU，使用SGLang或vLLM推理框架
