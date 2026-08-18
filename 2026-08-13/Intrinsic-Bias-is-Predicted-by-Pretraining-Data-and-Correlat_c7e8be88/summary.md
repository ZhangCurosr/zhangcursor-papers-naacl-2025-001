---
title: "Intrinsic-Bias-is-Predicted-by-Pretraining-Data-and-Correlat"
source: https://aclanthology.org/2025.naacl-long.148.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:05:04"
field: "多模态AI公平性与可解释性"
keywords: ["视觉-语言模型", "内在偏见", "Embedding Association Test", "预训练数据", "CLIP", "公平性", "零样本性能"]
innovations: ["提出改进的EAT评估框架，使用经人类验证的刺激材料降低4.8%方差", "首次系统性证明预训练数据集选择是偏见的首要预测因子（β=0.608）而架构影响不显著", "揭示性能优化与偏见放大存在正相关（r最高达0.82），构建偏见-性能联合评估范式"]
benchmarks: ["VTAB+", "ImageNet", "OpenCLIP"]
---

# 论文速读：Intrinsic-Bias-is-Predicted-by-Pretraining-Data-and-Correlat

## 一句话总结
本文对131个CLIP模型进行了迄今最大规模的内在偏见系统性分析，发现**预训练数据集选择**是影响模型内在偏见的决定性上游因素（模型架构和参数规模影响不显著）；同时揭示了**性能优化与偏见放大之间存在正相关**——使用高过滤策略构建的数据集（如dfn、DataComp）虽能提升下游零样本性能，却会加剧社会偏见。

## 研究问题与动机
1. **缺乏系统性分析**：现有CLIP偏见研究多聚焦单一模型或单一偏见维度（如仅研究性别偏见），难以回答"哪些上游预训练因素真正驱动偏见"这一核心问题。
2. **偏见测量信度不足**：传统SEAT/iEAT测试的效应量方差较高（0.62），且未针对跨模态组合进行系统扩展，限制了跨研究可比性。
3. **性能-偏见关联不明**：虽有个别研究发现偏见与下游性能相关，但尚未在大规模模型上建立定量关联，不清楚"优化性能是否以放大偏成为代价"。
4. **过滤策略的公平性盲区**：当前数据过滤技术（如DFN、DataComp）旨在提升ImageNet准确率，但缺乏对公平性与代表性偏差的系统评估。

## 核心贡献（创新点）
1. **提出改进的EAT评估框架**：使用经人类验证的OASIS图像和NRC-VAD词典替换传统刺激材料，将效应量方差平均降低4.8%，并在3,406个案例中实现78.86%与人类IAT一致性——相比Berg et al.的单一模型研究，提供了更高统计功效。
2. **首次系统性量化上游因素对偏见的影响**：通过混合效应回归模型，证明**数据集选择**是唯一显著预测偏见的因素（dfn数据集β=0.608, p<0.01），而架构和参数规模均不显著——这与Ladhak et al.在NLP中观察到的架构偏见效应形成对比，凸显了VLM偏见的独特数据驱动本质。
3. **揭示"性能优化悖论"**：发现高过滤策略数据集（自动化神经网络驱动的dfn、启发式规则的DataComp）虽在ImageNet等基准上表现优异，却显著放大偏见（β=0.360~0.608）；而CC12m采用超名词化策略（将人名替换为[PERSON]）后偏见最低——这表明当前SOTA数据工程实践忽视了公平性考量。
4. **建立偏见-性能定量关联**：首次在大范围模型上证明内在偏见与下游零样本性能显著相关（0.3≤r≤0.8），且相关性方向因测试类别而异（非人类概念正向相关、性别偏见负向相关）——扩展了Berg et al.的初步发现至更全面的偏见谱系。

## 方法详解
**1. 模型与因子范围**：131个CLIP模型（含OpenAI原始9个、Cherti et al. 29个、OpenCLIP 93个），涵盖26个预训练数据集、55种架构、参数量102M~5B。

**2. 偏见测量——改进的Embedding Association Test (EAT)**：
- 采用四类模态组合：All Image、All Text、Image as Target、Text as Target
- 五类测试类别：Flower-Insect/Valence、Instrument-Weapon/Valence、Race/Valence、Gender/Valence、Age/Valence
- **刺激材料改进**：使用OASIS数据集（Kurdi et al., 2017）提供经822人效价评分的图像，NRC-VAD词典（Mohammad, 2018）提供效价词汇，各取top 25积极/消极刺激，共26个EAT测试
- **效应量公式**：
  $$d = \frac{mean_{x \in X}s(x, A, B) - mean_{y \in Y}s(y, A, B)}{std\_dev_{w \in X \cup Y}s(w, A, B)}$$
  其中$s(w, A, B) = mean_{a \in A}cos(w, a) - mean_{b \in B}cos(w, b)$，正值表示与刻板印象一致的偏见

**3. 上游因素分析——混合效应回归模型**：
$$d_{ij} = \beta_0 + \beta_1\log(param)_{ij} + \beta_2 arch_{ij} + \beta_3 dataset_{ij} + \beta_4\log(dataset\_size)_{ij} + u_{0j} + u_{1j}\log(param)_{ij} + u_{2j}\log(dataset\_size)_{ij} + \epsilon_{ij}$$
- 固定效应：log_params、architecture_family、dataset_family、log_dataset_size
- 随机效应：按模态-测试组合分组的截距和斜率
- 估计方法：REML，Wald t检验

**4. 性能-偏见关联**：计算Pearson相关系数，连接EAT效应量与VTAB+基准（35个零样本分类/检索任务）上的性能表现。

## 实验与结果
**数据集与评估**：131模型×26测试=3,406个数据点；下游性能使用VTAB+（含ImageNet、Caltech-101、Diabetic Retinopathy、SmallNORB等35项任务）。

**关键结果**：
| 发现 | 数值 | 统计显著性 |
|------|------|-----------|
| 新刺激材料降低效应量方差 | 0.62→0.59（↓4.8%） | All Text模态方差降低33.96% |
| 与人类IAT一致的比例 | 78.86%（3,406案例中d>0） | 相比旧刺激（67.88%）显著提升 |
| dfn数据集偏见效应 | β=0.608 (95%CI: 0.380~0.835) | p<0.001 |
| DataComp数据集偏见效应 | β=0.360 (95%CI: 0.149~0.571) | p=0.001 |
| CC12m作为基线 | 参照组（偏见最低） | — |
| 架构影响 | 所有架构家族均不显著 | p>0.05 |
| 参数规模影响 | β=0.010, p=0.589 | 不显著 |
| 非人类偏见-性能相关 | Flower-Insect: r=0.56; Instrument-Weapon: r=0.78 | p<0.01 |
| 性别偏见-性能相关 | Image as Target: r=-0.51; All Text: r=-0.27 | p<0.05 |
| 最强性能-偏见关联 | Instrument-Weapon/Valence (Text as Target): r=0.82 | p<0.001 |

**结论**：①数据集选择是偏见的主要来源，过滤策略越"智能"（如dfn的神经网络过滤）偏见越大；②性能优化与偏见放大呈系统性正相关，暗示优化目标与公平性目标存在内在张力。

## 相关工作脉络
1. **Caliskan et al. (2017) Science**：开创性工作，证明静态词嵌入编码了人类隐性偏见——本文将其扩展至视觉-语言多模态嵌入，并引入控制更严格的效价刺激。

2. **May et al. (2019) NAACL - SEAT**：提出句子编码器偏见测试——本文复现其框架，但引入NRC-VAD词典和语义空白模板以提升刺激材料的可控性。

3. **Steed & Caliskan (2021) FAccT - iEAT**：首个图像嵌入偏见测试——本文扩展至跨模态组合（Image-Text/Text-Image），并新增Gender/Valence和Instrument/Weapon类别。

4. **Berg et al. (2022) NAACL-HLT**：首个将CLIP偏见与下游性能关联的研究（9个模型，仅性别偏见）——本文扩展至131模型、26个测试类别，揭示更广泛的性能-偏见关联模式。

5. **Gadre et al. (2024) NeurIPS - DataComp**：提出多模态数据集构建基准，使用启发式过滤提升零样本性能——本文证明此类"质量优先"策略会牺牲公平性（β=0.360）。

6. **Fang et al. (2023) arXiv - DFN**：使用神经网络过滤构建高质量数据集——本文发现此类自动化过滤反而加剧偏见（β=0.608，最高），揭示算法决策的公平性风险。

## 局限性与未来方向
1. **单语言限制**：仅分析英语模型，无法捕捉多语言/多元文化数据源（如WebLI）引入的文化特异性偏见。
2. **偏见谱系有限**：仅覆盖race、gender、age及non-human baseline，未探索socioeconomic status、intersectional identities等更广泛的偏见维度。
3. **数据集组成黑箱**：未深入分析数据集内部构成（如群体代表比例、图像-文本对齐质量）与偏见的因果机制。
4. **刺激材料频率控制不足**：未系统控制目标词/图像的出现频率（Wilson & Caliskan, 2024），可能影响效应量的解释。
5. **未来方向**：扩展至多语言设置；探索公平性约束下的预训练策略；研究性能-公平性权衡的量化边界。

## 研究启发与可借鉴点
1. **EAT刺激材料的精细化设计**：使用经人类效价评分验证的刺激（OASIS图像、NRC-VAD词汇）而非自由选取的词汇/图像，可将方差降低4.8%、提升与人类刻板印象的一致性11个百分点——此方法可直接迁移至其他模态（如音频、多模态生成模型）的偏见评估。

2. **混合效应回归建模框架**：采用"group-level随机截距+斜率"结构控制模态-测试组合的异质性，有效分离固定效应（数据集、架构）与随机变异——此框架适用于任何多因素分析的偏见研究，可推广至LLM偏见溯源。

3. **上游因素的重要性排序方法**：通过偏回归系数比较（而非简单显著性检验）量化各因素的相对贡献——发现"数据集>架构>规模"的层级，为资源分配提供依据：应优先投入数据策展而非架构搜索。

4. **性能-偏见联合评估范式**：在同一实验中并行测量偏见效应量与下游性能，揭示"性能优化≠公平性提升"的悖论——此双轨评估可成为VLM基准测试的标准组件，避免单一指标优化。

5. **超名词化（Hyponymization）作为公平性干预**：CC12m通过将人名替换为[PERSON]显著降低偏见，启发后续工作探索类似"属性泛化"策略在其他数据集上的可迁移性。

## 关键术语表
**Embedding Association Test (EAT)**：基于嵌入空间余弦相似度的偏见测量方法，通过比较目标群体与属性集合的相对距离量化隐性偏见，源自心理学中的Implicit Association Test (IAT)。

**Valence（效价）**：描述刺激情感倾向的维度（积极/消极），是人类认知中塑造态度和偏见的核心维度，本文作为属性类别的核心维度。

**Mixed Effects Regression（混合效应回归）**：同时包含固定效应（群体层面预测变量）和随机效应（组间变异）的统计模型，用于控制模态-测试组合的异质性。

**VTAB+**：包含35个图像分类与检索任务的零样本性能基准，覆盖自然图像（ImageNet、Caltech-101）、医学图像（Diabetic Retinopathy）和结构图像（SmallNORB）。

**DFN (Data Filtering Networks)**：使用神经网络自动过滤大规模图像-文本对的策略，在下游性能上优于启发式方法，但本文发现其显著放大社会偏见。

**OASIS (Open Affective Standardized Image Set)**：包含900张经822人效价和唤醒度评分的自然图像数据集，本文用于提供可控的视觉效价刺激。

**NRC-VAD Lexicon**：包含20,000英文单词的效价(V)、唤醒度(A)、优势度(D)人工评分词典，本文选用top 25积极/消极词汇作为文本刺激。

**Hyponymization（超名词化）**：将具体人名替换为泛化类别词（如[PERSON]）的数据处理策略，CC12m采用此方法以降低社会群体偏见。

## 可复现要素
- **数据集**：131个CLIP模型权重来自OpenCLIP（MIT许可）；OASIS图像（Kurdi et al., 2017公开）、NRC-VAD词典（Mohammad, 2018公开）、VTAB+基准（Schuhmann et al., 2022公开）
- **代码**：论文声明已开源，URL: https://github.com/kshitishghate/CLIP_bias
- **关键超参**：刺激材料选取top 25积极/消极；模板数量6个；混合效应模型使用REML估计、lbfgs优化器；log变换参数和数据集大小
- **依赖**：OpenCLIP (open-clip-torch 2.16.0)、timm (0.6.12)
