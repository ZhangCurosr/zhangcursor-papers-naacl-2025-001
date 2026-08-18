---
title: "On-Behalf-of-the-Stakeholders-Trends-in-NLP-Model-Interpreta"
source: https://aclanthology.org/2025.naacl-long.29.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:31:45"
field: "NLP模型可解释性"
keywords: ["NLP可解释性", "LLM", "利益相关者", "机械可解释性", "自然语言解释", "因果推断", "文献计量"]
innovations: ["从利益相关者视角系统分类NLP可解释性方法并提出四视角框架", "首次使用LLM大规模标注2000+篇论文进行跨领域可解释性趋势分析", "提出六维属性分类体系并揭示NLP开发者与非开发者需求差异"]
benchmarks: ["Semantic Scholar NLP Interpretability Corpus (2015-2024)"]
---

# 论文速读：On-Behalf-of-the-Stakeholders-Trends-in-NLP-Model-Interpreta

## 一句话总结
本文从利益相关者（stakeholders）视角系统回顾了LLM时代NLP模型可解释性的研究范式与趋势，通过LLM自动标注分析了过去十年超过2000篇论文，揭示了NLP开发者与非开发者用户在可解释性需求上的显著差异，并指出LLM正推动自然语言解释等范式的快速兴起。

## 研究问题与动机
1. **现有综述忽视利益相关者需求**：尽管NLP可解释性研究爆发式增长并涌现大量技术综述，但这些综述多聚焦方法本身，忽视了不同解释受众（利益相关者）的实际需求与视角差异。
2. **术语与定义缺乏共识**：可解释性（interpretability）与可解释性（explainability）在NLP与XAI文献中存在混用，缺乏统一框架来系统描述方法的性质与适用场景。
3. **跨领域需求鸿沟**：NLP开发者高度关注模型内部机制（如神经元、注意力头），而外部领域（医疗、神经科学、社会科学等）用户更关注输入-输出层面的解释，两者之间存在明显脱节。
4. **LLM带来的范式转变尚未被系统刻画**：LLM的普及是否以及如何改变了可解释性研究趋势，尚缺乏量化分析。

## 核心贡献（创新点）
1. **提出利益相关者驱动的可解释性分析框架**：将可解释性需求分解为算法、商业、科学、社会四个视角，并映射到不同利益相关者（开发者、医生、科学家、监管者等），与现有综述的方法中心视角形成本质区别。
2. **系统定义可解释性方法与解释的边界**：提出"提取机制洞察"作为可解释性方法的广义定义，并区分"解释"（communication to stakeholders）与"归因"（extraction）的概念差异。
3. **提出六维属性分类体系**：归纳出解释机制（mechanism）、范围（scope）、时间（time）、访问方式（access）、呈现方式（presentation）、因果性（causal-based）六个属性，为方法论分类提供统一语言。
4. **首次大规模LLM辅助的趋势分析**：利用LLM对2000+篇论文进行范式、属性、领域等多维度标注，验证了LLM标注与人工标注超过90%的一致性，为文献计量学研究提供了可复现范式。
5. **揭示跨领域差异并给出实践建议**：发现非开发者极少使用内部机制解释方法（如机械可解释性），而偏好LIME/SHAP等易用工具；建议NLP研究者加强用户友好代码、概念级解释和因果方法的研究。

## 方法详解
**1. 四视角利益相关者框架（§2）**
- **算法视角**：利益相关者为开发者，目标是通过可解释性进行调试、改进模型泛化能力、定位知识神经元等。
- **商业视角**：利益相关者为企业决策者与终端用户，目标包括法律合规（如GDPR"解释权"）、提升用户信任、支持决策。
- **科学视角**：利益相关者为跨学科科学家（社会科学、心理学、神经科学等），目标是通过NLP模型探索人类行为、认知模式等科学现象。
- **社会视角**：利益相关者为更广泛的社会公众，目标包括公平性验证、偏见检测、AI安全。

**2. 概念定义（§3）**
- **可解释性方法**：提取NLP系统机制洞察的任何方法（涵盖模型分析）。
- **解释**：提取机制洞察并以利益相关者可理解的方式传达给他们的过程。公式化表达为：解释 = 提取（interpretability method）+ 传达（communication）。

**3. 六维属性体系（§4 & Table 1）**
- **[what] 解释机制（Explained mechanism）**：
  - input-output：解释整个输入-输出系统
  - input-internal：解释输入表示（如probing）
  - internal-internal：解释内部组件（神经元、注意力头、电路）
  - concept-output：解释抽象概念对输出的影响
- **[what] 范围（Scope）**：local（单个样本）vs global（整个数据分布）
- **[how] 时间（Time）**：post-hoc（预测后）vs intrinsic（预测时内建）
- **[how] 访问（Access）**：model-agnostic（仅输入输出）vs model-specific（需访问内部）
- **[how] 呈现（Presentation）**：scores、visualization、examples、natural language text
- **[how] 因果性（Causal-based）**：是否基于因果推理提供忠实解释

**4. 七类可解释性范式（Appendix §B）**
1. Feature Attributions（特征归因）：扰动、梯度、传播、代理模型（LIME/SHAP）、注意力
2. Probing & Clustering（探测与聚类）：训练分类器预测属性、无监督聚类
3. Mechanistic Interpretability（机械可解释性）：刺激响应、稀疏自编码器、Patching、Logits Lens
4. Diagnostic Sets（诊断集）：Challenge sets、Test suites
5. Counterfactuals & Adversarial Attacks（反事实与对抗攻击）：对比示例、概念反事实
6. Natural Language Explanations（自然语言解释）：抽取式、生成式、Chain-of-Thought
7. Self-explaining Models（自解释模型）：经典ML、概念瓶颈模型、KNN-based

**5. 趋势分析方法论（§5.1）**
- 通过Semantic Scholar API检索14,676篇论文
- 使用LLM（gemini-1.5-pro-preview-0514）进行相关性过滤与多维度标注
- 五阶段流程：检索→LLM相关性判断→LLM属性标注→人工抽样验证（100篇）→统计分析
- 验证结果：LLM标注与人工标注在排除"unknown"后一致率超过95%

## 实验与结果
**数据集与规模**
- 检索来源：Semantic Scholar API
- 初始检索量：14,676篇
- 最终分析样本：2,009篇（2015-2024年）
- 领域分布：NLP领域1,495篇，其他领域514篇

**主要发现与数字**

| 发现 | 关键数据 |
|------|---------|
| NLP社区内部趋势稳定 | Feature Attributions从2017年~45%降至2024年~30%；Natural Language Explanations从~10%升至~25% |
| 非开发者更少关注内部机制 | NLP领域internal-internal机制论文是外部的5倍 |
| 外部领域偏好易用方法 | LIME/SHAP在非NLP领域占9.6%，而在NLP领域仅4.3%；Clustering占比9.7%（NLP仅2.3%） |
| LLM彻底改变趋势 | 2024年NLP领域66.7%论文使用LLM；外部领域从2023年18.2%跃升至2024年50.7% |
| 自然语言解释爆发 | 使用LLM的外部领域论文中，48.7%采用Natural Language Explanations（非LLM论文仅6.2%） |

**跨领域差异（Table 2）**
- Healthcare：local解释占66.5%（医生/患者关注个体决策）
- Neuroscience：global解释占75.0%（科学家关注宏观认知机制）
- 因果方法使用率低：NLP领域5.2%，外部领域1.9%

**引用影响力（Table 5）**
- Adversarial Attacks平均引用最高（NLP领域53.1次）
- NLP领域各范式平均引用7.8-32.3次不等

## 相关工作脉络
1. **Belinkov & Glass (2019)**：NLP模型分析综述，本文与其区别在于首次从利益相关者视角系统分类，并扩展到LLM时代与跨领域趋势分析。
2. **Lyu et al. (2022)**：忠实性（faithfulness）专项综述，本文将其纳入因果性属性讨论，但更侧重方法论分类与趋势而非单一质量维度。
3. **Räuker et al. (2023) / Bereska & Gavves (2024)**：机械可解释性综述，本文指出该范式在NLP外几乎不被采用，强调跨领域适配需求。
4. **Ribeiro et al. (2016, 2018) - LIME/SHAP**：特征归因的代理模型方法，本文分析其为何在非NLP领域更流行（易用性），而非仅作为技术方法讨论。
5. **Feder et al. (2022)**：NLP因果推断综述，本文将其思想纳入"因果性"属性，并量化了因果方法在实际研究中的低采用率。
6. **Rudin (2018) / Arrieta et al. (2020)**：XAI基础理论，本文吸收其post-hoc vs intrinsic的区分，但重新定义为更宽泛的"时间"属性并扩展至NLP语境。

## 局限性与未来方向
**论文自述局限（§7）**
1. **模态局限**：仅聚焦文本/NLP，未覆盖视觉、音频等多模态系统，结论可能不适用于多模态大模型的可解释性。
2. **LLM标注偏差**：虽经人工验证一致率高，但仍可能存在系统性偏差；2000+篇论文全部人工标注成本过高，未来需持续监督。

**合理推断的未来方向**
1. **概念级解释的深化**：当前仅2%论文采用concept-level方法，需发展更适合非专家用户的高层概念解释。
2. **因果方法的推广**：因果方法忠实性更高但使用率仅~5%，需降低技术门槛。
3. **自解释模型的性能-可解释性权衡**：仅7%论文关注self-explaining models，需探索如何在保持性能的同时实现内在可解释性。
4. **机械可解释性的跨领域适配**：NLP内部的热门方法（mechanistic interpretability）在外领域几乎无用武之地，需开发更适合外部用户的简化版本。
5. **自然语言解释的忠实性验证**：LLM时代自然语言解释激增，但其忠实性存疑，需建立评估标准。

## 研究启发与可借鉴点
1. **LLM辅助文献分析的可靠范式**：本文首次成功验证了LLM可用于大规模学术论文的属性标注（90%+一致性），为其他领域的文献计量研究提供了可复现的方法论模板。
2. **利益相关者视角的方法设计原则**：在开发新可解释性方法时，应明确目标受众（开发者/医生/科学家/监管者），并根据其需求匹配相应的属性组合（如医生需要local+input-output+易懂呈现）。
3. **易用性驱动技术扩散的洞察**：LIME/SHAP在非NLP领域的流行主要归功于scikit-learn等生态系统的易用封装，提示未来NLP可解释性方法的推广需配套友好API与可视化。
4. **跨领域差异的定量刻画方法**：通过LLM自动标注实现14,000+论文的领域、范式、属性三维分析，为"技术在不同学科间的 Adoption gap"研究提供了量化分析路径。
5. **LLM时代可解释性研究的新机会**：LLM的生成本身可作为解释手段（如CoT、自然语言解释），这一趋势在外部领域尤为明显（48.7%），为设计"生成式解释"方法提供了明确方向。

## 关键术语表
**Stakeholders（利益相关者）**：可解释性方法的最终受益者或受众，包括开发者、医生、科学家、监管者、终端用户等，不同角色对解释的需求存在显著差异。

**Interpretability Method（可解释性方法）**：提取NLP系统机制洞察的任何技术手段，本文广义定义涵盖模型分析（model analysis），不局限于生成人类可理解解释的方法。

**Explanation（解释）**：将可解释性方法提取的洞察筛选、处理后，以利益相关者可理解的方式呈现的过程，区别于单纯的"归因"（attribution）。

**Mechanism（机制）**：NLP系统中两个状态之间的关系过程，如input-output（整个模型）、input-internal（表征学习）、internal-internal（神经元功能）、concept-output（概念影响）。

**Scope（范围）**：解释的覆盖范围，local指单个样本级别的解释，global指整个数据分布或模型层面的解释。

**Causal-based（因果性）**：解释方法是否基于因果推断理论（如反事实、干预、调整），旨在提供忠实的归因而非仅相关性洞察。

**Natural Language Explanation（NLE）**：由NLP系统生成或提取的自然语言解释，包括Chain-of-Thought、extractive rationale、abstractive explanation等，在LLM时代迅速普及。

**Mechanistic Interpretability（机械可解释性）**：自底向上研究神经网络内部组件（神经元、注意力头、电路）功能的方法，主要在NLP社区内部流行，外部领域极少使用。

## 可复现要素
- **数据集**：Semantic Scholar API检索，查询词列表见Appendix Box D.1；最终2,009篇论文列表及标注结果论文未明确公开，但检索方法与LLM标注prompt（Box D.4）完整提供。
- **代码/权重**：论文未提及代码开源。
- **LLM标注工具**：使用gemini-1.5-pro-preview-0514，zero-shot prompt见Box D.4；采用三遍生成+多数投票策略。
- **关键超参**：未明确提及；标注策略为每个论文生成3个JSON响应取多数 vote。
- **人工验证**：随机抽样100篇论文进行人工标注，一致率统计见Table 4。
