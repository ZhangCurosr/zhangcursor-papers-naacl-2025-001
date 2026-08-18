---
title: "Beyond-End-to-End-VLMs-Leveraging-Intermediate-Text-Represen"
source: https://aclanthology.org/2025.naacl-long.180.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:59:34"
field: "多模态理解与图表推理"
keywords: ["流程图理解", "视觉语言模型", "双阶段框架", "文本化表示", "FlowVQA", "多模态推理"]
innovations: ["提出TEXTFLOW双阶段框架，将流程图理解解耦为视觉文本化与文本推理两阶段", "首次系统引入GRAPHVIZ/MERMAID/PLANTUML三种文本格式用于流程图表征并对比有效性", "支持将文本表示转为可执行图对象并调用工具函数增强拓扑推理能力"]
benchmarks: ["FlowVQA", "FlowLearn"]
---

# 论文速读：Beyond-End-to-End-VLMs-Leveraging-Intermediate-Text-Represen

## 一句话总结
论文提出 **TEXTFLOW** 双阶段框架，通过将流程图图像转换为结构化文本表示（GRAPHVIZ/MERMAID/PLANTUML），再交由 LLM 进行推理问答，解决了端到端 VLM 在流程图理解中可控性差、可解释性弱的问题，在 FlowVQA 和 FlowLearn 基准上达到 SOTA（82.74 vs 76.61）。

## 研究问题与动机
- **可控性不足**：端到端 VLM 训练对研究者不可控，用户只能调整输入图像，无法改进模型内部处理流程。
- **可解释性缺失**：VLM 出错时难以定位是视觉编码失败还是推理环节失误。
- **端到端方法瓶颈**：现有工作（如 FlowchartQA、FlowLearn、FlowVQA）主要依赖端到端 VLM，缺乏对中间表示的显式建模。
- **模块化需求**：若 VLM 端到端推理能力不足，可借助更强 LLM 弥补，但现有方法无法灵活替换组件。

## 核心贡献（创新点）
- **双阶段解耦框架**：将流程图理解拆分为 VISION TEXTUALIZER（图像转文本）和 TEXTUAL REASONER（文本推理问答），与传统端到端 VQA 形成本质区别。
- **首创三种流程图文本化格式**：首次系统引入 GRAPHVIZ、MERMAID、PLANTUML 用于流程图结构表征，并对比其有效性。
- **可执行图对象增强推理**：支持将文本表示转为 Python 可执行图对象，调用工具函数（如最短路径、入度/出度计算）提升拓扑类任务精度。
- **细粒度错误归因分析**：通过三源错误分类（TEXTUALIZER错/REASONER错/两者皆错）量化两阶段误差来源，发现 TEXTUALIZER 是主要瓶颈。
- **SOTA 性能与鲁棒性**：在 FlowVQA 和 FlowLearn 上超越所有端到端基线，且在流程图方向反转、节点数增加等条件下保持更强鲁棒性。

## 方法详解
**TEXTFLOW 由两个阶段组成：**

1. **VISION TEXTUALIZER**：使用 VLM 将流程图图像转换为结构化文本表示，支持三种格式：
   - **MERMAID**：基于链接的简单语法，节点用括号定义形状，适合线性流程。
   - **GRAPHVIZ**：节点与边分离定义，属性丰富（标签、形状），结构更清晰，适合复杂拓扑。
   - **PLANTUML**：类伪代码语法，支持条件、循环、嵌套结构，适合详细逻辑描述，但语法复杂。

2. **TEXTUAL REASONER**：使用 LLM 或 VLM 基于文本表示回答问题，支持两种输入模式：
   - **纯文本表示**：直接将 GRAPHVIZ/MERMAID/PLANTUML 代码与问题一同输入 LLM。
   - **可执行图对象**：将文本转为 Python 图对象，LLM 可通过工具调用获取精确图结构信息（如节点数、边数、前驱/后继、最短路径、最大入度/出度）。

**核心思想**：通过将视觉任务转化为文本任务，实现模块化解耦——VLM 专注视觉编码，LLM 专注逻辑推理，二者可独立替换与优化。

## 实验与结果
- **数据集**：
  - **FlowVQA**（主实验）：200 张流程图、2005 个 QA 对，涵盖 CODE/WIKI/INSTRUCT 三类来源及 T1（事实检索）、T2（应用场景）、T3（流向指代）、T4（拓扑结构）四类任务。
  - **FlowLearn**（辅助验证）：7 个任务、每任务 100 对，侧重 OCR、True/False、节点计数等。

- **评估基线**：Llama3.2-11B/90B、Llava-v1.6-110B、Qwen2-VL-7B/72B、GPT-4o、Claude-3.5-Sonnet 的端到端 VQA 表现。

- **主要结果**：
  - **TEXTFLOW (Claude-3.5-Sonnet + GPT-4o, GRAPHVIZ)**：FlowVQA 准确率 **82.74**，FlowLearn **80.57**，显著优于最强端到端基线 GPT-4o（76.61 / 77.00）。
  - **TEXTFLOW (GPT-4o + GPT-4o)**：FlowVQA **80.10**，仍大幅领先基线。
  - **工具增强效果**：使用 Gold 表示 + 工具时，TOP4 拓扑任务准确率接近完美（Graphviz 99.76%，Mermaid 100%）。
  - **文本格式对比**：GRAPHVIZ 整体最优，PLANTUML 最差但仍优于端到端基线。
  - **鲁棒性**：TEXTFLOW 在流程图反向（bottom-up）和节点数增加时性能下降幅度小于 VQA 基线。

- **错误分析**：使用 Claude-3.5 时，错误主要来源于 TEXTUALIZER（决策节点多连接、标签误写），REASONER 错误率较低。

## 相关工作脉络
- **FlowchartQA (Tannert et al., 2023)**：强调几何与拓扑推理，采用端到端方法；TEXTFLOW 聚焦文本化中间表示提升可控性。
- **FlowLearn (Pan et al., 2024)**：关注合成与科学流程图，使用 OCR 预处理；TEXTFLOW 直接生成结构化文本而非仅 OCR。
- **IconQA (Lu et al., 2021)**：针对抽象语义图表，使用金字塔 Cross-modal Transformer；TEXTFLOW 避免复杂视觉编码，转向文本推理。
- **FlowVQA (Singh et al., 2024)**：评估空间推理与决策流，当前 SOTA 基线；TEXTFLOW 通过双阶段分解超越其端到端方法。
- **问题分解方法 (Cao & Jiang, 2023; Khan et al., 2024; Barezi & Kordjamshidi, 2024)**：将问题拆分为子任务分配给不同模型；TEXTFLOW 不同之处在于拆分视觉编码与文本推理两个阶段，而非问题本身。
- **IdealGPT (You et al., 2023)**：LLM 生成子问题、VLM 生成子答案、LLM 最终推理；TEXTFLOW 更简洁，仅保留图像→文本→回答的线性流水线。

## 局限性与未来方向
- **提取精度瓶颈**：复杂或噪声流程图中，TEXTUALIZER 在决策节点多连接、标签改写等方面仍易出错。
- **数据集多样性不足**：现有基准（如 FlowVQA）多为标准 Mermaid 风格，缺乏真实世界复杂流程图。
- **复杂嵌套结构泛化有限**：对依赖图、甘特图、嵌入图像等复杂图表处理能力不足。
- **领域知识依赖**：部分流程图需外部知识或文档引用，未来可结合 RAG 增强。
- **工具依赖**：可执行图对象增强依赖外部工具链，增加系统复杂度与兼容风险。

## 研究启发与可借鉴点
- **双阶段解耦范式可迁移**：将视觉任务拆分为"视觉编码→文本推理"两阶段，可应用于其他图表理解任务（如电路图、UML 图、组织结构图）。
- **结构化文本表示的价值**：GRAPHVIZ 作为最有效格式的经验表明，将视觉结构显式编码为机器可读格式可显著提升下游推理精度。
- **工具增强策略**：将文本表示转为可执行图对象并调用精确计算工具，为拓扑类任务提供高可信答案，该设计可扩展至其他结构化数据理解。
- **错误归因分析框架**：三源错误分类（TEXTUALIZER/REASONER/两者）为多阶段系统的误差分析提供可复用方法论。
- **模块化替换优势**：允许 TEXTUALIZER 与 REASONER 使用不同模型（如强 VLM + 强 LLM），为资源受限场景下的灵活部署提供思路。

## 关键术语表
- **TEXTFLOW**：本文提出的双阶段流程图理解框架，包含 VISION TEXTUALIZER 和 TEXTUAL REASONER 两个模块。
- **VISION TEXTUALIZER**：第一阶段，使用 VLM 将流程图图像转换为结构化文本表示（GRAPHVIZ/MERMAID/PLANTUML）。
- **TEXTUAL REASONER**：第二阶段，使用 LLM/VLM 基于文本表示进行问答推理，可调用图工具增强。
- **GRAPHVIZ**：一种节点-边分离的结构化文本格式，本文实验表明其对流程图理解最有效。
- **MERMAID**：一种基于链接的简单文本语法，适合线性流程，渲染成功率较高。
- **PLANTUML**：类伪代码语法，支持复杂结构（循环、嵌套），但语法复杂、提取难度大。
- **FlowVQA**：评估流程图空间推理、决策流理解的多模态基准，本文主要评测数据集。
- **Topological Tasks (T₄)**：涉及节点/边计数、最短路径、前驱/后继关系等图结构分析的任务类别。

## 可复现要素
- **数据集**：FlowVQA（公开）、FlowLearn（公开）
- **代码/权重**：论文声明"All code and data are publicly available"（具体链接见原文脚注¹）
- **关键超参**：
  - 解码策略：greedy decoding（temperature = 0）
  - 最大 token 长度：4096
  - 评估 temperature：0.2（三次采样多数投票）
  - GPU：1-4 张 Nvidia A100 (80GB)
  - 图像分辨率：使用各 VLM 支持的最高分辨率模式
