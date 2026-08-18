---
title: "Investigating-the-De-Composition-Capabilities-of-Large-Langu"
source: https://aclanthology.org/2025.naacl-long.87.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:05:00"
field: "大语言模型能力评估"
keywords: ["自然到形式语言转换", "分解能力", "组合能力", "组合泛化", "in-context learning", "符号系统学习"]
innovations: ["提出DEDC框架解耦评估LLM在N2F中的分解与组合能力", "设计组合间隙与反直觉符号名称设置量化现实挑战的影响", "建立六类错误分类体系并提供归因分析"]
benchmarks: ["DEDC数据集（323样本表格推理形式语言）"]
---

# 论文速读：Investigating-the-De-Composition-Capabilities-of-Large-Langu

## 一句话总结
本文提出DEDC框架，通过半自动样本/任务构建实现大语言模型在自然语言到形式语言转换（N2F）中分解与组合能力的解耦评估，揭示了当前先进LLM在该任务中分解能力缺陷尤为严重，且组合间隙和反直觉符号名称会显著损害其表现。

## 研究问题与动机
1. 核心问题：当LLM面对陌生形式语言时，是否具备从示例中学习符号原语含义（分解能力）并将其组合成新表达式（组合能力）的基本能力，以完成泛化且鲁棒的N2F任务？
2. 现有方法不足：尽管LLM在常见形式语言（如SQL、逻辑表达式）上有一定N2F能力，但缺乏系统化评估框架来检验其应对任意新形式语言的分解与组合基础能力。
3. 实际应用动机：在in-context learning场景下，LLM需处理 uncommon formal languages，分解与组合是实现零样本泛化的关键能力。
4. 认知科学动机：该能力集合是确认LLM在N2F中具备真正智能的必要条件，当前无现有框架支持此类评估与分析。

## 核心贡献（创新点）
1. 提出DEDC框架，通过构造带演示样本的测试任务和可选的原语演示，实现对分解能力与组合能力的解耦度量，区别于以往仅报告单一准确率的做法。
2. 设计了组合间隙（0%/100% gap）与反直觉符号名称（Anomalous/Cross-mapping）两种额外评估设置，量化了这些现实挑战对LLM分解与组合的具体影响幅度。
3. 建立了六类错误分类体系（Primitive confusion/fiction、Variable misuse、Redundancy、Omission、Incorrect meaning），并给出了归因分析（语义误解 vs 符号系统学习缺陷），为后续改进提供可操作的诊断依据。

## 方法详解
1. 形式语言定义：基于表格推理场景，选取10个原语函数（f0–f9，如filter_gt、top_k、kth_max等），表达式为使用单个函数的赋值语句，多步表达式通过DAG连接构成复杂推理链。
2. 样本构建：识别6种4节点基图（Figure 2），枚举所有合法的节点-原语映射（共323种有效方案），对每种方案随机生成结构无关参数，生成323个(graph, expression, question)样本对。
3. 任务构造：每个样本作为测试样本，抽取3个演示样本（满足：每演示样本至少包含测试样本的一个原语，且三个演示样本的原语集合覆盖测试样本全部原语），LLM需基于演示将自然语言问题转换为对应表达式。
4. 解耦度量设计：令P_dc为需分解+组合的准确率，P_c为仅需组合（提供每个原语的最小演示样本）的准确率；定义D_c = 100 - P_c衡量组合缺陷率，D_d = P_c - P_dc衡量分解引入的额外错误率，两者越大表明对应能力越弱。
5. 额外设置度量：对于设置s，定义△_c^s = P_c^s - P_c和△_d^s = (P_dc^s - P_dc) - △_c^s，正值表示该设置使能力增强，负值表示难度增加。

## 实验与结果
- 数据集：自建323个样本（表格推理形式语言），代码与数据已开源（https://github.com/xzy-xzy/DEDC）。
- 评估模型：GPT-4o、Claude-3.5、DeepSeek-2.5、Mistral-large、Llama-3.1（temperature=0）。
- 基准结果（Table 1）：Claude-3.5表现最优（P_dc=91.02%，P_c=98.76%，D_c=1.24%，D_d=7.74%）；DeepSeek-2.5最差（P_dc=68.73%，D_c=13.31%，D_d=17.96%）。所有模型D_d > D_c，分解缺陷均重于组合缺陷。
- 组合间隙实验（Table 4）：0% gap时各模型△_c和△_d均为正（性能提升）；100% gap时均为负（性能下降）。Claude-3.5受影响最小（△_d=-0.62），Llama-3.1受影响最大（△_d=-4.64）。
- 反直觉符号名称实验（Table 5）：两种设置均导致显著性能下降；Cross-mapping比Anomalous影响更严重。Claude-3.5在Anomalous下△_d=-30.34，在Cross下△_d=-31.27；DeepSeek-2.5在Cross下△_d=-21.36。所有模型|△_d^s|>20，表明分解能力受符号名称干扰极为严重。
- 错误分析（Table 3）：Primitive confusion是最常见错误类型，且随分解需求增加最显著；Claude-3.5错误覆盖类型最少（仅2种）。

## 相关工作脉络
1. Compositionality研究（Lake & Baroni, 2018; Hupkes et al., 2020）关注训练-测试集组合间隙下的泛化能力，本文在N2F框架下解耦评估分解与组合，且避免预先分区数据。
2. In-context compositional generalization（Levy et al., 2023; An et al., 2023）通过采样预设partition数据集研究组合泛化，本文通过DEDC框架直接解耦两能力而无需预分区。
3. Decomposition of execution steps（Song et al., 2019）关注任务执行步骤的层次分解，本文聚焦于从示例中学习符号原语含义的分解过程，二者关注点不同。
4. Decomposition for reasoning/planning（Ye et al., 2023; Wu et al., 2024）针对大规模数据与复杂问题的内容分解，本文的分解针对形式语言符号原语的学习与提取。
5. N2F相关工作（Dozat & Manning, 2017; Katsogiannis-Meimarakis & Koutrika, 2023）多针对特定形式语言（句法解析、Text-to-SQL）的能力评估，通常依赖针对性训练语料；本文从无已知映射的新形式语言角度切入，评估LLM的基础学习能力。
6. Formal language proficiency评估（Liu et al., 2024）关注LLM对KBQA中已有形式语言（如SPARQL）的掌握程度，本文则聚焦于面对全新形式语言时的分解-组合基础能力。

## 局限性与未来方向
1. 仅在一个形式语言案例（表格推理场景）上进行评估，覆盖的形式语言类型不够全面，框架的泛化性有待验证（附录B.2讨论了框架可推广性）。
2. 未提供提升LLM分解与组合能力的具体方法，仅定位了问题和归因方向。
3. 缺乏从认知科学角度的人机对比实验，无法判断LLM的错误模式与人类是否存在系统性差异（附录B.3讨论了人类能力的预期行为）。
4. 问答模板存在轻微语法分歧（如使用"it"指代而非"whose"），虽经重实验验证影响较小，但仍可能引入微弱偏差。

## 研究启发与可借鉴点
1. 解耦评估思路可直接迁移到其他需要"学习-应用"分离评估的任务中，如程序合成、代码生成、工具调用学习等场景，通过构造对照实验分别测量"理解符号含义"与"组合应用"两个阶段的能力瓶颈。
2. 反直觉符号名称设置（Anomalous/Cross-mapping）揭示了LLM对符号名称与语义绑定过强的问题，可启发未来研究设计"名称无关"的训练数据或提示策略，增强符号系统的鲁棒性。
3. 六类错误分类体系及其归因（语义误解 vs 符号学习缺陷）可作为通用诊断工具，帮助团队定位自身研究中模型失败的根本原因，而非仅看整体准确率。
4. 组合间隙的设置思路可应用于评估LLM在in-context learning下的零样本泛化边界，为prompt engineering提供实证依据（如演示样本的选择策略应尽量减少组合间隙）。

## 关键术语表
**DEDC**：Decoupled Evaluation of Decomposition and Composition的缩写，本文提出的框架，用于解耦评估LLM在N2F中的分解与组合能力。
**形式语言（Formal Language）**：由符号原语按规则组合成表达式的符号系统，如SQL、逻辑表达式、形式句法等。
**原语（Primitive）**：形式语言中的基本函数单元，本文使用10个表格推理函数（f0–f9）作为原语。
**分解（Decomposition）**：从演示示例中抽取符号原语的含义与格式的能力。
**组合（Composition）**：将已学习的符号原语按照任务要求组装成新表达式的能力。
**组合间隙（Compositional Gap）**：测试样本与演示样本在表达式组合结构上存在差异的情况，衡量LLM的零样本泛化能力。
**反直觉符号名称（Counter-intuitive Symbolic Name）**：符号原语的名称与其实际语义不匹配的命名方式，用于评估LLM是否过度依赖名称启发式。
**D_c / D_d**：分别表示组合缺陷率（D_c = 100 - P_c）和分解引入的额外错误率（D_d = P_c - P_dc）。

## 可复现要素
- 数据集：自建，323个样本（表格推理形式语言），已开源（https://github.com/xzy-xzy/DEDC）
- 代码：已开源（https://github.com/xzy-xzy/DEDC）
- 权重：使用商用LLM API（GPT-4o、Claude-3.5、DeepSeek-2.5、Mistral-large、Llama-3.1），temperature=0
- 关键超参：演示样本数=3；测试样本323个；10个原语；6种基图结构（4节点DAG）
