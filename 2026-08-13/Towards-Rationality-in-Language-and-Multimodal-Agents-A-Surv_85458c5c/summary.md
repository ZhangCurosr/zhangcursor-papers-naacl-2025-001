---
title: "Towards-Rationality-in-Language-and-Multimodal-Agents-A-Surv"
source: https://aclanthology.org/2025.naacl-long.186.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:58:37"
field: "Agent 系统理性评估"
keywords: ["理性 Agent", "多模态 Agent", "RLAIF", "保形风险控制", "多 Agent 辩论", "逻辑一致性", "信息接地", "期望效用理论"]
innovations: ["提出四项理性公理作为统一分类框架", "将保形风险控制引入 Agent 事实性管理", "以公理透镜重新归类解读多模态/RAG/工具/神经符号/RLHF 等技术路线"]
benchmarks: ["POPE", "LLaVA-RLHF", "BLINK", "MileBench", "Seed-bench-2", "DEMON", "PaperswithcodeMCQA"]
---

# 论文速读：Towards-Rationality-in-Language-and-Multimodal-Agents-A-Survey

## 一句话总结
本文首次以**理性公理（Rationality Axioms）**为统一框架，系统梳理了语言与多模态 Agent 领域的前沿工作，提出理性 Agent 应满足四项必要条件（信息接地、逻辑一致、无关语境不变性、偏好序可排序），并以这一透镜重新归类解读了多模态融合、外部知识检索、多 Agent 协作、神经符号推理、RLHF/RLAIF、效用函数与保形风险控制等方法，同时指出当前理性评估指标的严重匮乏。

## 研究问题与动机
- **LLM 单体的理性缺陷**：单一 LLM 依赖参数化文本知识，缺乏与现实世界的接地与反馈机制，存在有界知识、输出不一致、易受偏置与框架效应影响等问题，在医疗、金融、法律等高 stakes 场景中可靠性不足。
- **现有 Survey 的视角缺口**：已有 Agent 类 Survey 聚焦架构/规划/通信/记忆，推理类 Survey 聚焦推理能力，仅少数触及理性；本文是国内/国际上**首个以理性公理为纲**全面审视 Agent 系统的综述。
- **评估基准缺失**：大量推理基准（如 MMLU、HellaSwag）并不直接测量理性；针对四项公理的量化评测方法几乎空白，制约了该方向的实证推进。
- **理性≠推理**：逻辑一致性仅是理性的必要条件而非充分条件；本文区分"推理（从前提得出结论）"与"理性（结论可靠一致且与证据对齐）"，确立新的研究坐标系。

## 核心贡献（创新点）
1. **提出四项理性公理**：信息接地（Information Grounding）、逻辑一致性（Logical Consistency）、无关语境不变性（Invariance from Irrelevant Context）、偏好序可排序（Orderability of Preference），构成理性 Agent 的必要条件集合。
2. **以公理为透镜的系统性分类学**：将多模态基础模型、RAG/工具调用、多 Agent 辩论、神经符号推理、RLAIF、效用最大化、保形风险控制等数十条技术路线归入四项公理，建立首份"理性-Agent 映射树"。
3. **揭示 RLHF 的理性局限并提出 RLAIF 替代路径**：指出人类偏好内在不一致性使 RLHF 无法保证理性排序；AI 反馈（RLAIF）在多轮迭代中可提供更稳定的偏好信号。
4. **保形风险控制（Conformal Risk Control）引入 Agent 决策**：将 Angelopoulos 等人的保形预测框架适配至 LLM 的事实性/幻觉频率控制，为高风险领域的理性决策提供可证明的风险上界。
5. **首次系统诊断理性评估的缺口**：逐项分析四项公理对应的评测难点，指出 token bias、扰动鲁棒性、数据污染、中间过程观测等关键问题，呼吁建立专门的理性基准。

## 方法详解
- **四项公理的形式刻画**：
  - *信息接地*：Agent 决策须以感知到的多模态事实为依据，避免幻觉；对应"picture is worth a thousand words"的多源 grounding。
  - *逻辑一致性*：避免自相矛盾，等价问题表征下输出不变；对应 System 2 式的反复推敲。
  - *无关语境不变性*：忽略无关上下文干扰（如"lost-in-context"现象），聚焦逻辑本质。
  - *偏好序可排序*：基于期望结果对备选方案排序并选择最优；形式化为期望效用理论（EUT）。
- **通向各项公理的技术路径**：
  - *多模态基础模型*（CLIP、LLaVA、GPT-4V、Gemini 1.5 Pro 等）：跨模态预训练在联合隐空间中对齐语义，提升接地与不变性。
  - *RAG 与外部知识*（Retrieval-Augmented Generation、Chain-of-Knowledge、知识图谱）：突破有界理性（Bounded Rationality）的工作记忆瓶颈，减少幻觉。
  - *工具调用*（Toolformer、VisProg、ViperGPT、Parsel、Binder）：将问题抽象为确定性子程序执行，确保一致输出。
  - *多 Agent 协作/辩论*（LM vs LM、FORD、AgentReview、Du et al. 2023）：通过 System 2 式跨 Agent 交叉质询收敛共识，Du 等报告事实准确率提升 7.2–15.9%。
  - *神经符号推理*（Logic-LM、KRISP、LEFT）：结合学习力与显式符号表示，保证演绎一致性。
  - *RLAIF*（Lee et al. 2024）：以 LLM 作为评估器迭代生成稳定偏好信号。
  - *期望效用理论*（DeLLMa）：将决策分解为子任务，分配效用值，选最大化期望效用的动作。
  - *保形风险控制*（Angelopoulos et al. 2022）：对任意非递增损失函数，保证期望损失被阈值 α 界定。
- **公理非充分性声明**：四项公理为必要但非充分条件；任何单项技术均不能单独保证真正理性，需组合使用。

## 实验与结果
> 本文为**综述论文**，无自建实验；但引用了多项基线工作的量化结果，以下为核心数字：
- **Chain-of-Knowledge**：整合多知识源较单源性能提升 **2.1%**。
- **FORD**（多 Agent 辩论）：相比单 LLM 准确率提升最高 **4.9%**。
- **Liang et al. 2023**：多 Agent 辩论优于单 Agent 自我反思，推理任务最终共识提升 **16.0%**。
- **Du et al. 2023**：多轮辩论后 LLM 收敛于单一共享答案，跨任务事实准确率提升 **7.2–15.9%**。
- **AGENT Review**：讨论引起最终审稿分布偏移（定性发现）。
- **Jiang et al. 2024a**：SOTA LLM 在面对 token bias 时仍表现出不一致行为（即使任务逻辑本质不变）。
- **Ross et al. 2024**：揭示 LLM 存在时间折扣、风险厌恶、损失厌恶等人类样 bias，既非完全拟人也非完全经济理性人（economicus）。
- 最强结果指向：**多 Agent 辩论/共识机制**在逻辑一致性和事实准确率维度上带来最大幅度提升（最高 +16.0%）。

## 相关工作脉络
1. **Agent 系统 Survey**（Xie et al. 2024a、Han et al. 2024、Guo et al. 2024）：聚焦架构/规划/通信/记忆；本文与之差异在于以"理性公理"为分类轴而非系统组件。
2. **LLM 推理 Survey**（Qiao et al. 2022、Huang & Chang 2022、Ahn et al. 2024）：聚焦推理能力本身；本文强调理性≠推理，理性包含证据对齐、一致性、不变性、偏好排序等更广维度。
3. **幻觉/事实性研究**（HaluEval、BEGIN、Dial-Fact、FaithDial）：聚焦单点检测；本文将其统一纳入"信息接地"公理，并提出多模态幻觉基准的缺失。
4. **RLHF/对齐工作**（Ouyang et al. 2022、Bai et al. 2022）：以人类反馈为主；本文指出其偏好不一致性局限，引出 RLAIF 作为补充。
5. **神经符号推理**（Logic-LM、Parsel、Binder）：早期工作聚焦特定任务；本文将其归入"逻辑一致性+无关语境不变性"双公理框架。
6. **保形预测**（Angelopoulos et al. 2022；Mohri & Hashimoto 2024）：原属统计学习；本文首次将其引入 Agent 理性决策的风险控制语境。
7. **认知偏差与 LLM**（Binz & Schulz 2023、Macmillan-Scott & Musolesi 2024、Echterhoff et al. 2024）：揭示 LLM 的启发式推理与人类相似 bias；本文将其纳入理性评估的实证基础。

## 局限性与未来方向
- **公理非充分性**：四项公理仅为必要条件，无法单独保证真正理性；需要更多公理（如完备性、传递性、单调性、可分解性）的严格形式化，但这些属于更基础的理论逻辑范畴，难以直接映射到语言/多模态 Agent。
- **外部工具依赖**：当前方法均通过外部工具/多 Agent 实现理性逼近，未解决如何将这些理性输出"烤回"（bake into）基础模型本身的参数化表示中。
- **多模态在多 Agent 系统中利用不足**：当前多 Agent 协作和符号推理尚未充分整合视觉、音频等多模态输入。
- **评估指标严重匮乏**：缺乏针对四项公理的标准化基准；现有评测过度关注最终性能而忽略中间推理过程；Token bias、扰动鲁棒性、数据污染等关键问题未被系统性解决。
- **领域覆盖局限**：未涵盖心智理论（Theory of Mind）、认知架构、形式逻辑与概率理论中的理性模型等核心理论，这些对深入理解 Agent 理性至关重要。

## 研究启发与可借鉴点
1. **公理化分类框架的可迁移性**：本文的"四项公理→技术路径映射"方法可复用于其他 AI 属性（如可信度、公平性、可解释性）的系统性分类学构建，为团队后续综述提供方法论模板。
2. **多 Agent 辩论作为 System 2 的实现**：Du et al. 和高效率共识收敛（+16.0%）表明，**多轮跨 Agent 辩论**是低成本提升逻辑一致性的有效手段，可直接迁移至本团队的高 stakes 决策场景。
3. **保形风险控制进入 Agent 工程**：将保形预测从学术概念推向 Agent 事实性/幻觉频率控制，为金融、医疗等高风险领域提供了**可证明风险上界**的工程路径，值得团队深入实践。
4. **RLAIF 替代/补充 RLHF**：人类反馈的不一致性是系统性问题；团队可在偏好对齐任务中探索以 LLM 评估器为核心的 RLAIF pipeline，提升偏好排序稳定性。
5. **理性基准建设的机遇**：当前四项公理评估几乎空白，团队可率先构建针对"token bias 鲁棒性""多模态幻觉率""跨任务偏好一致性"的基准套件，抢占评测标准制定先机。

## 关键术语表
- **理性公理（Rationality Axioms）**：本文提出的四项 Agent 理性必要准则，包括信息接地、逻辑一致性、无关语境不变性、偏好序可排序。
- **有界理性（Bounded Rationality）**：March & Simon 提出的概念，指决策受限于可用资源、计算能力和工作记忆，偏离最优主要源于容量限制。
- **RLAIF（Reinforcement Learning from AI Feedback）**：以 LLM 作为评估器迭代生成偏好信号的训练范式，相较 RLHF 可提供更稳定的跨格式偏好排序。
- **保形风险控制（Conformal Risk Control）**：基于保形预测的理论框架，对任意非递增损失函数保证期望损失被预定义阈值 α 界定，用于可控幻觉/事实性管理。
- **期望效用理论（Expected Utility Theory, EUT）**：Von Neumann & Morgenstern 提出的决策形式化框架，通过分配效用值并加权概率来计算期望效用，指导最优动作选择。
- **System 1 / System 2**：Kahneman 提出的双系统认知框架；System 1 快速直觉，System 2 慢速审慎——多 Agent 辩论模拟 System 2 式 deliberation。
- **信息接地（Information Grounding）**：Agent 决策须基于感知的多模态事实依据，而非仅依赖参数化文本知识，以避免幻觉。
- **逻辑一致性（Logical Consistency）**：Agent 避免自相矛盾，在等价问题表征下输出不变，是理性排序的推导基础。

## 可复现要素
- **数据集**：本文为综述，无自建数据集；引用数据包括 POPE、LLaVA-RLHF、BLINK（多模态幻觉）、MileBench、Seed-bench-2、DEMON（长上下文）、PaperswithcodeMCQA（偏好排序）等。
- **代码/权重**：开源仓库 https://github.com/bowenupenn/Agent_Rationality（论文声明）；各引用工作代码分散于各项目（如 Toolformer、VisProg、Parsel、FORD、AgentReview 等），需分别获取。
- **关键超参**：论文未统一声明超参（综述性质）；保形风险控制阈值 α 依具体任务设定（通常 0.05–0.1）；多 Agent 辩论轮次在 Du et al. 中通常为 2–5 轮。
