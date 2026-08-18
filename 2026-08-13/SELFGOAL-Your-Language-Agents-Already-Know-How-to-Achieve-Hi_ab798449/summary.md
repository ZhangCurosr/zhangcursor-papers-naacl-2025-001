---
title: "SELFGOAL-Your-Language-Agents-Already-Know-How-to-Achieve-Hi"
source: https://aclanthology.org/2025.naacl-long.36.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:37:23"
field: "多智能体系统中的 LLM 规划"
keywords: ["language agents", "goal decomposition", "LLM planning", "multi-agent games", "adaptive tree structure", "non-parametric learning"]
innovations: ["提出 SELFGOAL 框架，通过 Search-Act-Decompose 三模块动态构建和维护 GOALTREE 实现高层目标的实时分解", "引入余弦相似度阈值和停止机制控制子目标树的粒度与深度，避免过度细化或信息不足", "在非参数学习框架下使小模型也能有效达成延迟反馈的高水平目标"]
benchmarks: ["GAMA-Bench", "AucArena", "DealOrNotDeal", "ScienceWorld"]
---

# 论文速读：SELFGOAL-Your-Language-Agents-Already-Know-How-to-Achieve-Hi

## 一句话总结
本文提出 SELFGOAL，一种自适应的无参数学习框架，通过在与环境交互过程中动态构建、利用和更新子目标树（GOALTREE），使语言智能体能够在延迟反馈和高水平目标下自主适应并完成复杂任务。

## 研究问题与动机
- **高水平目标的动态分解需求**：实际场景中智能体常面临模糊且延迟反馈的高水平目标（如"赢得最多钱"），难以在没有详细指令和环境反馈的情况下持续达成目标。
- **现有方法的两极分化**：现有工作分为任务预分解（如 ReAct、ADAPT）和经验后总结（如 Reflexion、CLIN），前者脱离环境缺乏实证指导，后者过于简单且难以优先排序。
- **固定树深的泛化瓶颈**：无论先验分解还是经验总结，都无法应对场景差异导致的粒度需求变化，需要一个能自适应调整的树结构。
- **小模型能力受限问题**：小型 LLM（如 Mistral-7B、Qwen-7B）缺乏归纳和总结能力，导致现有经验总结方法在小模型上表现不佳。

## 核心贡献（创新点）
- **提出 SELFGOAL 自适应框架**：通过 Search、Act、Decompose 三个模块动态维护 GOALTREE，实现高层目标的实时分解与调整，与 ReAct/ADAPT 的事前分解和 Reflexion/CLIN 的事后总结形成本质区别。
- **设计基于语义相似度的粒度控制机制**：通过余弦相似度阈值 ξ 过滤冗余节点，结合停止机制控制树深度，避免过度细化或信息不足，这是已有方法所未涉及的自适应粒度控制。
- **在竞争/合作/延迟反馈多场景验证有效性**：在 GAMA-Bench（公共物品博弈、猜2/3平均值）、AucArena（一级价格拍卖）、DealOrNotDeal（谈判）及 ScienceWorld 等任务中，SELFGOAL 显著优于所有基线，且对小模型同样有效。

## 方法详解
SELFGOAL 是一个非参数学习算法，不更新模型权重，而是通过更新提示词 p 来适应当前情境。核心是维护一棵文本子目标树 T（GOALTREE）：

- **Search Module**：给定当前状态 s_{t-1}（对话历史描述）和 GOALTREE 的叶节点列表，要求 LLM 选出最合适的 K 个子目标，用于指导当前行动。选择依据是 LLM 对"哪些子目标在当前情境下有助于达成主目标"的判断。
- **Act Module**：将搜索到的子目标加入提示词 p_t，驱动 Actor M_a 生成动作 a_t，与环境交互得到新状态 s_t。
- **Decompose Module**：基于动作-状态对 {a_t, s_t}，对 Search Module 选中的子目标 g_{i,j} 进行分解，生成新的子目标集合 G。每个新子目标 g 与现有叶节点的余弦相似度若超过阈值 ξ，则过滤掉；否则作为 g_{i,j} 的子节点加入 GOALTREE。
- **停止机制**：若连续 N 轮没有新节点加入，则停止分解更新。

关键公式：
- 动作生成：a_t ~ π_θ(a_t | s_{t-1})
- 语义过滤：cosine(g, T_leafnodes) < ξ 时添加子节点

## 实验与结果
**实验环境**：
- Public Goods Game (GAMA-Bench, N=5, R=2)
- Guess 2/3 of the Average (GAMA-Bench, N=5)
- First-price Auction (AucArena, N=4, K=15, budget=$20000)
- Bargaining (DealOrNotDeal, N=2, K=10, M=50 items)
- 额外单智能体任务：ScienceWorld

**基线方法**：ReAct、ADAPT、Reflexion、CLIN
**使用模型**：GPT-3.5-Turbo、GPT-4-Turbo、Gemini 1.0 Pro、Mistral-7B、Mixtral-8x7B、Qwen-7B/72B

**主要结果**：
- **Public Goods Game**（越低越好）：GPT-4 下 SELFGOAL 得 11.95 vs 次优 16.70；Qwen-72B 下 8.45 vs 次优 20.75
- **Guess 2/3 Average**（越高越好）：GPT-4 下 94.54 vs 次优 94.41；GPT-3.5 下 83.28 vs 次优 92.57（ReAct）
- **First-price Auction**（TrueSkill 越高越好）：GPT-4 下 39.02 vs 次优 38.98；Qwen-72B 下 36.48 vs 次优 35.92
- **Bargaining**（利润差越低越好）：GPT-4 下 1.71 vs 次优 1.80
- **ScienceWorld**：GPT-4o-mini + SELFGOAL 平均得分 24.34 vs 基线 20.68

**最强提升**：在 Auction 任务中，SELFGOAL 相较 ReAct 获得 TrueSkill +5.9 提升（GPT-3.5 下 27.40 vs 22.90）。小模型如 Mistral-7B 在多个任务中从 SELFGOAL 获得显著提升，而 CLIN/Reflexion 在小模型上效果差。

## 相关工作脉络
- **ReAct (Yao et al., 2023)**：先推理再行动，但分解发生在交互之前，缺乏环境反馈驱动的动态调整能力。
- **ADAPT (Prasad et al., 2024)**：递归分解目标，同样属于事前分解，未考虑环境实证，产生的子目标粒度固定且可能脱离实际。
- **Reflexion (Shinn et al., 2023)**：从失败经验中反思并更新策略，但总结出的规则简单、无序，难以优先排序。
- **CLIN (Majumder et al., 2023)**：生成因果抽象记忆（"X may be necessary for Y"），但聚焦细节且与主目标关联性弱。
- **Voyager (Wang et al., 2023a)**：在 Minecraft 中构建代码技能库，依赖详细错误反馈，不适合延迟反馈场景。
- **OKR-Agent (Zheng et al., 2023)**：分层自协作/自纠错机制，仍基于任务开始前的一次性分解，缺乏运行时动态调整。

## 局限性与未来方向
- **计算成本较高**：SELFGOAL 约需基线方法 5 倍计算资源（Token 消耗和推理时间），但仍处于可接受范围。
- **小模型上限受限**：虽然对小模型有效，但其性能仍受限于模型本身理解和总结复杂能力，可能无法完全发挥 GOALTREE 潜力。
- **树结构规模依赖超参**：子节点数量、相似度阈值 ξ 和停止轮数 N 需根据场景调整，缺乏统一设定准则。
- **未来方向**：可调节约束的子节点数量以平衡成本与性能；探索更高效的树搜索与剪枝策略；结合 MCTS 等推理增强技术。

## 研究启发与可借鉴点
- **树结构 + 粒度控制的自生长机制**：GOALTREE 的设计思路可迁移至其他需要长期规划的领域（如机器人、多步骤编程任务），通过余弦相似度阈值控制树深度是一种简洁有效的策略。
- **Search-Act-Decompose 三模块解耦**：将"选择"、"执行"、"分解"分离为独立模块，便于分别优化和替换，为模块化智能体架构提供了参考范式。
- **小模型友好性**：通过提供结构化的子目标树，弥补了小模型归纳能力不足，证明结构化提示工程可有效提升小模型复杂任务表现，值得进一步探索。
- **非参数学习的性价比**：无需微调即可适配新任务，仅通过提示更新实现策略调整，适合快速迭代和低资源场景，可推广至更多动态环境。

## 关键术语表
**SELFGOAL**：本文提出的自适应语言智能体框架，通过动态维护 GOALTREE 实现高层目标的实时分解与调整。
**GOALTREE**：由文本子目标节点构成的层级树结构，根节点为主目标，叶节点为当前可用指导，随交互动态生长。
**Search Module**：从 GOALTREE 叶节点中选择当前情境下最相关的 K 个子目标的模块，利用 LLM 先验知识进行筛选。
**Decompose Module**：基于最新 action-state 对，将选定子目标分解为更具体的子目标并加入 GOALTREE 的模块。
**余弦相似度阈值 ξ**：用于过滤冗余子目标的语义相似度边界，超过阈值则不添加新节点，控制树的粒度与深度。
**TrueSkill Score**：基于贝叶斯统计的动态技能评分系统，用于 Auction 等竞争性任务的性能评估。
**GAMA-Bench**：评估 LLM 在多智能体博弈中决策能力的基准测试，包含公共物品博弈和猜2/3平均值等任务。
**延迟反馈 (Deferred Feedback)**：智能体仅在任务结束时才能获得评估分数的反馈类型，常见于多轮博弈和长期规划任务。

## 可复现要素
- **数据集/环境**：GAMA-Bench (Huang et al., 2024)、AucArena (Chen et al., 2023)、DealOrNotDeal (Lewis et al., 2017)、ScienceWorld (Wang et al., 2022) — 均已开源
- **代码/权重**：论文未明确提供开源链接，实验使用 GPT-3.5/4、Gemini、Mistral-7B、Mixtral-8x7B、Qwen-7B/72B 等商用/开源模型
- **关键超参**：子节点最大数量（默认 10）、相似度阈值 ξ（实验设为 0.6/0.7/0.8/0.9）、停止连续轮数 N、每轮选择 K 个子目标、温度参数 temperature=0
- **环境配置**：Public Goods Game R=2、Auction K=15 items budget=$20000、Bargaining K=10 rounds M=50 items
