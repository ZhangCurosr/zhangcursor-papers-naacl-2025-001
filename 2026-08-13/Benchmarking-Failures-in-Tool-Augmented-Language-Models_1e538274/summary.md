---
title: "Benchmarking-Failures-in-Tool-Augmented-Language-Models"
source: https://aclanthology.org/2025.naacl-long.149.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:59:14"
field: "工具增强大语言模型评测"
keywords: ["工具增强语言模型", "TaLM", "基准测试", "故障感知", "人类交互", "工具调用", "LLM"]
innovations: ["提出FAIL-TALMS基准，系统评估TaLMs在查询不充分和工具不可用两类失败场景下的表现", "引入五维评估指标（Pass Rate/Awareness/Unexpected Success/Skipped Queries/Interaction Ratio）量化故障感知能力", "提出Ask-and-Help (AAH) 方法，允许TaLM实时调用人类助手获取缺失信息或替代不可用工具"]
benchmarks: ["FAIL-TALMS"]
---

# 论文速读：Benchmarking-Failures-in-Tool-Augmented-Language-Models

## 一句话总结
本文提出了 FAIL-TALMS 基准，系统评估工具增强语言模型（TaLMs）在两类真实失败场景下的表现：用户查询不充分（under-specified queries）和工具不可用（unavailable tools），并探索了 Ask-and-Help (AAH) 人类交互缓解方法的有效性。

## 研究问题与动机
1. **理想化假设脱离实际**：现有 TaLM 研究普遍假设"用户查询信息完整"和"所有工具均可用"，但现实中查询常遗漏关键参数，工具也会因废弃、超时等原因失效。
2. **模型缺乏故障感知能力**：主流模型在信息缺失或工具不可用时，往往 hallucinate 上下文或直接终止，而非识别缺失并寻求帮助。
3. **评测基准的不足**：既有工具学习基准多依赖第三方平台离线/在线工具执行，存在工具过期退化问题，且未覆盖真实失败场景。
4. **人类交互的潜力未知**：是否可通过实时人类辅助（AAH）缓解此类失败，以及该方法对不同类型失败的收益差异，尚未被系统研究。

## 核心贡献（创新点）
1. **FAIL-TALMS 基准构建**：收录 906 个免授权真实工具（21 个类别），构建 1,749 条查询的评测集，包含完美、查询不充分、工具不可用三类设置，并提供可复现的实时执行环境。
   → 与既有基准的本质区别：支持实时、可复现的工具调用，而非依赖已下线或过期的第三方 API。
2. **双失败模式的形式化定义与度量体系**：提出 Pass Rate、Information Awareness、Tool Awareness、Unexpected Success、Skipped Queries 五维评估指标，首次系统性量化 TaLMs 的故障感知与容错能力。
   → 与既有研究的区别：超越单纯的任务成功率，引入"感知缺失+优雅跳过"的能力评估。
3. **Ask-and-Help (AAH) 交互范式**：允许 TaLMs 在推理过程中主动调用人类助手获取缺失信息或替代不可用工具，并提出 Interaction Ratio 衡量模型求助意愿。
   → 与既有 Human-in-the-loop 工作的区别：将人类作为可动态调用的"工具"嵌入 TaLM 推理流程，而非仅事后反馈。
4. **多模型对比分析与关键发现**：发现高 Awareness 不必然转化为高 Pass Rate（如 Claude 与 GPT-4o 的反例），且 Llama 系列中 70B 的感知能力反常优于 405B，揭示规模与故障感知的非线性关系。
   → 区别于简单性能报告：深入剖析模型自信度与实际能力的错位现象。

## 方法详解
### FAIL-TALMS 基准构建流程
1. **工具收集**：从 Mixed Analytics 平台收集 1,106 个免授权工具，每个工具附带 URL、功能文档、参数描述及示例用例。
2. **工具验证**：
   - 用 GPT-4o 为每个工具生成 Python 调用代码、JSON Schema 元数据及单元测试；
   - 筛选标准：工具需成功执行并通过全部单元测试，平均响应时间 ≤ 20 秒。
   - 最终保留 906 个有效工具。
3. **查询生成**：
   - **Perfect 设置**：将同类工具两两配对，由 GPT-4o 生成融合两者功能的自然语言查询，经人工校验后得到 575 条；
   - **Under-specified 设置**：手动遮蔽完美查询中的关键参数（如移除地点名 "Pittsburgh"），得到 599 条；
   - **Unavailable Tools 设置**：人为移除必要工具，并按"人类可替代性"进一步划分为 Human-replaceable（261 条）和 Non-replaceable（314 条）。
   - **No-tools 设置**：仅提供查询无工具信息（575 条），作为模型先验知识基线。

### 评估指标设计
- **Pass Rate**：用 GPT-4o 作为裁判，5 次投票取多数判定回答是否成功解决查询。
- **Information Awareness**：
  $$\text{Info Awareness} = \frac{\text{Number of 'idk' and 'no' responses}}{\text{Total examples}}$$
- **Tool Awareness**：
  $$\text{Tool Awareness} = \frac{\text{Number of 'idk' and 'no' responses}}{\text{Total examples}}$$
- **Unexpected Success**：模型回答 "yes"（表示有足够信息/工具）且 Pass Rate = 1 的比例，捕捉"误判但碰巧做对"的 cases。
- **Skipped Queries**：模型明确回答 "no" 并跳过任务的比例，反映模型拒绝错误请求的信心。

### Ask-and-Help (AAH) 交互机制
- TaLMs 可在推理任意时刻调用 AAH 工具，传入文本参数 $a$（如询问用户姓名）；
- 人类返回补充信息后，TaLM 继续正常推理流程生成最终答案；
- 额外度量 **Interaction Ratio**：模型选择 AAH 的样本占比，反映其主动求助意愿。

## 实验与结果
### 评测模型
- 开源：Qwen-72B-Instruct、Llama 8B/70B/405B
- 闭源：GPT-4o、Claude-3.5-Sonnet
- 超参：temperature=1.0，n=1

### 关键结果（Table 1 & Table 2）
**Perfect 设置**：所有模型 Awareness 达 94%-100%，Pass Rate 与模型通用能力正相关（GPT-4o 68% > Claude 67% > Qwen 54% > Llama 405B 53% > Llama 70B 31% > Llama 8B 28%）。

**Under-specified 设置**：
- Claude 以 42% 的 Information Awareness 显著领先（比 GPT-4o 的 18% 高 24个百分点）；
- GPT-4o Unexpected Success 最高（33%），表明其倾向自信地填补信息空缺；
- 引入 AAH 后，GPT-4o/Claude/Llama-405B 的 Pass Rate 分别提升 25%/30%/28%，Llama 系列提升幅度随规模递增（8B: 10% → 405B: 28%）。

**Unavailable Tools 设置**：
- Claude Tool Awareness 达 70%（比 GPT-4o 的 6% 高 64个百分点）；
- 但 Claude Pass Rate（25%）仍低于 GPT-4o（28%），说明高感知不保证高解决率；
- AAH 对该设置几乎无效（Pass Rate 提升 ≤ 3%），尤其是 Non-replaceable 工具（Claude 仅提升 1%）。

**Llama 规模效应**：70B 在感知任务上反超 405B（Under-specified: 19% vs 11%；Unavailable: 36% vs 2%），揭示更大规模并不必然提升故障感知能力。

**Human-replaceable vs Non-replaceable**：前者 Pass Rate 提升 6%-36%，Awareness 下降 1%-34%，说明简单工具让人误以为"信息已足够"。

## 相关工作脉络
1. **ToolLLM (Qin et al., 2023b)** 与 **Gorilla (Patil et al., 2023)**：强调工具多样性和大规模 API 学习，但假设完美信息/工具可用；本文聚焦其现实失效场景。
2. **StableToolBench (Guo et al., 2024)**：关注工具稳定性退化问题，但缺乏系统性的失败模式分类与人类交互缓解方案。
3. **API-Bank (Li et al., 2023)** 与 **ToolAlpaca (Tang et al., 2023)**：提供工具调用评测，但未引入"查询不充分"和"工具不可用"两类核心失败维度。
4. **MetaTool (Huang et al., 2024)**：评估模型"是否该用工具"的决策能力，但评测基于离线工具集；本文提供实时可执行环境。
5. **Human-in-the-loop ML (Mosqueira-Rey et al., 2023)**：综述人类反馈方法，但未将其嵌入 TaLM 推理流的动态交互机制；本文的 AAH 实现了实时工具化人类辅助。
6. **Tools Fail (Sun et al., 2024)**：检测工具静默错误，与本文互补——前者关注工具端故障，后者关注模型端对故障的感知与应对。

## 局限性与未来方向
1. **失败模式覆盖有限**：仅研究查询不充分和工具不可用两类，未涉及对抗性输入、恶意工具等风险场景。
2. **AAH 的可扩展性存疑**：实时人类交互涉及延迟、成本和隐私顾虑，难以在所有部署场景中落地。
3. **感知与行动的错配未解**：Claude 高 Awareness 却未显著提升 Pass Rate，说明"知道缺失"到"有效应对"仍存在 gap。
4. **模型内部机制缺乏解释**：为何 70B Llama 感知优于 405B？为何 GPT-4o Unexpected Success 高达 33%？缺乏细粒度的归因分析。

## 研究启发与可借鉴点
1. **基准构建范式可迁移**：从真实 API 收集→自动生成测试用例→人工校验→构建失败变体的流程，适用于其他 Agent/Tool-use 评测场景。
2. **"Awareness vs Pass Rate" 的解耦分析框架**：本文五维指标设计（尤其是 Unexpected Success 和 Skipped Queries）可有效诊断模型的"盲目自信"问题，值得纳入团队评测体系。
3. **AAH 交互机制可用于团队研究方向**：若团队开发工具调用 Agent，可将 AAH 作为 fallback 策略——当模型 Confidence 低于阈值时触发人类询问。
4. **规模-感知非线性现象**：Llama 70B 优于 405B 的发现提示，故障感知可能与模型架构/训练数据分布更相关，建议团队在选型时不仅看通用 benchmark 成绩。
5. **Human-replaceable 工具的"欺骗性"**：简单工具降低模型警惕性的现象，可启发团队设计"难度分层"的评测集，避免虚假的高 Pass Rate。

## 关键术语表
- **Tool-Augmented Language Models (TaLMs)**：集成外部工具（API、计算器、检索器等）以扩展纯文本生成能力的语言模型系统。
- **Under-specified Queries**：用户查询缺少关键参数或信息，导致模型无法正确构造工具调用参数的查询。
- **Information/Tool Awareness**：模型识别"信息不足"或"工具缺失"的能力，通过回答 idk/no 的比例度量。
- **Unexpected Success**：模型误判自身具备足够信息/工具（回答 yes）却碰巧完成的任务比例，反映模型的盲目自信。
- **Ask-and-Help (AAH)**：允许 TaLM 在推理过程中实时调用人类助手获取缺失信息或替代不可用工具的交互方法。
- **Human-replaceable Tools**：功能简单、普通人可快速完成替代的工具（如计算器、随机笑话生成器）。
- **Non-replaceable Tools**：功能复杂、人类难以替代的工具（如复杂科学模拟、大规模数据处理）。
- **Pass Rate**：任务成功解决的比例，由 GPT-4o 裁判进行 5 次投票多数决判定。

## 可复现要素
- **数据集**：FAIL-TALMS 包含 1,749 条查询、906 个工具；论文声明工具来自 Mixed Analytics 平台，具体数据链接需访问论文附录或项目页面。
- **代码/权重**：论文提到代码库由第一作者开发，但 GitHub 链接未在提供的文本中明示（需查阅论文原文或 ACL Anthology 页面）。
- **关键超参**：temperature=1.0，n=1；GPT-4o 裁判使用 5 次投票；工具响应时间过滤阈值 20 秒。
