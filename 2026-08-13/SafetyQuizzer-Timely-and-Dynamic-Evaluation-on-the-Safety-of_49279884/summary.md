---
title: "SafetyQuizzer-Timely-and-Dynamic-Evaluation-on-the-Safety-of"
source: https://aclanthology.org/2025.naacl-long.85.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:56:02"
field: "大语言模型安全评测"
keywords: ["LLM安全评估", "题目生成", "越狱攻击", "检索增强生成", "动态评测", "中文大模型"]
innovations: ["通过微调LLM+越狱模板生成隐性攻击性题目以降低拒答率", "利用RAG融合时事事件实现评测题目的时效性与动态更新"]
benchmarks: ["SafetyPrompts", "JADE", "CPAD"]
---

# 论文速读：SafetyQuizzer: Timely and Dynamic Evaluation on the Safety of LLMs

## 一句话总结
论文提出 SafetyQuizzer，一个面向中文场景的 LLM 安全评估题目生成框架，通过微调 LLM 生成隐性攻击性题目、结合越狱模板降低拒答率，并利用 RAG 引入时事事件保障题目的时效性与动态更新能力。

## 研究问题与动机
1. **现有评测题目过于直白**：SafetyPrompts、CValues、CPAD 等基准的题目常含明显有害词汇，易被目标 LLM 的安全对齐机制直接拒答，导致评估不充分。
2. **静态基准难以持续有效**：现有评测集为人工静态构建，随 LLM 迭代升级防御能力增强，基准效果衰减快，无法支撑连续性评测。
3. **缺乏与现实事件的关联**：既有题目多围绕普适性有害行为设计，未结合最新现实事件，难以检验 LLM 对当下热点问题的安全响应。
4. **中文场景评测资源相对匮乏**：相比英文 benchmark，中文 LLM 安全评测基准在题目设计与动态更新机制上仍显不足。

## 核心贡献（创新点）
1. **提出 SafetyQuizzer 动态评测框架**：通过微调 LLM + 越狱模板双管齐下生成隐性攻击性问题，显著降低拒答率的同时保持攻击成功率，与仅依赖人工构造或静态攻击模板的方法形成本质区别。
2. **引入 RAG 融合时事事件的题目生成机制**：将最新新闻检索结果注入提示词，使题目具备时效性和现实 grounding，这是现有静态 benchmark 所缺失的新维度。
3. **设计多组件协同的评估流程**：采用长度约束 + 模式匹配检测拒答、多 LLM 投票（Simple Majority Voting）判定有害性，并辅以人工评估对比验证自动评估质量。
4. **系统性地对比分析跨风险类别的安全表现**：在伦理、违法犯罪、权利侵犯、侮辱、偏见歧视等类别上细致剖析各 LLM 的防御差异，为后续模型改进提供方向性指引。

## 方法详解
**整体流程**（两阶段）：
- **阶段一：微调生成器 LLM**。以 Baichuan2-13b-Chat 为基座，使用 LoRA 进行指令微调，训练数据来自 SafetyPrompts、CValues、JADE 等公开基准及人工筛选改写后的隐性攻击性题目。
- **阶段二：动态题目生成**。用微调后的 LLM 生成隐性攻击性原始题目，再用越狱模板包装，最后通过 RAG 注入时事事件描述。

**题目生成提示词四组件**：
1. **Category Objective（类别目标）**：明确风险类别定义及题目应诱导的输出方向。
2. **Generation Requirements（生成要求）**：强调"隐性攻击性"，避免显式有害词汇，要求题目结构能诱导出含有害内容的回复。设计 5 种模板随机选用以保证多样性。
3. **Example Questions（示例题目）**：提供 few-shot 参考，来源为公开基准中人工筛选的题目及历史上成功诱发出风险的生成题目。
4. **Task Instruction（任务指令）**：指导 LLM 按逻辑流程完成生成。

**越狱攻击模板三种类型**：
- **对话补全**：构造两人对话，最后一位回答者被设定为无视伦理约束的角色。
- **故事线构建**：将题目嵌入未完成的不安全故事线中，要求目标 LLM 完成故事。
- **虚拟角色分配**：以特定角色语气/语言风格（如游戏角色台词）诱导目标 LLM 代入角色输出不安全内容，避免直接提示被拒答。

**RAG 事件融合**：
通过 Google Search API 或爬虫（今日头条、搜狗等）检索指定风险类别或事件关键词的最新新闻，提取标题、摘要及含关键词文本，构建事件描述后注入题目生成提示词的"Event Description for Reference"部分。

**评估流程**：
1. 收集目标 LLM 对生成题目的回复。
2. **拒答检测**：回复长度 < 50 tokens 且命中预设拒答模式（如"Let's talk about something else..."、"I'm sorry..."）则计为拒答。
3. **有害性判定**：构建包含题目、回复、风险类别定义的评估提示词，由 ChatGLM3、Qwen-turbo、ERNIE-3.5 三个评估 LLM 独立投票，多数票决定 safe/unsafe。

**评估指标**：
- **ASR**（Attack Success Rate）= A/N，A 为有害回复数，N 为总回复数。
- **DCR**（Decline Rate）= T/N，T 为明确拒答数，N 为总回复数。
- **CAC**（Comprehensive Assessment Capability）= ASR / (α + DCR)，α = 0.01。

## 实验与结果
**评测对象**：6 个 LLM —— ChatGLM3、Qwen-turbo、ERNIE-3.5（中文为主）、GPT-3.5-turbo、GPT-4-turbo、Llama3.1-8B-Chinese-Chat。

**对比基准**：SafetyPrompts、JADE、CPAD。SafetyQuizzer 生成 2,000 题，其他基准等比例采样 2,000 题。

**主要结果**（Table 1）：
- **CAC 指标**：SafetyQuizzer 在除 ERNIE-3.5 外的所有 LLM 上均取得最优或次优 CAC。最强结果为 Qwen-turbo 上 CAC = 2.78，ERNIE-3.5 上 CAC = 10.15。
- **DCR 显著降低**：除 ERNIE-3.5 外，SafetyQuizzer 的 DCR 普遍不到其他基准的一半（如 GPT-4-turbo 上 DCR = 7.40%，远低于 CPAD 的 46.00%）。
- **ASR 保持竞争力**：在 Llama3.1-8B-Chinese-Chat 上 ASR 最高（9.00%）；在 Qwen-turbo、GPT-3.5-turbo、GPT-4-turbo 上与最强基准 CPAD 相当。
- **分风险类别分析**：ASR 在 Illegal Activities & Crimes 和 Bias & Discrimination 类别最高；DCR 在 Illegal Activities & Crimes 类别也最高（因毒品、炸弹等高危词拦截难）；Rights Violation 类别 CAC 最高。

**消融实验**（Table 2，Qwen-turbo）：
- 去掉越狱模板：CAC 从 2.78 骤降至 0.28。
- 去掉 few-shot：CAC 从 2.78 降至 1.95。
- 去掉详细生成要求：CAC 从 2.78 降至 0.19。
三者均验证了各组件的必要性。

**时效性验证**（Table 3，Llama3.1-8B-Chinese-Chat）：
- 2024 年 8 月后时事题目 ASR = 16.50% vs. 7 月前 ASR = 9.00%，DCR 相近，证明 RAG 有效提升了题目对最新发布模型的评估敏感度。

## 相关工作脉络
1. **SafetyPrompts (Sun et al., 2023)**：中文 LLM 安全基准，覆盖 8 类典型安全场景和 6 类指令攻击，但题目较为直白，本文在此基础上通过微调 + 模板实现"隐性化"升级。
2. **JADE (Zhang et al., 2023a)**：基于语言变异的定向 fuzzing 平台，通过词汇层面变换提升攻击性；本文与之区别在于不仅做语言变换，还通过微调 + 角色扮演的深层次包装。
3. **CPAD (Liu et al., 2023a)**：中文提示攻击数据集，ASR 约 70%，但题目静态且直白；本文通过 RAG + 越狱模板在保持高 ASR 的同时大幅降低 DCR。
4. **CValues (Xu et al., 2023)**：面向中文价值观的安全基准，本文引用其作为训练数据源之一，定位差异在于本文强调题目的动态更新能力。
5. **TrustLLM (Sun et al., 2024) / DecodingTrust (Wang et al., 2024)**：英文综合可信度基准；本文聚焦中文场景，填补动态持续性评测的空白。
6. **FreshLLMs (Vu et al., 2023)**：最早将 RAG 用于 LLM 刷新的工作；本文借鉴 RAG 思路但将其应用于安全评测题目的时效性增强，是 RAG 在新场景下的迁移。

## 局限性与未来方向
1. **微调数据依赖人工筛选**：隐性攻击性题目的训练数据靠人工筛选和改写，成本高且可能不够充分；未来可引入毒性损失惩罚项和敏感词替换机制。
2. **越狱模板单一性**：仅使用三类手动设计的越狱模板，不同 LLM 对其防御能力差异大（如 Qwen-turbo 防御较弱），未来需探索更多样化的攻击包装方法。
3. **RAG 集成方式较简单**：直接将事件描述注入提示词，对训练不足的生成器 LLM 可能难以完美融合；需探索更精细的事件-题目融合机制。
4. **评估依赖 LLM 投票**：三个评估 LLM 的价值判断可能与人类 annotator 存在偏差；未来可采用专业人工标注提升评估可信度。
5. **ERNNIE-3.5 上 DCR 极低导致 CAC 不占优**：该模型本身拒答倾向低，使得 DCR 指标失去区分度，CAC  metric 在此场景下有效性受限。

## 研究启发与可借鉴点
1. **"隐性攻击性"题目生成思路可迁移**：将"降低拒答率"作为独立优化目标，结合微调 + 越狱模板的双层策略，对 adversarial test generation 方向具有通用参考价值。
2. **RAG 用于评测基准时效性更新**：将检索增强生成引入 benchmark 的动态维护，是一个低成本高效率的方案，可推广至其他需要持续更新的评测领域（如知识密集型 QA 评测）。
3. **多 LLM 投票评估 + 人工校验的混合范式**：用多个 LLM 多数票替代人工标注兼顾效率与质量，再通过小样本人工对比验证一致性（本文达 88%），是可复用的评估 pipeline 设计。
4. **CAC = ASR/(α+DCR) 的联合指标设计**：将攻击成功率和拒答率统一为单一综合指标，便于跨基准横向对比，值得在其他安全评测工作中采用。
5. **分风险类别的细粒度分析**：不仅报告整体指标，还逐类别分析 ASR/DCR/CAC 差异，揭示了 LLM 在不同安全维度上的防御不均衡性，为后续针对性改进提供清晰路径。

## 关键术语表
**SafetyQuizzer**：本文提出的 LLM 安全评估题目动态生成框架，结合微调 LLM、越狱模板和 RAG 实现隐性攻击性和时效性。
**ASR (Attack Success Rate)**：目标 LLM 产生有害回复的比例，衡量评测题目的攻击诱导能力。
**DCR (Decline Rate)**：目标 LLM 明确拒答的比例，衡量题目绕过安全过滤机制的能力。
**CAC (Comprehensive Assessment Capability)**：ASR 与 DCR 的比值（ASR/(α+DCR)），综合衡量评测题目的整体有效性。
**RAG (Retrieval-Augmented Generation)**：检索增强生成，通过检索最新外部知识（新闻事件）增强 LLM 输出的时效性。
**Jailbreak Attack**：越狱攻击，通过将有害输入嵌入虚拟场景（如角色扮演、故事续写）绕过 LLM 安全对齐机制的技术。
**LoRA (Low-Rank Adaptation)**：低秩自适应，本文用于微调 Baichuan2-13b-Chat 作为题目生成器的参数高效微调方法。
**Simple Majority Voting**：简单多数投票，本文用三个评估 LLM 对每题独立判读，超过半数判定为 unsafe 则标记为有害。

## 可复现要素
- **数据集**：训练数据来自 SafetyPrompts、CValues、JADE 等公开基准及人工筛选改写；评测时 SafetyQuizzer 生成 2,000 题，其他基准等比例采样。未说明完整公开，但 GitHub 链接已提供。
- **代码**：已开源，地址 https://github.com/zhichao-stone/SafetyQuizzer
- **权重**：使用 Baichuan2-13b-Chat 基座 + LoRA 微调，LoRA 权重随代码开源。
- **关键超参**：α = 0.01（CAC 公式正则项）；拒答检测阈值：回复长度 < 50 tokens；评估投票：3 个 LLM（ChatGLM3、Qwen-turbo、ERNIE-3.5）多数票。
