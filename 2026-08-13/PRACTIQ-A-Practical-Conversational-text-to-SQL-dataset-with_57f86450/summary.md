---
title: "PRACTIQ-A-Practical-Conversational-text-to-SQL-dataset-with"
source: https://aclanthology.org/2025.naacl-long.13.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:31:51"
field: "文本到SQL语义解析"
keywords: ["text-to-SQL", "ambiguous queries", "unanswerable questions", "conversational dataset", "large language models", "data generation"]
innovations: ["提出八类细粒度模糊/不可回答问题的对话式text-to-SQL数据集PRACTIQ", "设计程序化修改+LLM反向生成的三阶段数据构建框架", "建立分类+澄清SQL预测双阶段评测基线揭示模型能力差距"]
benchmarks: ["Spider", "BIRD", "CoSQL", "AmbiQT", "AMBROSIA"]
---

# 论文速读：PRACTIQ-A-Practical-Conversational-text-to-SQL-dataset-with

## 一句话总结
本文构建了首个面向实际应用的对话式text-to-SQL数据集PRACTIQ，包含2,812个覆盖模糊（ambiguous）和不可回答（unanswerable）问题的多轮对话样本，并提出基于大语言模型的分类+澄清SQL预测双阶段基线，揭示了当前SOTA模型在处理真实场景查询时的显著不足。

## 研究问题与动机
1. **现有text-to-SQL数据集缺乏模糊/不可回答查询**：Spider、BIRD等主流数据集仅包含意图清晰、可回答的问题，无法反映真实用户场景中常见的歧义或数据缺失问题。
2. **已有工作覆盖类别有限且非对话式**：AmbiQT仅覆盖4类模糊问题；CoSQL虽有对话形式，但不可回答问题占比仅约12%，且系统回复多为"抱歉无法回答"等无效回应，未提供有价值的澄清引导。
3. **模型在实际部署中暴露明显缺陷**：DIN-SQL等SOTA系统在测试时发现会幻觉出不存在的列（如Nonexistent SELECT Column类别下45%样本产生幻觉），缺乏对问题可回答性的判断能力。
4. **缺乏高质量训练数据制约开源模型能力提升**：闭源模型（如Claude 3.5 Sonnet）表现优于开源模型，但开源模型在模糊/不可回答问题上仍有约9%的性能差距，亟需针对性数据用于微调。

## 核心贡献（创新点）
1. **提出细粒度八类问题分类体系**：定义4种模糊类别（Ambiguous SELECT Column/WHERE Column/Values Within Column/Filter Criteria）和4种不可回答类别（Nonexistent SELECT/WHERE Column、Nonexistent Filter Value、Unsupported Join），相比现有工作覆盖更全面的现实场景。
2. **构建首个对话式模糊/不可回答text-to-SQL数据集**：基于Spider dev集生成2,812条四步对话（初始问题→澄清请求→用户澄清→最终SQL及结果解释），支持分类与SQL预测双任务评测。
3. **设计程序化+LLM混合的数据生成框架**：采用三阶段流程（数据库修改→SQL修改与反向生成澄清响应→对话润色），结合执行校验、LLM质量过滤与人工验证，确保数据质量（二元分类准确率93.75%，对话自然度/事实性/帮助性均值接近1分）。
4. **建立双阶段评测基线并揭示模型差距**：实现问题类别分类（9-way）和澄清SQL预测（execution accuracy）两个子任务，发现开源模型（Llama-3.1 70B）在模糊/不可回答问题上比Claude 3.5 Sonnet低3.7%~9%。

## 方法详解
**数据生成三阶段框架**：

1. **Stage 1: SQL解析与数据库修改**
   - 使用SQLGLOT自定义解析器提取SQL中的列名和单元格值
   - 针对目标类别修改数据库模式：如Ambiguous SELECT Column，用LLM生成两个语义相近但非等价的替代列替换原列；Nonexistent Filter Value则直接从数据库中删除目标单元格值

2. **Stage 2: SQL修改与澄清响应生成（反向生成策略）**
   - 先生成助手澄清回复（模板或prompt方式）
   - 采用reverse generation：先程序化修改原始SQL生成澄清后SQL，再提示LLM生成对应的用户澄清回复（而非直接让LLM生成SQL，保证SQL可执行性）
   - 执行过滤：丢弃无法执行的SQL样本

3. **Stage 3: 对话润色与质量控制**
   - 3-shot prompt改进对话自然度与连贯性，并添加执行结果的 NL 解释
   - 双重质量过滤：LLM分类验证（保留通过二元分类的样本）+ 执行校验
   - 对Ambiguous SELECT/WHERE Column还生成"直接有帮助的SQL"（返回所有可能解释），减少交互轮次

**评测任务与指标**：
- **问题类别分类**：9-way分类（含answerable），使用few-shot prompting，输入含schema及fuzzy匹配获取的相关cell values；指标为accuracy
- **澄清SQL预测**：基于DIN-SQL框架，输入为澄清后的对话上下文；指标为execution accuracy

**关键prompt设计要点**：
- 生成替代列时要求synonyms与原始列有相似lexical overlap，且为非简单变形（大小写/单复数等）
- 反向生成用户澄清回复时确保与最终SQL一致
- 分类prompt提供每类别定义+few-shot示例，含step-by-step reasoning

## 实验与结果
**数据集**：基于Spider dev集生成，共2,812条对话（1,802条模糊/不可回答 + 1,034条可回答）

**评估模型**：Claude 3.5 Sonnet、Claude 3 Sonnet、Llama-3.1 70B/8B、Mixtral-large-v2

**主要结果**：

| 任务 | 最佳模型 | 最佳指标 | 说明 |
|------|---------|---------|------|
| 问题分类（含oracle cell values） | Claude 3.5 Sonnet | 77.4% | 3-shot，75.9%排除answerable后 |
| 问题分类（仅lexical cell values） | Claude 3.5 Sonnet | 74.3% | - |
| SQL预测（模糊/不可回答平均） | Mixtral-large-v2 / Claude 3.5 Sonnet | 71.95% / 72.15% | DIN-SQL框架 |
| SQL预测（可回答Spider） | Claude 3.5 Sonnet | 79.21% | - |
| SQL预测（开源最佳Llama-3.1 70B） | Llama-3.1 70B | 67.58%（模糊/不可回答） | 较Claude 3.5 Sonnet低3.7% |

**关键发现**：
- 提供oracle cell values使三个高依赖值的类别（Ambig. Values Within Column、Ambig. WHERE Column、Ambig. Filter Criteria）分类准确率提升1.5%，说明**cell value retrieval是重要瓶颈**
- Llama-3.1 70B在提供2+ few-shot时出现重复输出导致性能暴跌（<20%）
- DIN-SQL直接在模糊/不可回答问题上测试时幻觉率极高（Nonexistent SELECT Column: 45%，Unsupported Join: 56%）

## 相关工作脉络
1. **Spider/BIRD**：标准text-to-SQL基准，仅含可回答、无歧义的单轮问题；PRACTIQ在其基础上扩展了真实场景所需的能力维度
2. **NoisySP**：引入一定噪声数据，但仅覆盖极少量不可回答类别（Nonexistent Filter Value仅2例），且非对话式
3. **AmbiQT**：定义4类模糊问题但不涵盖不可回答类别，且为非对话式；本文扩展至8类并引入对话交互
4. **AMBROSIA**：基于scope/attachment/vagueness定义模糊，未覆盖不可回答类别；本文分类体系更贴合实际数据库操作场景
5. **CoSQL/SPARC**：对话式text-to-SQL数据集，但模糊/不可回答样本极少（CoSQL仅~12%），且系统回复质量差（如"Sorry I can't answer"）
6. **Text2Analysis**：关注高级数据分析与模糊查询，但侧重分析技能而非基础text-to-SQL；本文聚焦SQL生成核心能力

## 局限性与未来方向
1. **数据规模受限**：仅基于Spider dev集生成，尚未扩展到Spider train、BIRD、WikiSQL等更大规模数据集
2. **微调验证未开展**：因时间限制，未使用生成数据微调开源模型并评估效果提升
3. **类别覆盖仍有扩展空间**：当前仅对Ambiguous SELECT/WHERE Column生成直接有帮助的SQL，其他类别尚需澄清交互
4. **生成质量可进一步提升**：承认可引入更复杂的agentic workflow改进数据质量
5. **环境成本**：大规模调用LLM API生成数据和评测带来CO2排放问题

## 研究启发与可借鉴点
1. **反向生成策略保证SQL可执行性**：先生成SQL再生成用户澄清回复，避免LLM直接生成SQL带来的语法错误，可有效复用于其他需要保证结构输出的数据生成场景
2. **程序化修改+LLM润色的混合生成范式**：数据库/SQL修改采用确定性程序操作确保正确性，对话生成利用LLM保证自然度，两者结合兼顾质量与效率
3. **执行校验作为硬性过滤手段**：所有修改后的SQL必须能成功执行，这一标准可直接迁移至其他text-to-SQL数据增强工作
4. **cell value retrieval作为分类辅助信号**：研究表明获取相关单元格值可显著提升模糊/不可回答检测精度，提示在特征工程中应重视值级信息的提取
5. **双阶段任务设计（分类+生成）**：将复杂问题拆解为先判断问题类型再执行对应策略，值得在构建类似benchmark时参考

## 关键术语表
**PRACTIQ**：本文提出的对话式text-to-SQL数据集，专门包含模糊和不可回答问题的多轮交互样本
**Ambiguous SELECT Column**：查询结果对应的列在数据库中存在多个合法解释的模糊类别
**Ambiguous Filter Criteria**：查询条件中的术语含义不明确（如"underage"需界定具体年龄阈值）的模糊类别
**Nonexistent SELECT Column**：查询所需结果列在数据库模式中根本不存在，导致不可回答
**Unsupported Join**：查询要求的表连接在数据库中外键关系不支持，无法执行
**Reverse Generation**：先生成最终SQL再反向推导用户澄清回复的数据生成策略
**Execution Accuracy**：预测SQL在数据库中实际执行后与ground truth结果的一致性指标

## 可复现要素
- **数据集**：PRACTIQ将公开（代码和数据承诺开源，MIT License）；基础数据来自Spider dev集（CC BY-SA 4.0）
- **代码**：GitHub链接在论文中提供（代码和数据生成脚本将开源）
- **模型**：数据生成使用Claude 3 Sonnet（via Amazon Bedrock）；评测使用Claude 3.5 Sonnet、Claude 3 Sonnet、Llama-3.1 70B/8B、Mixtral-large-v2
- **关键超参**：greedy decoding（top-p=1.0, temperature=0.0）；few-shot数量0-3；DIN-SQL使用与GPT-4相同的超参配置
