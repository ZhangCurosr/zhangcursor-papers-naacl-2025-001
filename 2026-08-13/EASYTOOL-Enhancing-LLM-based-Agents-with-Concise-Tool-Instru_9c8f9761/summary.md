---
title: "EASYTOOL-Enhancing-LLM-based-Agents-with-Concise-Tool-Instru"
source: https://aclanthology.org/2025.naacl-long.44.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:00:21"
---

# 论文速读：EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction

## 一句话总结
论文提出 EASYTOOL 框架，将来源各异、冗长嘈杂的工具文档精炼为统一、结构化的精简指令（含功能描述与场景化参数示例），在不修改模型权重的前提下即插即用，显著降低 Token 消耗并大幅提升 LLM 智能体在工具检索、选择与参数执行阶段的准确率。

## 研究问题与动机
1. **工具文档质量缺陷阻碍 LLM 调用**：现实中工具文档格式高度不一致、充斥无关信息（URL、ID 等），且普遍缺乏使用场景与参数示例，导致 LLM 难以准确理解工具功能与调用规范。
2. **直接拼接原始文档面临上下文与噪声双重瓶颈**：工具文档平均长度高达数千 Token（如 ToolBench 约 2,530），超出模型有效上下文窗口，且冗余内容干扰工具检索与选择，引发工具名幻觉与参数错误。
3. **现有微调方案泛化性差且依赖额外数据**：Toolformer、ToolLLaMA 等方法需大量工具调用合成数据进行微调，无法直接赋能闭源黑盒模型，也缺乏灵活接入新工具的即插即用能力。
4. **通用提示压缩方法不适配工具领域**：如 LLMLingua 等压缩技术主要针对纯文本提示，容易误删工具调用必需的函数签名或参数名，反而损害模型的工具理解能力。

## 核心贡献（创新点）
1. **首次系统剖析并量化工具文档的三大缺陷（不一致性、冗余性、不完整性）**，为工具调用优化确立了明确的文档治理方向；与已有工作多聚焦于检索策略或模型微调的本质区别在于，本文率先从上游输入材料的质量重构切入。
2. **提出两阶段工具指令生成框架（Tool Description Generation + Tool Functionality Guidelines Construction）**，利用强指令模型自动提炼功能并生成结构化说明；与现有提示增强方法的本质区别在于，不依赖人工编写或简单截断，而是通过 LLM 完成语义重述与格式统一。
3. **创新性地引入场景化参数示例生成机制**，为每个工具函数构造包含具体 Scenario 与合法 Parameter 值的使用示例，并经真实执行校验；与单纯罗列参数类型的已有方法相比，本文直接针对 LLM 最易出错的参数传递环节进行专项优化。
4. **在 ToolBench、RestBench 与 FuncQA 三个跨领域基准上实现广泛验证**，最大降幅达 97.35% 的 Token 消耗且多项指标刷新 SOTA；与依赖额外训练数据的微调路线相比，本文完全免训练、免适配，可直接复用至任意具备指令跟随能力的开源/闭源模型。

## 方法详解
EASYTOOL 整体分为两个流水线阶段，均由强指令模型（ChatGPT/GPT-4）驱动，输出统一 JSON 格式的结构化工具指令：

1. **Tool Description Generation（工具描述生成）**
   - 输入：原始工具文档（含参数列表、URL、调用示例等混杂内容）。
   - 处理：要求模型剔除与工具核心功能无关的信息（如域名、产品 ID、推荐语等），仅保留工具的总体用途说明，并按内置函数逐条概括其功能语义。
   - 输出结构：`{Tool_name} is a tool that can {General_Purposes}. This tool has {Number} multiple built-in functions: 1. {Function_1} is to...`

2. **Tool Functionality Guidelines Construction（工具功能指南构建）**
   - 输入：第一阶段生成的精简描述及原始参数列表（required/optional）。
   - 处理：针对每个函数提取参数清单，并要求模型生成对应的高频使用场景与合法参数取值示例。示例生成后需通过真实 API 执行验证，确保参数格式与返回值正确。
   - 输出结构：JSON 格式，包含 `required_parameters`、`optional_parameters` 与 `Example`（含 `Scenario` 与 `Parameters` 字段）。

3. **即插即用接入机制**
   - 生成的工具指令替代原始文档直接注入 Agent Prompt，全程无需微调模型参数。框架兼容 ReACT、DFSDT 等主流推理-行动范式，并在检索阶段可直接替换原有工具描述以提升向量相似度计算质量。

## 实验与结果
- **数据集**：ToolBench（I2-Category、I3-Instruction 子集）、RestBench（TMDB 子集）、FuncQA（one-hop / multi-hop）。
- **评估基线**：ReACT、DFSDT、ToolLLaMA-7B、BERT 密集检索器、Ada Embedding、LLMLingua 压缩方法。
- **Token 压缩效果**：ToolBench 单工具文档从 2,530 降至 748（-70.43%）；RestBench 从 3,881 降至 103（-97.35%）。
- **ToolBench 主结果**：GPT-4 + EASYTOOL 在 I2-Category 达到 Pass 76.5 / Win 83.8 / Success 70.0，I3-Instruction 达到 69.0 / 89.0 / 64.0，全面超越 ReACT 与 DFSDT 基线；Llama-3.1-8B 结合 EASYTOOL 平均 Success 达 48.5，反超经工具数据微调的 ToolLLaMA-7B（15.0）。
- **检索与选择提升**：替换原始描述后，BERT 检索器 NDCG@1 从 68.2 升至 73.4，Ada 从 36.8 跃升至 73.4；多候选工具下的工具选择准确率同步显著提高。
- **错误率分析**：工具名错误降至 0；参数错误大幅下降（ChatGPT 三示例条件下参数错误由 25 降至 5），幻觉评估显示输入/上下文/事实冲突幻觉均低于 1.2%。
- **跨域泛化**：RestBench 正确路径率（CP%）显著提升；FuncQA 上 ChatGPT 单步准确率由 85.0 提升至 91.66，多步由 41.17 提升至 48.53，错误率由 9.38 降至 2.34。

## 相关工作脉络
1. **工具微调范式**（Toolformer、ToolLLaMA、Gorilla）：依赖工具调用合成数据微调开放模型；EASYTOOL 与之本质不同，完全免训练且可直接应用于闭源黑盒模型。
2. **LLM 控制器范式**（HuggingGPT、RestGPT）：将原始文档拼入 Prompt 让 LLM 自主规划；EASYTOOL 针对此类方法受限于文档噪声与上下文长度的痛点，提供前置文档治理模块。
3. **零样本/提示优化**（ToolDoc、AnyTool）：尝试重写文档以支持零样本调用；本文进一步区分了“通用提示压缩”与“工具领域结构化重构”，指出后者对参数保真度的必要性。
4. **Prompt 压缩技术**（LLMLingua）：通过 gist token 去除冗余；消融实验表明其对工具文档可能误删关键函数签名，而 EASYTOOL 在保真前提下实现更高比例的语义压缩。
5. **工具检索增强**：传统方法依赖原始描述做语义匹配；本文证明经提炼的指令描述可显著提升向量检索排序质量（NDCG 提升可达 30%+）。

## 局限性与未来方向
- **输入长度限制**：当前依赖单轮 ChatGPT 处理，超长文档（超出模型输入窗口）需额外分段预处理。
- **忽略工具间依赖关系**：仅针对独立工具文档进行重构，未建模多工具组合调用时的依赖图谱与状态流转。
- **强依赖指令跟随能力**：方法仅对具备较好指令遵循能力的模型生效，弱模型或早期开源模型增益有限。
- **未来方向**：结合工具依赖图生成联动指令、针对特定垂直领域微调轻量专用模型、探索分段/层级式超长文档的自动化压缩流水线。

## 研究启发与可借鉴点
1. **上游数据清洗优于下游 Prompt 堆砌**：在 Agent 系统中，优先对 API 文档、知识库条目进行结构化提炼，往往比在设计复杂 CoT 或 Few-shot 模板上投入更多算力更有效。
2. **场景化参数示例是抑制幻觉的关键杠杆**：单纯给出参数类型列表不足以约束 LLM，构造“Scenario + 合法取值”的对偶示例能直接对齐模型的参数生成空间，值得迁移至代码生成、表格操作等同类任务。
3. **即插即用型前置模块设计范式**：EASYTOOL 不侵入模型训练与推理主干，仅作为预处理流水线存在，这种“低耦合、高收益”的设计模式非常适合快速集成到本团队现有的 Agent 框架中。
4. **多维度基准验证策略**：同时覆盖开放 API（ToolBench）、Web 服务调用（RestBench）与数值计算工具（FuncQA
