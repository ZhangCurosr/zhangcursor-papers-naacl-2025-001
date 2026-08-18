---
title: "Large-Language-Models-Can-Solve-Real-World-Planning-Rigorous"
source: https://aclanthology.org/2025.naacl-long.176.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:28:09"
field: "LLM规划与形式化方法结合"
keywords: ["大语言模型", "约束规划", "SMT求解器", "TravelPlanner", "形式化验证", "交互式修复"]
innovations: ["LLM将自然语言规划问题翻译为SMT代码并由求解器严格求解", "基于unsat core的交互式计划修复机制", "零样本泛化到新约束和新组合优化任务"]
benchmarks: ["TravelPlanner", "UnsatChristmas"]
---

# 论文速读：Large-Language-Models-Can-Solve-Real-World-Planning-Rigorous

## 一句话总结
本文提出了一种结合大语言模型（LLM）与形式化验证工具（SMT求解器）的规划框架，将复杂多约束规划问题形式化为约束可满足性问题（CSP）并严格求解；在TravelPlanner基准上取得93.9%的最终通过率，且具备零样本泛化到新约束与新领域的强大能力。

## 研究问题与动机
- **核心问题**：LLM难以直接生成正确满足所有约束的复杂规划方案，即使具备自我验证和自我批判能力亦然。
- **现有基线不足**：TravelPlanner上最强LLM（o1-preview）直接生成计划的通过率仅10.0%；LLM-Modulo框架将通过率提升至65%但仍依赖外部批判者。
- **算法规划器的局限**：传统约束规划方法（SAT/SMT）严谨但学习曲线陡峭，且用户必须手动反复修改不可行输入。
- **LLM与算法规划器的互补性**：LLM擅长解析自然语言与交互，算法求解器严谨完整但无法处理动态、歧义的自然语言需求。

## 核心贡献（创新点）
1. **LLM+ SMT求解器的规划框架**：通过3个示例教LLM将自然语言约束翻译为步骤、再将步骤翻译为调用Z3求解器的Python代码，保证可解时必能找到计划。
2. **交互式不可满足计划修复机制**：利用SMT求解器的`get_unsat_core`提取不可满足核心，结合LLM推理给出修改建议，并支持人机交互迭代。
3. **零样本泛化到新约束与新任务**：无需额外示例即可处理未见过的约束类型，并在Block Picking、Task Allocation、TSP、Warehouse四个新领域平均达到89.0%最优率。
4. **高Prompt鲁棒性**：对 paraphrased prompts 仍保持86.7%的最终通过率，证明非过度依赖特定Prompt设计。

## 方法详解
1. **Query-Step Generation**：LLM将自然语言查询分解为形式化约束的步骤描述（如"设置`departure_dates`变量，断言第一次交通发生在第0天"），通过3个在位示例教授。
2. **Step-Code Generation**：LLM将步骤翻译为Python代码，调用外部API（CitySearch、FlightSearch等）收集数据，并使用Z3 SMT求解器（`Optimize()`）编码约束并求解。
3. **SMT Solver**：使用Z3求解器执行生成的代码；若满足则输出经过形式化验证的计划；若不满足则通过`get_unsat_core`提取导致不可满足的具体约束子集。
4. **Interactive Plan Repair**：当遇到不可满足查询时，LLM根据不可满足原因调用信息收集API（如FlightCheck、AttractionSearch），分析现状，提出最小化修改建议（如提升预算、更换目的地城市），用户可接受/拒绝/提供偏好，最多迭代20次。

## 实验与结果
- **数据集**：TravelPlanner（验证集180 query，测试集1000 query）与自建UnsatChristmas（39个含新约束的不可满足查询）。
- **评估指标**：Final Pass Rate（通过所有约束的比例）、Delivery Rate、Optimal Rate。
- **TravelPlanner结果**：
  - 基线：Greedy Search 0%，TwoStage(GPT-4) 0.6%，Direct(GPT-4) 4.4%，Direct(o1-preview) 10.0%。
  - Ours(Claude-3)：验证集93.3%，测试集93.9%；Ours(GPT-4)：验证集93.3%，测试集90.2%。
  - Ours(Mistral-Large)：验证集66.7%，测试集67.8%（仍显著优于所有基线）。
- **UnsatChristmas交互式修复**：Ours平均解决78.6%不可满足查询，Ours-20（20次迭代）达81.6%。
- **零样本泛化到新任务**（Table 2）：Block Picking最优率92%，Task Allocation 92%，TSP 100%，Warehouse 72%，平均89.0%。
- **Prompt鲁棒性**：paraphrased prompts下GPT-4达到86.7%最终通过率。

## 相关工作脉络
1. **LLM Planning**：分解任务（CoT/ToT）、生成多计划择优、反思修正、借助外部规划器——本文采用最后一种思路但将其扩展至组合优化领域。
2. **Algorithm-based Planning**：启发式搜索（FF/fast-downward）与约束规划（SAT/SMT）——本文取其严谨性，通过LLM屏蔽其用户接口复杂性。
3. **LLM Tool-use**：ReAct、Toolformer等——本文的LLM生成的是调用API+求解器的代码而非直接调用工具，实现"code as policy"。
4. **LLM-Modulo Framework (Kambhampati et al., 2024)**：同样结合LLM与外部求解器，但本文更强调端到端的自动代码生成与交互式修复，且泛化能力更强。
5. **Code as Policies (Liang et al., 2023)**：将LLM生成代码用于机器人控制——本文借鉴此思路应用于规划问题。
6. **TravelPlanner (Xie et al., 2024)**：本文主要评测基准的来源，揭示了LLM在多约束规划中的根本缺陷。

## 局限性与未来方向
- **Prompt设计耗时**：从零构建需人工编写instruction steps和对应代码示例；但可通过泛化能力减轻。
- **SMT求解器运行时**：随问题复杂度增加，求解时间可能过长（本文设置上限30分钟，仅1.3%查询超时）；未来可引入启发式优先搜索或切换至MILP求解器。
- **危险数据风险**：当前框架依赖数据库信息，无法区分不安全或错误信息，可能生成有风险的计划。
- **未来方向**：结合MILP求解器、引入更多启发式策略、扩展至更大规模数据库场景。

## 研究启发与可借鉴点
1. **"LLM生成代码驱动形式化求解器"范式**：将LLM定位为"翻译器"而非"求解器"，适用于任何可将自然语言约束编码为数学/逻辑问题的场景。
2. **三示例In-context Learning即够用**：仅需3个示例即可让LLM学习通用的步骤-代码翻译模式，大幅降低Prompt工程成本。
3. **交互式修复机制的设计**：通过`get_unsat_core`获取精确失败原因后由LLM分析并提出修改建议，比纯LLM自我批判更有效。
4. **零样本泛化验证策略**：通过添加少量约束描述行（而非新增示例）来扩展新约束类型，验证了框架的灵活性。
5. **与团队方向结合机会**：可迁移至资源调度、物流配送、任务分配等组合优化场景；可与强化学习结合实现自动Prompt优化。

## 关键术语表
**SMT (Satisfiability Modulo Theories)**：在一阶逻辑可满足性问题上加入背景理论（如线性算术、数组）的扩展，广泛用于软件验证和规划。
**Z3 Solver**：微软研究院开发的高性能SMT求解器，支持多种理论组合，本文用于严格验证计划可行性。
**Unsat Core**：求解器返回的最小不可满足约束子集，用于精确定位导致计划失败的具体约束冲突。
**TravelPlanner**：Xie等人提出的真实世界旅行规划基准，包含多维度硬约束，用于评测LLM的多约束规划能力。
**Final Pass Rate**：评估指标，指生成的计划通过所有硬约束和常识约束的比例。
**In-context Learning**：通过在Prompt中提供少量输入-输出示例，使LLM学会特定任务模式而无需微调。

## 可复现要素
- **数据集**：TravelPlanner（公开，Xie et al., 2024）；UnsatChristmas（论文中提到自建，未提及开源）。
- **代码/权重**：项目页面 https://sites.google.com/view/llm-rwplanning，论文未明确声明代码开源状态。
- **关键超参**：SMT求解器最大运行时间30分钟/查询；交互式修复最大迭代次数10次（主实验）和20次（消融）；温度=0；主要使用GPT-4、Claude 3 Opus、Mistral-Large。
- **依赖库**：Z3 Solver（Python绑定）、OpenAI API/Claude API/Mistral API。
