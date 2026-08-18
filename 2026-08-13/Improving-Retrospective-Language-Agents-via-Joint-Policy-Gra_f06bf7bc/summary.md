---
title: "Improving-Retrospective-Language-Agents-via-Joint-Policy-Gra"
source: https://aclanthology.org/2025.naacl-long.6.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:04:05"
field: "语言智能体"
keywords: ["Language Agent", "Reinforcement Learning", "Imitation Learning", "Policy Gradient", "Self-Reflection", "Open-source LLM"]
innovations: ["提出 RetroAct 框架联合优化 Planner 与 Reflector 的 IL+RL 两阶段训练", "设计带 IL 正则化的离策略联合策略梯度算法提升数据效率与训练稳定性"]
benchmarks: ["HotpotQA", "ALFWorld", "InterCode"]
---

# 论文速读：Improving-Retrospective-Language-Agents-via-Joint-Policy-Gra

## 一句话总结
本文提出 RetroAct 框架，通过模仿学习与离策略联合策略梯度强化学习的两阶段优化，协同提升开源 LLM 智能体的任务规划与自我反思能力，使较小规模开源模型在复杂任务中达到甚至超越 ChatGPT 水平。

## 研究问题与动机
- **开源小模型智能体能力不足**：当前基于提示的智能体依赖大规模闭源 LLM（如 ChatGPT），而小规模开源 LLM 作为智能体时性能与鲁棒性显著不足。
- **微调智能体缺乏持续自我改进能力**：现有微调方法（IL/SFT）仅依赖预训练阶段习得的知识，无法在测试环境中根据环境反馈进行实时自我反思与迭代优化。
- **缺乏对规划与反思能力的联合优化研究**：现有工作（如 Reflexion、Retroformer）仅单独优化 Planner 或 Reflector，尚无同时增强任务规划与失败自我反思能力的综合框架（见表 1）。
- **离线/离策略 RL 在智能体任务中面临挑战**：在线策略梯度算法样本效率低、训练不稳定，且推理成本高、延迟大，需设计适配智能体任务的离策略优化方法。

## 核心贡献（创新点）
1. **提出 RetroAct 联合优化框架**：首次将模仿学习与离策略强化学习结合，同步优化开源 LLM 的智能体任务规划（Planner）与自我反思（Reflector）能力，使智能体无需闭源模型即可实现持续学习与进化。
2. **设计带 IL 正则化的离策略联合策略梯度算法**：针对智能体任务中标准 PPO 收敛困难的问题，构建基于 Replay Buffer 的离策略优化，并引入模仿学习损失作为正则项，缓解 RL 阶段对 IL 知识的遗忘，提升数据效率与训练稳定性。
3. **在多个基准上实现开源模型的显著突破**：基于 Llama-7b/13b 的 RetroAct 在 HotpotQA、ALFWorld、InterCode 上较最佳微调基线平均提升 13.4%，Llama-7b 版本平均性能超越 ChatGPT+Reflexion 约 8%。
4. **揭示 Planner-Reflector 联合优化的协同机制**：实验证明 Planner 与 Reflector 在联合优化中存在相互促进效应；单独优化 Planner 比单独优化 Reflector 更有效，但两者联合最优。

## 方法详解
**整体框架**：RetroAct 采用两阶段流程——第一阶段为模仿学习（IL），第二阶段为强化学习（RL）联合策略梯度优化（图 2）。

**阶段一：模仿学习（IL）**
- **专家数据收集**：使用 Mixtral-8×7b 作为教师智能体（$\pi_{\text{expert}}$, $\mu_{\text{expert}}$）在训练集上生成专家轨迹，通过规则评估器筛选正例，构建 $D_{\text{planner}}^{\text{IL}}$ 与 $D_{\text{reflector}}^{\text{IL}}$（表 4 显示 HotpotQA 规划器正例 6956 条、反思器 1304 条）。
- **IL 训练目标**：最小化学生模型与专家模型在动作/反思分布上的交叉熵：
  - 规划器：$\mathcal{L}_{\text{planner}}^{\text{IL}} = \mathbb{E}_{s}[-\pi_{\text{expert}}(a|s)\log \pi_\theta(a|s)]$
  - 反思器：$\mathcal{L}_{\text{reflector}}^{\text{IL}} = \mathbb{E}_{\tau}[-\mu_{\text{expert}}(f|\tau)\log \mu_\phi(f|\tau)]$

**阶段二：离策略联合策略梯度优化（RL）**
- **奖励设计**：
  - 规划器奖励 $R_{\pi_k} = R_{\tau_k}$，直接取自环境（如 HotpotQA 用 F1 Score，ALFWorld 用 Success Rate，InterCode 用 IoU）。
  - 反思器奖励 $R_{\mu_k} = R_{\tau_{k+1}} - R_{\tau_k}$，衡量反思后下一轮尝试相比当前失败尝试的奖励增益（式 5-6）。
- **离策略策略梯度损失**（基于 PPO-Clip 改进）：
  - 构建 Replay Buffer 存储历史轨迹，使用重要性采样权重 $w_\pi, w_\mu$ 并施加 Clip 操作限制在 $[1-\epsilon, 1+\epsilon]$ 区间内，缓解分布偏移。
  - 引入梯度裁剪系数控制优化速度，避免标准 PPO 因阈值过大而浪费大量样本的问题。
- **IL 正则化联合损失**：
  - $\mathcal{L}_{\text{planner, aug}}^{\text{RL}} = \mathcal{L}_{\text{planner}}^{\text{RL}} + \lambda_\pi \mathcal{L}_{\text{planner}}^{\text{IL}}$
  - $\mathcal{L}_{\text{reflector, aug}}^{\text{RL}} = \mathcal{L}_{\text{reflector}}^{\text{RL}} + \lambda_\mu \mathcal{L}_{\text{reflector}}^{\text{IL}}$
  - 其中 $\lambda_\pi, \lambda_\mu$ 平衡 RL 目标与 IL 知识保持，实验表明 $\lambda \approx 1.0$ 时效果最佳（表 3）。

**架构设计**：
- Planner（$\pi$）：生成 Thought（显式推理过程）与 Action（工具调用/执行），状态 $s_t$ 包含任务提示、环境描述与历史交互。
- Reflector（$\mu$）：输入失败轨迹 $\tau^k$，生成自然语言反思 $f^k$，作为语义梯度信号指导 Planner 更新初始状态 $s_0^{k+1}$，实现无参数更新的迭代改进。

## 实验与结果
**实验设置**：
- 基座模型：Llama-chat-7b、Llama-chat-13b（Touvron et al., 2023）
- 训练平台：4×Nvidia A800 80GB，LoRA（$r=8, \alpha=16$），batch size=1
- 测试环境：HotpotQA（复杂推理）、ALFWorld（具身决策）、InterCode-SQL（交互式编程），各 100/134 个测试任务，每任务 10 轮试验与反思
- 评估指标：HotpotQA 用 F1 Score，ALFWorld 用 Success Rate，InterCode 用 Reward Score (IoU)；报告 Initial Reward (IR)、Final Reward (FR)、Average Reward (AR)

**主要结果（表 2）**：
- **Llama-7b + RetroAct**：HotpotQA FR 71.51（基线 SFT+RL 60.70，提升 17.8%）、ALFWorld FR 97.01（基线 80.60，提升 20.4%）、InterCode FR 54.17（基线 39.42，提升 37.4%），平均准确率 68.91，较最佳微调基线 SFT+RL（60.24）提升 14.4%。
- **Llama-13b + RetroAct**：平均准确率 68.11，较 SFT+RL（60.59）提升 12.4%。
- **对比闭源模型**：Llama-7b+RetroAct 平均性能超越 ChatGPT+Reflexion（63.34）约 8%，首次实现开源小模型在智能体任务上追上闭源大模型。
- **提升幅度最大达 348.3%**（原文摘要所述，主要在 ALFWorld 等低基线场景）。

**消融与对比实验**：
- 多 Agent（独立 Planner/Reflector）vs 单 Agent：HotpotQA 上单 Agent 几乎无性能损失；ALFWorld 与 InterCode 上单 Agent 较双 Agent 下降约 10%，归因于任务类型差异导致知识冲突。
- Planner 单独优化 > Reflector 单独优化：前者直接优化任务执行，后者仅调整提示策略，效率较低。
- RL vs IL：RL 在 HotpotQA 与 InterCode 上提升更显著（因奖励信号更丰富），ALFWorld 二值奖励限制了 RL 效果。
- IL 正则化必要性：移除正则项（$\lambda=0$）导致性能下降（表 3：IR 从 56.52 降至 52.72），验证知识保持的作用。
- 标准 PPO 对比：标准 token-level PPO 在智能体任务上易导致指令跟随能力丧失，性能低于本文设计的离策略算法（附录 D.1）。

## 相关工作脉络
1. **ReAct (Yao et al., 2023a)**：结合推理与行动的提示智能体，无微调；RetroAct 在其基础上引入 IL+RL 联合微调，使小模型获得持续自我改进能力。
2. **Reflexion (Shinn et al., 2024)**：基于提示的自我反思智能体，依赖闭源大模型；RetroAct 将反思机制与规划能力一起微调至开源小模型，降低推理成本。
3. **Retroformer (Yao et al., 2023b)**：使用策略梯度优化反思模型，但仅针对 Planner 微调，未联合优化双组件；RetroAct 同时优化 Planner 与 Reflector，并引入 IL 正则化。
4. **FireAct (Chen et al., 2023a)**：纯模仿学习微调智能体，缺乏测试时的自我反思与进化能力；RetroAct 通过 RL 阶段赋予智能体持续学习能力。
5. **ArCHer (Xi et al., 2024)**：分层多轮 RL 仅优化 Planner；RetroAct 扩展到 Planner-Reflector 联合优化，并设计适配智能体的离策略算法。
6. **Self-Refine (Madaan et al., 2024)**：通过自我反馈迭代改进输出，但无环境交互与规划能力；RetroAct 将反思嵌入 Agent 闭环交互流程，与任务规划协同优化。

## 局限性与未来方向
- **依赖环境直接奖励**：当前方法使用环境提供的粗粒度奖励（如二值成功/失败），缺乏细粒度、信息丰富的反馈信号，限制了智能体的微调潜力。
- **未来方向**：训练独立的 Reward Model 分别用于 Planner 与 Reflector，提供更精细的奖励信号，进一步提升智能体在复杂任务中的表现与适应能力（论文 Section 4 与 Section 5 明确提及）。
- **单 Agent 架构的性能损失**：虽然单模型可联合学习规划与反思，但在任务类型差异大的环境（如 ALFWorld、InterCode）中约 10% 性能下降，表明轨迹一致性对单模型融合至关重要。
- **数据量不均衡**：规划器 IL 数据量（如 HotpotQA 6956 条）显著多于反思器（1304 条），可能导致 Reflector 训练不充分（附录 C 提及）。

## 研究启发与可借鉴点
1. **IL 正则化稳定 RL 训练**：在智能体 RL 微调中引入 IL 损失作为正则项，有效缓解策略梯度优化对预训练/监督知识的遗忘，这一策略可迁移至其他 LLM 微调场景。
2. **离策略 Clip 优化替代标准 PPO**：针对智能体任务中环境奖励稀疏、标准 PPO 收敛困难的问题，采用 Replay Buffer+Clip 重要性采样+梯度裁剪的组合，更适合离线/离策略智能体训练。
3. **反思器奖励设计**：将 Reflector 奖励定义为"反思后成功尝试与失败尝试的奖励差"（$R_{\mu_k} = R_{\tau_{k+1}} - R_{\tau_k}$），为反思能力提供明确的强化学习目标，可推广至其他具有"反思/回顾"模块的 Agent 设计。
4. **多 Agent vs 单 Agent 的权衡分析**：实验揭示了任务相似性对单模型联合学习的影响，为后续研究"何时应拆分 Agent 组件、何时可融合"提供了实证依据。
5. **开源小模型智能体的可行路径**：证明通过"专家蒸馏（IL）+ 探索优化（RL）"的两阶段策略，7B 级开源模型可逼近甚至超越闭源大模型（ChatGPT）的智能体性能，为低资源场景下的 Agent 部署提供可行方案。

## 关键术语表
- **RetroAct**：本文提出的语言智能体框架，通过模仿学习与离策略联合策略梯度强化学习，同步优化 Planner 的任务规划与 Reflector 的自我反思能力。
- **Planner（规划器）**：智能体的核心决策组件，根据当前状态生成 Thought 与 Action，直接与外部工具/环境交互。
- **Reflector（反思器）**：智能体的自我改进组件，输入失败轨迹，生成自然语言反思，作为语义梯度信号指导 Planner 调整后续策略。
- **Off-policy Joint Policy Gradient（离策略联合策略梯度）**：基于 PPO-Clip 改进的算法，利用 Replay Buffer 存储历史轨迹进行离策略优化，同时更新 Planner 与 Reflector。
- **IL Regularization（模仿学习正则化）**：在 RL 损失中加入 IL 交叉熵损失项（权重 $\lambda$），防止 RL 训练过程中遗忘 IL 阶段习得的规划与反思知识。
- **Reward Shaping（奖励塑造）**：为 Planner 直接使用环境奖励 $R_{\tau}$，为 Reflector 设计差值奖励 $R_{\tau_{k+1}} - R_{\tau_k}$，量化反思的实际贡献。
- **HotpotQA / ALFWorld / InterCode**：三个实验基准，分别代表复杂多跳推理、具身文本交互决策、交互式 SQL 编程任务。
- **IR / FR / AR**：评估指标，分别为 Initial Reward（首轮尝试奖励）、Final Reward（最终成功奖励）、Average Reward（十轮平均奖励）。

## 可复现要素
- **数据集**：HotpotQA（100 测试任务）、ALFWorld（134 OOD 测试任务）、InterCode-SQL（100 测试任务）；专家训练数据已生成（见附录 Table 4），但论文未公开原始专家轨迹数据集。
- **代码**：已开源（https://anonymous.4open.science/r/RetroAct-04E8），基于 transformers 实现。
- **权重**：论文未公开微调后的 Llama-7b/13b 权重。
- **关键超参**：LoRA $r=8, \alpha=16$；IL/RL 正则权重 $\lambda_\pi = \lambda_\mu = 1.0$；反思器奖励系数 $\alpha = 1.0$；学习率候选 $\{5e{-5}, 1e{-4}, 3e{-4}\}$；epoch 3/5；batch size=1；推理温度 0.0，RL 训练温度 1.0；最大步数 HotpotQA=5、ALFWorld=50、InterCode=10。
