---
title: "DAWN-ICL-Strategic-Planning-of-Problem-solving-Trajectories"
source: https://aclanthology.org/2025.naacl-long.96.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:02:21"
field: "上下文学习与规划"
keywords: ["zero-shot in-context learning", "MCTS", "demonstration selection", "planning", "large language models"]
innovations: ["首次将ZS-ICL形式化为规划问题", "提出演示感知Q值函数加速MCTS搜索", "设计校准增强聚合方法缓解token偏差"]
benchmarks: ["BBH", "MMLU"]
---

# 论文速读：DAWN-ICL-Strategic-Planning-of-Problem-solving-Trajectories

## 一句话总结
本文首次将零样本上下文学习（ZS-ICL）重新定义为规划问题，提出基于蒙特卡洛树搜索（MCTS）的DAWN-ICL方法，通过演示感知的Q值函数策略性规划问题求解顺序，显著提升ZS-ICL性能，甚至超越使用人工标注演示的few-shot方法。

## 研究问题与动机
- **现有ZS-ICL方法的随机性问题**：现有方法假设测试样本来自同一任务并以随机顺序遍历，但真实场景中样本来自多样化任务，仅少量属于同一任务，随机顺序易导致伪演示不可靠和错误累积。
- **状态空间爆炸与计算效率**：ZS-ICL的状态空间为n!（n为样本数），传统MCTS的Q值更新需要大量LLM推理来计算奖励，计算成本过高。
- **演示质量对ICL性能的关键影响**：ICL对演示选择高度敏感，如何选择和利用历史伪演示对后续预测至关重要。
- **大模型合成数据能力有限**：LLM在数据合成方面存在局限，生成的伪演示质量有限，导致ZS-ICL性能通常低于人类标注演示的ICL。

## 核心贡献（创新点）
- **首次将ZS-ICL形式化为规划问题**：提出基于MDP的框架，将问题遍历顺序优化作为规划任务，更贴近真实场景需求；与现有随机遍历方法本质不同。
- **提出演示感知的Q值函数（DQ）**：利用伪演示集的置信度和相似度信息初始化Q值，实现高效准确的估计；区别于传统仅依赖模拟回溯更新Q值的方法。
- **演示感知的MCTS算法（DUCT）**：将DQ函数整合到UCT选择策略中，增强选择阶段并加速扩展和模拟阶段；比传统UCT更具方向性指导。
- **校准增强的聚合方法**：结合多轮MCTS迭代的预测结果，并通过预训练先验概率去偏；有效缓解LLM常见token偏差问题。

## 方法详解
**问题建模**：
将ZS-ICL构建为马尔可夫决策过程(MDP)元组(S, A, T, r)，其中状态s_i包含已解决问题的集合及伪演示集D_i，动作a_i为选择下一个待解决问题，奖励函数r取模型预测的置信度。

**演示感知Q值函数(DQ)**：
$$\mathrm{DQ}(s, a) = Q_0(s, a) + w_Q \cdot Q(s, a)$$
其中初始值Q_0通过检索k_d个最相似的演示，聚合其置信度C(d_i)和相似度S(d_i, x_{i+1})计算得到，使用BGE模型计算相似度。

**选择阶段(DUCT)**：
$$\mathrm{DUCT}(s, a) = \mathrm{DQ}(s, a) + w_a \sqrt{\frac{\ln N(s)}{N(\mathrm{c}(s, a))}}$$
平衡探索（访问次数少的节点）与利用（高DQ值节点）。

**扩展阶段**：
使用DQ函数选择top-k_a个动作进行扩展，构建伪演示时首先检索k个最相似样本提高相关性，然后随机选择不同伪标签样本增强多样性。

**模拟阶段**：
采用与扩展相同的高DQ值动作选择策略，引入基于DQ阈值的缓存机制加速——当DQ_max超过阈值ε时，缓存(action, pseudo-demonstration)对，后续直接读取避免重复推理。

**校准增强聚合**：
$$y^* = \arg\max_{y_i \in \mathcal{V}} \frac{1}{N_i} \sum_{j=1}^{N_i} \frac{\mathrm{Pr}(y_i|x, d_j; \mathcal{M})}{\mathrm{Pr}(y_i|\mathcal{M})}$$
其中分母为标签的先验概率，通过所有测试样本的零样本预测平均值得到。

## 实验与结果
**数据集**：BIG-Bench Hard (BBH)、Massive Multitask Language Understanding (MMLU)，包括域内和跨域两种场景。

**基线方法**：Zero-shot、Few-shot、Self-ICL、DAIL。

**主要结果**：
- BBH域内场景：Llama3.1-8B上DAWN-ICL达到48.01（平均），超越DAIL的42.66，甚至优于few-shot的45.42；Qwen2.5-7B上达到53.20，超越few-shot的52.06。
- MMLU域内场景：Llama3.1-8B上DAWN-ICL达到63.79，超越few-shot的62.18。
- BBH跨域场景（BBH-mini）：Llama3.1-8B上达到43.48，超越DAIL随机顺序的35.33和顺序处理的39.13。
- 消融实验显示：移除校准策略性能下降最显著，验证其重要性。
- 搜索策略对比：DAWN-ICL在效率和准确率上均优于MC搜索、贪婪搜索和束搜索。

**最强结果**：在BBH上Llama3.1-8B达48.01（超越few-shot 2.59分），在MMLU上Llama3.1-8B达63.79（超越few-shot 1.61分）。

## 相关工作脉络
- **Self-ICL (Chen et al., 2023)**：独立为每个示例生成伪演示，未利用历史预测，DAWN-ICL通过规划解决顺序问题。
- **DAIL (Su et al., 2024)**：使用历史预测作为演示源但随机遍历，DAWN-ICL通过MCTS策略性选择顺序。
- **MCTS在LLM中的应用 (Yao et al., 2023; Wan et al., 2024)**：传统方法将MCTS用于推理步搜索，本文将其应用于演示选择规划。
- **演示选择研究 (Liu et al., 2022; Ye et al., 2023)**：关注演示质量评估，本文从规划角度系统化解决ZS-ICL中的演示利用问题。
- **Q值函数估计 (Coulom, 2006)**：传统MCTS通过模拟回溯更新Q值，本文引入演示信息预初始化以加速收敛。

## 局限性与未来方向
- 仅使用了MCTS一种规划算法，其他先进规划算法（如AlphaZero式方法）有待探索。
- Q值估计仍依赖模拟和回溯，计算开销较大，未来可训练价值模型进行高效评估。
- 仅在部分代表性LLM上实验，更大规模模型的泛化性需验证。
- 缓存机制虽加速但未完全消除计算负担，极端场景下效率仍有提升空间。

## 研究启发与可借鉴点
- **规划视角的创新迁移**：将ICL问题形式化为MDP并使用MCTS求解的思路，可迁移至多步推理、agent任务规划等场景。
- **演示质量评估的量化方法**：置信度+相似度双维度评估伪演示质量的设计，可作为通用演示筛选框架。
- **缓存加速策略**：基于阈值的action-cache机制，适用于任何需要重复评估相似状态的场景。
- **校准去偏技术**：先验概率除法去偏策略可有效缓解LLM token偏差，可集成到任意ICL系统中。
- **多样性引入**：在扩展阶段选择不同伪标签样本以增强演示多样性的策略，避免ICL复制偏见。

## 关键术语表
**ZS-ICL (Zero-shot In-context Learning)**：不使用人工标注演示，通过模型自生成伪演示进行上下文学习的范式。
**MCTS (Monte Carlo Tree Search)**：蒙特卡洛树搜索，通过迭代选择、扩展、模拟和回溯四个阶段进行树搜索的规划算法。
**DQ (Demonstration-aware Q-value)**：演示感知的Q值函数，利用伪演示集的置信度和相似度信息初始化和增强Q值估计。
**DUCT (Demonstration-aware UCT)**：演示感知的UCT选择策略，将DQ值整合到传统UCT公式中以指导搜索方向。
**Calibration**：校准，通过除以其先验概率来纠正LLM输出分布偏差的技术。
**MDP (Markov Decision Process)**：马尔可夫决策过程，由状态、动作、转移函数和奖励函数组成的规划框架。

## 可复现要素
- **数据集**：BBH和MMLU为标准公开基准，无需额外申请。
- **代码**：已开源，地址为 https://github.com/txy77/MCTS4ZSICL。
- **关键超参数**：MCTS迭代次数=5，k_d=30，w_Q=1，w_a=5，k_a=3，缓存阈值ε=1.5；BBH演示数=3，MMLU演示数=4。
- **模型**：Llama3.1-8B、Qwen2.5-7B、Mistral-7B-v0.3、GPT-4o-mini。
