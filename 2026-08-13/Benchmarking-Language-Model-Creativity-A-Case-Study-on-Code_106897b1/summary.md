---
title: "Benchmarking-Language-Model-Creativity-A-Case-Study-on-Code"
source: https://aclanthology.org/2025.naacl-long.141.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:59:13"
field: "大语言模型创造力评估"
keywords: ["LLM创造力", "代码生成", "收敛发散思维", "NEOGAUGE", "DENIAL PROMPTING", "NEOCODER", "Creative Benchmarking"]
innovations: ["提出DENIAL PROMPTING迭代约束方法激发LLM非常规解题策略", "定义NEOGAUGE联合度量收敛性与发散性创造力的统一指标", "构建NEOCODER数据集并提供系统性人机创造力对比分析"]
benchmarks: ["NEOCODER", "Codeforces"]
---

# 论文速读：Benchmarking-Language-Model-Creativity-A-Case-Study-on-Code

## 一句话总结
本文提出了 **DENIAL PROMPTING** 迭代约束提示方法与 **NEOGAUGE** 创造力评估指标，在 Codeforces 代码生成任务上系统评测了多种 LLM 的创造力，发现即使是 GPT-4 也尚未展现出类人创造力，且现有推理策略（MCTS、自我修正等）均无法同时提升收敛性与发散性创造力。

## 研究问题与动机
1. **创造力评估缺失**：现有 LLM 创造力工作多聚焦开放式生成（如故事写作），针对问题解决场景的创造力自动评估方法几乎空白。
2. **激发创意困难**：LLM 生成往往重复或复现训练数据，难以 eliciting 多样且创造性的解法。
3. **度量不可靠**：缺乏同时兼顾"收敛性"（正确性+约束遵循）与"发散性"（新颖性+历史对比）的量化指标。
4. **认知科学启发**：从认知科学出发，创造力包含收敛思维（目的性解决问题）与发散思维（适应新约束环境），二者缺一不可。

## 核心贡献（创新点）
1. **提出 DENIAL PROMPTING 方法**：通过迭代检测前序解法中的原子技术并施加约束，迫使 LLM 探索非常规策略；与已有工作（如 Tian et al. 2024 的单约束问题）的本质区别在于支持多轮迭代约束演化，形成随状态递增难度的测试序列。
2. **定义 NEOGAUGE 指标**：联合量化收敛性与发散性创造力；与前作（如 CreativeVal、Torrance Tests 自动化）的本质区别在于引入"人类历史解法"作为参照基线，实现 human-grounded 的新颖性度量。
3. **发布 NEOCODER 数据集**：包含 199 道最新 Codeforces 题目与 5.9K 个人类正确解法，支持多状态约束测试；与现有评测（如 HumanEval、MBPP）的本质区别在于提供历史人类解法对比与逐状态创造力追踪能力。
4. **系统性创造力基准分析**：覆盖 8 种主流/开源模型与 4 种推理策略，揭示收敛-发散权衡规律及大模型尺寸与创造力的非线性关系。

## 方法详解
**DENIAL PROMPTING（迭代约束提示）**：
- 给定初始问题 $x$ 与空约束列表 $\mathcal{C}_0 = \{\}$。
- 使用增强模型 $\mathbf{P}_{LM}$（GPT-4）生成初始解 $y_1$，再用同一模型检测其中使用的原子技术集合 $\mathcal{T}_1$（如 for loop、recursion、hashmap 等）。
- 随机采样一个未在历史约束中出现过的技术 $\tau_1 \sim \mathcal{T}_1 \setminus \mathcal{C}_0$，将其加入约束列表 $\mathcal{C}_1 = \{\tau_1\}$，得到新问题 $x \oplus \tau_1$（禁止使用该技术）。
- 重复 $t$ 次，获得包含递增难度约束的状态序列 $\mathcal{C}_t = \{\tau_1, \tau_2, \cdots, \tau_t\}$，最大迭代轮数 $T=5$。
- 关键设计：保留完整对话历史以利用上下文信息，但技术检测阶段仅关注当前轮次。

**NEOGAUGE 指标**：
- 对目标模型 $\mathbf{G}_{LM}$ 在状态 $t$ 的所有生成解集 $\mathcal{V}_t$，定义：
  - **收敛性创造力**（Eq.2）：$\text{convergent}(\mathbf{G}_{LM}, t) = \frac{1}{|\mathcal{V}_t|}\sum \mathbb{1}^{\mathcal{T}_t^i \cap \mathcal{C}_t^i = \emptyset} \times \mathbb{1}^{\text{Correct}(y_t^i)}$，要求解法既通过测试用例又未违反任何约束技术。
  - **发散性创造力**（Eq.3）：$\text{divergent}(\mathbf{G}_{LM}, t) = \frac{1}{|\mathcal{V}_t|}\sum \frac{|\mathcal{T}_t^i \setminus \widehat{\mathcal{T}}^i|}{|\mathcal{T}_t^i|}$，用模型解法中未出现在人类历史解法集 $\widehat{\mathcal{T}}^i$ 中的原子技术比例度量 H-creativity。
  - **NEOGAUGE@t**（Eq.4）：两者乘积的平均值，即 $\text{NEOGAUGE}@t = \mathbb{E}[\text{Convergent} \times \text{Divergent}]$，等价于联合概率。

## 实验与结果
- **数据集**：199 道 Codeforces 题目（difficulty 800，近期以避免训练数据泄漏），每题 30 个正确人类解法（共 5.9K），平均 4.5 个测试用例（共 2.2K）。
- **评测模型**：GPT-4、GPT-3.5、Claude 3 Sonnet、Llama3-70B、Llama2-70B、CodeLlama-34B-Python、CodeGemma-7B、Mistral-7B，temperature=1。
- **主要结果**（GPT-4，Table 3）：
  - 随状态 $t$ 增加，pass@1 从 16.1% 降至 2.1%，constraint following 从 100% 降至 14.4%，convergent creativity 从 16.2% 降至 0%。
  - divergent creativity 持续上升（4.5 → 15.3%），证明约束能激发新颖策略。
  - NEOGAUGE@0 = 1.0，NEOGAUGE@1 = 1.4，之后递减至 NEOGAUGE@4 = 0。
- **模型对比**（Figure 4）：GPT-4 在各状态均表现最优；Claude-3 与 Llama3-70B 与 GPT-4 在 t=0 接近，但 NEOGAUGE 快速降至 0。
- **人机对比**（Figure 5）：LLM 发散创造力仅略优于人类最低水平；人类在早期状态（t<3）的收敛创造力显著高于 LLM，结论"LLM 尚未展现类人创造力"。
- **推理策略评估**（Figure 6, Table 5）：
  - MCTS 唯一显著提升发散创造力，但对 NEOGAUGE 提升有限（t>2 时为零），因发散解常缺乏正确性。
  - Self-correction、Planning、Sampling 均牺牲发散能力换取收敛提升，无策略能同时优化二者，NEOGAUGE 整体无显著改善。

## 相关工作脉络
1. **Tian et al. (2024) MacGyver**：提出创造性问题解决数据集但缺乏自动评估方法；本文的 DENIAL PROMPTING 支持多轮约束迭代，并提供可量化的 NEOGAUGE 指标。
2. **Atmakuru et al. (2024) CS4**：同样使用多约束，但聚焦语言创造力（故事写作）；本文聚焦代码问题解决，引入人类历史解法作为对照基线。
3. **Zhu et al. (2024) DyVal / Xu et al. (2024a)**：动态生成挑战性问题，但评估侧重准确率而非创造力；本文明确以创造力为核心度量目标。
4. **DeLorenzo et al. (2024) CreativeVal / Zhao et al. (2024)**：基于 Torrance Tests 自动化评估四维度创造力；本文引入 P/H-creativity 框架，以原子技术层面与人类历史解法对比，更贴合问题解决场景。
5. **Chakrabarty et al. (2024a)**：评估 Torrance 测试四子成分；本文指出该测试专为人类发散思维设计，对机器创造力的适用性存疑，转而采用收敛-发散统一框架。

## 局限性与未来方向
1. **任务范围受限**：NEOGAUGE 需要历史人类解法集，仅适用于有足够人类参考的任务（如代码），难以直接迁移至开放式文本生成等任务。
2. **数据泄漏风险**：NEOCODER 基于最新 Codeforces 题目构建，未来模型可能通过预训练接触到这些数据；可通过更高难度题目或更高状态 $t$ 缓解，或定期更新题目批次。
3. **人类发散创造力评估不足**：目前人类发散创造力仅在 t=0 状态下计算（Eq.6），未来需在多状态下测量以实现公平对比。
4. **推理策略创新空间**：现有 MCTS、自我修正等策略均无法同时提升收敛与发散创造力，需探索专为创造力增强的推理方法。

## 研究启发与可借鉴点
1. **迭代约束生成范式**：DENIAL PROMPTING 的"检测-约束-再生成"循环设计可迁移至其他领域（如数学证明、算法设计），作为激发模型非常规思路的通用框架。
2. **原子技术分解评估**：将解法分解为原子技术集合再进行对比，比 sentence-level 相似度更 interpretable 且跨领域通用，适用于代码、数学公式等多种结构化输出。
3. **收敛-发散联合度量**：NEOGAUGE 的乘积形式避免了单一维度的偏差，可作为创造力评测的一般性设计原则，后续研究可探索加权或归一化变体。
4. **人机创造力对比框架**：以人类历史解法集为参照基线的思路，为评估 LLM 在专业领域（如编程、科学发现）的真实创新能力提供了可复用的方法论。

## 关键术语表
**DENIAL PROMPTING**：一种迭代约束提示方法，通过逐步禁止前序解法中的原子技术来迫使 LLM 探索新颖策略。
**NEOGAUGE**：联合量化收敛性与发散性创造力的评估指标，定义为两者乘积的期望值。
**收敛性创造力（Convergent Creativity）**：模型生成正确且遵循约束的解法的能力，体现目的性与可行性。
**发散性创造力（Divergent Creativity）**：模型生成与人类历史解法不同的新颖策略的比例，体现 H-creativity。
**H-creativity（Historical Creativity）**：指人类历史上从未出现过的原创性，通过对比模型解法与历史人类解法集合来度量。
**原子技术（Atomic Techniques）**：代码解法中的基本编程构造单元（如 for loop、recursion、hashmap、dynamic programming 等），用于细粒度解法分解。
**NEOCODER**：本文发布的数据集，包含 199 道 Codeforces 题目及其多状态约束版本，以及 5.9K 个人类正确解法。
**pass@1**：单次采样通过所有单元测试的概率，用于衡量代码生成的基础正确性。

## 可复现要素
- **数据集**：NEOCODER 已发布（论文声明 "we release NEOCODER dataset"），包含 199 道 Codeforces 问题与约束序列。
- **代码/权重**：论文未明确提供开源代码链接；使用 GPT-4 作为增强模型 $\mathbf{P}_{LM}$，目标模型通过 HuggingFace Transformers 访问。
- **关键超参**：采样温度 temperature=1；最大迭代轮数 T=5；每问题人类解法数 m=30；题目难度 800；推理策略实验中 k=5 次采样（Sampling）。
