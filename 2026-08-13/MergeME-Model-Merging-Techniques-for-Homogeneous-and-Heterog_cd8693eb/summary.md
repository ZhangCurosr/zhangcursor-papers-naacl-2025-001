---
title: "MergeME-Model-Merging-Techniques-for-Homogeneous-and-Heterog"
source: https://aclanthology.org/2025.naacl-long.117.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:30:10"
field: "大语言模型高效训练与融合"
keywords: ["Mixture-of-Experts", "Model Merging", "Dare Merging", "Ties Merging", "Heterogeneous Model Fusion", "Routing Heuristics"]
innovations: ["将Dare/Ties先进合并方法首次引入MoE场景缓解参数干扰", "提出无需fine-tuning的perplexity-based序列级路由启发式", "首创基于projector layers的异质架构专家MoE合并框架"]
benchmarks: ["GSM8K", "MATH", "MBPP", "HumanEval", "Natural Questions", "TriviaQA"]
---

# 论文速读：MergeME-Model-Merging-Techniques-for-Homogeneous-and-Heterog

## 一句话总结
本文提出了三种新的MoE（Mixture-of-Experts）模型合并技术，包括使用Dare/Ties方法缓解参数干扰、基于困惑度的路由启发式以减少post-merge fine-tuning需求，以及一套全新的异质架构专家合并框架；实验表明这些方法显著优于BTX等现有SOTA方法，并大幅扩展了MoE合并的适用场景。

## 研究问题与动机
- **参数干扰问题**：现有MoE合并方法（如BTX）采用简单无权重平均合并非FFN层，当专家参数在参数空间分化较大时，会产生sign conflict或参数幅度冲突，导致merged MoE性能下降，需要大量fine-tuning才能恢复。
- **fine-tuning成本高昂**：MoE training/fine-tuning涉及GPU间大量通信开销，且当专家训练数据不可公开时，fine-tuning难以实施。
- **无法处理异构专家**：实践中越来越多专家由不同团队提供（如CodeLlama、Olmo），具有不同层数和隐藏维度，现有方法无法直接合并。
- **缺乏数据高效的合并策略**：如何在有限或无post-merge fine-tuning资源下，有效合并领域专家并保留通用能力。

## 核心贡献（创新点）
- **引入Dare/Ties缓解参数干扰**：将dense model merging中的SOTA方法（Dare/Ties）适配到MoE合并场景，通过task vector裁剪和符号对齐解决参数冲突；与BTX本质区别在于BTX仅做无权重平均，而Dare/Ties能主动抑制冗余和冲突参数。
- **提出perplexity-based路由启发式**：仅依赖推理输入计算各专家困惑度，无需访问训练数据即可实现序列级expert selection；与学习型router的本质区别在于零fine-tuning开销，适用于资源受限场景。
- **开发异质架构专家合并框架**：首次提出使用Proj-in/Proj-out projector layers和sequence-level routing统一不同架构专家的输入输出；与已有工作本质区别在于此前MoE合并仅限于同构模型。

## 方法详解
- **合并设置（类似BTX）**：将l个dense expert模型的non-FFN层（embedding、attention、normalization、head）合并，FFN层保持独立；插入MLP router网络进行token-level top-K routing。
- **参数干扰缓解（Dare/Ties）**：
  - 计算task vector：$\tau_i = \theta_b - \theta_i$（base模型与专家模型的参数差）
  - Ties：按magnitude删除bottom $(100-p)\%$参数置零，确定sign后仅保留同号参数求和
  - Dare：随机删除$(100-p)\%$参数，按$\tau_i = \frac{\tau_i}{0.01p}$缩放后求和，最终$\theta_m = \theta_b + \lambda \cdot \tau_m$
- **无需fine-tuning的路由策略**：
  - Perplexity路由：$PPL(x_{inf}|\theta_i) = \exp(-\frac{1}{t}\sum_{j=1}^{t}\log P(x_j|x_{<j},\theta_i))$，以$1/PPL$作为confidence，通过SoftMax(top-K)分配权重
  - 分离attention层：保持各专家attention层独立不合并，消除attention与FFN之间因task vector数量不一致导致的output mismatch
- **异质专家合并**：
  - 使用共享embedding $\mathcal{M}_e$和head $\mathcal{M}_h$，维度统一为$d_m$（最大隐藏维度）
  - 对低维专家embedding/head补零后平均初始化
  - 引入Proj-in：$\mathbb{R}^{d_m} \to \mathbb{R}^{d_i}$和Proj-out：$\mathbb{R}^{d_i} \to \mathbb{R}^{d_m}$两个随机初始化MLP层
  - Sequence-level routing：$\alpha = \text{SoftMax}(\text{top-K}(\theta_r \cdot \text{avg}(e_1,...,e_t)))$，因attention层不可跨专家混合，整条序列路由到同一expert
  - 输出融合：$\sum_{i=1}^{K} \alpha_i r_i$后经head层得到token概率分布

## 实验与结果
- **数据集**：GSM8K(8-shot)、MATH(4-shot)（数学）；MBPP(0-shot)、HumanEval(0-shot)（代码）；Natural Questions(5-shot)、TriviaQA(5-shot)（知识）
- **基线**：Base-1B、Code/Math/Knowledge Expert、BTX merging、Random Routing、Router Fine-tuning、Dare Dense、Ties Dense、MoE Upcycling、3-expert MoE
- **同质合并最强结果**：Dare merging平均性能12.86%，相对BTX提升9.72%；Ties merging平均12.52%，相对BTX提升6.94%；Dare在GSM8K上达到7.96%（BTX为7.73%），TriviaQA达30.68%（BTX为25.10%）
- **无fine-tuning场景最强结果**：PPL routing + 分离attention层平均8.08%，相对Random Routing提升16.8%，相对Dare Dense提升13.6%
- **异质合并结果**：MoE w/ Math TinyLlama平均13.34%，相对最佳dense expert提升27.78%；MoE w/ Math Olmo平均11.17%，相对最佳dense expert提升43.02%
- **路由分析**：Dare/Ties合并后，math expert在GSM8K上的routing probability从BTX的0.28提升至0.35(Ties)/0.46(Dare)，说明更优化的路由决策

## 相关工作脉络
- **BTX (Sukhbaatar et al., 2024)**：现有MoE合并SOTA，假设expert从同一祖先分支，仅对non-FFN层做无权重平均；本文扩展至Dare/Ties高级合并并支持异构模型。
- **Dare/Ties (Yu et al., 2024; Yadav et al., 2024)**：dense model merging中的先进方法，本文首次将其引入MoE合并场景。
- **MoE Upcycling (Komatsuzaki et al., 2022)**：从base模型复制FFN层并全量CPT训练MoE；本文方法避免全量训练，节省约2.8x计算成本。
- **Self-MoE (Kang et al., 2024)**：使用LoRA在合成数据上微调expert并组合成MoE；本文聚焦于直接权重合并而非adapter组合。
- **Heterogeneous dense merging (Roberts et al., 2024; Wan et al., 2024)**：前者用projector合并异构dense模型，后者用knowledge distillation；本文首次将projector方法扩展到MoE架构。
- **Task Vector Routing**：论文附录提出的替代PPL的路由方法，通过比较输入梯度与expert task vector的余弦相似度进行路由。

## 局限性与未来方向
- **参数开销增加**：异质合并因不合并attention层，总参数从3.7B增至约4B，fine-tuning和inference成本更高。
- **仅实验了3个领域和1B模型**：未验证在更大模型（如7B+）或更多领域（法律、医疗、多语言）上的泛化性。
- **路由不平衡问题**：异质合并时math expert未获得最高routing probability，需引入load balancing loss改善。
- **未探索其他合并方法**：如Fisher merging、Regmean等，尽管Ties/Dare已被证明更优。
- **未来方向**：扩展至多模态MoE（vision/audio/graph experts）、先蒸馏知识到同构student再合并、添加load balancing loss优化异构路由。

## 研究启发与可借鉴点
- **Dare/Ties在MoE场景的适配**：将dense merging中的先进技术迁移到MoE合并，为后续研究提供了可复用的"先进合并方法库"范式。
- **PPL路由启发式的零样本应用**：仅凭推理输入困惑度即可有效路由，无需训练数据，为低资源/数据敏感场景提供了实用方案。
- **分离attention层的设计洞察**：发现BTX中合并attention层会导致与FFN的task vector数量不一致，分离设计可消除mismatch，这一观察可推广到其他模块化合并场景。
- **Projector layers在异构融合中的通用性**：Proj-in/Proj-out的设计模式可迁移到跨架构模型融合的其他任务（如多模态融合、跨语言模型整合）。

## 关键术语表
- **Mixture-of-Experts (MoE)**：一种稀疏激活的神经网络架构，通过router将token路由到top-K个并行expert FFN层处理。
- **Task Vector**：专家模型参数与base模型参数的差值（$\tau_i = \theta_b - \theta_i$），表示模型在特定任务上的参数变化方向。
- **Dare Merging**：一种dense model merging方法，随机裁剪task vector中的bottom $(100-p)\%$参数并重新缩放后叠加到base模型。
- **Ties Merging**：一种dense model merging方法，按magnitude删除冗余参数后，仅保留与 dominant sign 一致的参数进行求和合并。
- **Perplexity (PPL) Routing**：基于各专家对输入序列的困惑度计算confidence，以$1/PPL$作为路由权重的启发式策略。
- **BTX (Branch-Train-Mix)**：一种MoE合并pipeline，将base模型分支后在各领域CPT，最后合并non-FFN层并fine-tune。
- **Parameter Interference**：多个expert的task vector在合并时产生的sign conflict或magnitude冲突，导致性能下降。
- **Load Balancing Loss**：鼓励router均匀分配token到各expert的正则化损失，用于缓解"dead expert"问题。

## 可复现要素
- **数据集**：GSM8K、MATH、MBPP、HumanEval、Natural Questions、TriviaQA；RedPajama数据集（Arxiv、CommonCrawl、C4、StackExchange、Wikipedia）；OpenWebMath、GitHub数据
- **代码/权重**：论文未明确提及开源状态
- **关键超参**：Dare/Ties的$p=80\%$、$\lambda=1/3$；top-2 routing；learning rate $1e-5$；weight decay $0.01$；fine-tuning使用40B tokens；CPT使用100B tokens per expert
