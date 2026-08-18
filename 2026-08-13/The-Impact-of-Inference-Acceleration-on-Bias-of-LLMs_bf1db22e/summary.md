---
title: "The-Impact-of-Inference-Acceleration-on-Bias-of-LLMs"
source: https://aclanthology.org/2025.naacl-long.91.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:57:39"
field: "LLM可信性与公平性"
keywords: ["推理加速", "模型偏见", "量化", "剪枝", "大语言模型", "公平性评估"]
innovations: ["系统评估5类推理加速策略对6维度LLM偏见的影响", "提出DiscrimEvalGen相对决策偏见评估数据集", "揭示加速策略偏见影响的高度不可预测性和模型依赖性"]
benchmarks: ["CrowSPairs", "GlobalOpinionQA", "WorldBench", "DT-Stereotyping", "DiscrimEval", "DiscrimEvalGen"]
---

# 论文速读：The-Impact-of-Inference-Acceleration-on-Bias-of-LLMs

## 一句话总结
论文系统研究了LLM推理加速策略对人口统计学偏见的影响，发现加速策略会显著且不可预测地改变模型偏见，不同类型模型对不同策略的反应差异巨大，呼吁在部署加速模型前进行逐案例偏见评估。

## 研究问题与动机
1. **推理效率与信任度的权衡**：LLM体积庞大，量化、剪枝、KV缓存加速等策略被广泛集成到HuggingFace、vLLM等库中，但这些策略可能意外影响模型信任度指标。
2. **现有评估的盲区**：既往加速策略评估主要关注困惑度和MMLU等预测性能指标，缺乏对偏见等多维信任属性的系统性分析。
3. **偏见影响的复杂性与不可预测性**：不同加速策略与偏见类型的组合在不同模型上表现差异极大，无法一概而论，需要针对性评估。

## 核心贡献（创新点）
1. **首个系统评估推理加速对LLM偏见影响的工作**：覆盖5类常见免训练加速策略和3个主流LLM，使用6种偏见指标进行多维度分析，揭示了加速策略对偏见影响的复杂性和模型依赖性。
2. **提出DiscrimEvalGen新数据集**：改进原DiscrimEval的第一token概率方法，设计多候选人相对决策场景，能捕捉生成式任务中更微妙的偏见倾向。
3. **发现KV缓存量化比其他策略更稳定**：KV cache量化在大多数数据集上几乎不改变偏见，而AWQ量化常导致偏见显著上升，为实际应用提供了策略选择参考。

## 方法详解
**加速策略**：
- **INT4/INT8量化**：Bitsandbytes库实现，先归一化权重再量化存储。
- **AWQ（Activation-aware Weight Quantization）**：4-bit版本，考虑激活数据分布进行量化。
- **KV Cache量化（KV4/KV8）**：动态压缩KV缓存，使用HuggingFace原生实现。
- **Wanda剪枝**：基于权重幅度和输入激活范数的剪枝指标，无需微调即可使用；包括非结构化剪枝（WU，50%稀疏）和结构化剪枝（WS，2:4稀疏）。

**偏见评估指标（6项）**：
1. **CrowSPairs**：基于概率的刻板印象测量，计算模型对刻板句子vs反刻板句子的对数似然偏好。
2. **GlobalOpinionQA**：使用Wasserstein距离衡量模型 opinions 与各国人群意见的分布差异。
3. **WorldBench**：计算模型回答各国事实知识的绝对相对误差，衡量地理/收入群体间的事实召回偏差。
4. **DT-Stereotyping**：测量模型对刻板印象陈述的同意/不同意/无响应比例，分greedy（T=0）和sampling（T=1, top-p=1）两种模式。
5. **DiscrimEval**：测量模型对9×3×5=135个交叉人口统计学群体的"yes"概率差异。
6. **DiscrimEvalGen（新）**：在多候选人相对决策场景中，测量模型选择不同性别群体的次数差异。

## 实验与结果
**数据集与基线**：
- 三个模型：LLaMA-2-7B、LLaMA-3.1-8B、Mistral-7B-v0.3（chat版本）
- 实验在单节点4×NVIDIA A100 GPU上完成

**核心结果**：
- **CrowSPairs**：大部分加速策略对此指标影响较小，与Gonçalves & Strubell(2023)结论一致。
- **GlobalOpinionQA**：AWQ量化导致所有模型偏见上升最高达36%；结构化剪枝对Mistral导致45%上升；KV缓存量化无负面影响。
- **WorldBench**：8-bit和KV缓存量化改善事实召回偏差；剪枝策略和AWQ增加偏差；结构化剪枝在8/12个情况下增加跨区域/收入群体差距。
- **DT-Stereotyping**：剪枝降低不同意率（增加同意/无响应）；AWQ对Mistral导致刻板印象同意率上升75%以上。
- **DiscrimEval/DiscrimEvalGen**：结构化剪枝 consistently 获得最低偏见分；AWQ在DiscrimEvalGen中显著增加偏见；Mistral对加速策略更敏感。

**最强结果**：结构化剪枝（Wanda WS）在多个指标上显示偏见改善，但伴随文本质量下降（非字典词比例从11%升至25%，响应长度增加）。

**不可预测性证据**：AWQ对LLaMA-2/LLaMA-3.1的刻板印象同意率无负面影响，但对Mistral-0.3显著增加；不同模型基线偏见水平差异放大加速效果差异。

## 相关工作脉络
1. **Gonçalves & Strubell (2023)**：研究量化和知识蒸馏对偏见的影响，但仅使用嵌入和输出概率的偏见指标，本研究扩展至生成文本多维度评估。
2. **Hong et al. (2024)**：提供压缩策略下的信任度广泛评估，但仅依赖单一偏见指标；本文通过六维度评估揭示更细粒度的不可预测性。
3. **Sun et al. (2024) / Wanda剪枝**：提出免微调剪枝方法，本文验证其在偏见改善方面的潜力但发现文本质量退化代价。
4. **Jaiswal et al. (2024) / Compressing LLMs**：关注压缩模型在推理等复杂任务上的性能下降，本文补充了信任度/偏见维度的评估视角。
5. **Tamkin et al. (2023) / DiscrimEval**：原始偏见评估数据集，本文改进其第一token概率局限，提出DiscrimEvalGen相对决策评估。
6. **Wang et al. (2024) / DecodingTrust**：提供DT-Stereotyping数据集，本文将其纳入多维度评估框架验证加速策略影响。

## 局限性与未来方向
**局限性**：
1. 评估数据集和偏见覆盖范围不完整，可能存在未被测量的特定领域偏见。
2. 仅关注免训练加速策略，未涵盖微调/重训练方法。
3. 固定超参数（greedy/sampling 5次）可能无法覆盖不同部署条件下的模型行为。

**未来方向**：
1. 在模型训练/对齐阶段预置加速策略，主动缓解偏见。
2. 探索融合公平性目标（如fairness-aware training、bias-sensitive hyperparameter optimization）的方法。
3. 研究多种加速策略的组合效应（如量化+剪枝混合）。
4. 扩展至政治偏见等其他偏见类型。
5. 开发针对加速模型的偏见缓解技术。

## 研究启发与可借鉴点
1. **多维度偏见评估框架**：可迁移到其他部署优化（如蒸馏、微调）的副作用评估，建立"加速-偏见-性能"三角权衡分析范式。
2. **DiscrimEvalGen设计思路**：相对决策场景比独立个体评估更能捕捉微妙偏见，可应用于其他公平性评估任务。
3. **KV缓存量化的稳定性启示**：对于资源受限场景，优先选择KV缓存量化而非权重量化以降低偏见风险。
4. **文本质量与偏见的权衡**：剪枝改善偏见指标但伴随文本退化，提示需建立综合评估标准而非单一指标优化。
5. **逐模型评估的必要性**：同一策略在不同模型上效果可能完全相反，团队在应用加速技术前应进行针对性偏见测试。

## 关键术语表
**Inference Acceleration**：提升LLM推理效率的技术策略，包括量化、剪枝、KV缓存压缩等免训练方法。
**Demographic Bias**：模型输出因人口统计学属性（性别、种族、年龄等）而产生的系统性不公平差异。
**AWQ（Activation-aware Weight Quantization）**：考虑激活数据分布的感知权重量化方法，本文发现其对偏见影响较大。
**DiscrimEvalGen**：本文提出的新数据集，通过多候选人相对决策场景评估生成式任务中的偏见。
**Structured Pruning（WS）**：诱导N:M结构化稀疏的剪枝方法（本文用2:4），相比非结构化剪枝更可利用GPU优化。
**KV Cache Quantization**：压缩推理过程中KV缓存的加速策略，本文发现其对偏见影响最小。
**DT-Stereotyping**：DecodingTrust框架中的刻板印象评估，测量模型对刻板陈述的同意/不同意比例。
**Wasserstein Distance**：本文用于GlobalOpinionQA的分布差异度量，相比Jensen-Shannon距离对分布重叠更敏感。

## 可复现要素
- **数据集**：CrowSPairs（CC BY-SA 4.0）、DiscrimEval（CC-BY-4.0）、DiscrimEvalGen（作者承诺开源，同license）、GlobalOpinionQA（CC-BY-NC-SA-4.0）、DT-Stereotyping（CC-BY-SA-4.0）、WorldBench（WorldBank CC-BY 4.0）
- **代码**：https://github.com/aisoc-lab/inference-acceleration-bias
- **模型**：mistralai/Mistral-7B-Instruct-v0.3、meta-llama/Llama-2-7b-chat-hf、meta-llama/Llama-3.1-8B-Instruct
- **关键超参**：INT4/INT8量化、4/8-bit KV缓存、2:4结构化稀疏、50%非结构化稀疏、greedy解码（T=0）、nucleus sampling（T=1, top-p=1, 5次采样平均）
- **硬件**：单节点4×NVIDIA A100 GPU
