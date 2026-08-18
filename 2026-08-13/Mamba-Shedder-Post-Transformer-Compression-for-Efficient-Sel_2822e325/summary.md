---
title: "Mamba-Shedder-Post-Transformer-Compression-for-Efficient-Sel"
source: https://aclanthology.org/2025.naacl-long.195.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:30:01"
field: "高效序列建模与模型压缩"
keywords: ["Mamba", "Selective SSM", "Model Pruning", "Structured State Space Model", "Model Compression", "Efficient Inference", "Hybrid Architecture"]
innovations: ["提出首个针对选择性SSM的结构化剪枝框架Mamba-Shedder，支持Mamba块/SSM模块/Transformer子组件的多粒度剪枝", "揭示Mamba-1与Mamba-2在块级剪枝和SSM模块剪枝上的互补敏感性", "设计训练-free剪枝+轻量恢复微调流程，在10-15%剪枝率下实现最高1.4x推理加速"]
benchmarks: ["Lambada", "HellaSwag", "PIQA", "ARC-e", "ARC-c", "WinoGrande", "OBQA"]
---

# 论文速读：Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models

## 一句话总结
本文提出了 Mamba-Shedder，一种针对选择性状态空间模型（SSM，如 Mamba 及其混合架构）的训练-free 结构化剪枝方法，通过移除冗余的 Mamba 块、SSM 模块（S6/SSD）、Transformer 子组件及 MLP 通道组，在几乎不损失精度的前提下实现推理加速。实验表明，在约 10-15% 剪枝率下可获得 1.1-1.4× 的解码加速，配合两阶段恢复微调后精度可恢复至与稠密模型相当水平。

## 研究问题与动机
- **Transformer 的扩展性瓶颈**：Transformer 在序列建模上取得巨大成功，但训练成本随序列长度呈二次增长，生成阶段需存储大量 KV Cache，导致推理效率受限。
- **SSM 类模型的压缩研究空白**：选择性 SSM（如 Mamba/Mamba-2）作为 Transformer 的高效替代方案，其结构化冗余性尚不清楚；现有压缩工作集中于 Transformer，缺乏针对 SSM 的剪枝研究。
- **多粒度剪枝策略缺失**：现有 BlockPruner 等方法仅针对 Transformer 块级或通道级剪枝，未探索 SSM 内部模块（如 S6/SSD）与混合架构中 Transformer 子组件的协同剪枝。
- **架构差异导致剪枝敏感性不同**：Mamba-1 与 Mamba-2 在结构设计上存在显著差异，两者对块级剪枝和 SSM 模块剪枝的容忍度呈现相反趋势，需系统性评估。

## 核心贡献（创新点）
- **提出 Mamba-Shedder，首个针对选择性 SSM 的结构化剪枝框架**：与已有 Transformer 剪枝方法（如 BlockPruner）的本质区别在于其目标对象为 SSM 模块（S6/SSD）、混合架构中的 Mamba-Transformer 双组件，并支持多粒度联合剪枝。
- **揭示 Mamba-1 与 Mamba-2 在剪枝敏感度上的互补特性**：Mamba-1 对移除整个 Mamba 块更具鲁棒性，而 Mamba-2 对移除 SSM 模块表现出更强容忍度，为不同架构选择剪枝策略提供指导。
- **设计多粒度协同剪枝搜索空间**：从块级（Mamba 块/Transformer 块）、子组件级（MHA/MLP）、通道级（MLP channel group）到 SSM 模块级，系统性探索不同剪枝组合对精度-效率权衡的影响。
- **验证训练-free 剪枝 + 轻量微调的实用可行性**：仅需两个 epoch 的 Alpaca 校准数据微调即可将剪枝模型的 perplexity 和平均准确率恢复至接近原始稠密模型水平。

## 方法详解
- **整体流程**：Mamba-Shedder 采用训练-free 的迭代重要性评估策略，基于校准数据集（Alpaca）计算各候选结构的"重要性分数"，逐步移除最不重要的结构。
- **块级/模块级剪枝（Algorithm 1）**：给定候选结构集合、校准数据集 C 和评估指标 ϕ（perplexity），对每个结构 Mᵢ 计算重要性 Sᵢ = Importance(Mᵢ, m, C, ϕ)，选择 Sᵢ 最小的结构移除，重复 t 步。
- **MLP 通道剪枝（Algorithm 2）**：基于通道组大小 g，每次迭代评估每个 MLP 块中相邻 g 个通道的去除影响，移除重要性最低的通道组，实现宽度维度的结构化剪枝。
- **多阶段剪枝策略**："&"表示在同一迭代中联合评估多个目标，"+"表示分阶段串行剪枝（如先剪 Mamba 块再剪 SSM 模块），允许更灵活地探索不同粒度。
- **恢复微调**：剪枝完成后，在清理后的 Alpaca 数据集上进行两个 epoch 的 post-training，以恢复因结构移除导致的性能下降。

## 实验与结果
- **模型与数据集**：使用 Mamba-2.8B、Mamba2-2.7B、Zamba2-2.7B、Falcon-Mamba-7B、Hymba-1.5B-Base 五个开源模型；评测基于 lm-eval-harness 的 Lambada（PPL）、HellaSwag、PIQA、ARC-e、ARC-c、WinoGrande、OBQA 共七个任务。
- **Mamba 块剪枝**：Mamba-2.8B 在 20.86% 剪枝率下，Lambada PPL 从 4.23 升至 7.51（+3.28），平均准确率从 59.9 降至 53.8（-6.1），仍表现稳健；Mamba2-2.7B 和 Zamba2-2.7B 对块级剪枝更敏感，精度下降更明显。
- **SSM 模块剪枝**：Mamba2-2.7B 在 24/64 SSM 被剪枝时，PPL 仅从 4.10 升至 14.95，平均准确率从 60.2 降至 55.5（-4.7）；Zamba2-2.7B 在 24/54 SSM 剪枝时 PPL 为 5.46，平均准确率 64.7（-2.5），显示混合架构的强鲁棒性。
- **多粒度协同剪枝**：在 Zamba2-2.7B 上，联合剪枝 Mamba 块、MLP、MHA 及 MLP 通道（10.27% 比率）后，再移除 18 个 SSM 模块，PPL 为 5.18，平均准确率 65.9；经微调后 PPL 降至 4.58，准确率提升至 67.0（仅-0.2）。
- **推理加速**：Mamba-2.8B 剪枝 14 个块后解码加速达 1.29×；Mamba2-2.7B 剪枝 24 个 SSM 后预填充加速 1.20×、解码加速 1.18×；Zamba2-2.7B 多粒度剪枝后解码加速最高达 1.39×；整体最高实现 1.4× 推理加速。

## 相关工作脉络
- **BlockPruner（Zhong et al., 2024）**：针对 Transformer 的全局重要性度量块级剪枝方法，本文将其思路扩展至 SSM 和混合架构。
- **ShortGPT（Men et al., 2024）**：发现 Transformer 层存在大量冗余，通过移除低重要性层实现压缩；本文研究与之互补，但聚焦于 Mamba 类架构。
- **LLM-Pruner（Ma et al., 2023）**：早期结构化剪枝 Transformer 的研究，本文在此基础上引入 SSM 模块级剪枝维度。
- **Mamba-1/Mamba-2 架构**：Gu & Dao（2023, 2024）提出的选择性 SSM，本文首次系统研究其冗余性。
- **Zamba/Hymba 混合模型**：Glorioso et al.（2024）和 Dong et al.（2024）分别提出的 Mamba-Transformer 混合架构，本文验证了剪枝在这些新架构上的有效性。

## 局限性与未来方向
- **仅评估训练-free 剪枝**：未深入探索在线剪枝或与量化/蒸馏等技术的联合优化。
- **未覆盖大规模下游任务**：评测集中于 NLU 基准，缺乏对长文本生成、代码生成等任务的验证。
- **搜索空间有限**：当前仅考察固定粒度的剪枝组合，未探索更复杂的拓扑结构搜索。
- **硬件加速依赖特定推理引擎**：速度提升主要在 Tesla V100 上验证，未测试最新硬件（如 H100/A100）或专用 SSM 推理内核。
- **未来方向**：探索动态剪枝、跨模型迁移剪枝策略、与 MoE 架构结合、以及在更大规模模型（>7B）上验证。

## 研究启发与可借鉴点
- **训练-free 结构化剪枝框架可直接迁移**：基于校准数据集的重要性评估范式适用于任意序列模型架构，只需适配对应的模块定义。
- **多粒度协同剪枝策略设计值得借鉴**：从粗到细的分阶段剪枝（块→子组件→通道→SSM模块）可有效平衡精度与效率，可作为其他模型压缩工作的参考模板。
- **SSM 模块剪枝的发现具有启发性**：Mamba-2 对 SSM 模块剪枝的鲁棒性提示其内部存在冗余计算路径，可进一步探索模块内参数稀疏化。
- **恢复微调仅需少量数据**：两个 epoch 的 Alpaca 校准即可显著恢复精度，说明剪枝引入的误差可通过轻量微调快速修正，降低了实际部署成本。
- **可结合本团队方向**：若团队关注 SSM 模型压缩或高效推理，可在此基础上探索与低秩适配（LoRA）、量化（INT4/INT8）的联合压缩方法。

## 关键术语表
- **Selective State Space Model (SSM)**：一类时间可变的状态空间模型，其参数依赖于输入，典型代表为 Mamba。
- **S6 / SSD**：Mamba-1 的 SSM 模块（S6）和 Mamba-2 的改进版（SSD，State Space Duality）。
- **Structured Pruning**：移除模型中完整结构组件（如层、头、通道组）而非单个权重的剪枝方式。
- **Training-free Pruning**：无需重新训练即可执行的结构化剪枝，依赖校准数据和重要性度量。
- **Recovery Tuning**：剪枝后在少量数据上进行轻量微调，以恢复模型性能。
- **Hybrid Model**：同时包含 Transformer 和 SSM 模块的混合架构，如 Zamba、Hymba。
- **lm-eval-harness**：OpenAI 开源的大语言模型零样本评测框架。

## 可复现要素
- **数据集**：Alpaca（校准数据集，公开）；评测基准均为公开数据集（Lambada、HellaSwag 等，通过 lm-eval-harness 访问）。
- **代码/权重**：论文未明确声明代码开源；模型权重来自官方发布（Mamba、Mamba-2、Zamba2、Falcon-Mamba、Hymba）。
- **关键超参**：校准样本数 256，MLP 通道组大小 g=1024，MLP 通道剪枝步数 20，恢复微调 2 epoch，评测 prompt 长度 512，生成 token 数 16/256。
