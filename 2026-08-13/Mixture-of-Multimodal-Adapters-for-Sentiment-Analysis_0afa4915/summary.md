---
title: "Mixture-of-Multimodal-Adapters-for-Sentiment-Analysis"
source: https://aclanthology.org/2025.naacl-long.90.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:30:53"
field: "多模态情感分析"
keywords: ["多模态情感分析", "Mixture of Experts", "参数高效微调", "适配器", "选择性融合"]
innovations: ["将MoE结构扩展为多模态混合专家（MOME），实现选择性多模态特征融合", "设计插件式轻量适配器专家，可训练参数仅为SOTA方法的1/22", "引入模态内负载均衡损失，避免跨模态强制均衡削弱选择能力"]
benchmarks: ["MOSI", "MOSEI"]
---

# 论文速读：Mixture-of-Multimodal-Adapters-for-Sentiment-Analysis

## 一句话总结
本文提出MMA（Mixture of Multimodal Adapters），将MoE结构扩展至多模态情感分析任务，通过多模态注意力路由器选择性融合视频和音频中的情感特征，以极少的可训练参数（仅6M，为SOTA方法AcFormer的1/22）在MOSI和MOSEI数据集上达到最优性能。

## 研究问题与动机
1. **现有MSA方法忽略多模态噪声**：视频和音频数据中存在大量与情感无关的背景噪声（如背景动作、环境噪音），传统融合方法会将其一并纳入，阻碍情感判断。
2. **全量微调预训练模型参数开销巨大**：将PLM适配到MSA任务需要训练大量参数，资源消耗大且效率低下。
3. **缺乏针对MSA的轻量级选择性融合机制**：现有方法（如TFN、LMF、MulT、MISA、AcFormer等）侧重于全面融合多模态特征，未充分考虑MSA任务中"哪些模态在何时有用"的选择性问题。

## 核心贡献（创新点）
1. **将MoE结构扩展为多模态混合专家（MOME）**：设计多模态注意力路由器，在每个文本token时刻动态选择最合适的专家进行融合，与传统的全面融合方法本质不同，实现了选择性多模态特征集成。
2. **提出插件式轻量适配器作为专家**：用适配器（Adapter）充当各模态专家，通过下投影-激活-上投影的低秩结构捕获情感特征，仅训练极少参数（6M vs AcFormer的130M），与全量微调方法形成鲜明对比。
3. **引入模态内负载均衡损失（Intra-modality Load Balancing Loss）**：在同一模态内的专家间强制负载均衡，而非跨模态统一均衡，避免抑制路由器对不同模态有用性的差异化选择能力。
4. **验证方法在LLM上的通用性**：将MMA应用于LLaMA2-7B，在MOSI和MOSEI上全面超越ConFEDE和AcFormer，证明插件式设计的跨模型适用性。

## 方法详解

**整体框架**：MMA插件式嵌入到L层Transformer的每个block中，冻结PLM预训练权重，仅训练适配器参数和预测头。

**特征提取**：
- 文本：PLM的Embedding层 $\mathbf{T} = \Phi_t(\mathcal{D}_t)$
- 视频/音频：分别使用FACET和COVAREP工具提取 $\mathbf{V} = \Phi_v(\mathcal{D}_v), \mathbf{A} = \Phi_a(\mathcal{D}_a)$

**1D卷积与跨注意力融合**（无参数）：
$$\hat{\mathbf{V}}^l = \text{Conv1D}(\mathbf{V}), \quad \hat{\mathbf{A}}^l = \text{Conv1D}(\mathbf{A})$$
$$\mathbf{X}_v^l = \text{softmax}\left(\frac{\mathbf{T}^l \hat{\mathbf{V}}^{l\top}}{\sqrt{d_t}}\right)\hat{\mathbf{V}}^l + \mathbf{T}^l$$
$$\mathbf{X}_a^l = \text{softmax}\left(\frac{\mathbf{T}^l \hat{\mathbf{A}}^{l\top}}{\sqrt{d_t}}\right)\hat{\mathbf{A}}^l + \mathbf{T}^l$$

**MOME模块**（核心创新）：
- **多模态专家**：每个模态有$N$个适配器专家，第$m$模态第$n$个专家的输出：
$$\mathbf{H}_{m,i}^{l,n} = \delta(\mathbf{D}_m^{l,n}\mathbf{X}_{m,i}^l + b_{m,\text{down}}^{l,n}), \quad \mathbf{E}_{m,i}^{l,n} = s_m^{l,n}(\mathbf{U}_m^{l,n}\mathbf{H}_{m,i}^{l,n} + b_{m,\text{up}}^{l,n})$$
- **多模态注意力路由器**：将三种模态特征拼接后通过无参数Self-Attention得到$\mathbf{G}_i^l$，再经线性层计算gating向量$\mathbf{g}_i^l \in \mathbb{R}^{3 \times N}$，通过Top-K离散路由选择$K$个专家进行加权融合：
$$\mathbf{C}_i^l = \frac{\alpha}{r}\sum_{j=1}^{K}\text{softmax}(\hat{\mathbf{g}}_i^l)_j \hat{\mathbf{E}}_{i,j}^l$$
- **Transformer块输出**：$\mathbf{B}_{\text{out}}^l = \text{FFN}(\mathbf{X}_t^l) + \text{MOME}(\mathbf{X}_t^l, \mathbf{X}_v^l, \mathbf{X}_a^l)$

**负载均衡损失**（模态内）：
$$\mathcal{L}_{lb,m}^l = N\sum_{n \in \mathbb{I}_m} f_m^{l,n} P_m^{l,n}, \quad \mathcal{L}_{lb} = \frac{1}{L}\sum_{l=1}^{L}(\mathcal{L}_{lb,t}^l + \mathcal{L}_{lb,v}^l + \mathcal{L}_{lb,a}^l)$$

**最终损失**：
$$\mathcal{L} = \text{MAE}(\mathbf{y}, \mathbf{p}) + \lambda \mathcal{L}_{lb}$$

## 实验与结果

**数据集**：MOSI（2199个clip，标注[-3,+3]）、MOSEI（23453个clip，标注[-3,+3]）

**评估指标**：MAE、Corr、ACC-7、ACC-2、F1

**主要结果**：
- **MOSI（BERT-base）**：MAE=0.693（SOTA）、Corr=0.803（SOTA）、ACC-7=46.9%（SOTA）、ACC-2=86.4%（SOTA）、F1=86.4%（SOTA）；可训练参数仅**5.7M**，为AcFormer（130.2M）的**~4.4%**
- **MOSEI（BERT-base）**：MAE=0.529、Corr=0.766、ACC-7=55.2%（SOTA）、ACC-2=85.7%、F1=85.7%；可训练参数**8.1M**
- **LLaMA2-7B上**：MOSI MAE=0.536（SOTA，超越ConFEDE的0.569和AcFormer的0.612）；MOSEI MAE=0.471（SOTA）；可训练参数81.2M vs AcFormer的141.6M

**消融结论**：
- 三模态专家组合效果最佳
- 最优超参：$r=32, N=2, K=3$
- 多模态注意力路由器 > 线性路由器 > 无路由器
- 模态内负载均衡损失 > 统一负载均衡损失 > 无损失

## 相关工作脉络

1. **TFN / LMF / MulT**：传统多模态融合方法，侧重于全面的跨模态交互建模，但未考虑多模态噪声问题；MMA通过选择性专家路由实现噪声过滤。
2. **MISA / Self-MM**：强调多模态表示学习，学习模态不变/特有序子空间；MMA不依赖子空间分解，而是通过MoE选择性融合。
3. **ConFEDE**：引入对比学习增强表示；MMA与其定位不同，MMA关注的是"如何选择性融合"而非"如何学习更好表示"，且参数效率更高。
4. **AcFormer**：当前SOTA方法，使用紧凑Transformer和pivot attention融合；MMA参数仅为AcFormer的1/22，且通过MoE实现更精细的选择性融合。
5. **Switch Transformer / GShard**：传统NLP中的MoE架构；MMA将MoE扩展到多模态场景，设计了跨模态注意力路由器，与单模态MoE本质不同。
6. **Adapter / LoRA**：参数高效微调方法；MMA借鉴其思想，但将其嵌入MoE框架实现多模态选择融合，是PEFT在MSA领域的创新应用。

## 局限性与未来方向

1. **视频/音频表征受限于特征提取工具**：使用FACET和COVAREP提取视觉/声学特征，可能限制了整体性能上限；作者计划未来引入预训练视频/音频模型实现端到端网络。
2. **未考虑模态缺失场景**：当前方法假设文本、视频、音频三模态均存在，对缺失模态的鲁棒性不足，实际应用场景可能面临模态不完整的问题。
3. **专家数量与负载均衡设计的潜在限制**：Top-K路由和模态内负载均衡可能在某些场景下限制了跨模态特征交互的灵活性。

## 研究启发与可借鉴点

1. **MoE结构迁移到多模态任务的新思路**：将稀疏激活的MoE思想用于多模态特征选择，可启发其他多模态任务（如多模态情感理解、多模态对话）中的噪声过滤与选择性融合设计。
2. **模态内负载均衡损失的设计**：仅在相同模态专家间施加负载均衡，而非跨模态统一均衡，这一设计更合理且效果更好，可推广到其他多模态MoE场景。
3. **插件式轻量适配器的跨模型通用性**：MMA在BERT和LLaMA2上均有效验证，证明该设计具有良好的模型无关性，可作为通用多模态适配模块使用。
4. **可复用的实验配置**：$r=32, N=2, K=3, \alpha=32, \lambda=0.01$等超参设定可作为后续研究的基线参考。

## 关键术语表

**MMA（Mixture of Multimodal Adapters）**：本文提出的多模态适配器混合方法，将MoE结构嵌入PLM以实现轻量级多模态情感分析。

**MOME（Mixture of Multimodal Experts）**：MMA的核心模块，由多模态适配器和多模态注意力路由器组成，实现选择性多模态特征融合。

**多模态注意力路由器（Multimodal Attention Router）**：通过无参数Self-Attention计算gating向量，动态决定每个文本token时刻应激活哪些多模态专家。

**Intra-modality Load Balancing Loss**：仅在相同模态内的专家之间施加负载均衡损失，避免跨模态强制均衡削弱路由器的选择能力。

**PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调，仅训练少量额外参数即可适配大规模预训练模型。

**Top-K Discrete Routing**：从所有专家中选择gating值最大的K个进行加权输出，实现稀疏激活。

**COVAREP**：开源语音特征提取工具包，用于从音频中提取声学特征。

**FACET**：开源面部行为分析工具包，用于从视频中提取视觉特征。

## 可复现要素

- **数据集**：MOSI和MOSEI（公开数据集）
- **代码**：已开源，https://github.com/MMA4MSA/MMA
- **预训练模型**：BERT-base-uncased、LLaMA2-7B-base
- **关键超参**：$N=2$（每模态专家数）、$\alpha=32$（adapter缩放因子）、$\lambda=0.01$（负载均衡损失权重）、LoRA rank $r=32$
- **训练细节**：MOSI/MOSEI（BERT）batch size=128，学习率{1e-3, 2e-4}，优化器{AdamW, Adam}；LLaMA2实验batch size=16/4，学习率{1e-4, 1e-5}，优化器AdamW；共训练25个epoch
