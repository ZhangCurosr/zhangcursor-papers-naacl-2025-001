---
title: "CORRECT-Context-and-Reference-Augmented-Reasoning-and-Prompt"
source: https://aclanthology.org/2025.naacl-long.154.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:01:27"
field: "事实验证与 misinformation detection"
keywords: ["fact-checking", "three-layer evidence graph", "prompt tuning", "context-augmented reasoning", "reference-aware verification", "few-shot fact verification"]
innovations: ["首次同时建模上下文和参考文献的三层证据图推理框架", "证据条件化连续提示编码用于事实核查", "基于图语义的提示嵌入初始化策略"]
benchmarks: ["FEVEROUS-S", "BearFact", "Check-COVID", "SciFact"]
---

# 论文速读：CORRECT-Context-and-Reference-Augmented-Reasoning-and-Prompt

## 一句话总结
本文提出 CORRECT 模型，通过构建包含证据、上下文和参考文献的三层证据图，并结合层内/跨层推理与证据条件化提示编码，将辅助上下文和参考文献整合入多证据推理，从而提升自动事实核查的准确率。

## 研究问题与动机
1. **证据句子信息不完整**：从大规模语料库检索出的证据句子脱离原文后往往缺乏自足性，需要额外的上下文（如未定义缩略词的解释）和参考文献（如被引用论文以理解共指关系）才能正确判断声明的真伪。
2. **现有方法仅利用单一辅助文档**：MultiVerS 仅建模上下文文档，Transformer-XH 和 HESM 仅建模参考文献，尚无任何工作同时聚合上下文和参考文献到统一证据嵌入中。
3. **离散自然语言提示设计困难**：ProToCo 等基于自然语言提示的方法依赖人工设计，单个词的更改即可能导致性能显著下降，且难以优化；连续可学习提示在事实核查领域尚未被探索。
4. **长文本直接输入效率低下**：多数现有方法将证据与上下文/参考文献简单拼接后输入长文本给语言模型进行编码，效率较低。

## 核心贡献（创新点）
1. **提出 CORRECT 模型，首次同时整合上下文和参考文献文档进入证据推理**：与 MultiVerS（仅上下文）和 Transformer-XH/HESM（仅参考文献）的本质区别在于构建了三层图统一建模三类文档。
2. **设计三层证据图与层内/跨层推理机制**：通过证据层、上下文层、参考文献层的图结构及层内聚合（同声明多证据间注意力）和跨层聚合，将三类信息整合为统一证据嵌入，区别于以往单层或简单拼接方式。
3. **提出证据条件化提示编码器（Evidence-Conditioned Prompt Encoder）**：将证据嵌入作为条件生成针对每个声明的独特连续提示嵌入，并与声明 token 嵌入结合用于预测，这是提示调优技术在事实核查中的首次探索，区别于 ProToCo 等基于离散自然语言提示的方法。
4. **设计基于三层图的提示初始化策略**：利用图结构中同类标签训练声明的语义相关性，以词嵌入均值初始化基础提示嵌入，优于随机初始化，避免了离散提示选择困难的问题。

## 方法详解
**三层证据图构建（Fig. 2a）**：针对每个声明 x 及其证据句子集合 N_evid(x)，构建三层图——证据层（证据句子间全连接）、上下文层（每证据与其对应上下文文档相连）、参考文献层（每证据与其多篇参考文献相连），三层间通过跨层边连接。

**层内推理（Intra-layer，Eq. 1-4）**：对每个证据句子 e，取 [CLS] token 经投影 W₁ 后，使用类型特定注意力 a_{e,e'} = softmax(LeakyReLU(b₁ᵀ[h̃_{e,CLS}‖h̃_{e',CLS}])) 聚合同声明其他证据句子的表示，再通过 mean pooling 得到聚合嵌入 ĥ_e，并引入虚拟 token 将其与原始 token 拼接。

**跨层推理（Cross-layer，Eq. 5-6）**：使用类型特定参数 W₂、b₂ 将参考文献文档的 [CLS] 嵌入聚合到证据句子上，同理处理上下文文档，得到 ĥ_r 和 ĥ_c，然后拼接为增强嵌入矩阵 H̃_e = [ĥ_c ‖ ĥ_r ‖ ĥ_e ‖ H_e]。

**非对称多头自注意力（Asymmetric MSA，Eq. 7）**：Query 来自原始证据 token 嵌入，Key/Value 来自增强嵌入（含虚拟 token），防止上下文/参考文献嵌入被证据嵌入覆盖，经 MLP 和 LayerNorm 得到下一层输出 H_e^{(l+1)}，重复 L 次后取最终 [CLS] 作为证据嵌入 h_e，再对所有证据做 mean pooling 得到声明级证据嵌入 h_E。

**证据条件化提示编码（Eq. 9-11）**：为每个标签 y ∈ {SUPPORT, REFUTE, NEI} 初始化 M 个基础提示嵌入 h_{m,y}，将证据嵌入 h_E 投影后通过 tanh 函数生成缩放和平移参数 α_x、β_x，按 π_{m,y} = h_{m,y} ⊙ (α_x + 1) + β_x 生成条件化提示嵌入，拼入声明 token 嵌入序列后输入声明编码器，得到 |V| 个声明嵌入。

**对比损失（Eq. 13）**：L = -Σ log exp(h_{x,y}ᵀh_E) / [exp(h_{x,y}ᵀh_E) + Σ_{y'≠y} exp(h_{x,y'}ᵀh_E)]，拉近正确标签的声明-证据嵌入对，推远错误标签对。

**提示初始化**：对每个标签 y，截断其训练声明的证据/上下文/参考文献为 M 个词，取词嵌入均值后在所有同标签声明上平均，作为基础提示嵌入的初始化。

## 实验与结果
**数据集**：FEVEROUS-S（通用领域，23,912/5,978 训练/测试）、BearFact（生物医学，1,158/290）、Check-COVID（COVID-19，1,275/229）、SciFact（生物医学，809/300），均提供上下文和参考文献文档（见表2）。

**评估设置**：Gold（使用金标准证据）和 Retrieved（BM25 检索 top-3 证据）两种设置，分别在 Fully Supervised 和 5-shot Few-shot 下报告 Macro/Micro F1。

**主要结果（Fully Supervised, Macro F1, Gold）**：
- CORRECT 在 BearFact 达 **59.88%**（+5.34 over P-Tuning v2 的 54.54）、Check-COVID **75.34%**（+2.31）、SciFact **83.20%**（+1.87）、FEVEROUS-S **88.41%**（+1.40 over ProgramFC 的 86.84），全面领先。
- 在 Retrieved 设置下，CORRECT 同样最强：BearFact **44.25%**、Check-COVID **80.59%**、SciFact **60.26%**、FEVEROUS-S **74.95%**。
- Transformer-XH++（本文扩展基线，同时建模上下文和参考文献）验证了双重辅助文档的有效性，但仍逊于 CORRECT，说明证据条件化提示进一步提升了性能。
- SciFact 上 MultiVerS 在 Gold 设置 Micro F1（83.68%）略优于 CORRECT（85.17% Macro 但 Micro 83.20 vs 85.17），因该数据集证据本身信息较充分。
- Few-shot（5-shot, Macro F1, Gold）：CORRECT 在 BearFact **40.91%**、Check-COVID **40.77%**、SciFact **49.12%**、FEVEROUS-S **61.00%**，全面领先；唯一稍弱于 MultiVerS 的是 SciFact（52.29 vs 49.12）。

## 相关工作脉络
1. **Multi-hop 事实验证（GEAR, KGAT, DREAM, SaGP, CausalWalk）**：聚焦证据句子内部推理，忽略辅助上下文和参考文献；本文通过三层图扩展了这一方向。
2. **上下文建模（ParagraphJoint, ARSJoint, MultiVerS）**：仅利用上下文文档，本文首次同时建模上下文和参考文献两类辅助文档。
3. **参考文献建模（Transformer-XH, HESM）**：仅利用参考文献，本文将其与上下文统一纳入三层图框架。
4. **提示式事实验证（ProToCo, ProgramFC, Varifocal）**：依赖人工设计的离散自然语言提示，本文首次将连续可学习提示嵌入引入事实核查，并通过证据条件化使提示自适应不同声明。
5. **Prompt Tuning（P-Tuning v2, Prefix-Tuning）**：通用任务的连续提示方法，本文将其适配到事实核查并设计了证据条件化机制和专用初始化策略。
6. **检索增强生成（JustiLM）**：联合证据检索与验证，本文聚焦验证阶段，依赖外部检索工具，与 MultiVerS 等设定一致。

## 局限性与未来方向
1. **数据集依赖**：模型假设证据的上下文和参考文献文档可从数据集中获取或通过标识符（如 PubMed ID、CORD ID）在线检索，若数据集中不提供此类信息则无法发挥优势。
2. **仅支持文本证据**：模型仅对文本证据句子进行推理，不适用于表格或多模态证据的事实验证。
3. **未来方向**：将三层图扩展到多模态图以支持表格/多模态证据推理（论文自述）。

## 研究启发与可借鉴点
1. **三层图结构设计**：将证据、上下文、参考文献分置不同图层并通过跨层边连接，可有效区分不同类型文档的角色，该分层图思想可迁移到其他需要多源信息整合的 NLP 任务（如问答、摘要）。
2. **非对称 MSA 机制**：Query 来自主文本、Key/Value 来自增强嵌入的设计，既利用了辅助信息又避免了主信息被覆盖，可作为通用的"主-辅信息融合"模块复用。
3. **证据条件化提示编码**：将任务相关嵌入（此处为证据嵌入）作为条件生成个性化提示，替代固定提示，对 few-shot 场景下适配不同输入分布具有通用价值，可迁移到分类、生成等任务。
4. **基于数据语义的提示初始化**：利用同类标签样本的词嵌入均值初始化提示嵌入，比随机初始化更稳健，该策略可推广到其他 prompt tuning 应用场景。
5. **与 RAG 解耦的设计**：本文专注验证阶段、依赖外部检索工具，这种模块化设计便于与先进检索器组合，适合构建可插拔的事实核查 pipeline。

## 关键术语表
**CORRECT**：Context- and Reference-augmented Reasoning and Prompting 的缩写，本文提出的事实验证模型名称。
**三层证据图**：由证据层、上下文层、参考文献层构成的异构图结构，用于统一建模事实验证所需的多源文档。
**Intra-layer reasoning**：证据层内部的推理机制，通过注意力聚合同一声明下的多个证据句子。
**Cross-layer reasoning**：跨层推理机制，将上下文和参考文献文档的信息聚合到证据句子中。
**Evidence-conditioned prompt encoder**：以证据嵌入为条件生成连续提示嵌入的模块，使提示自适应不同声明。
**Asymmetric MSA**：非对称多头自注意力，Query 来自原始证据而 Key/Value 来自增强嵌入，避免辅助信息覆盖主信息。
**Gold/Retrieved evidence**：Gold 指使用人工标注的金标准证据，Retrieved 指通过 BM25 自动检索的证据。
**SUPPORT/REFUTE/NEI**：事实验证的三类标签，分别表示证据支持、反驳或信息不足无法判定。

## 可复现要素
- **数据集**：FEVEROUS-S、BearFact、Check-COVID、SciFact；代码和数据集已公开于 https://github.com/cezhang01/correct
- **代码/权重**：代码已开源（GitHub 链接见 Abstract），预训练权重未明确说明开源方式
- **关键超参**：Transformer 层数 L=12，隐藏维度 d=768，提示嵌入数量 M=8，温度 τ=100；生物医学数据集使用 biomedical 预训练参数（BioBERT），通用数据集使用 general domain 预训练参数（BERT-base）
- **实验环境**：4× NVIDIA A100 80GB GPU，PyTorch 1.14.0，Transformers 4.43.3
