---
title: "DREAM-Improving-Video-Text-Retrieval-Through-Relevance-Based"
source: https://aclanthology.org/2025.naacl-long.156.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:02:56"
---

# 论文速读：DREAM-Improving-Video-Text-Retrieval-Through-Relevance-Based

## 一句话总结
本文针对视频-文本检索（VTR）基准数据中标签一对一对应关系模糊、质量受限的瓶颈，提出 DREAM 框架，利用大语言模型（LLMs）与视觉生成模型（VGMs）进行三重递进式数据增强，显著提升跨模态特征对齐能力，在 MSR-VTT、MSVD 与 ActivityNet 上刷新多项 SOTA。

## 研究问题与动机
1. **数据质量制约表征学习**：现有 VTR 工作多聚焦架构与训练策略优化，但底层瓶颈在于训练数据标注质量低、规模有限，导致模型难以学到鲁棒的跨模态表示。
2. **一一对应假设失效**：主流 benchmark（如 MSR-VTT）隐含视频与文本严格一对一匹配的假设，但实际数据存在严重语义模糊性（一个视频可匹配多条合法描述，反之亦然），造成训练信号噪声大、表示学习不稳定。
3. **重新采集高质量数据不现实**：受限于视频-文本数据天然的歧义性，构建大规模精确配对数据集成本极高，因此需从现有数据内部挖掘潜力。
4. **大模型赋予数据增强新可能**：LLMs 与 VGMs 在语义理解与内容生成上已取得突破性进展，为对现有视频/文本进行高质量、可控的语义扩充提供了可行路径。

## 核心贡献（创新点）
1. **指出 VTR 核心痛点并开辟数据增强新视角**：明确将性能瓶颈归因于基准数据的一一对应假设失效，首次系统性探索“大模型驱动的数据增强”路线，与以往纯改进网络架构的工作形成正交互补。
2. **提出 DREAM 框架与三类递进式增强策略**：设计简单增强（SA）、文本改写与视频风格化（TPVS）、相关性增强（RE）三种方法，逐步扩大相似样本间的语义细微差异，缓解标签模糊带来的表征混淆。
3. **开创性地将 LLMs/VGMs 引入 VTR 数据构建**：利用 LLaMA2/OLMo 结合提示工程生成高质量 paraphrase，利用 ControlNet 进行无文本引导的逐帧风格化，是业内首批将前沿基础模型直接用于视频-文本检索数据增强的工作之一。
4. **确立多项权威基准上的 SOTA 性能**：在 MSR-VTT、MSVD、ActivityNet 上全面超越现有方法，MSR-VTT 文本到视频 R@1 从 46.1 跃升至 60.8（+14.7），验证了数据质量对跨模态检索的决定性作用。

## 方法详解
- **整体范式**：对每个原始视频 $V$ 和文本 $T$，生成正样本视图 $\tilde{V}$ 与 $\tilde{T}$，将其与原始数据拼接后统一输入基座模型（X-CLIP）训练。不采用多查询检索，以保持与已有方法的公平对比。训练目标为对称 InfoNCE 对比损失，拉近配对特征、推远负样本。
- **简单增强（SA）**：无需任何预训练模型，通过随机重复或删除帧/子词（保持原始顺序）生成自相似数据，作为轻量级 baseline 验证扩充本身的有效性。
- **文本改写与视频风格化（TPVS）**：
  - *文本侧*：使用 LLM（LLaMA2-7b）生成语义相近的改写版本。Prompt 包含两项关键设计：① 前缀 `"This is a hard problem"` 激活模型的 zero-shot 推理潜力；② 约束 `"capturing the key information and main themes in one sentence with up to twenty words"` 防止生成冗余或偏离主题的内容。
  - *视频侧*：逐帧输入 ControlNet 进行风格迁移（如卡通化），因视频生成模型计算昂贵且时序一致性仍待提升，故采用高效的图像风格化方案替代。
- **相关性增强（RE）**：
  - *文本侧*：在 TPVS 提示基础上加入 `"feel free to add more relevant details based on your knowledge and speculation"`，鼓励模型注入潜在但语义相关的补充信息，体现“可控的不确定性”。
  - *视频侧*：使用 ControlNet 的 **“猜测模式”（guess mode，无 text guidance）**，仅依赖原始帧内容生成富含相关视觉线索的变体，避免外部文本提示引入的语义漂移。
- **损失函数**：采用对称 InfoNCE $\ell_{sim} = \ell_{v2t} + \ell_{t2
