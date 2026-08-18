---
title: "Mitigating-Hallucinations-in-Multi-modal-Large-Language-Mode"
source: https://aclanthology.org/2025.naacl-long.75.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:30:16"
field: "多模态大语言模型幻觉缓解"
keywords: ["Multi-modal Large Language Models", "Hallucination Mitigation", "Contrastive Decoding", "Image Token Attention", "Inference-time Intervention"]
innovations: ["提出iTaV（图像token注意力向量）并基于JSD度量层间差异，设计注意力引导的动态层选择策略", "首次从输出token对图像token注意力衰减角度揭示MLLM幻觉机制并用于推理阶段无训练缓解", "提出轻量级即插即用解码框架iTaD，延迟仅增加1.08-1.18倍，在四个MLLM和多个基准上取得SOTA"]
benchmarks: ["CHAIR", "POPE", "GPT-4V Assisted Evaluation", "MME", "VG-100K"]
---

# 论文速读：Mitigating-Hallucinations-in-Multi-modal-Large-Language-Mode

## 一句话总结
本文提出了一种名为**iTaD**（image Token attention-guided Decoding）的即插即用推理阶段方法，通过挖掘MLLM内部表示中输出token对图像token的注意力差异，结合层间对比解码技术，有效缓解多模态大语言模型的幻觉问题。

## 研究问题与动机
1. **核心问题**：MLLM在生成过程中容易产生语法通顺但事实错误的"幻觉"内容，严重影响其在医疗、自动驾驶等高风险场景的实际部署。
2. **现有方法不足**：早期工作依赖额外训练数据或外部知识，人力与计算成本高；近期无训练方法主要关注输入端扰动（如VCD）或输出端内部表示识别，**极少探索输入与输出token的交互关系**。
3. **关键洞察缺失**：作为MLLM区别于LLM的核心特征——图像理解能力——在现有方法的内部表示利用中被普遍忽略。
4. **核心观察**：通过实验发现，幻觉段落的输出token对图像token的注意力权重显著低于非幻觉段落（表1），这表明幻觉与注意力向图像token的衰减密切相关。

## 核心贡献（创新点）
1. **发现幻觉与图像注意力的内在关联**：首次从输出token对输入图像token的注意力衰减角度揭示了MLLM幻觉的产生机制，并提出基于此的新缓解思路。
2. **提出iTaV构建与层间距离度量**：定义了**Image Token attention Vector（iTaV）**，利用Jensen-Shannon散度（JSD）量化不同解码层间对图像理解程度的差异。
3. **设计注意力引导的动态层选择策略**：提出选择与最后一层iTaV距离最大的中间层进行对比解码，而非静态或随机选层，从而突出层间图像理解的进步。
4. **实现轻量级即插即用解码框架iTaD**：无需额外训练或数据，仅需最小推理开销（延迟增幅1.08-1.18倍）即可提升多种MLLM的幻觉缓解效果。

## 方法详解
**整体框架**：iTaD基于层间对比解码（Inter-layer Contrastive Decoding）范式，核心流程包括iTaV构建、层选择、对比解码三步。

1. **iTaV构建**：
   - 对于第n层解码步骤t，获取多头注意力（MHA）的注意力权重$\mathbf{w}_{t,h}^n$。
   - 选取每个位置j在各头中的最大注意力权重：$\hat{v}_{t,j}^n = \max_{h} w_{t,h,j}^n$。
   - 针对图像token位置区间$[I_s, I_e]$，提取对应权重并经softmax归一化，得到iTaV：
     $$\mathbf{iTaV}_t^n = \text{softmax}([\hat{v}_{t,I_s}^n, \hat{v}_{t,I_s+1}^n, \dots, \hat{v}_{t,I_e}^n])$$

2. **层间距离度量**：
   - 使用JSD度量任意两层iTaV的差异：
     $$\text{dist}(\mathbf{iTaV}_t^i, \mathbf{iTaV}_t^j) = \text{JSD}(\mathbf{iTaV}_t^i || \mathbf{iTaV}_t^j)$$

3. **动态层选择**：
   - 从候选集合$\mathcal{M}=\{2,4,6,8,10,12,14\}$中选择与最后一层N的iTaV距离最大的层M：
     $$M = \arg\max_{j \in \mathcal{M}} \text{dist}(\mathbf{iTaV}_t^j, \mathbf{iTaV}_t^N)$$

4. **层间对比解码**：
   - 将第M层和第N层的输出分布做logit差值对比，结合约束集$\mathcal{C}_{t+1}$避免假阳性：
     $$\hat{p}(x_{t+1}|x_{<t+1}) = \text{softmax}\left(\mathbb{I}(x_{t+1}) \cdot \log\frac{p_N}{p_M}\right)$$

## 实验与结果
**数据集与模型**：
- **模型**：LLaVA-1.5、InstructBLIP、MiniGPT-4、mPLUG-Owl（均约7B参数）
- **数据集**：MSCOCO（训练/调参），CHAIR、POPE、GPT-4V辅助评估、MME、VG-100K等作为评测基准

**主要结果**：

| 基准 | 指标 | iTaD最佳表现 | 对比提升 |
|------|------|-------------|---------|
| **CHAIR** (max_l=512) | $C_S \downarrow, C_I \downarrow$ | LLaVA-1.5: 45.4/13.4；MiniGPT-4: 26.4/9.6 | 全面优于Beam Search、DoLa、OPERA |
| **CHAIR** (max_l=64) | $C_S \downarrow, C_I \downarrow$ | 短文本场景下依然有效 | 一致性提升 |
| **POPE** | F1↑ | LLaVA-1.5: 85.5；MiniGPT-4: 75.5 | 所有模型一致性提升 |
| **GPT-4V评估** | C↑ (correctness) | LLaVA-1.5: 5.9；MiniGPT-4: 6.7 | 正确性显著提升，详细度基本持平 |
| **MME** | 分数↑ | MiniGPT-4: 772.3；mPLUG-Owl: 1259.7 | 优于OPERA |

**延迟开销**：iTaD推理延迟仅增加**1.08-1.18倍**（Table 8），而OPERA达**5-6倍**，iTaD在效率上优势显著。

## 相关工作脉络
1. **DoLa**（Chuang et al., 2024）：层间对比解码应用于LLM事实性提升，但仅基于token概率选层，未考虑MLLM特有的图像注意力特性。
2. **OPERA**（Huang et al., 2024）：针对MLLM的幻觉缓解，通过过信任惩罚和 retrospection-allocation机制，但推理延迟大幅增加。
3. **VCD**（Leng et al., 2024）：视觉对比解码，通过处理噪声和原始图像双倍推理成本来缓解幻觉，iTaD无需额外图像处理。
4. **Contrastive Decoding系列**（Li et al., 2023d; Shi et al., 2024）：利用概率差异优化文本生成质量，但未探索MLLM多模态注意力结构。
5. **HA-DPO**（Zhao et al., 2023）：基于训练的幻觉感知直接偏好优化，需要额外训练数据。

## 局限性与未来方向
1. **机制解释不足**：iTaD主要为经验性方法，对"注意力为何在幻觉时向图像token衰减"的底层原因（预训练/微调阶段机制）尚未深入探究。
2. **全局图像注意力**：当前方法将图像token作为一个整体处理，未细粒度分析不同图像区域在不同上下文下的注意力差异及其与幻觉的关系。
3. **仅限视觉-语言模态**：目前仅针对图像理解场景，视频、音频等多模态幻觉问题有待探索。
4. **超参数敏感性**：α参数需在小验证集上调优，不同模型的适配策略可进一步优化。

## 研究启发与可借鉴点
1. **注意力导向的层选择策略**：将内部表示的语义差异（如注意力分布）作为层选择的依据，而非仅依赖token概率，可推广至其他多模态或跨模态任务。
2. **层间对比解码的模块化设计**：iTaD可与多种基础解码策略（Beam Search、Nucleus Sampling）兼容，证明了该框架的可迁移性。
3. **消融实验的系统性**：对iTaV构成（只选图像token vs. 文本token vs. 全部token）和层选择策略（静态/随机/距离最大化/最小化）的全面消融，为方法论证提供了扎实支撑。
4. **多基准交叉验证**：同时使用CHAIR（captioning）、POPE（VQA）、GPT-4V（属性/位置幻觉）、MME（综合评测），从多维度验证方法普适性。

## 关键术语表
- **Hallucination（幻觉）**：MLLM生成语法正确但事实错误或与输入图像不一致的内容的现象。
- **iTav（Image Token Attention Vector）**：对每层解码中输出token对图像token的最大注意力权重经softmax归一化后得到的向量。
- **Inter-layer Contrastive Decoding（层间对比解码）**：利用模型不同层的输出分布进行logit差值对比，以提升生成质量的解码策略。
- **Jensen-Shannon Divergence (JSD)**：用于度量两个概率分布之间相似性的对称散度，本文用于计算层间iTaV差异。
- **CHAIR基准**：包含$C_S$（句子级幻觉率）和$C_I$（图像级幻觉率）两个指标，评估图像描述中的对象幻觉。
- **POPE基准**：通过是/否问题评估模型对图像中对象存在的判断准确性，包含random/popular/adversarial三种划分。
- **GPT-4V辅助评估**：利用GPT-4V从correctness（正确性）和detailedness（详细度）两个维度对图像描述进行评分。

## 可复现要素
- **数据集**：MSCOCO（COCO validation set随机采样）、CHAIR、POPE、GPT-4V评估、MME、VG-100K（均公开可用）
- **代码/权重**：论文未提供开源代码仓库链接，模型权重来自官方发布
- **关键超参**：
  - 候选层集合$\mathcal{M} = \{2, 4, 6, 8, 10, 12, 14\}$
  - α值：LLaVA-1.5=0.03, InstructBLIP=0.05, MiniGPT-4=0.05, mPLUG-Owl=0.7
  - Nucleus Sampling的p=0.9，Beam Search的beam数=5
  - 模型温度=1（POPE/MME实验）
